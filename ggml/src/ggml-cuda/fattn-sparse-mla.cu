// DSV4 sparse-MLA flash attention (the k4 standalone kernel wired into ggml).
// Behind DSV4_SPARSE_ATTN=1, overrides the round-3 vec route for the per-query top-512 sparse
// compressed segment. MQA: 1 KV head, HQ Q heads, D=512 latent (K==V mirrored). One CTA per
// (query token, stream) processes ALL HQ heads, gathering the selected keys ONCE (MQA amortization)
// and running bf16 WMMA QK^T/PV + flash online softmax over: the dense raw window [0,n_raw) AND the
// sparse comp segment (top_k rows at absolute k_all row n_raw + kv_idx[ord]).
//
// Option (a): gather bf16/F16 directly from the MATERIALIZED k_all (src[1]) — the standalone k3
// variant proved cos 0.999999 doing exactly this. No raw-FP8-cache plumbing. Capture-safe:
// indexed loads from k_all (no TMA tensor-map over a moving graph tensor; the standalone TMA-gather
// was a perf primitive — here the gather is plain indexed bf16, still correct + fast on the
// already-resident k_all; can be upgraded to TMA later with persistent staging).
//
// Default path (DSV4_SPARSE_ATTN unset / src[6] null) is untouched: this file's launcher is only
// reached from the new dispatch case, which only fires when both the gate and src[6] are set.

#include "common.cuh"
#include "fattn-sparse-mla.cuh"
#include <mma.h>
#include <cuda_fp16.h>
#include <map>
#include <cstdint>

#if defined(FLASH_ATTN_AVAILABLE) && (__CUDA_ARCH__ >= 1000 || !defined(__CUDA_ARCH__))
#define SPARSE_MLA_BUILD 1
#endif

#ifdef SPARSE_MLA_BUILD
using namespace nvcuda;

#define SM_HQ   64
#define SM_D    512
#define SM_KB   16          // keys per flash block
#define SM_WARPS 8
#define SM_NM   (SM_HQ/16)  // 4 head row-tiles
#define SM_NN   (SM_KB/16)  // 1 key col-tile
#define SM_ND   (SM_D/16)   // 32 D contraction tiles

// One CTA = one (query token, stream). gridDim.x = n_tokens * n_stream.
// smem: sQ[HQ*D]bf16 64KB + sK[KB*D]bf16 16KB + sP[HQ*KB]bf16 2KB + sS[HQ*KB]f32 4KB = 86KB.
extern __shared__ char sparse_mla_smem[];

__global__ void __launch_bounds__(256) sparse_mla_kernel(
        const char * __restrict__ Q,   // [D, n_tokens, n_head, n_stream] f16/f32(->we read f16)
        const char * __restrict__ K,   // [D, n_kv, 1, n_stream] bf16/f16 (k_all)
        const int  * __restrict__ kv_idx, // [top_k, n_tokens] i32 (VIEW: row stride = kv_row_elems, NOT top_k!)
        float * __restrict__ dst,      // [D, n_head, n_tokens, n_stream] f32
        const float scale, const int n_raw, const int top_k, const int n_comp_view,
        const int n_tokens, const int n_head, const int n_kv,
        const int64_t Qnb1, const int64_t Qnb2, const int64_t Qnb3, // byte strides of Q dims 1,2,3
        const int64_t Knb1, const int64_t Knb3,                     // K row stride, stream stride
        const int64_t kv_row_elems,                                 // kv_idx row stride in INT elements
        const int Ktype, const int Qtype) {                         // type: 0=f16, 1=bf16, 2=f32
    const int t=threadIdx.x, warp=t>>5, lane=t&31;
    const int gid = blockIdx.x;
    const int it  = gid % n_tokens;     // query token index
    const int is  = gid / n_tokens;     // stream index

    __nv_bfloat16* sQ=(__nv_bfloat16*)sparse_mla_smem;
    __nv_bfloat16* sK=sQ+SM_HQ*SM_D;
    __nv_bfloat16* sP=sK+SM_KB*SM_D;
    float*         sS=(float*)(sP+SM_HQ*SM_KB);
    __shared__ float m_run[SM_HQ], l_run[SM_HQ];
    __shared__ float wscratch[SM_WARPS][256];
    __shared__ int   sValid[SM_KB];   // per key-slot validity for the current flash block

    // Load this token's Q for all heads -> sQ[h*D + d]. Q layout: element (d, it, h, is) at
    // Q + d*2 + it*Qnb1 + h*Qnb2 + is*Qnb3 (f16). (Q is f16 after build_attn_mha cast.)
    const int Qesz = (Qtype==2)?4:2;
    for(int i=t; i<SM_HQ*SM_D; i+=256){ int h=i/SM_D, d=i%SM_D;
        const char* qp=Q + (size_t)d*Qesz + (size_t)it*Qnb1 + (size_t)h*Qnb2 + (size_t)is*Qnb3;
        float qv = (Qtype==2) ? *(const float*)qp
                 : (Qtype==1) ? __bfloat162float(*(const __nv_bfloat16*)qp)
                 :              __half2float(*(const half*)qp);
        sQ[i]=__float2bfloat16(qv);
    }
    for(int i=t;i<SM_HQ*SM_D;i+=256){ /* O accum in dst */ }
    if(t<SM_HQ){ m_run[t]=-INFINITY; l_run[t]=0.f; }
    // dst element (d, h, it, is): dst + (((is*n_tokens + it)*n_head + h)*D + d)
    float* Ot = dst + ((size_t)is*n_tokens + it)*n_head*SM_D; // base for [h*D + d] within this token
    for(int i=t;i<SM_HQ*SM_D;i+=256) Ot[i]=0.f;
    __syncthreads();

    const char* Kbase = K + (size_t)is*Knb3;     // this stream's K
    // kv_idx is a VIEW from ggml_argsort_top_k: ne[0]=top_k but the ROW STRIDE is the FULL argsort
    // width (n_comp_view), passed as kv_row_elems — NOT top_k. Using it*top_k read the wrong query's
    // row (and ran OFF THE BUFFER for late tokens at prefill -> illegal access). Decode (it=0) hid it.
    const int*  idx_q = kv_idx + (size_t)it*kv_row_elems; // this token's top_k list (correct stride)
    const int   total = n_raw + top_k;            // logical keys: dense raw window + sparse comp

    for(int kb0=0; kb0<total; kb0+=SM_KB){
        // gather SM_KB logical keys into sK (bf16). pos in [0,n_raw): raw row pos (always valid,
        // 0<=pos<n_raw<=n_kv). [n_raw,total): comp slot; ord=pos-n_raw; the selected comp ROW is
        // idx_q[ord]; absolute k_all row = n_raw + idx_q[ord]. At SHORT context n_comp_view < top_k,
        // so most top_k slots are PADDING (idx out of [0,n_comp_view)) -> OOB k_all read (the crash).
        // GUARD: a comp slot is valid only if 0 <= idx < n_comp_view; otherwise zero-fill + mark
        // invalid (softmax -inf), never dereferencing the OOB row. pos>=total also invalid.
        for(int r=warp; r<SM_KB; r+=SM_WARPS){
            int pos = kb0 + r;
            int krow = 0; bool valid = false;
            if(pos < total){
                if(pos < n_raw){ krow = pos; valid = true; }            // raw window: in-bounds
                else {
                    int idx = idx_q[pos - n_raw];                        // selected comp row
                    if(idx >= 0 && idx < n_comp_view){ krow = n_raw + idx; valid = true; }
                    // else: padding/OOB top_k slot -> invalid, krow stays 0 (safe, won't read OOB)
                }
            }
            // HARD SAFETY: never dereference a row outside k_all [0, n_kv). Defends against any
            // index/stride surprise (n_kv = padded comp-cache length). If out of range, drop the slot.
            if(valid && (krow < 0 || krow >= n_kv)){ valid = false; krow = 0; }
            if(lane==0) sValid[r] = valid ? 1 : 0;
            const char* kp = Kbase + (size_t)krow*Knb1;
            for(int d=lane; d<SM_D; d+=32){
                float kv = 0.f;
                if(valid){
                    if(Ktype==1) kv=__bfloat162float(((const __nv_bfloat16*)kp)[d]);
                    else         kv=__half2float(((const half*)kp)[d]);
                }
                sK[r*SM_D+d]=__float2bfloat16(kv);
            }
        }
        __syncthreads();

        // QK^T (WMMA over D)
        for(int tile=warp; tile<SM_NM*SM_NN; tile+=SM_WARPS){
            int mt=tile/SM_NN, nt=tile%SM_NN;
            wmma::fragment<wmma::accumulator,16,16,16,float> acc; wmma::fill_fragment(acc,0.f);
            for(int dt=0;dt<SM_ND;dt++){
                wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major> a;
                wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::col_major> b;
                wmma::load_matrix_sync(a,sQ+(mt*16)*SM_D+dt*16,SM_D);
                wmma::load_matrix_sync(b,sK+(nt*16)*SM_D+dt*16,SM_D);
                wmma::mma_sync(acc,a,b,acc);
            }
            wmma::store_matrix_sync(sS+(mt*16)*SM_KB+nt*16,acc,SM_KB,wmma::mem_row_major);
        }
        __syncthreads();

        // online softmax. Mask = sValid[j] (covers BOTH pos>=total AND invalid/padding comp slots
        // whose top_k index was out of [0,n_comp_view)). Invalid slots contribute -inf score -> 0 prob.
        for(int h=warp; h<SM_HQ; h+=SM_WARPS){
            float* sr=sS+h*SM_KB;
            float bmx=-INFINITY; for(int j=lane;j<SM_KB;j+=32){ float v=sValid[j]?sr[j]*scale:-INFINITY; if(v>bmx)bmx=v; }
            for(int o=16;o>0;o>>=1) bmx=fmaxf(bmx,__shfl_xor_sync(0xffffffff,bmx,o));
            float mo=m_run[h],mn=fmaxf(mo,bmx),corr=(mo==-INFINITY)?0.f:expf(mo-mn);
            float bs=0; for(int j=lane;j<SM_KB;j+=32){ float pp=sValid[j]?expf(sr[j]*scale-mn):0.f; sP[h*SM_KB+j]=__float2bfloat16(pp); bs+=pp; }
            for(int o=16;o>0;o>>=1) bs+=__shfl_xor_sync(0xffffffff,bs,o);
            if(lane==0){ m_run[h]=mn; l_run[h]=l_run[h]*corr+bs; }
            for(int d=lane;d<SM_D;d+=32) Ot[h*SM_D+d]*=corr;
        }
        __syncthreads();

        // PV (V==K)
        for(int tile=warp; tile<SM_NM*SM_ND; tile+=SM_WARPS){
            int mt=tile/SM_ND, dt=tile%SM_ND;
            wmma::fragment<wmma::accumulator,16,16,16,float> acc; wmma::fill_fragment(acc,0.f);
            for(int kt=0;kt<SM_NN;kt++){
                wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major> a;
                wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::row_major> b;
                wmma::load_matrix_sync(a,sP+(mt*16)*SM_KB+kt*16,SM_KB);
                wmma::load_matrix_sync(b,sK+(kt*16)*SM_D+dt*16,SM_D);
                wmma::mma_sync(acc,a,b,acc);
            }
            wmma::store_matrix_sync(wscratch[warp],acc,16,wmma::mem_row_major);
            for(int e=lane;e<256;e+=32){ int rr=e/16,cc=e%16; Ot[(mt*16+rr)*SM_D+dt*16+cc]+=wscratch[warp][e]; }
        }
        __syncthreads();
    }
    for(int h=warp;h<SM_HQ;h+=SM_WARPS){ float inv=(l_run[h]>0.f)?1.0f/l_run[h]:0.f; for(int d=lane;d<SM_D;d+=32) Ot[h*SM_D+d]*=inv; }
}

// ---------------------------------------------------------------------------------------------
// TMA-Gather4 variant of the K-loop (proven standalone k3_tma/k4pf_tma: cos 0.999998 at PREFILL,
// 0.999999 at SHORT). Identical bf16 WMMA QK^T/PV + online softmax; ONLY the per-key gather is
// replaced by the hardware cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4 instruction.
// D=512 bf16 = 1024B exceeds the TMA box width cap (256 elems / 512B) -> each row is gathered as
// TWO 256-wide column tiles. gather4 pulls 4 rows at a time -> KB/4 groups x 2 col-tiles = 8 ops.
//
// The tensor-map (tmK) targets an ADDRESS-STABLE per-layer staging buffer that the launcher fills
// with the current k_all each call (capture-safe memcpy on the stream). The kernel never touches a
// moving graph tensor address; row indices into the stable buffer carry the gather (raw window:
// contiguous krow=pos; comp segment: krow = n_raw + idx). Invalid/padding slots gather SAFE row 0
// (clamped, never OOB) and are masked to -inf via sValid -> byte-equivalent to the plain kernel.
#define SM_TMA_W 256                 // bf16 elems per gather4 col-tile (512B). D=512 => 2 tiles.
#define SM_NTILE (SM_D/SM_TMA_W)     // 2

__device__ __forceinline__ void sparse_cp_gather4(uint32_t dst_s, const CUtensorMap* tm,
        int c0, int r0,int r1,int r2,int r3, uint32_t bar_s){
    asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes "
        "[%0],[%1,{%2,%3,%4,%5,%6}],[%7];"
        ::"r"(dst_s),"l"(tm),"r"(c0),"r"(r0),"r"(r1),"r"(r2),"r"(r3),"r"(bar_s):"memory");
}

__global__ void __launch_bounds__(256) sparse_mla_kernel_tma(
        const __grid_constant__ CUtensorMap tmK,  // [width=D, height=stage_rows] bf16, box {256,1} gather4
        const char * __restrict__ Q,
        const int  * __restrict__ kv_idx,
        float * __restrict__ dst,
        const float scale, const int n_raw, const int top_k, const int n_comp_view,
        const int n_tokens, const int n_head, const int n_kv,
        const int64_t Qnb1, const int64_t Qnb2, const int64_t Qnb3,
        const int64_t kv_row_elems,
        const int Qtype) {
    const int t=threadIdx.x, warp=t>>5, lane=t&31;
    const int gid = blockIdx.x;
    const int it  = gid % n_tokens;
    const int is  = gid / n_tokens;

    __nv_bfloat16* sQ=(__nv_bfloat16*)sparse_mla_smem;
    __nv_bfloat16* sK=sQ+SM_HQ*SM_D;
    __nv_bfloat16* sP=sK+SM_KB*SM_D;
    float*         sS=(float*)(sP+SM_HQ*SM_KB);
    __shared__ float m_run[SM_HQ], l_run[SM_HQ];
    __shared__ float wscratch[SM_WARPS][256];
    __shared__ int   sValid[SM_KB];
    __shared__ int   sRow[SM_KB];                          // resolved (clamped) gather row per slot
    __shared__ alignas(128) __nv_bfloat16 sStage[4*SM_TMA_W]; // gather4 dst (128B-aligned static)
    __shared__ alignas(8)   uint64_t bar;

    const int Qesz = (Qtype==2)?4:2;
    for(int i=t; i<SM_HQ*SM_D; i+=256){ int h=i/SM_D, d=i%SM_D;
        const char* qp=Q + (size_t)d*Qesz + (size_t)it*Qnb1 + (size_t)h*Qnb2 + (size_t)is*Qnb3;
        float qv = (Qtype==2) ? *(const float*)qp
                 : (Qtype==1) ? __bfloat162float(*(const __nv_bfloat16*)qp)
                 :              __half2float(*(const half*)qp);
        sQ[i]=__float2bfloat16(qv);
    }
    if(t<SM_HQ){ m_run[t]=-INFINITY; l_run[t]=0.f; }
    float* Ot = dst + ((size_t)is*n_tokens + it)*n_head*SM_D;
    for(int i=t;i<SM_HQ*SM_D;i+=256) Ot[i]=0.f;
    uint32_t bar_s=(uint32_t)__cvta_generic_to_shared(&bar);

    // Per-stream staging: stream is gathered from a [n_kv] row block at offset is*n_kv in the stable
    // buffer (the launcher lays streams back-to-back). Single stream is the validated case (is==0).
    const int row_base = is * n_kv;

    const int*  idx_q = kv_idx + (size_t)it*kv_row_elems;
    const int   total = n_raw + top_k;
    __syncthreads();

    for(int kb0=0; kb0<total; kb0+=SM_KB){
        // Resolve validity + clamped gather row for this block's SM_KB slots (same guards as plain).
        for(int r=t; r<SM_KB; r+=256){
            int pos = kb0 + r;
            int krow = 0; bool valid = false;
            if(pos < total){
                if(pos < n_raw){ krow = pos; valid = true; }
                else { int idx = idx_q[pos - n_raw];
                    if(idx >= 0 && idx < n_comp_view){ krow = n_raw + idx; valid = true; } }
            }
            if(valid && (krow < 0 || krow >= n_kv)){ valid = false; krow = 0; }
            sValid[r] = valid ? 1 : 0;
            sRow[r]   = row_base + krow;   // absolute row into the staging buffer (invalid -> row_base, safe)
        }
        __syncthreads();
        // TMA gather4 the SM_KB rows x D as KB/4 groups x 2 col-tiles into sK.
        for(int g=0; g<SM_KB/4; g++){
            for(int ct=0; ct<SM_NTILE; ct++){
                if(t==0){
                    asm volatile("mbarrier.init.shared.b64 [%0],1;"::"r"(bar_s));
                    asm volatile("fence.proxy.async.shared::cta;");
                    uint32_t dst_s=(uint32_t)__cvta_generic_to_shared(sStage);
                    sparse_cp_gather4(dst_s,&tmK, ct*SM_TMA_W,
                        sRow[g*4+0],sRow[g*4+1],sRow[g*4+2],sRow[g*4+3], bar_s);
                    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;"
                        ::"r"(bar_s),"r"((uint32_t)(4*SM_TMA_W*2)));
                    asm volatile("{.reg .pred p;L:mbarrier.try_wait.parity.shared::cta.b64 p,[%0],0;@!p bra L;}"
                        ::"r"(bar_s));
                }
                __syncthreads();
                for(int e=t; e<4*SM_TMA_W; e+=256){ int i=e/SM_TMA_W, c=e%SM_TMA_W;
                    sK[(g*4+i)*SM_D + ct*SM_TMA_W + c] = sStage[e]; }
                __syncthreads();
            }
        }

        for(int tile=warp; tile<SM_NM*SM_NN; tile+=SM_WARPS){
            int mt=tile/SM_NN, nt=tile%SM_NN;
            wmma::fragment<wmma::accumulator,16,16,16,float> acc; wmma::fill_fragment(acc,0.f);
            for(int dt=0;dt<SM_ND;dt++){
                wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major> a;
                wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::col_major> b;
                wmma::load_matrix_sync(a,sQ+(mt*16)*SM_D+dt*16,SM_D);
                wmma::load_matrix_sync(b,sK+(nt*16)*SM_D+dt*16,SM_D);
                wmma::mma_sync(acc,a,b,acc);
            }
            wmma::store_matrix_sync(sS+(mt*16)*SM_KB+nt*16,acc,SM_KB,wmma::mem_row_major);
        }
        __syncthreads();

        for(int h=warp; h<SM_HQ; h+=SM_WARPS){
            float* sr=sS+h*SM_KB;
            float bmx=-INFINITY; for(int j=lane;j<SM_KB;j+=32){ float v=sValid[j]?sr[j]*scale:-INFINITY; if(v>bmx)bmx=v; }
            for(int o=16;o>0;o>>=1) bmx=fmaxf(bmx,__shfl_xor_sync(0xffffffff,bmx,o));
            float mo=m_run[h],mn=fmaxf(mo,bmx),corr=(mo==-INFINITY)?0.f:expf(mo-mn);
            float bs=0; for(int j=lane;j<SM_KB;j+=32){ float pp=sValid[j]?expf(sr[j]*scale-mn):0.f; sP[h*SM_KB+j]=__float2bfloat16(pp); bs+=pp; }
            for(int o=16;o>0;o>>=1) bs+=__shfl_xor_sync(0xffffffff,bs,o);
            if(lane==0){ m_run[h]=mn; l_run[h]=l_run[h]*corr+bs; }
            for(int d=lane;d<SM_D;d+=32) Ot[h*SM_D+d]*=corr;
        }
        __syncthreads();

        for(int tile=warp; tile<SM_NM*SM_ND; tile+=SM_WARPS){
            int mt=tile/SM_ND, dt=tile%SM_ND;
            wmma::fragment<wmma::accumulator,16,16,16,float> acc; wmma::fill_fragment(acc,0.f);
            for(int kt=0;kt<SM_NN;kt++){
                wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major> a;
                wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::row_major> b;
                wmma::load_matrix_sync(a,sP+(mt*16)*SM_KB+kt*16,SM_KB);
                wmma::load_matrix_sync(b,sK+(kt*16)*SM_D+dt*16,SM_D);
                wmma::mma_sync(acc,a,b,acc);
            }
            wmma::store_matrix_sync(wscratch[warp],acc,16,wmma::mem_row_major);
            for(int e=lane;e<256;e+=32){ int rr=e/16,cc=e%16; Ot[(mt*16+rr)*SM_D+dt*16+cc]+=wscratch[warp][e]; }
        }
        __syncthreads();
    }
    for(int h=warp;h<SM_HQ;h+=SM_WARPS){ float inv=(l_run[h]>0.f)?1.0f/l_run[h]:0.f; for(int d=lane;d<SM_D;d+=32) Ot[h*SM_D+d]*=inv; }
}
#endif // SPARSE_MLA_BUILD

bool ggml_cuda_fattn_sparse_mla_supported(const ggml_tensor * dst) {
#ifdef SPARSE_MLA_BUILD
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * kv = dst->src[6];
    if (!kv) return false;
    // MQA D=512, 64 heads, K is f16/bf16 (materialized k_all). top_k must be 16-aligned (KB=16).
    // Q at the op (post view_4d + permute(0,2,1,3)) = [D, n_tokens, n_head, n_stream].
    if (Q->ne[0] != SM_D || Q->ne[2] != SM_HQ) return false; // D=512, 64 Q heads
    if (Q->ne[3] != 1) return false;                       // single stream only (validated case);
                                                           // multislot (n_stream>1) -> fall back to VEC
    if (K->ne[2] != 1 || K->ne[3] != 1) return false;      // single latent KV head (MQA), single stream
    if (K->type != GGML_TYPE_F16 && K->type != GGML_TYPE_BF16) return false;
    if (kv->type != GGML_TYPE_I32) return false;
    if (kv->ne[0] % SM_KB != 0) return false;              // top_k multiple of 16
    if (kv->ne[1] != Q->ne[1]) return false;               // one top_k list per query token

    // Set the 86KB opt-in smem attribute HERE — supported() is called during graph build /
    // op-supported checks, which run OUTSIDE CUDA graph capture, so this is the capture-safe place
    // to do it (cudaFuncSetAttribute is illegal mid-capture). Idempotent; once per process.
    static bool s_attr_set = false;
    if (!s_attr_set) {
        const size_t smem = (size_t)SM_HQ*SM_D*2 + SM_KB*SM_D*2 + SM_HQ*SM_KB*2 + SM_HQ*SM_KB*4;
        cudaFuncSetAttribute(sparse_mla_kernel,     cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        cudaFuncSetAttribute(sparse_mla_kernel_tma, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        s_attr_set = true;
    }
    return true;
#else
    (void)dst; return false;
#endif
}

void ggml_cuda_flash_attn_ext_sparse_mla(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
#ifdef SPARSE_MLA_BUILD
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * kv = dst->src[6];

    const int n_tokens = (int) Q->ne[1];
    const int n_head   = (int) Q->ne[2];
    const int n_stream = (int) Q->ne[3];
    const int n_kv     = (int) K->ne[1];
    const int top_k    = (int) kv->ne[0];

    float scale = 1.0f; memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));
    int n_raw = ggml_get_op_params_i32(dst, 4);
    // kv_idx (topk) is a VIEW of ggml_argsort over the comp scores [argsort_w, n_tokens]: ne[0]=top_k
    // but nb[1] = argsort_w * 4 = the ROW STRIDE. argsort_w is the TRUE valid comp-row count the
    // indices range over (= n_comp_view at the indexer, NOT the 256-padded k_all length). Use it for
    // BOTH the per-token row offset and the in-kernel validity bound. (k_all's padded length = n_kv
    // is a separate, larger number; the hard krow<n_kv clamp still guards the absolute row.)
    const int64_t kv_row_elems = (int64_t)(kv->nb[1] / sizeof(int32_t));
    const int     n_comp_view  = (int) kv_row_elems;

    const int Ktype = (K->type == GGML_TYPE_BF16) ? 1 : 0;
    const int Qtype = (Q->type == GGML_TYPE_F32) ? 2 : ((Q->type == GGML_TYPE_BF16) ? 1 : 0);

    const size_t smem = (size_t)SM_HQ*SM_D*2 + SM_KB*SM_D*2 + SM_HQ*SM_KB*2 + SM_HQ*SM_KB*4;
    cudaStream_t stream = ctx.stream();

    // CAPTURE-SAFETY: cudaFuncSetAttribute is ILLEGAL during CUDA graph capture. The graph is
    // captured on first use, so a naive "set on first call" lands inside capture -> the 86KB
    // (>48KB default) opt-in never takes effect -> kernel launches with too little smem ->
    // ILLEGAL MEMORY ACCESS. Set the attribute ONLY when NOT capturing; track per-device that
    // it's been set so steady-state (captured) launches just run. The very first reach of this op
    // is always a non-captured warmup/eager pass in this engine, so the attribute is set there.
    static int s_attr_set_dev = -1;
    cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
    cudaStreamIsCapturing(stream, &cap);
    const bool capturing = (cap != cudaStreamCaptureStatusNone);
    if (s_attr_set_dev != ctx.device && !capturing) {
        cudaFuncSetAttribute(sparse_mla_kernel,     cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        cudaFuncSetAttribute(sparse_mla_kernel_tma, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        s_attr_set_dev = ctx.device;
    }

    const dim3 grid(n_tokens * n_stream);

    // ---- TMA-Gather4 path (OPT-IN via DSV4_SPARSE_TMA=1; plain indexed gather is the default) ----
    // HONEST PERF FINDING (standalone A/B, GB10 sm_121a, perf_ab.cu, M=2000, top_k=512, repeated):
    //   plain gather ~28.0k q/s vs TMA gather ~27.0k q/s -> TMA is 0.96-0.98x (marginally SLOWER),
    //   FLAT for both across n_comp 1k..40k. The kernel is OCCUPANCY-bound (64KB Q -> 1 CTA/SM), not
    //   gather-bound (Round 5: removing the gather entirely lifts only ~15%). The single-issue
    //   gather4 (barrier init+wait+staging copy per op) does not beat the already-coalesced per-row
    //   bf16 load from the contiguous k_all. The 3.6-13x in ROUND6 was sparse-vs-DENSE (the plain
    //   wired kernel ALREADY delivers that); it was never TMA-vs-plain. So the plain gather stays the
    //   DEFAULT (no regression). The TMA path is wired + numerically validated (cos 0.999998 PREFILL,
    //   0.999999 SHORT, 0.999998 raw-window) and available for A/B / future pipelined-gather work.
    // Only for bf16 K (the in-server DSV4 latent cache). The gather4 instruction moves raw 2-byte
    // elements reinterpreted as bf16; F16 K (or anything exotic) -> plain kernel.
    static const bool s_use_tma = (getenv("DSV4_SPARSE_TMA") != nullptr);
    const bool use_tma = s_use_tma && (Ktype == 1);

    if (use_tma) {
        // Persistent, ADDRESS-STABLE staging of the k_all rows, per (device,stream-count). The TMA
        // tensor-map targets THIS stable buffer (not the moving graph tensor K->data), so the map
        // baked into a captured launch never goes stale. Each call we memcpy the current K into the
        // staging buffer on the stream (capture-safe; the copy is captured and re-runs on replay,
        // refreshing the staged contents). Map dims = [width=D, height=stage_rows] over the stable
        // buffer; row stride = K->nb[1] (the cache's true bf16 row stride). The map is rebuilt ONLY
        // when the buffer grows or the row stride changes (outside capture). The kernel reads only
        // rows < n_stream*n_kv; padding rows up to capacity are never gathered (krow<n_kv guard).
        struct tma_stage {
            void *     buf      = nullptr;   // device staging (cudaMalloc, ADDRESS-STABLE: never freed)
            size_t     cap_rows = 0;         // allocated rows (n_stream*n_kv high-water)
            int64_t    row_nb   = 0;         // bf16 row stride the map was built for
            CUtensorMap map{};
            bool       map_valid = false;
        };
        static std::map<int, tma_stage> s_stage;   // keyed by device

        tma_stage & st = s_stage[ctx.device];
        const size_t need_rows = (size_t) n_stream * (size_t) n_kv;
        const int64_t row_nb   = K->nb[1];         // bytes per K row (D bf16 = 1024 normally)

        // Grow / (re)build the map only OUTSIDE capture (cudaMalloc + map encode are host ops; doing
        // them mid-capture would either be illegal or bake a transient state). The first reach of a
        // given (shape) is a non-captured warmup, so growth happens there; steady captured replays
        // just memcpy + launch against the already-stable buffer + map.
        //
        // CAPTURE-SAFE GROWTH: the staging buffer is NEVER cudaFree'd. A captured graph bakes the map
        // (which embeds the buffer address); if we freed+realloc'd on growth, an OLD captured graph
        // (smaller shape) would replay against a freed pointer -> use-after-free. So on growth we
        // allocate a NEW buffer and LEAK the old one (it stays alive for any still-cached graph that
        // baked it). Growth is rare (4K-row granularity, monotonic high-water) -> the leak is bounded
        // and tiny vs the model. Once n_kv changes the graph key changes -> that shape re-captures
        // the launcher and bakes the new (current) map, so it never reads the stale buffer's content.
        if (!capturing) {
            if (st.buf == nullptr || need_rows > st.cap_rows || row_nb != st.row_nb) {
                size_t new_cap = need_rows;
                if (new_cap < st.cap_rows) new_cap = st.cap_rows;
                new_cap = ((new_cap + 4095) / 4096) * 4096;          // 4K-row granularity
                if (st.buf == nullptr || new_cap > st.cap_rows || row_nb != st.row_nb) {
                    void * newbuf = nullptr;
                    if (cudaMalloc(&newbuf, (size_t) new_cap * (size_t) row_nb) == cudaSuccess) {
                        st.buf      = newbuf;            // leak any prior buffer on purpose (see above)
                        st.cap_rows = new_cap;
                        st.row_nb   = row_nb;
                        st.map_valid = false;
                    }
                }
            }
            if (!st.map_valid) {
                uint64_t dims[2]   = { (uint64_t) SM_D, (uint64_t) st.cap_rows };
                uint64_t strides[1]= { (uint64_t) st.row_nb };       // bf16 row stride
                uint32_t box[2]    = { (uint32_t) SM_TMA_W, 1 };
                uint32_t es[2]     = { 1, 1 };
                CUresult r = cuTensorMapEncodeTiled(&st.map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2,
                    st.buf, dims, strides, box, es,
                    CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
                    CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
                st.map_valid = (r == CUDA_SUCCESS);
            }
        }

        if (st.buf != nullptr && st.map_valid && need_rows <= st.cap_rows) {
            // Refresh the staged copy of k_all (capture-safe D2D on the stream). Copies the dense
            // n_stream*n_kv row block; if K rows are non-contiguous (row_nb != D*2) the staging keeps
            // the same stride, so the map's stride matches. K is a freshly concatenated contiguous
            // cache in the DSV4 graph, so this is a single linear copy.
            cudaMemcpyAsync(st.buf, K->data, need_rows * (size_t) row_nb,
                            cudaMemcpyDeviceToDevice, stream);
            sparse_mla_kernel_tma<<<grid, 256, smem, stream>>>(
                st.map, (const char*)Q->data, (const int*)kv->data, (float*)dst->data,
                scale, n_raw, top_k, n_comp_view, n_tokens, n_head, n_kv,
                Q->nb[1], Q->nb[2], Q->nb[3], kv_row_elems, Qtype);
            return;
        }
        // else: map not ready yet (e.g. first call landed mid-capture) -> fall through to plain.
    }

    sparse_mla_kernel<<<grid, 256, smem, stream>>>(
        (const char*)Q->data, (const char*)K->data, (const int*)kv->data, (float*)dst->data,
        scale, n_raw, top_k, n_comp_view, n_tokens, n_head, n_kv,
        Q->nb[1], Q->nb[2], Q->nb[3], K->nb[1], K->nb[3], kv_row_elems, Ktype, Qtype);
#else
    (void)ctx; (void)dst; GGML_ABORT("sparse-mla kernel not built");
#endif
}
