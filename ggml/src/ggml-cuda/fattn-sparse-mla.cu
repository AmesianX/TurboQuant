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
        cudaFuncSetAttribute(sparse_mla_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
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
        cudaFuncSetAttribute(sparse_mla_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        s_attr_set_dev = ctx.device;
    }

    const dim3 grid(n_tokens * n_stream);
    sparse_mla_kernel<<<grid, 256, smem, stream>>>(
        (const char*)Q->data, (const char*)K->data, (const int*)kv->data, (float*)dst->data,
        scale, n_raw, top_k, n_comp_view, n_tokens, n_head, n_kv,
        Q->nb[1], Q->nb[2], Q->nb[3], K->nb[1], K->nb[3], kv_row_elems, Ktype, Qtype);
#else
    (void)ctx; (void)dst; GGML_ABORT("sparse-mla kernel not built");
#endif
}
