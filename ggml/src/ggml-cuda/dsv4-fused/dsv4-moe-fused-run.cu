// DSV4 fused MoE runner glue: maps our sidecar NVFP4 expert registry into
// flashinfer's CutlassMoeFCRunner<__nv_fp4_e2m1,__nv_fp4_e2m1,bf16,bf16>::runMoe.
//
// Round-2 implementation. Builds inside the scoped dsv4-moe-fused-cutlass static
// lib (see CMakeLists). Exposes extern "C" dsv4_moe_fused_run(...) called by
// ggml-cuda/dsv4-moe-fused.cu when DSV4_MOE_FUSED_CUTLASS is defined.
//
// MAPPING (from the runner's verified layout requirements):
//  * fc1 weights = [E, 2*inter, hidden] fp4, ordered UP-rows first then GATE-rows
//    (doActivation reads first half as the linear multiplier, second half as the
//    silu'd gate -> SiLu(gate)*up). hidden/2 bytes/row, consecutive nibbles.
//  * fc2 weights = [E, hidden, inter] fp4 (= our dq_down, already that layout).
//  * weight block scales must be PRE-SWIZZLED to CUTLASS SWIZZLED_128x4 atom
//    layout (computeSFIndex), per expert, padded rows->128 cols->4. We swizzle
//    our PLAIN dsf_*_simple [E][n][k/16] ourselves.
//  * fc1 needs ONE per-expert global for gate+up. Our gate/up have separate
//    globals -> g_common = max(g_gate,g_up); renormalize each projection's block
//    scales to g_common (one e4m3 round-trip).
//  * activations: pass bf16 hidden, NO input_sf -> runner quantizes internally
//    (need_nvfp4_quant=true). We provide fc1/fc2 act global scales.
//
// The repack is done ONCE per layer (keyed by il) and cached. Per-call we convert
// F32 hidden->bf16, run runMoe, convert bf16 out->F32.

#include "cutlass_fused_moe_kernels.cuh"
#include "moe_kernels.h"

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <unordered_map>
#include <mutex>
#include <vector>
#include <memory>
#include <cstdio>
#include <cmath>
#include <cinttypes>

namespace tk  = tensorrt_llm::kernels;
namespace tkc = tensorrt_llm::kernels::cutlass_kernels;

// Free the grouped-path source buffers we've folded into fused-format copies
// (dq_gate/up + dsf_*_simple). Implemented in dsv4-moe-grouped.cu. Keeps dq_down
// (fc2 alias) and dglobal_* (read live). Halves the fused memory overhead.
extern "C" bool dsv4_moe_grouped_free_superseded_by_fused(int il);
extern "C" int64_t dsv4_moe_grouped_gu_estride(int il);

// Defer-free retire list (shared with the grouped op): freed at the next backend
// synchronize (ggml_backend_cuda_synchronize), NOT immediately, so a grow-realloc
// never cudaFree's a pointer a still-pending captured graph might reference (UAF).
// Implemented in dsv4-moe-grouped.cu; drained by dsv4_moe_grouped_drain_retired().
extern "C" void dsv4_moe_grouped_retire_ptr(void* p);

// [ep2-dp] EP (expert-parallel) config, set once at load from the EP sidecar header.
// Returns ep flag; out-params get this rank's GLOBAL expert_base, the GLOBAL expert count,
// and the LOCAL (registered) expert count. ep_size = global/local, ep_rank = base/local.
// Implemented in dsv4-moe-grouped.cu. ep==0 => no parallelism (byte-identical).
extern "C" int dsv4_moe_get_ep_config(int* expert_base, int* n_expert_global, int* n_expert_local);

// Pre-size cap: the server publishes the prefill ubatch into DSV4_MOE_PREFILL_MAX
// at startup (BEFORE any graph capture), so we allocate the per-call scratch ONCE
// to that max and never realloc mid-capture. 0 => grow-once with retire-free.
static int dsv4_fused_prefill_max() {
    static const int v = []{ const char* e=getenv("DSV4_MOE_PREFILL_MAX"); return e?atoi(e):0; }();
    return v;
}

#define CK(x) do { cudaError_t _e=(x); if(_e!=cudaSuccess){ \
    fprintf(stderr,"[DSV4_MOE_FUSED] CUDA err %s @ %s:%d\n",cudaGetErrorString(_e),__FILE__,__LINE__); \
    return _e; } } while(0)

static __host__ int padUp(int x,int a){ return (x + a - 1)/a*a; }
static int grid_for(long n){ long b=(n+255)/256; return (int)(b>65535?65535:(b<1?1:b)); }

// SWIZZLED_128x4 destination index (mirror of computeSFIndex / fp4Op.cpp).
__device__ __forceinline__ int sf_swizzled_index(int rowIdx, int colIdx, int totalColumn) {
    const int kColumnGroup0Size = 4;
    const int kRowGroup0Size = 32;
    const int kRowGroup1Size = kRowGroup0Size * 4; // 128
    int paddedColumn = ((totalColumn + 3) / 4) * 4;
    int columnIdxInGroup0 = colIdx % kColumnGroup0Size;
    int columnGroupIdx    = colIdx / kColumnGroup0Size;
    const int columnGroupStride = kColumnGroup0Size * kRowGroup1Size; // 512
    int rowIdxInGroup0 = rowIdx % kRowGroup0Size;
    int rowIdxInGroup1 = (rowIdx % kRowGroup1Size) / kRowGroup0Size;
    int rowGroupIdx    = rowIdx / kRowGroup1Size;
    const int rowGroup1Stride = kColumnGroup0Size;                    // 4
    const int rowGroup0Stride = kColumnGroup0Size * rowGroup1Stride;  // 16
    int rowGroupStride = kRowGroup1Size * paddedColumn;
    return columnIdxInGroup0 + columnGroupIdx * columnGroupStride
         + rowIdxInGroup0 * rowGroup0Stride + rowIdxInGroup1 * rowGroup1Stride
         + rowGroupIdx * rowGroupStride;
}

// (1) concat up||gate packed-fp4 -> fused fc1 [E][2*inter][hidden/2].
__global__ void k_concat_fc1(const uint8_t* up, const uint8_t* gate, uint8_t* fused,
                             int E, int inter, int hbytes) {
    long total = (long)E * inter * hbytes;
    for (long i = blockIdx.x*(long)blockDim.x + threadIdx.x; i < total; i += (long)gridDim.x*blockDim.x) {
        int hb = (int)(i % hbytes);
        long rr = i / hbytes;
        int r   = (int)(rr % inter);
        int e   = (int)(rr / inter);
        long src = ((long)e*inter + r)*hbytes + hb;
        long base = (long)e*2*inter*hbytes;
        fused[base + (long)r*hbytes + hb]            = up[src];   // up first
        fused[base + (long)(inter+r)*hbytes + hb]    = gate[src]; // then gate
    }
}

// (2) swizzle one projection's plain [E][n][cols] e4m3 SF into SWIZZLED_128x4,
//     dst rows padded to padUp(n_fused,128), cols to padUp(cols,4). Optional
//     rescale by (g_proj/g_common)[e] for the fc1 global reconcile.
__global__ void k_swizzle_sf(const uint8_t* plain, uint8_t* swz,
                             int E, int n, int cols, int n_fused, int row_off,
                             const float* g_proj, const float* g_common) {
    int padN = ((n_fused + 127)/128)*128;
    int padC = ((cols + 3)/4)*4;
    long expert_out = (long)padN * padC;
    long total = (long)E * n * cols;
    for (long i = blockIdx.x*(long)blockDim.x + threadIdx.x; i < total; i += (long)gridDim.x*blockDim.x) {
        int c  = (int)(i % cols);
        long rr = i / cols;
        int r  = (int)(rr % n);
        int e  = (int)(rr / n);
        uint8_t v = plain[((long)e*n + r)*cols + c];
        if (g_proj && g_common) {
            __nv_fp8_e4m3 fp; fp.__x = v;
            float s = (float)fp;
            __nv_fp8_e4m3 fp2 = __nv_fp8_e4m3(s * (g_proj[e] / g_common[e]));
            v = fp2.__x;
        }
        int dst = sf_swizzled_index(row_off + r, c, cols);
        swz[(long)e*expert_out + dst] = v;
    }
}

__global__ void k_gmax(const float* a, const float* b, float* out, int E) {
    for (int e = blockIdx.x*blockDim.x+threadIdx.x; e<E; e+=gridDim.x*blockDim.x)
        out[e] = fmaxf(a[e], b[e]);
}
__global__ void k_fill(float* p, float v, int n) {
    for (int i=blockIdx.x*blockDim.x+threadIdx.x; i<n; i+=gridDim.x*blockDim.x) p[i]=v;
}

// ---- NVFP4 global-scale convention (flashinfer / fp4Quantize.cpp:27) ---------
//   globalScale = (448*6) / amax     (act and weight both)
//   GEMM alpha (fp4.fcX.global_scale[e]) = 1 / (act_gs * weight_gs[e])
// Our weight block SF dsf_*_simple already == flashinfer's weight block SF
// (e4m3(S_mx/g) with g=w_amax/448 == e4m3(globalScale_w*S_mx/6)). Our per-expert
// weight global g[e]=w_amax/448 -> flashinfer weight_gs[e] = 6/g[e]. So
//   alpha[e] = 1/( act_gs * 6/g[e] ) = g[e] / (6 * act_gs).
#define E2M1_MAX 6.0f
#define E4M3_MAX 448.0f

// per-tensor activation absmax over the F32 hidden [n] (block-reduced partials).
__global__ void k_absmax_part(const float* x, float* part, long n) {
    __shared__ float sm[256];
    float m = 0.f;
    for (long i = blockIdx.x*(long)blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x*blockDim.x)
        m = fmaxf(m, fabsf(x[i]));
    sm[threadIdx.x] = m; __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) { if (threadIdx.x < s) sm[threadIdx.x]=fmaxf(sm[threadIdx.x],sm[threadIdx.x+s]); __syncthreads(); }
    if (threadIdx.x == 0) part[blockIdx.x] = sm[0];
}
// finalize: act_global = (448*6)/amax (clamped); store to act_gs[0].
__global__ void k_absmax_final(const float* part, int nparts, float* act_gs) {
    __shared__ float sm[256];
    float m = 0.f;
    for (int i = threadIdx.x; i < nparts; i += blockDim.x) m = fmaxf(m, part[i]);
    sm[threadIdx.x] = m; __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) { if (threadIdx.x < s) sm[threadIdx.x]=fmaxf(sm[threadIdx.x],sm[threadIdx.x+s]); __syncthreads(); }
    if (threadIdx.x == 0) { float am = sm[0]; act_gs[0] = am > 0.f ? (E4M3_MAX*E2M1_MAX)/am : 1.f; }
}
// per-expert GEMM alpha. MATCHES the proven-coherent grouped op exactly:
//   grouped: alpha = gscaleA * gscaleB, gscaleA=act_amax/(6*448), gscaleB=w_amax/448.
//   here:   act_gs (the runner's SFScaleVal) = (6*448)/act_amax = 1/gscaleA, and our
//   weight global g = w_amax/448 = gscaleB. So alpha = gscaleA*gscaleB = g/act_gs.
//   (NO extra /6 -- that was the bug producing 6x-too-small output.)
__global__ void k_fc1_alpha(const float* g_common, const float* act_gs, float* alpha, int E) {
    float a = act_gs[0];
    for (int e = blockIdx.x*blockDim.x+threadIdx.x; e<E; e+=gridDim.x*blockDim.x)
        alpha[e] = g_common[e] / a;
}
__global__ void k_fc2_alpha(const float* g_down, const float* act_gs, float* alpha, int E) {
    float a = act_gs[0];
    for (int e = blockIdx.x*blockDim.x+threadIdx.x; e<E; e+=gridDim.x*blockDim.x)
        alpha[e] = g_down[e] / a;
}
// F32 -> bf16 (same contiguous index order; ggml [n_embd,n_tokens] == rowmajor [n_tokens,n_embd]).
__global__ void k_f32_to_bf16(const float* src, __nv_bfloat16* dst, long n) {
    for (long i=blockIdx.x*(long)blockDim.x+threadIdx.x; i<n; i+=(long)gridDim.x*blockDim.x)
        dst[i]=__float2bfloat16(src[i]);
}
__global__ void k_bf16_to_f32(const __nv_bfloat16* src, float* dst, long n) {
    for (long i=blockIdx.x*(long)blockDim.x+threadIdx.x; i<n; i+=(long)gridDim.x*blockDim.x)
        dst[i]=__bfloat162float(src[i]);
}

// Per-layer PERSISTENT weights (built once, kept resident).
struct FusedLayer {
    bool fc1_w_aliased=false;   // [fc1-fused-layout] fc1_w aliases the grouped registry (do not free)
    int E=0, hidden=0, inter=0;
    uint8_t* fc1_w=nullptr;
    uint8_t* fc2_w=nullptr;       // alias dq_down (registry-owned, not freed here)
    uint8_t* fc1_sf=nullptr;
    uint8_t* fc2_sf=nullptr;
    float*   g_common=nullptr;    // [E] per-expert fc1 weight global = max(g_gate,g_up)
    float*   g_down=nullptr;      // alias g_down (fc2 weight global)
    float    swiglu_limit_val=0.f;
    float*   swiglu_limit=nullptr;// [E]
    float*   fc2_act_global=nullptr; // [1] estimate from clamp limit (per-layer const)
    int      tactic_idx=0;        // selected getTactics() index (for crash diagnostics)
    std::unique_ptr<tkc::CutlassMoeFCRunnerInterface> runner;
    // legacy unused (kept to avoid touching free path); transient scratch moved to g_scratch
    float*   fc1_act_global=nullptr;
    float*   fc1_alpha=nullptr;
    float*   fc2_alpha=nullptr;
    float*   act_part=nullptr; int act_nparts=0;
    __nv_bfloat16* d_hidden_bf16=nullptr; long hidden_bf16_cap=0;
    __nv_bfloat16* d_out_bf16=nullptr;    long out_bf16_cap=0;
    char* d_workspace=nullptr; size_t workspace_cap=0;
    int*  d_src2dst=nullptr; long src2dst_cap=0;
};
static std::mutex g_fmu;
static std::unordered_map<int,FusedLayer*> g_fcache;

// ---- internal sub-phase profiler (DSV4_MOE_FUSED_PROF=1) ---------------------
// Deferred cudaEvent timing of the per-call phases so we can see GEMM vs glue.
// Same no-per-call-sync design as DSV4_OPPROF: record event pairs, drain ready
// ones lazily, dump after N calls + at exit. NEVER syncs in the hot path. Skips
// the FIRST call per layer (the one-time build) so steady-state isn't contaminated.
namespace fprof {
    struct ev { cudaEvent_t e0,e1; int phase; };  // phase: 0 cvt_in 1 route 2 runMoe 3 cvt_out
    struct prof {
        std::mutex m;
        std::vector<ev> pend;
        double ms[4]={0,0,0,0}; int64_t cnt[4]={0,0,0,0};
        bool on=false; int64_t calls=0, dump_after=0; bool dumped=false;
        const char* names[4]={"cvt_in(f32->bf16)","route(absmax+alpha)","runMoe(GEMM+sort+finalize)","cvt_out(bf16->f32)"};
        prof(){ on=getenv("DSV4_MOE_FUSED_PROF")!=nullptr;
                if(const char*d=getenv("DSV4_MOE_FUSED_PROF_AFTER")) dump_after=atoll(d); else if(on) dump_after=4000;
                if(on) fprintf(stderr,"[DSV4_MOE_FUSED_PROF] active, will auto-dump after %lld calls\n",
                               (long long)dump_after); }
        // drain: when force, sync ONLY our own events (cudaEventSynchronize) — NEVER a full
        // cudaDeviceSynchronize (that hangs under the 2-node SPMD meta-backend mid-async-dispatch).
        void drain(bool force){ size_t k=0; for(size_t i=0;i<pend.size();++i){ ev&e=pend[i];
            cudaError_t s = force ? cudaEventSynchronize(e.e1) : cudaEventQuery(e.e1);
            if(s==cudaSuccess){ float t=0; if(cudaEventElapsedTime(&t,e.e0,e.e1)==cudaSuccess){ ms[e.phase]+=t; cnt[e.phase]++; }
                cudaEventDestroy(e.e0); cudaEventDestroy(e.e1);} else pend[k++]=e; } pend.resize(k); }
        // dump_locked: caller MUST already hold m. No full-device sync (force-drains own events only).
        void dump_locked(const char* why){ drain(true);
            double tot=ms[0]+ms[1]+ms[2]+ms[3];
            fprintf(stderr,"\n[DSV4_MOE_FUSED_PROF] (%s) per-call phase totals %.1f ms over %lld calls:\n",
                    why,tot,(long long)calls);
            for(int p=0;p<4;p++) fprintf(stderr,"  %8.2f ms  %5.1f%%  %8" PRId64 "x  %s\n",
                ms[p], tot>0?100.0*ms[p]/tot:0.0, cnt[p], names[p]);
            fflush(stderr); }
        ~prof(){ if(on){ std::lock_guard<std::mutex> lk(m); dump_locked("exit"); } }
        // one-shot auto-dump after dump_after calls (so it prints without clean shutdown).
        void tick_and_maybe_dump(){ if(!on||dumped||dump_after<=0) return;
            std::lock_guard<std::mutex> lk(m);
            if(++calls>=dump_after && !dumped){ dumped=true; dump_locked("auto"); } }
    };
    static prof g;
    // record one phase around a lambda body (events on `stream`, no sync).
    struct scope {
        cudaStream_t s; int phase; cudaEvent_t e0=nullptr,e1=nullptr; bool ok=false;
        scope(cudaStream_t st,int p):s(st),phase(p){ if(!g.on)return;
            ok=(cudaEventCreate(&e0)==cudaSuccess)&&(cudaEventCreate(&e1)==cudaSuccess);
            if(ok)cudaEventRecord(e0,s); }
        ~scope(){ if(!ok)return; cudaEventRecord(e1,s); std::lock_guard<std::mutex> lk(g.m);
            g.pend.push_back({e0,e1,phase}); g.drain(false); }
    };
}

// SHARED per-call transient scratch — ONE pool for ALL layers (they run
// sequentially, so the runMoe workspace + bf16 buffers + routing scratch are
// reused, not duplicated 58x). This is what lets UB scale: the per-layer footprint
// drops from (per-layer workspace x58) to a single workspace. Sized grow-once to
// the prefill cap (retire-free on grow). All buffers re-written every call.
struct FusedScratch {
    int   E_max=0;                       // max experts seen (for alpha/act arrays)
    float* fc1_act_global=nullptr;       // [1]
    float* fc1_alpha=nullptr;            // [E] recomputed per layer per call
    float* fc2_alpha=nullptr;            // [E]
    float* act_part=nullptr; int act_nparts=256;
    __nv_bfloat16* d_hidden_bf16=nullptr; long hidden_bf16_cap=0;
    __nv_bfloat16* d_out_bf16=nullptr;    long out_bf16_cap=0;
    char* d_workspace=nullptr; size_t workspace_cap=0;
    int*  d_src2dst=nullptr; long src2dst_cap=0;
    long  tok_cap_hw=0;                   // monotonic high-water token cap (multi-request graph-stable)
};
static FusedScratch g_scratch;

static cudaError_t build_fused_layer(FusedLayer* L, int il,
        int E,int hidden,int inter,
        const void* dq_gate,const void* dq_up,const void* dq_down,
        const void* sf_gate,const void* sf_up,const void* sf_down,
        const float* g_gate,const float* g_up,const float* g_down,
        float swiglu_limit, cudaStream_t stream) {
    L->E=E; L->hidden=hidden; L->inter=inter;
    const int hbytes = hidden/2;
    const int colsD  = hidden/16;   // fc1 SF cols (K=hidden)
    const int colsF  = inter/16;    // fc2 SF cols (K=inter)

    // [fc1-fused-layout] If the loader stored gate/up in the fused interleave already
    // (DSV4_MOE_FC1_FUSED=1: per expert up rows then gate rows, stride 2*inter*hbytes),
    // fc1_w is EXACTLY that buffer: alias it. No 1.07GB/layer malloc, no concat, no free
    // -> kills the repack fragmentation OOM (+46GB over 43 layers) on big prefills.
    if (dsv4_moe_grouped_gu_estride(il) == (int64_t)2*inter*hbytes) {
        L->fc1_w = (uint8_t*)dq_up;   // base = expert0 up rows; gate at +inter*hbytes
        L->fc1_w_aliased = true;
    } else {
        CK(cudaMalloc(&L->fc1_w, (size_t)E*2*inter*hbytes));
        { long tot=(long)E*inter*hbytes; k_concat_fc1<<<grid_for(tot),256,0,stream>>>(
            (const uint8_t*)dq_up,(const uint8_t*)dq_gate,L->fc1_w,E,inter,hbytes); }
    }
    L->fc2_w = (uint8_t*)dq_down;

    CK(cudaMalloc(&L->g_common,(size_t)E*4));
    k_gmax<<<grid_for(E),256,0,stream>>>(g_gate,g_up,L->g_common,E);
    L->g_down = (float*)g_down;

    const int padN_fc1 = padUp(2*inter,128), padC_fc1 = padUp(colsD,4);
    CK(cudaMalloc(&L->fc1_sf,(size_t)E*padN_fc1*padC_fc1));
    CK(cudaMemsetAsync(L->fc1_sf,0,(size_t)E*padN_fc1*padC_fc1,stream));
    { long tot=(long)E*inter*colsD;
      k_swizzle_sf<<<grid_for(tot),256,0,stream>>>((const uint8_t*)sf_up,  L->fc1_sf,E,inter,colsD,2*inter,0,    g_up,  L->g_common);
      k_swizzle_sf<<<grid_for(tot),256,0,stream>>>((const uint8_t*)sf_gate,L->fc1_sf,E,inter,colsD,2*inter,inter,g_gate,L->g_common);
    }

    const int padN_fc2 = padUp(hidden,128), padC_fc2 = padUp(colsF,4);
    CK(cudaMalloc(&L->fc2_sf,(size_t)E*padN_fc2*padC_fc2));
    CK(cudaMemsetAsync(L->fc2_sf,0,(size_t)E*padN_fc2*padC_fc2,stream));
    { long tot=(long)E*hidden*colsF;
      k_swizzle_sf<<<grid_for(tot),256,0,stream>>>((const uint8_t*)sf_down,L->fc2_sf,E,hidden,colsF,hidden,0,nullptr,nullptr);
    }

    L->swiglu_limit_val = swiglu_limit>0.f?swiglu_limit:1e30f;
    CK(cudaMalloc(&L->swiglu_limit,(size_t)E*4));
    k_fill<<<grid_for(E),256,0,stream>>>(L->swiglu_limit, L->swiglu_limit_val, E);

    // fc2 act global (per-layer const): estimate from the clamp limit. The SwiGLU
    // intermediate silu(clamp(gate))*clamp(up) is bounded ~ limit; use (448*6)/lim_est
    // so the e4m3 block SF lands in range. (Dynamic per-block SF corrects the rest.)
    CK(cudaMalloc(&L->fc2_act_global,4));
    { float lim = L->swiglu_limit_val>1e29f ? 8.0f : L->swiglu_limit_val;
      float fc2_amax_est = lim;
      k_fill<<<1,1,0,stream>>>(L->fc2_act_global, (E4M3_MAX*E2M1_MAX)/fc2_amax_est, 1); }

    L->runner = std::make_unique<tkc::CutlassMoeFCRunner<__nv_fp4_e2m1,__nv_fp4_e2m1,__nv_bfloat16,__nv_bfloat16>>();
    auto tactics = L->runner->getTactics();
    if (tactics.empty()) { fprintf(stderr,"[DSV4_MOE_FUSED] no tactics, layer %d\n",il); return cudaErrorNotSupported; }
    // Tactic selection: DEFAULT = front() (the only config measured stable; the 256x128x64B tile
    // CRASHES at load on sm121 — confirmed by the coordinator, reverted). DSV4_MOE_FUSED_TACTIC=<i>
    // overrides to sweep the SM120 tiles for A/B (layer-0 log prints the index->tile mapping).
    int ti = []{ const char* e=getenv("DSV4_MOE_FUSED_TACTIC"); return e?atoi(e):0; }();
    if (ti < 0 || ti >= (int)tactics.size()) ti = 0;
    L->tactic_idx = ti;
    // [256x128 SMEM DIAG] On layer 0, print the device smem opt-in cap so a tile whose dynamic smem
    // exceeds it (the documented GB10 101376-byte limit -> cudaFuncSetAttribute fails) is obvious.
    if (il == 0) {
        int dev=0, smem_optin=0, smem_block=0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&smem_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
        cudaDeviceGetAttribute(&smem_block, cudaDevAttrMaxSharedMemoryPerBlock, dev);
        fprintf(stderr,"[DSV4_MOE_FUSED] dev %d smem: MaxPerBlockOptin=%d MaxPerBlock=%d (selected tactic idx=%d sm=%d tileCfg=%d)\n",
                dev, smem_optin, smem_block, ti, tactics[ti].sm_version, tactics[ti].getTileConfigAsInt());
    }
    L->runner->setTactic(tactics[ti], tactics[ti]);
    CK(cudaStreamSynchronize(stream));   // ensure all repack reads of source buffers done
    // Free the grouped source weights we've folded in (dq_gate/up + simple SFs) to
    // halve the fused memory overhead. dq_down (fc2) + dglobal_* are kept (aliased).
    dsv4_moe_grouped_free_superseded_by_fused(il);
    if (il == 0) {
        fprintf(stderr,"[DSV4_MOE_FUSED] %zu tactics; using index %d:\n", tactics.size(), ti);
        for (size_t i=0;i<tactics.size();++i)
            fprintf(stderr,"[DSV4_MOE_FUSED]   tactic[%zu] = %s\n", i, tactics[i].toString().c_str());
    }
    fprintf(stderr,"[DSV4_MOE_FUSED] layer %d repacked: E=%d hidden=%d inter=%d (tactic %d)\n",
            il,E,hidden,inter,ti);
    return cudaSuccess;
}

// ---- accessor for the GROUPED HP-DECODE op (dsv4-moe-grouped.cu) -------------
// ORTHODOX SINGLE-WEIGHT-SET DESIGN: once a layer is folded into the fused fc1/fc2
// layout, the grouped path's dq_gate/dq_up (+ simple SFs) are FREED (real memory
// reclaimed, no duplicate). The HP DECODE kernels therefore must read the SAME
// fused weight tensors. This hands the grouped TU the live fused pointers + dims so
// its decode kernels can slice gate/up out of fc1 and read fc2, using the fused
// swizzled SF + the per-expert g_common/g_down globals (numerically == prefill).
// Returns false if layer not built in the fused cache (caller keeps grouped buffers).
//   fc1_w  : [E][2*inter][hidden/2] e2m1, rows [0,inter)=UP then [inter,2*inter)=GATE
//   fc2_w  : [E][hidden][inter/2]   e2m1 (alias dq_down)
//   fc1_sf : SWIZZLED_128x4 ue4m3, per-expert stride padUp(2*inter,128)*padUp(hidden/16,4)
//   fc2_sf : SWIZZLED_128x4 ue4m3, per-expert stride padUp(hidden,128)*padUp(inter/16,4)
//   g_common: [E] fc1 weight global (=max(g_gate,g_up)); g_down: [E] fc2 weight global
extern "C" bool dsv4_moe_fused_get_layer(
        int il, int* E, int* hidden, int* inter,
        const void** fc1_w, const void** fc2_w,
        const void** fc1_sf, const void** fc2_sf,
        const float** g_common, const float** g_down) {
    std::lock_guard<std::mutex> lk(g_fmu);
    auto it = g_fcache.find(il);
    if (it == g_fcache.end() || !it->second) return false;
    FusedLayer* L = it->second;
    if (!L->fc1_w || !L->fc2_w || !L->fc1_sf || !L->fc2_sf) return false;
    if (E)        *E        = L->E;
    if (hidden)   *hidden   = L->hidden;
    if (inter)    *inter    = L->inter;
    if (fc1_w)    *fc1_w    = (const void*)L->fc1_w;
    if (fc2_w)    *fc2_w    = (const void*)L->fc2_w;
    if (fc1_sf)   *fc1_sf   = (const void*)L->fc1_sf;
    if (fc2_sf)   *fc2_sf   = (const void*)L->fc2_sf;
    if (g_common) *g_common = L->g_common;
    if (g_down)   *g_down   = L->g_down;
    return true;
}

extern "C" cudaError_t dsv4_moe_fused_run(
        int il,
        const float* hidden, const int* sel, const float* weights, float* moe_out,
        int n_tokens, int n_embd, int n_ff_exp, int n_expert, int n_expert_used,
        float swiglu_limit,
        const void* dq_gate, const void* dq_up, const void* dq_down,
        const void* sf_gate, const void* sf_up, const void* sf_down,
        const float* g_gate, const float* g_up, const float* g_down,
        cudaStream_t stream) {
    FusedLayer* L=nullptr;
    {
        std::lock_guard<std::mutex> lk(g_fmu);
        auto it=g_fcache.find(il);
        if (it==g_fcache.end()) {
            L=new FusedLayer();
            cudaError_t e=build_fused_layer(L,il,n_expert,n_embd,n_ff_exp,
                dq_gate,dq_up,dq_down,sf_gate,sf_up,sf_down,g_gate,g_up,g_down,swiglu_limit,stream);
            if (e!=cudaSuccess){ delete L; return e; }
            g_fcache[il]=L;
        } else L=it->second;
    }

    // SHARED scratch grow (ONE pool for all 58 layers). Sized to the prefill cap so it
    // never reallocs inside a captured region; grow RETIRES the old pointer (defer-free).
    // This is the lever-2 fix: per-layer workspace x58 -> single shared workspace.
    //
    // [LONG MULTI-CHUNK PREFILL — CAPTURE SAFETY] The CUTLASS runMoe's internal workspace
    // LAYOUT is M-keyed: configureWsPtrs(num_rows=n_tokens) recomputes every sub-buffer offset
    // per call (cutlass_fused_moe_kernels.cuh:2806), and the prologue sort grid/kernel template
    // is chosen host-side from M (computeNumTokensPerBlock, kernels:524/843). Sizing the arena
    // to tok_cap (the MAX ubatch, >= every chunk's M) keeps the ALLOCATION big enough for any
    // layout, but the OFFSETS still shift with M -> NOT safe to bake into a replayed CUDA graph.
    // The engine therefore runs this op with graphs OFF by default (ggml-cuda.cu GGML_OP_DSV4_MOE_FUSED
    // gate; DSV4_MOE_FUSED_GRAPH_ON=1 to re-enable for A/B). With graphs off, each chunk's runMoe
    // is launched eagerly with its true M, exactly like Aiden's vLLM chunked prefill -> crash-free.
    // We KEEP the fixed-cap pre-sizing below so the arena never reallocs mid-stream regardless.
    //
    // [MULTI-REQUEST CRASH FIX] tok_cap MUST be a MONOTONIC high-water mark, not per-request.
    // When DSV4_MOE_PREFILL_MAX is unset (pf_max=0), the old `max(pf_max, n_tokens)` made the
    // workspace size + getWorkspaceSize(tok_cap) + the captured graph shape REQUEST-SIZE-DEPENDENT.
    // A large request-1 sized everything to its M and captured a graph; a small request-2 (M=17)
    // recomputed a SMALLER tok_cap -> getWorkspaceSize shrank -> mismatched the already-captured
    // workspace layout -> illegal memory access in graph evaluate/capture. Holding tok_cap at the
    // max ever seen keeps the workspace, the getWorkspaceSize result, and the graph shape STABLE
    // across requests of any size -> capture-safe + multi-request-safe. Grows are still retire-free.
    FusedScratch& S = g_scratch;
    const int  pf_max = dsv4_fused_prefill_max();
    long       tok_cap = pf_max > n_tokens ? pf_max : n_tokens;
    if (tok_cap < S.tok_cap_hw) tok_cap = S.tok_cap_hw;   // never shrink below the high-water mark
    S.tok_cap_hw = tok_cap;
    long hcap=(long)tok_cap*n_embd;
    if (S.hidden_bf16_cap < hcap) {
        if (S.d_hidden_bf16) dsv4_moe_grouped_retire_ptr(S.d_hidden_bf16);
        CK(cudaMalloc(&S.d_hidden_bf16,(size_t)hcap*sizeof(__nv_bfloat16))); S.hidden_bf16_cap=hcap;
    }
    if (S.out_bf16_cap < hcap) {
        if (S.d_out_bf16) dsv4_moe_grouped_retire_ptr(S.d_out_bf16);
        CK(cudaMalloc(&S.d_out_bf16,(size_t)hcap*sizeof(__nv_bfloat16))); S.out_bf16_cap=hcap;
    }
    if (S.E_max < n_expert) {
        if (S.fc1_alpha) dsv4_moe_grouped_retire_ptr(S.fc1_alpha);
        if (S.fc2_alpha) dsv4_moe_grouped_retire_ptr(S.fc2_alpha);
        CK(cudaMalloc(&S.fc1_alpha,(size_t)n_expert*4));
        CK(cudaMalloc(&S.fc2_alpha,(size_t)n_expert*4));
        S.E_max=n_expert;
    }
    if (!S.fc1_act_global) CK(cudaMalloc(&S.fc1_act_global,4));
    if (!S.act_part)       CK(cudaMalloc(&S.act_part,(size_t)S.act_nparts*4));

    long hreal=(long)n_tokens*n_embd;
    { fprof::scope _p(stream,0);
      k_f32_to_bf16<<<grid_for(hreal),256,0,stream>>>(hidden,S.d_hidden_bf16,hreal); }

    // dynamic act global + per-(layer,call) alphas into the SHARED buffers.
    { fprof::scope _p(stream,1);
      k_absmax_part<<<S.act_nparts,256,0,stream>>>(hidden, S.act_part, hreal);
      k_absmax_final<<<1,256,0,stream>>>(S.act_part, S.act_nparts, S.fc1_act_global);
      k_fc1_alpha<<<grid_for(n_expert),256,0,stream>>>(L->g_common, S.fc1_act_global, S.fc1_alpha, n_expert);
      k_fc2_alpha<<<grid_for(n_expert),256,0,stream>>>(L->g_down,   L->fc2_act_global, S.fc2_alpha, n_expert); }

    // [ep2-dp] Expert-parallel: when this rank holds an EXPERT SHARD (g_ep set at load from the EP
    // sidecar), n_expert above is the LOCAL count (e.g. 128). flashinfer's runMoe wants the GLOBAL
    // expert count + an MOEParallelismConfig(ep_size, ep_rank); internally it does
    // num_experts_per_node = num_experts_global / ep_size, indexes the (local) weight arrays [0,local),
    // and setLocalExperts() remaps the GLOBAL token_selected_experts (our sel, [0,global)) to local
    // ids while SKIPPING remote experts (their per-token contribution = 0 on this rank). The per-rank
    // partial MoE output is then summed by the existing GGML_OP_DSV4_MOE_FUSED -> PARTIAL AllReduce
    // (1/layer). ep==0 (default / FF-split sidecar) => ep_size=1, GLOBAL==LOCAL => byte-identical. [ep2-dp]
    int ep_base=0, ep_n_global=0, ep_n_local=0;
    const int ep = dsv4_moe_get_ep_config(&ep_base, &ep_n_global, &ep_n_local);
    const int ep_size = (ep && ep_n_local>0) ? (ep_n_global / ep_n_local) : 1;
    const int ep_rank = (ep && ep_n_local>0) ? (ep_base    / ep_n_local) : 0;
    const int n_expert_global = (ep && ep_n_global>0) ? ep_n_global : n_expert;

    tkc::MOEParallelismConfig pc(1,0,ep_size,ep_rank);
    size_t ws = L->runner->getWorkspaceSize((int)tok_cap, n_embd, n_ff_exp, n_expert_global,
        n_expert_used, tkc::ActivationType::Swiglu, pc, false, false, false, false);
    if (S.workspace_cap < ws) {
        if (S.d_workspace) dsv4_moe_grouped_retire_ptr(S.d_workspace);
        CK(cudaMalloc(&S.d_workspace, ws)); S.workspace_cap=ws;
    }
    long s2d = (long)n_expert_used*tok_cap;
    if (S.src2dst_cap < s2d) {
        if (S.d_src2dst) dsv4_moe_grouped_retire_ptr(S.d_src2dst);
        CK(cudaMalloc(&S.d_src2dst,(size_t)s2d*sizeof(int))); S.src2dst_cap=s2d;
    }

    auto qp = tkc::QuantParams::FP4(
        S.fc1_act_global,
        reinterpret_cast<tkc::TmaWarpSpecializedGroupedGemmInput::NVFP4ElementSF const*>(L->fc1_sf),
        S.fc1_alpha,
        L->fc2_act_global,
        reinterpret_cast<tkc::TmaWarpSpecializedGroupedGemmInput::NVFP4ElementSF const*>(L->fc2_sf),
        S.fc2_alpha,
        false, false);

    tkc::ActivationParams act(tkc::ActivationType::SwigluBias, nullptr, nullptr, L->swiglu_limit);
    tkc::MoeMinLatencyParams mlp{};
    tk::LoraParams lora{};

    { fprof::scope _p(stream,2);
      // [256x128 TACTIC CRASH DIAG] runMoe THROWS (TLLM_CHECK_WITH_INFO -> std::runtime_error) on a
      // CUTLASS can_implement / initialize failure (e.g. a tile whose dynamic smem exceeds GB10's
      // 101376-byte opt-in cap -> cudaFuncSetAttribute fails -> Status != kSuccess). An uncaught
      // throw through this extern "C" = "died at warmup". Catch it, log the exact CUTLASS message +
      // the active tactic, and return an error so the op falls back to grouped instead of crashing.
      try {
        L->runner->runMoe(
          S.d_hidden_bf16, nullptr,
          sel, weights,
          L->fc1_w, nullptr, act,
          L->fc2_w, nullptr, qp,
          // [ep2-dp] GLOBAL num_experts (runMoe derives per-node = global/ep_size internally); the
          // weight/alpha/SF arrays are the LOCAL shard (L->E). pc carries ep_size/ep_rank.
          n_tokens, n_embd, n_ff_exp, n_expert_global, n_expert_used,
          S.d_workspace, S.d_out_bf16, S.d_src2dst,
          pc, false, false, lora,
          false, false, mlp, false, stream);
      } catch (const std::exception& ex) {
        fprintf(stderr, "[DSV4_MOE_FUSED] runMoe THREW at layer %d M=%d tactic=%d: %s\n",
                il, n_tokens, L->tactic_idx, ex.what());
        fflush(stderr);
        return cudaErrorLaunchFailure;
      }
    }

    { fprof::scope _p(stream,3);
      k_bf16_to_f32<<<grid_for(hreal),256,0,stream>>>(S.d_out_bf16, moe_out, hreal); }
    fprof::g.tick_and_maybe_dump();
    return cudaGetLastError();
}
