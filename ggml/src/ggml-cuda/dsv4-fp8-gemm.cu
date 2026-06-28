// DSV4 native SM120 FP8 dense GEMM. See dsv4-fp8-gemm.cuh for the design.
//
// Self-contained CUTLASS sm120 blockwise-scaled FP8 GEMM (no flashinfer sync, no
// per-call alloc). Built only when the flashinfer/CUTLASS include dirs are present
// (same gate as dsv4-moe-grouped.cu, wired in ggml-cuda/CMakeLists.txt).

#include "dsv4-fp8-gemm.cuh"
#include "ggml.h"
#include "ggml-cuda.h"
#include "ggml-cuda/common.cuh"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <map>
#include <mutex>
#include <vector>
#include <cuda_fp8.h>

// Only compile the native kernel where the CUTLASS sm120 path is available.
#if defined(__CUDACC__) && (__CUDACC_VER_MAJOR__ >= 12)
#define DSV4_FP8_HAVE_CUTLASS 1
#endif

#ifdef DSV4_FP8_HAVE_CUTLASS
#include "cutlass/cutlass.h"
#include "cutlass/detail/blockwise_scale_layout.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"
#endif

// ---- ggml block layout (must match ggml-common.h block_f8_e4m3_b128) ---------
#define DSV4_F8_QK 128
struct dsv4_block_f8 { uint8_t e; uint8_t qs[DSV4_F8_QK]; };

// E8M0 -> fp32 (matches ggml_cuda_e8m0_to_fp32 in common.cuh, software path).
static __device__ __forceinline__ float dsv4_e8m0_to_fp32(uint8_t x) {
#if CUDART_VERSION >= 12080
    const nv_bfloat16 e = __nv_cvt_e8m0_to_bf16raw(x);
    return (float) e;
#else
    uint32_t bits = (x == 0) ? 0x00400000u : ((uint32_t) x << 23);
    float r; memcpy(&r, &bits, sizeof(float)); return r;
#endif
}

#define DSV4_FP8_CK(x) do { cudaError_t e_=(x); if(e_!=cudaSuccess){ \
    fprintf(stderr,"[dsv4-fp8] CUDA %s @%s:%d\n",cudaGetErrorString(e_),__FILE__,__LINE__); abort(); } } while(0)

// =============================================================================
// (1) weight unpack: ggml [K,N] F8_E4M3_B128 (row-major over N, K-blocked) ->
//     contiguous e4m3 weight [N rows x K] + per-row-per-Kblock fp32 SFA scale.
//     SFA layout (MN-major, ScaleGranularityM=1): SFA[m + kblk*N].
// =============================================================================
__global__ void dsv4_fp8_unpack_weight(const dsv4_block_f8 * __restrict__ src, // [N rows][K/128 blocks]
                                       uint8_t * __restrict__ wq,   // [N*K] raw e4m3 bytes (row-major)
                                       float   * __restrict__ sfa,  // [N * (K/128)] : sfa[m + kblk*N]
                                       int N, int K) {
    const int kb = K / DSV4_F8_QK;
    const int row = blockIdx.y;               // 0..N-1
    const int blk = blockIdx.x;               // 0..kb-1
    if (row >= N || blk >= kb) return;
    const dsv4_block_f8 & b = src[(int64_t) row * kb + blk];
    if (threadIdx.x == 0) {
        sfa[(int64_t) blk * N + row] = dsv4_e8m0_to_fp32(b.e);
    }
    // copy 128 e4m3 bytes verbatim into the contiguous row
    uint8_t * dst = wq + ((int64_t) row * K + (int64_t) blk * DSV4_F8_QK);
    for (int i = threadIdx.x; i < DSV4_F8_QK; i += blockDim.x) dst[i] = b.qs[i];
}

// =============================================================================
// (2) activation quant: F32 [K, ntok] (ggml src1, col-major over tok: element
//     (k,t) at t*K + k) -> e4m3 [ntok rows x K] + per-128-tok-block fp32 SFB.
//     One block per (token, kblk) pair; block computes amax over 128 K-values,
//     scale = amax/448, writes e4m3 bytes. SFB[nblk + kblk*ceil(ntok/128)] holds
//     the per-128-token-block scale (granularity N=128): we use the MAX of the
//     128 per-token scales in the block so every token is representable.
// =============================================================================
#define DSV4_E4M3_MAX 448.0f

__global__ void dsv4_fp8_quant_act_pertok_scale(const float * __restrict__ x, // [K, ntok], (k,t)->t*K+k
                                                float * __restrict__ tok_scale, // [ntok]
                                                int K, int ntok) {
    const int t = blockIdx.x;                 // token
    if (t >= ntok) return;
    const float * row = x + (int64_t) t * K;
    // amax over K
    float amax = 0.f;
    for (int k = threadIdx.x; k < K; k += blockDim.x) amax = fmaxf(amax, fabsf(row[k]));
    // warp+block reduce
    __shared__ float sm[32];
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, o));
    if ((threadIdx.x & 31) == 0) sm[threadIdx.x >> 5] = amax;
    __syncthreads();
    if (threadIdx.x < 32) {
        float v = (threadIdx.x < (blockDim.x + 31) / 32) ? sm[threadIdx.x] : 0.f;
        for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
        if (threadIdx.x == 0) tok_scale[t] = (v > 0.f) ? (v / DSV4_E4M3_MAX) : 1.f;
    }
}

// reduce per-token scales -> per-128-token-block scale (max), SFB[nblk + kblk*nblk_n] for all kblk
__global__ void dsv4_fp8_block_scale(const float * __restrict__ tok_scale, // [ntok]
                                     float * __restrict__ sfb, // [ ceil(ntok/128) * (K/128) ]
                                     int ntok, int K) {
    const int nblk = blockIdx.x;              // token-block index
    const int nblk_n = (ntok + 127) / 128;
    if (nblk >= nblk_n) return;
    const int t0 = nblk * 128;
    float s = 0.f;
    for (int i = threadIdx.x; i < 128; i += blockDim.x) {
        int t = t0 + i; if (t < ntok) s = fmaxf(s, tok_scale[t]);
    }
    for (int o = 16; o > 0; o >>= 1) s = fmaxf(s, __shfl_xor_sync(0xffffffff, s, o));
    __shared__ float sm[32];
    if ((threadIdx.x & 31) == 0) sm[threadIdx.x >> 5] = s;
    __syncthreads();
    if (threadIdx.x == 0) {
        float v = 0.f; int nw = (blockDim.x + 31) / 32;
        for (int w = 0; w < nw; ++w) v = fmaxf(v, sm[w]);
        if (v <= 0.f) v = 1.f;
        const int kb = K / DSV4_F8_QK;
        for (int kblk = 0; kblk < kb; ++kblk) sfb[(int64_t) kblk * nblk_n + nblk] = v;
    }
}

// quantize activations using the per-128-token-block scale: e4m3[ntok x K] row-major
__global__ void dsv4_fp8_quant_act_bytes(const float * __restrict__ x, // [K,ntok] (k,t)->t*K+k
                                         const float * __restrict__ sfb, // block scales
                                         uint8_t * __restrict__ aq,    // [ntok*K] row-major
                                         int K, int ntok) {
    const int t = blockIdx.y;
    const int k0 = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= ntok || k0 >= K) return;
    const int nblk_n = (ntok + 127) / 128;
    const int nblk = t / 128;
    const float s = sfb[(int64_t) 0 * nblk_n + nblk]; // kblk=0 (all kblk identical here)
    const float inv = 1.f / s;
    float v = x[(int64_t) t * K + k0] * inv;
    // clamp to e4m3 range and round-to-nearest via __nv_fp8_e4m3
    __nv_fp8_e4m3 q = (__nv_fp8_e4m3) fminf(fmaxf(v, -DSV4_E4M3_MAX), DSV4_E4M3_MAX);
    aq[(int64_t) t * K + k0] = *reinterpret_cast<uint8_t*>(&q);
}

#ifdef DSV4_FP8_HAVE_CUTLASS
// ---- CUTLASS sm120 blockwise FP8 GEMM (no internal sync; persistent wsp) -----
using namespace cute;
using DTypeIn  = cutlass::float_e4m3_t;
using DTypeOut = cutlass::half_t;     // accumulate to fp16 out, convert to f32 after

// granularity M=1, N=128, K=128, MN-major (ScaleMajorK=false)
using ScaleConfig = cutlass::detail::Sm120BlockwiseScaleConfig<1, 128, 128,
        cutlass::UMMA::Major::MN, cutlass::UMMA::Major::MN>;
using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());

using ElementA = DTypeIn;  using LayoutA = cutlass::layout::RowMajor;
using ElementB = DTypeIn;  using LayoutB = cutlass::layout::ColumnMajor;
using ElementC = DTypeOut; using LayoutC = cutlass::layout::RowMajor;
using ElementD = ElementC; using LayoutD = LayoutC;
constexpr int AlignA = 128 / cutlass::sizeof_bits<ElementA>::value;
constexpr int AlignB = 128 / cutlass::sizeof_bits<ElementB>::value;
constexpr int AlignC = 128 / cutlass::sizeof_bits<ElementC>::value;
using ElementAcc = float; using ElementCompute = float;
using MmaTile = Shape<_128,_128,_128>;
using Cluster = Shape<_1,_1,_1>;

using CollEpi = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp, MmaTile, Cluster,
    cutlass::epilogue::collective::EpilogueTileAuto, ElementAcc, ElementCompute, ElementC,
    LayoutC, AlignC, ElementD, LayoutD, AlignC,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;
using StageCount = cutlass::gemm::collective::StageCountAutoCarveout<
    static_cast<int>(sizeof(typename CollEpi::SharedStorage))>;
using CollMain = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp, ElementA,
    cute::tuple<LayoutA, LayoutSFA>, AlignA, ElementB, cute::tuple<LayoutB, LayoutSFB>,
    AlignB, ElementAcc, MmaTile, Cluster, StageCount,
    cutlass::gemm::KernelScheduleSm120Blockwise>::CollectiveOp;
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<Shape<int,int,int,int>, CollMain, CollEpi, void>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
#endif // DSV4_FP8_HAVE_CUTLASS

// =============================================================================
// persistent per-weight cache (keyed by the ggml weight device ptr)
// =============================================================================
struct dsv4_fp8_weight {
    uint8_t * wq  = nullptr;   // [N*K] e4m3 (persistent, ~6GB total across all dense F8)
    float   * sfa = nullptr;   // [N*(K/128)]
    int N = 0, K = 0;
#ifdef DSV4_FP8_HAVE_CUTLASS
    // per-weight CUTLASS Gemm + workspace ONLY: the Params/workspace are the mutable
    // state that raced when shared across the 7 projection shapes. The transient I/O
    // buffers (aq/sfb/dout) are SHARED globally (g_st) — they are written then consumed
    // within one call on the SAME stream, so stream-ordering makes sharing safe AND
    // avoids ~2GB of per-weight dout duplication (watchdog OOM otherwise).
    void * wsp = nullptr;  size_t wsp_cap = 0;
    Gemm gemm;
#endif
};

struct dsv4_fp8_state {
    std::mutex mtx;
    std::map<const void*, dsv4_fp8_weight> weights;  // keyed by ggml weight device ptr
    // shared transient I/O buffers (grown to the max shape seen; same-stream-ordered).
    uint8_t * aq = nullptr;  size_t aq_cap = 0;       // [ntok*K]
    float * tok_scale = nullptr; size_t ts_cap = 0;   // [ntok]
    float * sfb = nullptr; size_t sfb_cap = 0;        // [ceil(ntok/128)*(K/128)]
#ifdef DSV4_FP8_HAVE_CUTLASS
    half * dout = nullptr; size_t dout_cap = 0;       // [N*ntok] fp16 out
#endif
    int cc = 0;
    bool enabled_checked = false, enabled = false;
};
static dsv4_fp8_state g_st;

bool dsv4_fp8_native_enabled(void) {
    if (!g_st.enabled_checked) {
        g_st.enabled_checked = true;
        g_st.enabled = getenv("DSV4_FP8_NATIVE") != nullptr;
#ifndef DSV4_FP8_HAVE_CUTLASS
        g_st.enabled = false;
#endif
        if (g_st.enabled) {
            int dev = 0; cudaGetDevice(&dev);
            cudaDeviceProp p; cudaGetDeviceProperties(&p, dev);
            g_st.cc = p.major * 10 + p.minor;
            if (g_st.cc < 120) g_st.enabled = false; // sm120/121 only
            fprintf(stderr, "[dsv4-fp8] native FP8 dense GEMM %s (cc=%d)\n",
                    g_st.enabled ? "ENABLED" : "disabled", g_st.cc);
        }
    }
    return g_st.enabled;
}

static dsv4_fp8_weight & get_weight(const void * key, const dsv4_block_f8 * src, int N, int K, cudaStream_t s) {
    auto it = g_st.weights.find(key);
    if (it != g_st.weights.end()) return it->second;
    // construct the entry in place (dsv4_fp8_weight holds a CUTLASS Gemm which is
    // not trivially copyable -> std::map::operator[] default-constructs in place).
    dsv4_fp8_weight & w = g_st.weights[key];
    w.N = N; w.K = K;
    const int kb = K / DSV4_F8_QK;
    DSV4_FP8_CK(cudaMalloc(&w.wq,  (size_t) N * K));
    DSV4_FP8_CK(cudaMalloc(&w.sfa, (size_t) N * kb * sizeof(float)));
    dim3 grid(kb, N);
    dsv4_fp8_unpack_weight<<<grid, 128, 0, s>>>(src, w.wq, w.sfa, N, K);
    DSV4_FP8_CK(cudaStreamSynchronize(s)); // one-time, outside capture
    return w;
}

void dsv4_fp8_native_free_all(void) {
    std::lock_guard<std::mutex> lk(g_st.mtx);
    auto F=[](void*&p){ if(p){ cudaFree(p); p=nullptr; } };
    for (auto & kv : g_st.weights) {
        auto & w = kv.second;
        F((void*&)w.wq); F((void*&)w.sfa);
#ifdef DSV4_FP8_HAVE_CUTLASS
        F((void*&)w.wsp);
#endif
    }
    g_st.weights.clear();
    F((void*&)g_st.aq); g_st.aq_cap=0;
    F((void*&)g_st.tok_scale); g_st.ts_cap=0;
    F((void*&)g_st.sfb); g_st.sfb_cap=0;
#ifdef DSV4_FP8_HAVE_CUTLASS
    F((void*&)g_st.dout); g_st.dout_cap=0;
#endif
}

// ---- reference dequant GEMM (for DSV4_FP8_VERIFY only) ----------------------
// Naive D[n,t] = sum_k (e8m0_scale[n,kblk]*e4m3(qs)) * x[k,t]. Slow; verify-only.
__global__ void dsv4_fp8_ref_gemm(const dsv4_block_f8 * __restrict__ W, // [N][K/128]
                                  const float * __restrict__ X,         // [K,ntok] (k,t)->t*K+k
                                  float * __restrict__ Dref,            // [ne0=N,ne1=ntok] (n,t)->t*N+n
                                  int N, int K, int ntok) {
    const int t = blockIdx.y;
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= ntok || n >= N) return;
    const int kb = K / DSV4_F8_QK;
    const dsv4_block_f8 * row = W + (int64_t) n * kb;
    const float * xcol = X + (int64_t) t * K;
    float acc = 0.f;
    for (int b = 0; b < kb; ++b) {
        const float d = dsv4_e8m0_to_fp32(row[b].e);
        const uint8_t * qs = row[b].qs;
        #pragma unroll 4
        for (int i = 0; i < DSV4_F8_QK; ++i) {
            // e4m3fn decode (matches ggml_cuda_f8_e4m3fn_to_fp32)
            uint8_t x = qs[i]; float val;
            if ((x & 0x7F) == 0) val = 0.f;
            else { int exp=(x>>3)&0xF, man=x&0x7;
                   val = exp==0 ? ldexpf((float)man,-9) : ldexpf(1.f+(float)man*0.125f, exp-7);
                   if (x & 0x80) val = -val; }
            acc += d * val * xcol[b*DSV4_F8_QK + i];
        }
    }
    Dref[(int64_t) t * N + n] = acc;
}

#ifdef DSV4_FP8_HAVE_CUTLASS
// convert fp16 dout [N x ntok] row-major -> ggml dst f32 [ne0=N, ne1=ntok] (col-major over tok: (n,t)->t*N+n)
__global__ void dsv4_fp8_out_to_f32(const half * __restrict__ din, float * __restrict__ dout, int N, int ntok) {
    const int t = blockIdx.y;
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= ntok || n >= N) return;
    // din row-major [N, ntok]: (n,t) -> n*ntok + t ; dst ggml: (n,t) -> t*N + n
    dout[(int64_t) t * N + n] = __half2float(din[(int64_t) n * ntok + t]);
}
#endif

bool ggml_cuda_dsv4_fp8_native_mul_mat(ggml_backend_cuda_context & ctx,
                                       const ggml_tensor * src0,
                                       const ggml_tensor * src1,
                                       ggml_tensor * dst) {
#ifndef DSV4_FP8_HAVE_CUTLASS
    (void)ctx;(void)src0;(void)src1;(void)dst; return false;
#else
    if (!dsv4_fp8_native_enabled()) return false;
    if (src0->type != GGML_TYPE_F8_E4M3_B128) return false;
    if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) return false;
    // 2D only (batched / 3D handled by the default path)
    if (src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1) return false;
    const int K = (int) src0->ne[0];
    const int N = (int) src0->ne[1];   // n_out
    const int ntok = (int) src1->ne[1];
    if ((K % DSV4_F8_QK) != 0) return false;
    if ((int) src1->ne[0] != K) return false;
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) return false;
    // CUTLASS sm120 e4m3 requires 16-element alignment on every GEMM extent
    // (AlignmentA/B/C = 128/8). K is always a multiple of 128; guard N and ntok.
    // Non-aligned tails (e.g. the last prefill chunk, or a 1-token decode) fall
    // back to the cuBLAS path -> still correct, just not native.
    if ((N % 16) != 0 || (ntok % 16) != 0) return false;

    cudaStream_t s = ctx.stream();

    // [CAPTURE SAFETY] cudaMalloc / cudaStreamSynchronize / cutlass init are illegal
    // during CUDA graph capture. If a cold weight or a workspace grow would be needed
    // while capturing, bail to the cuBLAS path for THIS op (returns false). The very
    // first (non-captured) prefill chunk warms every weight + the largest workspace,
    // so steady-state captured replays never hit an allocation here.
    cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
    cudaStreamIsCapturing(s, &cap);
    const bool capturing = (cap != cudaStreamCaptureStatusNone);

    const int kb = K / DSV4_F8_QK;
    const int nblk_n = (ntok + 127) / 128;

    std::lock_guard<std::mutex> lk(g_st.mtx);

    // cold weight needs unpack (malloc+sync) -> only outside capture
    const bool cold = g_st.weights.find(src0->data) == g_st.weights.end();
    if (capturing && cold) return false;

    dsv4_fp8_weight & w = get_weight(src0->data, (const dsv4_block_f8*) src0->data, N, K, s);

    // grow shared transient buffers + per-weight wsp -> only outside capture
    const bool need_grow =
        g_st.aq_cap   < (size_t) ntok * K ||
        g_st.ts_cap   < (size_t) ntok * sizeof(float) ||
        g_st.sfb_cap  < (size_t) nblk_n * kb * sizeof(float) ||
        g_st.dout_cap < (size_t) N * ntok * sizeof(half);
    if (capturing && need_grow) return false;

    auto ensure = [&](void ** p, size_t * cap_, size_t need) {
        if (*cap_ < need) { if (*p) cudaFree(*p); DSV4_FP8_CK(cudaMalloc(p, need)); *cap_ = need; }
    };
    ensure((void**)&g_st.aq, &g_st.aq_cap, (size_t) ntok * K);
    ensure((void**)&g_st.tok_scale, &g_st.ts_cap, (size_t) ntok * sizeof(float));
    ensure((void**)&g_st.sfb, &g_st.sfb_cap, (size_t) nblk_n * kb * sizeof(float));
    ensure((void**)&g_st.dout, &g_st.dout_cap, (size_t) N * ntok * sizeof(half));

    // (2) activation quant (shared transient buffers; same-stream-ordered)
    {
        int thr = 256;
        dsv4_fp8_quant_act_pertok_scale<<<ntok, thr, 0, s>>>((const float*)src1->data, g_st.tok_scale, K, ntok);
        dsv4_fp8_block_scale<<<nblk_n, 256, 0, s>>>(g_st.tok_scale, g_st.sfb, ntok, K);
        dim3 g((K + 255)/256, ntok); dsv4_fp8_quant_act_bytes<<<g, 256, 0, s>>>((const float*)src1->data, g_st.sfb, g_st.aq, K, ntok);
    }

    // (3) native GEMM: A=weight[M=N,K] rowmajor e4m3, B=act[N=ntok,K] colmajor e4m3,
    //     D=[N,ntok] rowmajor fp16. SFA=w.sfa (m+kblk*N), SFB=g_st.sfb (nblk+kblk*nblk_n).
    const int M = N, Ngemm = ntok, Kg = K, L = 1;
    auto stride_A = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideA{}, cute::make_shape(M, Kg, L));
    auto stride_B = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideB{}, cute::make_shape(Ngemm, Kg, L));
    auto stride_C = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideC{}, cute::make_shape(M, Ngemm, L));
    auto stride_D = stride_C;
    auto layout_SFA = ScaleConfig::tile_atom_to_shape_SFA(cute::make_shape(M, Ngemm, Kg, L));
    auto layout_SFB = ScaleConfig::tile_atom_to_shape_SFB(cute::make_shape(M, Ngemm, Kg, L));

    ElementA * A = reinterpret_cast<ElementA*>(w.wq);
    ElementB * B = reinterpret_cast<ElementB*>(g_st.aq);
    ElementD * D = reinterpret_cast<ElementD*>(g_st.dout);

    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, Ngemm, Kg, L},
        {A, stride_A, B, stride_B, w.sfa, layout_SFA, g_st.sfb, layout_SFB},
        {{}, D, stride_C, D, stride_D}};
    args.epilogue.thread.alpha = 1.0f;
    args.epilogue.thread.beta  = 0.0f;

    size_t need_wsp = Gemm::get_workspace_size(args);
    if (w.wsp_cap < need_wsp) {
        if (capturing) return false; // can't grow workspace mid-capture; fall back
        if (w.wsp) cudaFree(w.wsp);
        DSV4_FP8_CK(cudaMalloc(&w.wsp, need_wsp)); w.wsp_cap = need_wsp;
    }

    cutlass::Status st = w.gemm.can_implement(args);
    if (st != cutlass::Status::kSuccess) return false; // fall back to cuBLAS
    st = w.gemm.initialize(args, w.wsp, s);
    if (st != cutlass::Status::kSuccess) return false;
    st = w.gemm.run(s);
    if (st != cutlass::Status::kSuccess) return false;

    // (4) fp16 [N,ntok] rowmajor -> f32 ggml dst [ne0=N, ne1=ntok]
    dim3 go((N + 255)/256, ntok);
    dsv4_fp8_out_to_f32<<<go, 256, 0, s>>>(g_st.dout, (float*) dst->data, N, ntok);

    // [DSV4_FP8_DEBUG_SYNC] force a stream sync + error check after every native
    // GEMM (only when NOT capturing). Diagnoses async illegal-access / ordering bugs:
    // if the path is stable WITH this and crashes WITHOUT, the bug is async. Default
    // off (DSV4_FP8_DEBUG_SYNC unset) so steady state stays capture-safe + fast.
    static const bool dbg_sync = getenv("DSV4_FP8_DEBUG_SYNC") != nullptr;
    if (dbg_sync && !capturing) {
        cudaError_t le = cudaStreamSynchronize(s);
        if (le != cudaSuccess) {
            fprintf(stderr, "[dsv4-fp8] ASYNC ERROR N=%d K=%d ntok=%d : %s\n",
                    N, K, ntok, cudaGetErrorString(le)); fflush(stderr);
        }
    }

    // [DSV4_FP8_VERIFY] one-time per-shape numeric check vs the dequant reference.
    // ONLY when NOT capturing (does its own malloc/sync). Reports max/mean rel error.
    static const bool verify = getenv("DSV4_FP8_VERIFY") != nullptr;
    if (verify && !capturing) {
        // surface any prep/GEMM/out launch error precisely (warmup only)
        cudaError_t le = cudaStreamSynchronize(s);
        if (le != cudaSuccess) {
            fprintf(stderr,"[dsv4-fp8] LAUNCH ERROR after GEMM N=%d K=%d ntok=%d : %s\n",
                    N,K,ntok,cudaGetErrorString(le));
            fflush(stderr);
            return false; // fall back; do not trust
        }
        static std::map<uint64_t,bool> seen;
        uint64_t shapekey = ((uint64_t)N<<40) ^ ((uint64_t)K<<20) ^ (uint64_t)ntok;
        if (seen.find(shapekey) == seen.end()) {
            seen[shapekey] = true;
            float * dref = nullptr; DSV4_FP8_CK(cudaMalloc(&dref, (size_t)N*ntok*sizeof(float)));
            dim3 gr((N+255)/256, ntok);
            dsv4_fp8_ref_gemm<<<gr,256,0,s>>>((const dsv4_block_f8*)src0->data,(const float*)src1->data,dref,N,K,ntok);
            DSV4_FP8_CK(cudaStreamSynchronize(s));
            std::vector<float> hn((size_t)N*ntok), hr((size_t)N*ntok);
            DSV4_FP8_CK(cudaMemcpy(hn.data(), dst->data, hn.size()*sizeof(float), cudaMemcpyDeviceToHost));
            DSV4_FP8_CK(cudaMemcpy(hr.data(), dref, hr.size()*sizeof(float), cudaMemcpyDeviceToHost));
            double maxrel=0, sumrel=0, refnorm=0, errnorm=0; int cnt=0;
            for (size_t i=0;i<hn.size();++i){ double r=hr[i],v=hn[i]; refnorm+=r*r; errnorm+=(v-r)*(v-r);
                double den=fabs(r)>1e-3?fabs(r):1e-3; double rel=fabs(v-r)/den; maxrel=rel>maxrel?rel:maxrel; sumrel+=rel; ++cnt; }
            fprintf(stderr,"[dsv4-fp8] VERIFY N=%d K=%d ntok=%d : max_rel=%.4f mean_rel=%.5f rel_l2=%.5f\n",
                    N,K,ntok,maxrel,sumrel/cnt, sqrt(errnorm/(refnorm+1e-12)));
            cudaFree(dref);
        }
    }
    return true;
#endif
}
