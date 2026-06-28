// DSV4 dedicated MoE grouped-GEMM wrapper (STEP 1: build wiring + callable entry).
//
// Wraps flashinfer's SM120 grouped-GEMM core with the SAME template
// instantiation proven by the de-risk bench (GranM=1, GranN=128, GranK=128,
// ScaleMajorK=false, FP8 e4m3 in / BF16 out). No MoE op logic yet -- this is the
// minimal torch-free entry the future DSV4 MoE op will call.
//
// CUTLASS is heavy; this translation unit pulls in the full flashinfer/CUTLASS
// grouped-GEMM template tree. Long compile time here is expected and isolated to
// this one .cu file.

#include "dsv4-moe-gemm.cuh"

// flashinfer's SM120 grouped FP8 groupwise-scaled GEMM. This header transitively
// includes CUTLASS (cutlass_utils.cuh, allocator.h, utils.cuh). Include paths for
// flashinfer-src/include and the CUTLASS include + tools/util/include trees are
// wired into the ggml-cuda target in CMakeLists.txt.
#include "flashinfer/gemm/group_gemm_fp8_groupwise_sm120.cuh"

// Template parameters matching the proven bench instantiation (bench_v2.cu):
//   ScaleGranularityM = 1, ScaleGranularityN = 128, ScaleGranularityK = 128
//   ScaleMajorK       = false (UMMA::Major::MN scale layout)
//   DTypeIn           = cutlass::float_e4m3_t  (FP8 E4M3)
//   DTypeOut          = cutlass::bfloat16_t    (BF16)
namespace {
constexpr int kGranM = 1;
constexpr int kGranN = 128;
constexpr int kGranK = 128;
constexpr bool kScaleMajorK = false;
using DTypeIn  = cutlass::float_e4m3_t;
using DTypeOut = cutlass::bfloat16_t;
}  // namespace

extern "C" cudaError_t dsv4_grouped_gemm_fp8_bf16(const void* A_fp8,
                                                  const void* B_fp8,
                                                  const float* SFA,
                                                  const float* SFB,
                                                  void* D_bf16,
                                                  int n,
                                                  int k,
                                                  int max_m,
                                                  const int* m_indptr_dev,
                                                  int num_groups,
                                                  void* int_workspace,
                                                  size_t int_workspace_bytes,
                                                  void* float_workspace,
                                                  size_t float_workspace_bytes,
                                                  cudaStream_t stream) {
    // The flashinfer entry takes non-const operand pointers; the GEMM only reads
    // A/B/SFA/SFB, so the const_cast here is safe.
    return flashinfer::group_gemm::CutlassFP8GroupwiseScaledGroupGEMMSM120<
        kGranM, kGranN, kGranK, kScaleMajorK, DTypeIn, DTypeOut>(
        int_workspace, int_workspace_bytes,
        float_workspace, float_workspace_bytes,
        const_cast<DTypeIn*>(reinterpret_cast<const DTypeIn*>(A_fp8)),
        const_cast<DTypeIn*>(reinterpret_cast<const DTypeIn*>(B_fp8)),
        const_cast<float*>(SFA),
        const_cast<float*>(SFB),
        reinterpret_cast<DTypeOut*>(D_bf16),
        const_cast<int*>(m_indptr_dev),
        max_m, n, k, num_groups, stream);
}
