#pragma once

// DSV4 dedicated MoE grouped-GEMM (FP8 in / BF16 out, groupwise block scales).
//
// STEP 1 of the DSV4 MoE GEMM port: a torch-free C++ entry that wraps
// flashinfer's SM120 grouped-GEMM core
// (flashinfer::group_gemm::CutlassFP8GroupwiseScaledGroupGEMMSM120).
//
// This header is intentionally torch/CUTLASS-free so it can be #included from
// any ggml-cuda .cu file. All heavy CUTLASS/flashinfer template machinery lives
// in dsv4-moe-gemm.cu. The op logic is NOT implemented here yet.
//
// Hardware target: DGX Spark GB10 (sm_121a). The wrapper compiles for whatever
// the ggml-cuda target's CMAKE_CUDA_ARCHITECTURES is (already sm_121a).

#include <cstddef>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// Grouped FP8 -> BF16 GEMM with groupwise (block) scales, SM120/SM121 path.
//
// Computes, for each expert group g in [0, num_groups):
//     D[g] = (A[g] * SFA[g])  @  (B[g] * SFB[g])^T        (FP8 inputs, BF16 out)
//
// Layout (matches flashinfer SM120 grouped path):
//   A_fp8      : row-major  [m_total, k]   FP8 e4m3, all groups concatenated on M
//   B_fp8      : per-group  [num_groups, n, k] column-major FP8 e4m3 weights
//   SFA        : float groupwise scales for A (granularity M=1, K=128)
//   SFB        : float groupwise scales for B (granularity N=128, K=128)
//   D_bf16     : row-major  [m_total, n]   BF16 output, concatenated on M
//   m_indptr_dev : device int[num_groups+1] cumulative row offsets per group
//   max_m      : max rows among groups (for SFA layout when !ScaleMajorK)
//
// Workspaces (caller-owned device buffers, persistent / reusable):
//   int_workspace   : scratch for per-group pointer/stride/problem-size arrays
//   float_workspace : CUTLASS GEMM workspace
//
// Returns cudaSuccess on success; otherwise a cudaError_t describing the failure
// (e.g. cudaErrorNotSupported if can_implement fails or arch is unsupported).
cudaError_t dsv4_grouped_gemm_fp8_bf16(const void* A_fp8,
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
                                       cudaStream_t stream);

#ifdef __cplusplus
}
#endif
