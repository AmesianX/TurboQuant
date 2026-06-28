#pragma once

// DSV4 NVFP4 (W4A4) FUSED MoE op (flashinfer CUTLASS CutlassMoeFCRunner).
//
// Replaces the staging-arena grouped-GEMM path (dsv4-moe-grouped.cu) with the
// flashinfer fused MoE runner that does, in a SINGLE call:
//   dispatch/permute -> grouped W1 GEMM -> SwiGLU(+per-expert limit) -> grouped
//   W2 GEMM -> finalize (token_final_scales + unpermute).
// The fusion eliminates the per-layer x per-token bf16 staging arena
// (pf_gate/pf_up/pf_act_full/pf_down) that is the ub-ceiling / OOM cause of the
// grouped path, replacing 58 persistent per-layer arenas with ONE batch-scoped
// CUTLASS workspace (getWorkspaceSize).
//
// Hardware target: GB10 sm_121a. The valid SM120/121 CUTLASS MoE combo is NVFP4
// weight + NVFP4 activation ONLY (isValidSM120MOESpecialisation); tile config
// from the 1x1x1-cluster, K<=128 allowlist (256x128x64 = the 356-TFLOPS forum
// config). See turboquant/DSV4_CUTLASS_FUSED_MOE_PORT.md.
//
// The expert weights are the SAME per-layer device registry the grouped op fills
// from the NVFP4 sidecar (dsv4-moe-grouped-blob.h). This op only carries the
// layer index + per-layer SwiGLU clamp in op_params and looks the weights up.
//
// Gated behind DSV4_MOE_FUSED=1 in the model graph. Default OFF -> grouped path
// is selected and this code is never reached (byte-identical regression-safe).

#include "common.cuh"

// ---- the op -----------------------------------------------------------------
bool ggml_cuda_op_dsv4_moe_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// Whether the fused MoE op can run for the given build/runtime (sm_121a, the
// CUTLASS FP4 MoE TU compiled in, and the requested layer's weights registered).
bool ggml_cuda_op_dsv4_moe_fused_supported(void);
