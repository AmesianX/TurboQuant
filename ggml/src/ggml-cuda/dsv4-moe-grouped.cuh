#pragma once

// DSV4 NVFP4 (W4A4) grouped-GEMM MoE op (STEP 2b).
//
// A self-contained DSV4 MoE expert path: sort tokens by selected expert ->
// 3x NVFP4 grouped GEMM (gate, up, then down) -> SwiGLU -> router-weight scale
// -> scatter-combine. The whole pipeline (kernels + grouped-GEMM instantiation +
// SF swizzle layout) is the validated reference from dsv4_moe_nvfp4_test.cu
// (mean_rel_err 0.024 vs an NVFP4-quantized W4A4 reference).
//
// The expert weights live in a per-layer device-side REGISTRY populated by the
// load adapter (dsv4_moe_grouped_set_expert_weights), NOT in ggml tensors, since
// they are NVFP4 (packed e2m1 + ue4m3 block scales in a hardware-swizzled CUTLASS
// tile-atom layout + per-expert fp32 global scale). The ggml custom op only
// carries the layer index in op_params; the CUDA dispatch looks the weights up.
//
// Everything here is gated behind DSV4_MOE_GROUPED=1 in the model graph; with the
// flag off this code path is never reached and the default mul_mat_id MoE is
// byte-identical.

#include "common.cuh"

#include <cstdint>
#include <cstddef>
#include <vector>

// ---- load adapter -----------------------------------------------------------
// Convert one DSV4 layer's MXFP4 expert weights -> NVFP4 and stash on the device.
//
//   gate/up host blocks : [n_expert][n_ff_exp * n_embd] MXFP4 (ggml block_mxfp4, QK=32)
//   down  host blocks   : [n_expert][n_embd  * n_ff_exp] MXFP4
// Each weight tensor is row-major [n, k] per expert (n = output dim, k = input dim),
// matching ggml's expert tensor storage [k(=ne0), n(=ne1), n_expert(=ne2)].
//
// host pointers are raw ggml MXFP4 bytes (sizeof(block_mxfp4)=18, 32 elems/block).
// This is a one-time conversion at model load; safe to call once per layer.
void dsv4_moe_grouped_set_expert_weights(int il,
                                         const void * gate_mxfp4,
                                         const void * up_mxfp4,
                                         const void * down_mxfp4,
                                         int n_expert,
                                         int n_embd,
                                         int n_ff_exp);

// ---- OFFLINE pre-conversion (sidecar) ---------------------------------------
// To avoid holding MXFP4 experts resident next to the NVFP4 registry (~2x expert
// memory -> OOM) the conversion is done ONCE OFFLINE into per-rank sidecar files.
// The exact NVFP4 bytes the device registry holds for ONE layer / ONE rank are a
// deterministic function of (n_expert, n_embd, n_ff_half). These two host-side
// entry points let the offline tool produce that blob and the engine upload it
// with NO conversion and NO MXFP4 ever touching the device.
//
// Blob format + the host entry points (convert_layer / set_expert_weights_blob)
// live in this CUTLASS-free POD header, shared with the offline tool + the engine.
#include "dsv4-moe-grouped-blob.h"

// Whether layer il already has NVFP4 weights registered.
bool dsv4_moe_grouped_have_layer(int il);

// ---- EP (expert-parallel) config --------------------------------------------
// Set ONCE at load (from the EP sidecar file header) so the FUSED op can hand
// flashinfer's runMoe the right MOEParallelismConfig. When ep=0 (default / FF-split
// sidecar) the fused op runs the full local expert set with no parallelism (ep_size=1).
//   expert_base    : GLOBAL id of this rank's local expert 0
//   n_expert_global: total experts across all ranks (e.g. 256)
//   n_expert_local : experts registered on THIS rank (e.g. 128); ep_size = global/local.
extern "C" void dsv4_moe_set_ep_config(int ep, int expert_base, int n_expert_global, int n_expert_local);
// Read back the EP config (returns ep flag; out-params may be null). ep_size = global/local.
extern "C" int  dsv4_moe_get_ep_config(int* expert_base, int* n_expert_global, int* n_expert_local);

// Free all registered NVFP4 weights (called on model free / between runs).
void dsv4_moe_grouped_free_all(void);

// ---- the op -----------------------------------------------------------------
bool ggml_cuda_op_dsv4_moe_grouped(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
bool ggml_cuda_op_dsv4_moe_grouped_supported(void);
