#pragma once

// DSV4 native SM120 FP8 (F8_E4M3_B128) dense GEMM.
//
// The DSV4-Flash dense projections (MLA attn_q_a/q_b/kv_latent/output_a/output_b
// and the shared-expert FFN gate/up/down) are stored as GGML_TYPE_F8_E4M3_B128:
// per-row, per-128-K-block layout { uint8 e (E8M0 scale); uint8 qs[128] (e4m3fn) }.
//
// The default ggml path dequantizes the whole FP8 weight to F16 every forward and
// runs a cuBLAS F16 GEMM (ggml_cuda_op_mul_mat_cublas). On a 13k prefill this dense
// FP8 GEMM class is the #1 floor term (~38% of GPU time, measured DSV4_KERNEL_PROF).
//
// This module routes those GEMMs to a NATIVE sm120 FP8 tensor-core GEMM (CUTLASS
// blockwise-scaled, flashinfer CutlassGroupwiseScaledGEMMSM120 core). The weight FP8
// mantissa bytes are used verbatim (already e4m3fn); the per-row E8M0 scale is
// expanded to FP32 ONCE per weight (persistent, keyed by the weight device ptr).
// Activations (F32) are quantized to FP8 with per-128-token-block FP32 scales each
// call into a persistent workspace. No per-call cudaMalloc / thrust / sync -> the
// path is CUDA-graph capture-safe.
//
// Operand mapping (fits the kernel's scale granularity M in {1,128}, N=128, K=128):
//   A = weight      [M = n_out, K]  ScaleGranularityM = 1   (exact per-row weight scale)
//   B = activations [N = n_tok, K]  ScaleGranularityN = 128 (per-128-token block)
//   D = [n_out, n_tok] row-major  (== ggml dst [ne0=n_out, ne1=n_tok])
//
// Everything is gated behind DSV4_FP8_NATIVE=1. With the flag off this code path is
// never reached and the dense FP8 GEMMs are byte-identical to the cuBLAS dequant path.

#include "common.cuh"
#include <cstdint>

// Whether the native FP8 path is enabled (env DSV4_FP8_NATIVE) AND the device is sm120/121.
bool dsv4_fp8_native_enabled(void);

// Try to run dst = src0(F8_E4M3_B128 weight) x src1(F32 activations) natively.
// Returns true if handled, false to fall through to the default cuBLAS path
// (e.g. unsupported shape, k not a multiple of 128, or native disabled).
struct ggml_backend_cuda_context;
struct ggml_tensor;
bool ggml_cuda_dsv4_fp8_native_mul_mat(ggml_backend_cuda_context & ctx,
                                       const ggml_tensor * src0,
                                       const ggml_tensor * src1,
                                       ggml_tensor * dst);

// Free all persistent state (weight scale caches, workspaces). Called on backend free.
void dsv4_fp8_native_free_all(void);
