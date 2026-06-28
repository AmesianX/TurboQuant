#pragma once
#include "common.cuh"

// DSV4 sparse-MLA flash attention (k4 kernel wired into ggml), gated by DSV4_SPARSE_ATTN + src[6].
bool ggml_cuda_fattn_sparse_mla_supported(const ggml_tensor * dst);
void ggml_cuda_flash_attn_ext_sparse_mla(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
