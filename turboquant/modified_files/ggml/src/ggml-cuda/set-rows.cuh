#pragma once

#include "common.cuh"

#define CUDA_SET_ROWS_BLOCK_SIZE 256

void ggml_cuda_op_set_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// AMX3 cosine-optimal λ: call once at server init.
// λ=0.7 for reasoning (default), λ=2.5 for embedding.
void ggml_cuda_set_amx3_lambda(float lambda);
