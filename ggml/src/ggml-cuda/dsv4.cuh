#include "common.cuh"

bool ggml_cuda_op_dsv4_hc_split_sinkhorn(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
bool ggml_cuda_op_dsv4_hc_expand(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
bool ggml_cuda_op_dsv4_hc_weighted_sum(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
bool ggml_cuda_op_dsv4_rope_tail_supported(void);
bool ggml_cuda_op_dsv4_hc_split_sinkhorn_supported(void);
bool ggml_cuda_op_dsv4_hc_expand_supported(void);
bool ggml_cuda_op_dsv4_hc_weighted_sum_supported(void);
bool ggml_cuda_op_dsv4_fp8_kv_quantize_supported(void);

bool ggml_cuda_op_dsv4_rope_tail(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
bool ggml_cuda_op_dsv4_hc_split_sinkhorn(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
bool ggml_cuda_op_dsv4_hc_expand(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
bool ggml_cuda_op_dsv4_hc_weighted_sum(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
bool ggml_cuda_op_dsv4_fp8_kv_quantize(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
bool ggml_cuda_op_dsv4_rope_tail(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
// fused split_sinkhorn + weighted_sum (the split dst is still fully written so the
// later post/comb views stay valid); returns false if the shapes don't qualify
bool ggml_cuda_op_dsv4_hc_split_sinkhorn_ws_fused(ggml_backend_cuda_context & ctx, ggml_tensor * split, ggml_tensor * ws);

// Fused DSA lightning-indexer logits (mul_mat+relu+mul+cont+sum_rows fused into one kernel).
bool ggml_cuda_op_dsv4_indexer_logits(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
bool ggml_cuda_op_dsv4_indexer_logits_supported(const ggml_tensor * dst);
bool ggml_cuda_op_dsv4_norm_rope(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
bool ggml_cuda_op_dsv4_norm_rope_supported(const ggml_tensor * dst);
