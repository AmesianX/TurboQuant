// TriAttention scoring for AMX3_1 polar cache (Part B direct consumption, dequant-free).
// Reference: Mao et al. 2026 (arxiv 2604.04921) + domvox/triattention-ggml.
#pragma once

#include "common.cuh"
#include "ggml-backend.h"

#include <cstdint>

#define TRIA_MAGIC       0x54524941u  // "TRIA"
#define TRIA_HEADER_SIZE 64
#define TRIA_N_OFFSETS   17           // geometric: 1, 2, 4, ..., 2^16

// Full calibration data (host-owned struct, data in device memory).
struct tria_stats {
    uint32_t num_layers;
    uint32_t num_heads;        // Query heads (attention heads)
    uint32_t num_kv_heads;     // KV heads (after GQA)
    uint32_t head_dim;
    uint32_t freq_count;       // head_dim / 2
    float    rope_theta;
    float    attn_scale;

    // Device memory
    float * d_layer_budget_scales;      // [num_layers]
    float * d_omega;                    // [freq_count]  RoPE frequencies
    float * d_q_mean_real_flat;         // [num_layers * num_heads * freq_count]
    float * d_q_mean_imag_flat;
    float * d_q_abs_mean_flat;
    float * d_qma_flat;                 // |E[q_f]|
};

// Load TRIA binary stats file (host-side binary read → device upload).
// Returns nullptr on error. Caller frees with tria_free().
struct tria_stats * tria_load(const char * path);
void tria_free(struct tria_stats * stats);

// Global scoring kernel launcher — computes per-slot importance scores for one layer.
void tria_score_layer(
    const struct tria_stats * stats,
    const char * K_cache,        // block_amx3_1 array on GPU
    int          n_kv,
    int          cur_pos,
    const int  * key_pos,
    int          layer_idx,
    float      * out_scores,
    float      * tmp_scores,
    cudaStream_t stream);

// Top-B selection — histogram-based threshold + mask kernel.
void tria_topb_mask(
    const float * scores,
    int           n_kv,
    int           budget,
    uint8_t     * out_mask,
    cudaStream_t  stream);

// Host-facing trigger API (called from llama-kv-cache.cpp; no CUDA types in sig).
// GGML_BACKEND_API → __declspec(dllexport/import) on Windows shared builds, plain
// extern elsewhere. Without it, libllama can't resolve these symbols at link time
// when the CUDA backend is built as a separate DLL (Windows builds failed in v1.7.0).
struct llama_tria_stats;

#ifdef __cplusplus
extern "C" {
#endif

// Read per-model-layer budget scale (host copy).  Returns 1.0 if out of range.
GGML_BACKEND_API float tria_layer_scale(struct llama_tria_stats * stats, int model_layer_idx);

// Prepare one trigger: upload positions once, reset aggregate counters.
GGML_BACKEND_API void tria_trigger_prepare(
    struct llama_tria_stats * stats,
    int                       n_kv,
    const int *               host_key_pos);

// Score + mask + attention-sink override + physical eviction for one layer.
// budget: per-layer Top-B.  keep_first: always-keep slot count (attention sink).
// K_cache_gpu is modified in place (d_wht, d_r zeroed for evicted slots).
GGML_BACKEND_API void tria_trigger_score_layer(
    struct llama_tria_stats * stats,
    int                       model_layer_idx,
    void *                    K_cache_gpu,
    int                       n_kv,
    int                       cur_pos,
    int                       budget,
    int                       keep_first);

// Finish trigger: flush aggregate log, sync stream.
GGML_BACKEND_API void tria_trigger_finish(struct llama_tria_stats * stats);

#ifdef __cplusplus
}
#endif
