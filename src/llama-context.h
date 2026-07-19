#pragma once

#include "llama.h"
#include "llama-ext.h"
#include "llama-cparams.h"
#include "llama-graph.h"
#include "llama-adapter.h"
#include "llama-impl.h"

#include "ggml-cpp.h"
#include "ggml-opt.h"

#include <map>
#include <vector>

struct llama_model;
class llama_batch_allocr;

class llama_io_read_i;
class llama_io_write_i;

// "memory" as in abstract memory for the context
struct llama_memory_i;
struct llama_memory_context_i;

// stores copy of the memory in device buffer. used for fast state save/load
struct llama_memory_buffer {
    int n_tensors = 0;
    size_t total_size = 0;

    ggml_backend_buffer_ptr buf;

    ggml_context_ptr ctx;

    std::vector<ggml_tensor *> org;
    std::vector<ggml_tensor *> cpy;
};

using llama_memory_buffers = std::map<ggml_backend_buffer_type_t, llama_memory_buffer>;

struct llama_context {
    // init scheduler and compute buffers, reserve worst-case graphs
    llama_context(
            const llama_model & model,
                  llama_context_params params);

    ~llama_context();

    // reserve a new backend scheduler (if needed)
    // for example, when:
    //   - changing loras
    //   - changing samplers
    //   - changing attention type
    //   - etc.
    void sched_reserve();

    void synchronize();

    const llama_model   & get_model()   const;
    const llama_cparams & get_cparams() const;

    ggml_backend_sched_t get_sched() const;

    uint32_t n_ctx()     const;
    uint32_t n_ctx_seq() const;
    uint32_t n_batch()   const;
    uint32_t n_ubatch()  const;
    uint32_t n_seq_max() const;

    uint32_t n_threads()       const;
    uint32_t n_threads_batch() const;

    llama_memory_t get_memory() const;

    // return true if the memory was updated
    bool memory_update(bool optimize);

    enum llama_pooling_type pooling_type() const;

    float * get_logits();
    float * get_logits_ith(int32_t i);

    float * get_embeddings();
    float * get_embeddings_ith(int32_t i);
    float * get_embeddings_seq(llama_seq_id seq_id);

    float * get_embeddings_pre_norm();
    float * get_embeddings_pre_norm_ith(int32_t i);

    llama_token get_mtp_draft_ith(int32_t i); // [DSV4_MTP_FOLD] folded greedy draft token at token i

    llama_token * get_sampled_tokens() const;
    llama_token   get_sampled_token_ith(int32_t idx);

    float * get_sampled_logits_ith(int32_t idx);
    size_t  get_sampled_logits_count(int32_t idx);

    float * get_sampled_probs_ith(int32_t idx);
    size_t  get_sampled_probs_count(int32_t idx);

    const llama_token * get_sampled_candidates_ith(int32_t idx);
    size_t get_sampled_candidates_count(int32_t idx);

    void attach_threadpool(
            ggml_threadpool_t threadpool,
            ggml_threadpool_t threadpool_batch);

    void detach_threadpool();

    void set_n_threads(int32_t n_threads, int32_t n_threads_batch);

    void set_abort_callback(bool (*abort_callback)(void * data), void * abort_callback_data);

    void set_embeddings (bool value);
    void set_embeddings_pre_norm(bool value, bool masked);
    void set_causal_attn(bool value);
    void set_warmup(bool value);

    // inject cross-attention context embeddings (host copy) — used by the DFlash drafter to
    // feed the raw stacked target-layer features that its decode graph fuses (fc) into target_ctx.
    void set_cross_embd(const float * data, int32_t n_embd, int32_t n_enc);

    // set the per-node evaluation callback after context creation (DFlash uses it on ctx_tgt to
    // capture hc_ffn_post at the drafter's target layers during normal target decode).
    void set_eval_callback(ggml_backend_sched_eval_callback cb, void * user);

    // DFlash: enable in-graph capture of stacked n_hc-collapsed hc_ffn_post (no per-op eval cb,
    // so CUDA graphs stay enabled). Read back via dflash_feat after decode.
    void set_dflash_capture(bool value);
    const float * get_dflash_feat_host(int32_t * n_tokens, int32_t * dim);

    void set_adapters_lora(llama_adapter_lora ** adapters, size_t n_adapters, float * scales);

    bool adapters_lora_are_same(llama_adapter_lora ** adapters, size_t n_adapters, float * scales);

    bool set_adapter_cvec(
            const float * data,
                 size_t   len,
                int32_t   n_embd,
                int32_t   il_start,
                int32_t   il_end);

    // process a single ubatch with a specific graph type
    // if memory_context is provided, it will be applied first to the context's memory
    // ret contains the status of the graph computation
    // returns nullptr only if ret != GGML_STATUS_SUCCESS
    llm_graph_result * process_ubatch(
                const llama_ubatch & ubatch,
                    llm_graph_type   gtype,
            llama_memory_context_i * mctx,
                       ggml_status & ret);

    int encode(const llama_batch & batch_inp);
    int decode(const llama_batch & batch_inp);

    //
    // state save/load
    //

    size_t state_get_size();
    size_t state_get_data(      uint8_t * dst, size_t size);
    size_t state_set_data(const uint8_t * src, size_t size);

    size_t state_seq_get_size(llama_seq_id seq_id, llama_state_seq_flags flags);

    size_t state_seq_get_data(llama_seq_id seq_id,       uint8_t * dst, size_t size, llama_state_seq_flags flags);
    size_t state_seq_set_data(llama_seq_id seq_id, const uint8_t * src, size_t size, llama_state_seq_flags flags);

    bool state_load_file(
            const char * filepath,
           llama_token * tokens_out,
                size_t   n_token_capacity,
                size_t * n_token_count_out);

    bool state_save_file(
            const char * filepath,
     const llama_token * tokens,
                size_t   n_token_count);

    size_t state_seq_load_file(
          llama_seq_id   seq_id,
            const char * filepath,
           llama_token * tokens_out,
                size_t   n_token_capacity,
                size_t * n_token_count_out);

    size_t state_seq_save_file(
          llama_seq_id   seq_id,
            const char * filepath,
     const llama_token * tokens,
                size_t   n_token_count);

    //
    // perf
    //

    llama_perf_context_data perf_get_data() const;
    void perf_reset();

    llama_memory_breakdown memory_breakdown() const;

    //
    // training
    //

    void opt_init(struct llama_model * model, struct llama_opt_params lopt_params);

    // TODO: more flexible combinations of logical/physical batch size and context size
    void opt_epoch(
            ggml_opt_dataset_t      dataset,
            ggml_opt_result_t       result_train,
            ggml_opt_result_t       result_eval,
            int64_t                 idata_split,
            ggml_opt_epoch_callback callback_train,
            ggml_opt_epoch_callback callback_eval);

    void opt_epoch_iter(
            ggml_opt_dataset_t               dataset,
            ggml_opt_result_t                result,
            const std::vector<llama_token> & tokens,
            const std::vector<llama_token> & labels_sparse,
            llama_batch                    & batch,
            ggml_opt_epoch_callback          callback,
            bool                             train,
            int64_t                          idata_in_loop,
            int64_t                          ndata_in_loop,
            int64_t                          t_loop_start);

private:
    //
    // output
    //

    // Make sure enough space is available for outputs.
    // Returns max number of outputs for which space was reserved.
    uint32_t output_reserve(int32_t n_outputs);

    void output_reorder();

    // map the output row index `i` to batch index
    int64_t output_resolve_row(int32_t i) const;

    //
    // graph
    //

public:
    uint32_t graph_max_nodes(uint32_t n_tokens) const;

    // can reuse the llm_graph_result instance of the context (for example to update a memory module)
    llm_graph_result * get_gf_res_reserve() const;

    // returns the result of ggml_backend_sched_graph_compute_async execution
    ggml_status graph_compute(ggml_cgraph * gf, bool batched);

    // reserve a graph with a dummy ubatch of the specified size
    ggml_cgraph * graph_reserve(
        uint32_t n_tokens, uint32_t n_seqs, uint32_t n_outputs, const llama_memory_context_i * mctx, bool split_only = false, size_t * sizes = nullptr,
        llama_pos pos0 = 0);

    bool set_sampler(llama_seq_id seq_id, llama_sampler * sampler);

    void tria_set(struct llama_tria_stats * stats, int32_t budget, int32_t interval, int32_t keep_first);

private:
    llm_graph_params graph_params(
                        llm_graph_result * res,
                      const llama_ubatch & ubatch,
            const llama_memory_context_i * mctx,
                          llm_graph_type   gtype) const;

    llm_graph_cb graph_get_cb() const;

    // TODO: read/write lora adapters and cvec
    size_t state_write_data(llama_io_write_i & io);
    size_t state_read_data (llama_io_read_i  & io);

    size_t state_seq_write_data(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags);
    size_t state_seq_read_data (llama_io_read_i  & io, llama_seq_id seq_id, llama_state_seq_flags flags);

    //
    // members
    //

    const llama_model & model;

    llama_cparams cparams;

    llama_adapter_cvec_ptr  cvec;
    llama_adapter_loras_ptr loras;

    llama_cross cross; // TODO: tmp for handling cross-attention - need something better probably

    // DFlash feature capture. dflash_raw = raw 5-layer hc_ffn_post read back async (NO sync — drained
    // by the per-token sampling sync, like MTP). The n_hc SUM-collapse + stack happens lazily in
    // get_dflash_feat_host (consumer side). dflash_feat = the collapsed result.
    // IMPORTANT: the raw readback MUST land in a PINNED host buffer (buf_dflash). An async
    // device->pageable-host copy silently falls back to a *synchronous* blocking copy in CUDA,
    // which stalls the decode pipeline ~2x (the std::vector path measured 9.8 vs 19.8 t/s).
    ggml_backend_buffer_ptr buf_dflash;     // pinned host buffer backing dflash_raw_ptr
    float *  dflash_raw_ptr   = nullptr;    // base of pinned buffer (n_target*per floats)
    size_t   dflash_raw_cap   = 0;          // capacity in floats currently allocated
    std::vector<float> dflash_feat;
    int32_t dflash_feat_dim      = 0;
    int32_t dflash_feat_n_tokens = 0;
    int32_t dflash_n_target      = 0;
    int32_t dflash_ne            = 0;
    int32_t dflash_nh            = 0;

    std::unique_ptr<llama_memory_i> memory;

    // TriAttention runtime state (opaque handle + config)
    struct llama_tria_stats * tria_stats      = nullptr;
    int32_t                   tria_budget     = 0;
    int32_t                   tria_interval   = 128;
    int32_t                   tria_keep_first = 4;   // attention sink size
    int64_t                   tria_counter    = 0;

    // decode output (2-dimensional array: [n_outputs][n_vocab])
    buffer_view<float> logits = {nullptr, 0};

    // embeddings output (2-dimensional array: [n_outputs][n_embd])
    // populated only when pooling_type == LLAMA_POOLING_TYPE_NONE
    buffer_view<float> embd = {nullptr, 0};

    // hidden state before the final output norm (2-dimensional array: [n_outputs][n_embd])
    // populated only when cparams.embeddings_pre_norm is enabled and the model graph
    // sets llm_graph_result::t_h_pre_norm
    buffer_view<float> embd_pre_norm = {nullptr, 0};

    // [DSV4_MTP_FOLD] one greedy folded-draft token per batch position (dense by token index),
    // populated from llm_graph_result::t_mtp_draft. Empty unless the MTP head is folded.
    buffer_view<int32_t> mtp_draft = {nullptr, 0};

    struct sampling_info {
        // !samplers.empty() to check if any samplers are active
        std::map<llama_seq_id, llama_sampler *> samplers;

        buffer_view<float>       logits     = {nullptr, 0};
        buffer_view<llama_token> sampled    = {nullptr, 0};
        buffer_view<float>       probs      = {nullptr, 0};
        buffer_view<llama_token> candidates = {nullptr, 0};

        std::vector<uint32_t> logits_count;
        std::vector<uint32_t> probs_count;
        std::vector<uint32_t> candidates_count;

        // optimization
        std::vector<llama_token> token_ids_full_vocab;
    };

    sampling_info sampling;

    // sequence embeddings output (map of [n_embd] vectors)
    // populated only when pooling_type != LLAMA_POOLING_TYPE_NONE
    std::map<llama_seq_id, std::vector<float>> embd_seq;

    // reuse the batch_allocr to avoid unnecessary memory allocations
    std::unique_ptr<llama_batch_allocr> balloc;

    uint32_t n_outputs = 0; // number of actually-used outputs in the current ubatch or last logical batch

    std::vector<int32_t> output_ids; // map batch token positions to ids of the logits and embd buffers

    struct swap_info {
        uint32_t i0;
        uint32_t i1;
    };

    std::vector<swap_info> output_swaps;

    ggml_backend_sched_ptr sched;

    bool sched_need_reserve = true;

    ggml_backend_t backend_cpu = nullptr;
    std::vector<ggml_backend_ptr> backends;

    // training
    ggml_opt_context_t opt_ctx = nullptr;

    ggml_threadpool_t threadpool       = nullptr;
    ggml_threadpool_t threadpool_batch = nullptr;

    ggml_abort_callback abort_callback      = nullptr;
    void *              abort_callback_data = nullptr;

    std::vector<std::pair<ggml_backend_t, ggml_backend_set_n_threads_t>> set_n_threads_fns;

    // pointers and buffer types used for the compute buffer of each backend
    std::vector<ggml_backend_t>             backend_ptrs;
    std::vector<ggml_backend_buffer_type_t> backend_buft;
    std::vector<size_t>                     backend_buf_exp_size; // expected buffer sizes

    llm_graph_result_ptr gf_res_prev;
    llm_graph_result_ptr gf_res_reserve;

    // MTP graph-reuse slot pool.
    // MTP speculative decode alternates ubatch shapes (n_tokens, n_outputs) every step
    // (verify vs draft, plus variable draft-acceptance widths), which defeats the single
    // gf_res_prev reuse slot -> graphs reused = 0 -> per-step graph recapture whose cost
    // scales with the allocated KV size -> large-context generation collapses.
    // Keep a small pool of (sched, graph-result) slots keyed by shape so each recurring
    // shape's graph persists and is reused. Inactive slots live here; the active slot stays
    // in (sched, gf_res_prev) so the ~40 other sched users need no change. Each slot owns its
    // own sched because ggml_backend_sched_reset() deallocates globally (can't hold two graph
    // allocations in one sched). A fresh slot needs no worst-case reserve: it only ever sees
    // one small shape, so its first alloc_graph sizes the buffer exactly.
    struct graph_slot {
        ggml_backend_sched_ptr sched;
        llm_graph_result_ptr   res;
        uint64_t key = UINT64_MAX;
        uint64_t gen = 0;   // invalidated when sched_generation bumps (model/state change)
        uint64_t lru = 0;
    };
    std::vector<graph_slot> graph_slots;          // inactive slots only
    uint64_t active_graph_key = UINT64_MAX;
    uint64_t active_graph_gen = 0;
    uint64_t sched_generation = 0;                // bumped on every full sched_reserve()
    uint64_t graph_lru        = 0;
    size_t   max_graph_slots  = 1;                // K; 1 = pool disabled (env DSV4_GRAPH_SLOTS)
    int64_t  graph_slot_max_nodes = 0;            // node budget for fresh slot scheds/results

    // select (or lazily create) the pool slot whose graph matches this ubatch shape, and
    // make it the active (sched, gf_res_prev). No-op when max_graph_slots <= 1.
    void select_graph_slot(llm_graph_type gtype, const llama_ubatch & ubatch);

    // host buffer for the model output (logits and embeddings)
    ggml_backend_buffer_ptr buf_output;

    // keep copies of the per-sequence memory on the device
    std::map<llama_seq_id, llama_memory_buffers> mem_storage;

    bool has_evaluated_once = false;

    // env: LLAMA_GRAPH_REUSE_DISABLE
    bool graph_reuse_disable = false;

    // perf
    mutable int64_t t_start_us  = 0;
    mutable int64_t t_load_us   = 0;
    mutable int64_t t_p_eval_us = 0;
    mutable int64_t t_eval_us   = 0;

    mutable int64_t t_compute_start_us = 0;
    mutable int64_t n_queued_tokens    = 0;

    mutable int32_t n_p_eval = 0; // number of tokens in eval calls for the prompt (with batch size > 1)
    mutable int32_t n_eval   = 0; // number of eval calls

    mutable int32_t n_reused = 0; // number of times the previous graph was reused
};
