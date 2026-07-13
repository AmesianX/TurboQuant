#include "llama-memory-hybrid-iswa.h"

#include "ggml-backend.h"

#include "llama-impl.h"
#include "llama-io.h"
#include "llama-model.h"
#include "llama-context.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <vector>

namespace {

constexpr uint32_t DSV4_COMPRESSED_KV_STATE_MAGIC   = 0x44535634; // "DSV4"
// v2: the attn_k plane gained a front raw-window mirror (see dsv4_n_raw) — the state now carries
// n_raw in the header and the mirror rows after each layer's compressed rows.
constexpr uint32_t DSV4_COMPRESSED_KV_STATE_VERSION = 2;
constexpr uint32_t DSV4_COMPRESSED_DECODE_UBATCH_MAX = 512;

struct dsv4_row_range {
    uint32_t begin = 0;
    uint32_t end   = 0;

    uint32_t size() const {
        GGML_ASSERT(end >= begin);
        return end - begin;
    }
};

static dsv4_row_range dsv4_make_row_range(uint32_t n_comp, uint32_t ratio, llama_pos p0, llama_pos p1) {
    GGML_ASSERT(ratio > 0);

    if (n_comp == 0) {
        return {};
    }

    if (p0 < 0) {
        p0 = 0;
    }
    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }
    if (p0 >= p1) {
        return {};
    }

    const uint64_t row_begin = (uint64_t) p0 / ratio;
    uint64_t row_end;
    if (p1 == std::numeric_limits<llama_pos>::max()) {
        row_end = n_comp;
    } else {
        row_end = ((uint64_t) p1 + ratio - 1) / ratio;
    }

    return {
        (uint32_t) std::min<uint64_t>(row_begin, n_comp),
        (uint32_t) std::min<uint64_t>(row_end,   n_comp),
    };
}

static size_t dsv4_cache_row_size(const ggml_tensor * t) {
    GGML_ASSERT(t != nullptr);

    const size_t row_size = ggml_row_size(t->type, t->ne[0]);
    GGML_ASSERT((size_t) t->nb[1] == row_size);
    GGML_ASSERT((size_t) t->nb[2] == row_size*(size_t) t->ne[1]);

    return row_size;
}

static size_t dsv4_cache_offset(const ggml_tensor * t, llama_seq_id seq_id, uint32_t row) {
    GGML_ASSERT(seq_id >= 0);
    GGML_ASSERT(row <= (uint32_t) t->ne[1]);

    return (size_t) seq_id*(size_t) t->nb[2] + (size_t) row*(size_t) t->nb[1];
}

static void dsv4_zero_cache_rows(ggml_tensor * t, llama_seq_id seq_id, uint32_t row_start, uint32_t n_rows) {
    if (t == nullptr || n_rows == 0) {
        return;
    }

    const size_t row_size = dsv4_cache_row_size(t);
    const size_t n_bytes  = (size_t) n_rows*row_size;
    const size_t offset   = dsv4_cache_offset(t, seq_id, row_start);

    std::vector<uint8_t> zeros(n_bytes, 0);
    ggml_backend_tensor_set(t, zeros.data(), offset, n_bytes);
}

static void dsv4_copy_cache_rows(ggml_tensor * t, llama_seq_id seq_id_src, llama_seq_id seq_id_dst, uint32_t row_start, uint32_t n_rows) {
    if (t == nullptr || n_rows == 0 || seq_id_src == seq_id_dst) {
        return;
    }

    const size_t row_size   = dsv4_cache_row_size(t);
    const size_t n_bytes    = (size_t) n_rows*row_size;
    const size_t src_offset = dsv4_cache_offset(t, seq_id_src, row_start);
    const size_t dst_offset = dsv4_cache_offset(t, seq_id_dst, row_start);

    std::vector<uint8_t> tmp(n_bytes);
    ggml_backend_tensor_get(t, tmp.data(), src_offset, n_bytes);
    ggml_backend_tensor_set(t, tmp.data(), dst_offset, n_bytes);
}

} // namespace

//
// llama_memory_hybrid_iswa
//

llama_memory_hybrid_iswa::llama_memory_hybrid_iswa(
        const llama_model & model,
                            /* attn */
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                     bool   swa_full,
                 uint32_t   kv_size,
                 uint32_t   n_ubatch,
                 uint32_t   n_pad,
                            /* recurrent */
                ggml_type   type_r,
                ggml_type   type_s,
                 uint32_t   rs_size,
                            /* common */
                 uint32_t   n_seq_max,
                 uint32_t   n_rs_seq,
                     bool   offload,
                     bool   unified,
                            /* layer filters */
    const layer_filter_cb & filter_attn,
    const layer_filter_cb & filter_recr) :
    hparams(model.hparams),
    mem_attn(new llama_kv_cache_iswa(
        model,
        type_k,
        type_v,
        v_trans,
        offload,
        swa_full,
        unified,
        kv_size,
        n_seq_max,
        n_ubatch,
        n_pad,
        filter_attn == nullptr ?
            [&](int32_t il) { return !hparams.is_recurrent(il); }
            : filter_attn,
        nullptr
    )),
    mem_recr(new llama_memory_recurrent(
        model,
        type_r,
        type_s,
        offload,
        rs_size,
        n_seq_max,
        n_rs_seq,
        filter_recr == nullptr ?
            [&](int32_t il) { return hparams.is_recurrent(il); }
            : filter_recr
    )) {
    if (model.arch != LLM_ARCH_DEEPSEEK4) {
        return;
    }

    dsv4_n_seq_max = n_seq_max;
    dsv4_cache_layers.resize(hparams.n_layer);

    // [DSV4_KV_ADJACENT] front-pad every attn_k plane with the SWA window so that the attention's
    // k_all = [raw window | compressed rows] is one contiguous region (a view, not a concat copy).
    // The mirror rows are addressed by the SWA cell index, so dsv4_n_raw must be the SWA cache size.
    dsv4_n_raw = mem_attn->get_swa()->get_size();

    struct ggml_backend_buft_comparator {
        bool operator()(const ggml_backend_buffer_type_t & lhs, const ggml_backend_buffer_type_t & rhs) const {
            return strcmp(ggml_backend_buft_name(lhs), ggml_backend_buft_name(rhs)) < 0;
        }
    };
    std::map<ggml_backend_buffer_type_t, ggml_context_ptr, ggml_backend_buft_comparator> ctx_map;

    auto ctx_for_buft = [&](ggml_backend_buffer_type_t buft) -> ggml_context * {
        auto it = ctx_map.find(buft);
        if (it == ctx_map.end()) {
            ggml_init_params params = {
                /*.mem_size   =*/ size_t(2u*hparams.n_layer*ggml_tensor_overhead()),
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ true,
            };

            ggml_context * ctx = ggml_init(params);
            if (!ctx) {
                return nullptr;
            }

            ctx_map.emplace(buft, ctx);
            return ctx;
        }

        return it->second.get();
    };

    for (uint32_t il = 0; il < hparams.n_layer; ++il) {
        const uint32_t ratio = hparams.attn_compress_ratio[il];
        if (ratio == 0) {
            continue;
        }

        const uint32_t n_comp = std::max<uint32_t>(1, (kv_size + ratio - 1) / ratio);

        const char * dev_name = "CPU";
        ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();

        if (offload) {
            auto * dev = model.dev_layer(il);
            buft = ggml_backend_dev_buffer_type(dev);
            dev_name = ggml_backend_dev_name(dev);
        }

        LLAMA_LOG_DEBUG("%s: DeepSeek4 compressed KV layer %3d: dev = %s, ratio = %u, rows = %u\n",
                __func__, il, dev_name, ratio, n_comp);

        ggml_context * ctx = ctx_for_buft(buft);
        if (!ctx) {
            throw std::runtime_error("failed to create ggml context for DeepSeek4 compressed KV cache");
        }

        auto & cache = dsv4_cache_layers[il];
        cache.n_comp = n_comp;
        // attn_k plane layout: [ raw SWA mirror : dsv4_n_raw | compressed : n_comp | scratch : 1 ].
        //
        // The scratch row (index n_comp of the COMPRESSED region) takes the phase-uniform decode
        // graph's off-boundary (discarded) compress results, keeping the graph topology
        // token-invariant. Views are capped at n_comp, so scratch is never visible.
        //
        // The raw region is the SWA window mirror — see dsv4_n_raw. It makes k_all a view.
        //
        // Both side caches are pinned to F16 regardless of -ctk:
        //  - attn_k rows are attended together with SWA cache rows in the graph, and the SWA
        //    cache is force-upgraded to F16 in llama_kv_cache_iswa — the types must match.
        //  - index_k feeds regular mul_mat (indexer scores) and its row width
        //    (indexer_head_size=128) is smaller than the TBQ*_0 block size (256).
        cache.attn_k = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, hparams.n_embd_head_k(il), dsv4_n_raw + n_comp + 1, dsv4_n_seq_max);
        ggml_format_name(cache.attn_k, "cache_dsv4_attn_k_l%d", il);

        if (ratio == 4) {
            cache.index_k = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, hparams.indexer_head_size, n_comp + 1, dsv4_n_seq_max);
            ggml_format_name(cache.index_k, "cache_dsv4_index_k_l%d", il);
        }
    }

    for (auto & [buft, ctx] : ctx_map) {
        ggml_backend_buffer_t buf;
        if (model.hparams.no_alloc) {
            buf = ggml_backend_buft_alloc_buffer(buft, 0);
            for (ggml_tensor * t = ggml_get_first_tensor(ctx.get()); t != nullptr; t = ggml_get_next_tensor(ctx.get(), t)) {
                t->buffer = buf;
            }
        } else {
            buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft);
        }
        if (!buf) {
            throw std::runtime_error("failed to allocate DeepSeek4 compressed KV cache buffer");
        }

        LLAMA_LOG_INFO("%s: %10s DeepSeek4 compressed KV buffer size = %8.2f MiB\n", __func__,
                ggml_backend_buffer_name(buf), ggml_backend_buffer_get_size(buf)/1024.0/1024.0);

        ggml_backend_buffer_clear(buf, 0);
        dsv4_ctxs_bufs.emplace_back(std::move(ctx), buf);
    }
}

llama_memory_context_ptr llama_memory_hybrid_iswa::init_batch(llama_batch_allocr & balloc, uint32_t n_ubatch, bool embd_all) {
    const bool dsv4_compressed = has_dsv4_compressed_kv();

    do {
        balloc.split_reset();

        // follow the recurrent pattern for creating the ubatch splits
        std::vector<llama_ubatch> ubatches;

        while (true) {
            llama_ubatch ubatch;

            if (dsv4_compressed) {
                // DeepSeek V4 compressed attention keeps sequence-local compressor
                // state and compressed cache rows. Process one sequence set per
                // ubatch while still allowing multi-sequence batches at the API
                // level.
                uint32_t n_ubatch_dsv4 = n_ubatch;
                const auto & batch = balloc.get_batch();
                const bool first_split = balloc.get_n_used() == 0;
                const bool starts_at_zero = batch.pos == nullptr || batch.pos[0] == 0;
                const bool decode_regime = !first_split || !starts_at_zero;

                // [DSV4_RESUMED_UB] THE chunked-prefill forward-width cap. The 512 clamp was meant for
                // true DECODE (1 new token per seq -> one compressor update/token -> graph-arena risk on
                // long ctx). But `decode_regime` is also true for the 2nd+ chunk of a CHUNKED PREFILL
                // (multi-token, pos>0 resumed prefill) -> it was clamping every resumed prefill chunk to
                // 512 tokens, so the MoE never saw -ub tokens in-server (M~512 -> ~12 tok/expert,
                // tile-starved) and the standalone large-batch gain (91 TF/s @ 192 tok/expert) never
                // materialized. The build has a full RESUMED-CHUNK path (deepseek4.cpp ~2743,
                // n_tokens>1 && !is_prefill) that handles wide resumed chunks correctly.
                //
                // Distinguish TRUE DECODE from a RESUMED-PREFILL chunk: count the remaining tokens still
                // to be split (batch.n_tokens - n_used) and the distinct sequences. A resumed prefill has
                // many remaining tokens for few sequences (tokens_per_seq >> 1); decode has ~1 token/seq.
                // Only clamp when this is genuinely decode-sized. Default keeps the wide prefill; set
                // DSV4_RESUMED_UB_CAP=1 to restore the old always-512 behaviour for A/B.
                bool clamp_to_decode = decode_regime;
                if (decode_regime) {
                    static const bool force_old_cap = getenv("DSV4_RESUMED_UB_CAP") != nullptr;
                    if (!force_old_cap) {
                        // Distinguish TRUE DECODE from a RESUMED-PREFILL chunk by the whole batch's
                        // token-per-sequence density. A chunked prefill submits the full remaining
                        // prompt in ONE batch (many tokens, few sequences) and the splitter walks it in
                        // n_ubatch pieces; decode submits ~1 token per in-flight sequence. So
                        // tokens_per_seq >> 1 => prefill continuation -> keep the full -ub width.
                        std::set<llama_seq_id> seqs;
                        for (int32_t i = 0; i < batch.n_tokens; ++i) {
                            seqs.insert(batch.seq_id ? batch.seq_id[i][0] : (llama_seq_id) i);
                        }
                        const uint32_t n_seqs_b = (uint32_t) std::max<size_t>(seqs.size(), 1);
                        // Require BOTH high per-seq density (a prompt, not 1-token decode) AND a batch
                        // larger than the decode cap itself: any batch that already fits in <=512 tokens
                        // gains nothing from lifting the cap, and this keeps MTP-verify decode batches
                        // (K tokens/seq, but small total) on the safe capped path.
                        const bool resumed_prefill =
                            (uint32_t) batch.n_tokens > DSV4_COMPRESSED_DECODE_UBATCH_MAX &&
                            (uint32_t) batch.n_tokens > n_seqs_b * 2;
                        if (resumed_prefill) {
                            clamp_to_decode = false;
                        }
                        if (getenv("DSV4_PHASE_PROF") || getenv("DSV4_PREFILL_PROF")) {
                            fprintf(stderr, "DSV4_UBCAP: batch_n_tokens=%d n_seqs=%u resumed_prefill=%d -> n_ubatch_dsv4=%u (cap=%u)\n",
                                    batch.n_tokens, n_seqs_b, (int) resumed_prefill,
                                    clamp_to_decode ? std::min<uint32_t>(n_ubatch, DSV4_COMPRESSED_DECODE_UBATCH_MAX) : n_ubatch,
                                    DSV4_COMPRESSED_DECODE_UBATCH_MAX);
                            fflush(stderr);
                        }
                    }
                }
                if (clamp_to_decode) {
                    // Non-prefill compressed-attention chunks build one
                    // compressor update per token and can otherwise exhaust the
                    // graph metadata arena on long contexts.
                    n_ubatch_dsv4 = std::min<uint32_t>(n_ubatch_dsv4, DSV4_COMPRESSED_DECODE_UBATCH_MAX);
                }
                // Phase 2 multi-slot: batch multiple in-flight sequences into ONE ubatch (n_seqs>1)
                // instead of serializing them one sequence per ubatch. The compressed-layer build
                // path handles n_seqs>1 (per-seq recurrent state + block-diagonal masks); everything
                // else (MoE/dense/norm/lm_head) batches over n_tokens. Gated by DSV4_MULTISLOT and
                // restricted to the non-rollback regime (n_rs_seq==0): recurrent state rollback
                // [TAG_RECURRENT_ROLLBACK_SPLITS] does not support equal splits.
                //
                // CRITICAL: only PURE-DECODE batches (every token at pos>0, one new token per
                // sequence) may take split_equal — the compressed build asserts !is_prefill for
                // n_seqs>1. A batch containing any pos==0 (prefill) token, or a mixed prefill+decode
                // batch, falls back to split_seq (correct, just serialized). Default OFF -> byte-
                // identical single-slot behaviour (split_seq).
                static const bool dsv4_multislot = getenv("DSV4_MULTISLOT") != nullptr;
                // MULTI-DECODE test: split_equal lays the batch out sequence-major with a UNIFORM
                // token count K per sequence, and the compressed build requires every token be a
                // fresh continuation. Two regimes batch into one decode:
                //   - plain multi-slot (K==1, n_rs_seq==0): one new token per sequence
                //   - multi-slot + MTP verify (K>1, n_rs_seq>0): each seq verifies 1 accept + draft
                // Both require: all pos>0 (no prefill token), every token n_seq_id==1, >1 distinct
                // sequence, and an EQUAL token count K across all sequences. A prefill batch, a single
                // sequence, or RAGGED per-seq counts (split_equal would mis-align) fall back to
                // split_seq (correct, just serialized). n_rs_seq>0 is now allowed: the compressed
                // build writes per-seq rollback snapshot planes (dsv4_store_rollback_planes_multi).
                bool pure_decode = (batch.pos != nullptr) && (batch.n_tokens > 0);
                {
                    std::map<llama_seq_id, int64_t> seq_counts;
                    for (int32_t i = 0; pure_decode && i < batch.n_tokens; ++i) {
                        if (batch.pos[i] == 0) { pure_decode = false; break; }
                        if (batch.n_seq_id && batch.n_seq_id[i] != 1) { pure_decode = false; break; }
                        const llama_seq_id sid = batch.seq_id ? batch.seq_id[i][0] : (llama_seq_id) i;
                        seq_counts[sid]++;
                    }
                    if (pure_decode) {
                        if (seq_counts.size() < 2) { pure_decode = false; } // single slot -> normal path
                        int64_t k_uniform = -1;
                        for (const auto & kv : seq_counts) {
                            if (k_uniform < 0)            { k_uniform = kv.second; }
                            else if (kv.second != k_uniform) { pure_decode = false; break; } // ragged -> split_seq
                        }
                    }
                }
                const bool can_batch_seqs = dsv4_multislot && pure_decode;
                // Step C observation: log the pre-split batch shape so we can see whether the server
                // batches multiple slots' MTP-verify tokens into one decode (n_distinct_seqs>1) or
                // serializes them (the upstream n_parallel==1 MTP limit). Gated by DSV4_MS_DBG.
                if (getenv("DSV4_MS_DBG")) {
                    std::set<llama_seq_id> ds; int64_t maxpos = -1;
                    for (int32_t i = 0; batch.pos && i < batch.n_tokens; ++i) {
                        ds.insert(batch.seq_id ? batch.seq_id[i][0] : (llama_seq_id) i);
                        maxpos = std::max<int64_t>(maxpos, batch.pos[i]);
                    }
                    fprintf(stderr, "[MS_SPLIT] n_tokens=%d distinct_seqs=%zu maxpos=%lld n_rs_seq=%d pure_decode=%d can_batch=%d -> %s\n",
                            batch.n_tokens, ds.size(), (long long) maxpos, (int) mem_recr->n_rs_seq,
                            (int) pure_decode, (int) can_batch_seqs, can_batch_seqs ? "split_equal" : "split_seq");
                    fflush(stderr);
                }
                if (can_batch_seqs) {
                    // unified attention KV (n_stream==1) -> non-sequential equal split,
                    // matching the standard unified path; sequences are separated by the
                    // block-diagonal masks, not by per-stream KV partitions.
                    const bool unified = (mem_attn->get_base()->get_n_stream() == 1);
                    ubatch = balloc.split_equal(n_ubatch_dsv4, !unified);
                } else {
                    ubatch = balloc.split_seq(n_ubatch_dsv4);
                }
            } else if (embd_all) {
                // if all tokens are output, split by sequence
                ubatch = balloc.split_seq(n_ubatch);
            } else {
                if (mem_recr->n_rs_seq > 0) {
                    // [TAG_RECURRENT_ROLLBACK_SPLITS]
                    // TODO: recurrent state rollback does not support equal splits
                    ubatch = balloc.split_seq(n_ubatch);
                } else {
                    // Use non-sequential split when KV cache is unified (needed for hellaswag/winogrande/multiple-choice)
                    const bool unified = (mem_attn->get_base()->get_n_stream() == 1);
                    ubatch = balloc.split_equal(n_ubatch, !unified);
                }
            }

            if (ubatch.n_tokens == 0) {
                break;
            }

            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            // failed to find a suitable split
            break;
        }

        // prepare the recurrent batches first
        if (!mem_recr->prepare(ubatches)) {
            // TODO: will the recurrent cache be in an undefined context at this point?
            LLAMA_LOG_ERROR("%s: failed to prepare recurrent ubatches\n", __func__);
            return std::make_unique<llama_memory_hybrid_iswa_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        // prepare the attention cache (iswa version returns both base and swa slot infos)
        auto sinfos_base = mem_attn->get_base()->prepare(ubatches);
        if (sinfos_base.empty()) {
            LLAMA_LOG_ERROR("%s: failed to prepare attention base ubatches\n", __func__);
            return std::make_unique<llama_memory_hybrid_iswa_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        auto sinfos_swa = mem_attn->get_swa()->prepare(ubatches);
        if (sinfos_swa.empty()) {
            LLAMA_LOG_ERROR("%s: failed to prepare attention swa ubatches\n", __func__);
            return std::make_unique<llama_memory_hybrid_iswa_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        return std::make_unique<llama_memory_hybrid_iswa_context>(
                this, std::move(sinfos_base), std::move(sinfos_swa), std::move(ubatches));
    } while(false);

    return std::make_unique<llama_memory_hybrid_iswa_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

llama_memory_context_ptr llama_memory_hybrid_iswa::init_full() {
    return std::make_unique<llama_memory_hybrid_iswa_context>(this);
}

llama_memory_context_ptr llama_memory_hybrid_iswa::init_update(llama_context * lctx, bool optimize) {
    return std::make_unique<llama_memory_hybrid_iswa_context>(this, lctx, optimize);
}

bool llama_memory_hybrid_iswa::get_can_shift() const {
    if (has_dsv4_compressed_kv()) {
        return false;
    }

    // Shifting is trivially supported for recurrent
    return mem_attn->get_can_shift();
}

void llama_memory_hybrid_iswa::clear(bool data) {
    mem_attn->clear(data);
    mem_recr->clear(data);

    if (data) {
        for (auto & [_, buf] : dsv4_ctxs_bufs) {
            ggml_backend_buffer_clear(buf.get(), 0);
        }
    }
}

bool llama_memory_hybrid_iswa::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    // The compressed cache cannot honor head/interior removals: row validity is derived from
    // seq_pos_max, so zeroed interior rows would stay visible to attention (and a mid-row p0
    // also destroys data of kept positions sharing that compressed row). Only tail removals
    // (p1 covers through the end) and full clears are representable — reject the rest BEFORE
    // any sub-cache mutates so the caller can fall back to a full reprocess.
    if (has_dsv4_compressed_kv() && p1 >= 0) {
        const llama_pos pos_max = seq_pos_max(seq_id);
        if (pos_max >= 0 && p1 <= pos_max) {
            return false;
        }
    }

    // Try removing from the recurrent cache first since it may fail. If it does
    // fail, the cache will not have been mutated.
    if (!mem_recr->seq_rm(seq_id, p0, p1)) {
        return false;
    }
    // Clear the compressed rows BEFORE mem_attn trims the window: dsv4_clear_rows derives the
    // written high-water from mem_attn->seq_pos_max to bound the zeroed range, so it must read it
    // while still pre-trim. (The validity pre-check above guarantees this tail removal succeeds.)
    dsv4_seq_rm(seq_id, p0, p1);
    if (!mem_attn->seq_rm(seq_id, p0, p1)) {
        return false;
    }
    return true;
}

void llama_memory_hybrid_iswa::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    mem_attn->seq_cp(seq_id_src, seq_id_dst, p0, p1);
    mem_recr->seq_cp(seq_id_src, seq_id_dst, p0, p1);
    dsv4_seq_cp(seq_id_src, seq_id_dst, p0, p1);
}

void llama_memory_hybrid_iswa::seq_keep(llama_seq_id seq_id) {
    mem_attn->seq_keep(seq_id);
    mem_recr->seq_keep(seq_id);
    dsv4_seq_keep(seq_id);
}

void llama_memory_hybrid_iswa::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    if (has_dsv4_compressed_kv() && shift != 0) {
        GGML_ABORT("DeepSeek V4 compressed KV cache does not support K-shift");
    }

    mem_attn->seq_add(seq_id, p0, p1, shift);
    mem_recr->seq_add(seq_id, p0, p1, shift);
}

void llama_memory_hybrid_iswa::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    if (has_dsv4_compressed_kv() && d != 1) {
        GGML_ABORT("DeepSeek V4 compressed KV cache does not support position division");
    }

    mem_attn->seq_div(seq_id, p0, p1, d);
    mem_recr->seq_div(seq_id, p0, p1, d);
}

llama_pos llama_memory_hybrid_iswa::seq_pos_min(llama_seq_id seq_id) const {
    // the min of the total cache is the max of the two caches' min values
    if (has_dsv4_compressed_kv()) {
        // DSV4's compressed KV (recr) keeps a ratio-compressed copy of the whole
        // sequence from pos 0, so it never constrains the resume point — only the SWA
        // attention window does. Folding mem_recr in via max() pegs pos_min at the
        // recurrent state's position, which makes every context checkpoint
        // unrestorable (pos_min == pos_max) and forces a full prompt re-prefill on
        // every chat turn (multi-turn KV reuse breaks; cache_n stuck at 1).
        return mem_attn->seq_pos_min(seq_id);
    }
    return std::max(mem_attn->seq_pos_min(seq_id), mem_recr->seq_pos_min(seq_id));
}

llama_pos llama_memory_hybrid_iswa::seq_pos_max(llama_seq_id seq_id) const {
    // the max of the total cache is the min of the two caches' max values
    return std::min(mem_attn->seq_pos_max(seq_id), mem_recr->seq_pos_max(seq_id));
}

std::map<ggml_backend_buffer_type_t, size_t> llama_memory_hybrid_iswa::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> mb = mem_attn->memory_breakdown();
    for (const auto & buft_size : mem_recr->memory_breakdown()) {
        mb[buft_size.first] += buft_size.second;
    }
    for (const auto & [_, buf] : dsv4_ctxs_bufs) {
        mb[ggml_backend_buffer_get_type(buf.get())] += ggml_backend_buffer_get_size(buf.get());
    }
    return mb;
}

void llama_memory_hybrid_iswa::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    const bool dbg = getenv("DSV4_STATE_DEBUG") != nullptr;
    if (dbg) {
        fprintf(stderr, "DSV4DBG hybrid state_write enter seq=%d flags=%d\n", (int) seq_id, (int) flags);
    }
    size_t o0 = io.n_bytes();
    mem_attn->state_write(io, seq_id, flags);
    size_t o1 = io.n_bytes();
    mem_recr->state_write(io, seq_id, flags);
    size_t o2 = io.n_bytes();
    dsv4_state_write(io, seq_id);
    if (dbg) {
        fprintf(stderr, "DSV4DBG state_write seq=%d flags=%d attn=%zu recr=%zu dsv4=%zu total=%zu\n",
                (int) seq_id, (int) flags, o1 - o0, o2 - o1, io.n_bytes() - o2, io.n_bytes() - o0);
    }
}

void llama_memory_hybrid_iswa::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    const bool dbg = getenv("DSV4_STATE_DEBUG") != nullptr;
    size_t o0 = io.n_bytes();
    if (dbg) {
        fprintf(stderr, "DSV4DBG state_read enter seq=%d flags=%d\n", (int) seq_id, (int) flags);
    }
    mem_attn->state_read(io, seq_id, flags);
    size_t o1 = io.n_bytes();
    if (dbg) {
        fprintf(stderr, "DSV4DBG state_read attn=%zu\n", o1 - o0);
    }
    mem_recr->state_read(io, seq_id, flags);
    size_t o2 = io.n_bytes();
    if (dbg) {
        fprintf(stderr, "DSV4DBG state_read recr=%zu\n", o2 - o1);
    }
    dsv4_state_read(io, seq_id);
    if (dbg) {
        fprintf(stderr, "DSV4DBG state_read dsv4=%zu total=%zu\n", io.n_bytes() - o2, io.n_bytes() - o0);
    }
}

void llama_memory_hybrid_iswa::dsv4_seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    if (!has_dsv4_compressed_kv()) {
        return;
    }

    if (seq_id >= 0) {
        GGML_ASSERT((uint32_t) seq_id < dsv4_n_seq_max);
        for (int32_t il = 0; il < (int32_t) dsv4_cache_layers.size(); ++il) {
            dsv4_clear_rows(seq_id, il, p0, p1);
        }
        return;
    }

    for (uint32_t seq = 0; seq < dsv4_n_seq_max; ++seq) {
        for (int32_t il = 0; il < (int32_t) dsv4_cache_layers.size(); ++il) {
            dsv4_clear_rows(seq, il, p0, p1);
        }
    }
}

void llama_memory_hybrid_iswa::dsv4_seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    if (!has_dsv4_compressed_kv() || seq_id_src == seq_id_dst) {
        return;
    }

    GGML_ASSERT(seq_id_src >= 0 && (uint32_t) seq_id_src < dsv4_n_seq_max);
    GGML_ASSERT(seq_id_dst >= 0 && (uint32_t) seq_id_dst < dsv4_n_seq_max);

    for (int32_t il = 0; il < (int32_t) dsv4_cache_layers.size(); ++il) {
        dsv4_copy_rows(seq_id_src, seq_id_dst, il, p0, p1);
    }
}

void llama_memory_hybrid_iswa::dsv4_seq_keep(llama_seq_id seq_id) {
    if (!has_dsv4_compressed_kv()) {
        return;
    }

    GGML_ASSERT(seq_id >= 0 && (uint32_t) seq_id < dsv4_n_seq_max);

    for (uint32_t seq = 0; seq < dsv4_n_seq_max; ++seq) {
        if ((llama_seq_id) seq == seq_id) {
            continue;
        }

        dsv4_clear_seq(seq);
    }
}

void llama_memory_hybrid_iswa::dsv4_clear_seq(llama_seq_id seq_id) {
    GGML_ASSERT(seq_id >= 0 && (uint32_t) seq_id < dsv4_n_seq_max);

    for (const auto & layer : dsv4_cache_layers) {
        // attn_k rows are offset by the raw-window mirror; clear the mirror too (a full-sequence
        // clear is rare — unlike dsv4_clear_rows, which runs every MTP rollback).
        dsv4_zero_cache_rows(layer.attn_k,  seq_id, 0, dsv4_n_raw + layer.n_comp);
        dsv4_zero_cache_rows(layer.index_k, seq_id, 0, layer.n_comp);
    }
}

void llama_memory_hybrid_iswa::dsv4_clear_rows(llama_seq_id seq_id, int32_t il, llama_pos p0, llama_pos p1) {
    GGML_ASSERT(seq_id >= 0 && (uint32_t) seq_id < dsv4_n_seq_max);
    GGML_ASSERT(il >= 0 && il < (int32_t) dsv4_cache_layers.size());

    const uint32_t ratio = hparams.attn_compress_ratio[il];
    if (ratio == 0) {
        return;
    }

    const auto & layer = dsv4_cache_layers[il];
    const auto range = dsv4_make_row_range(layer.n_comp, ratio, p0, p1);

    // Open-ended tail removals (p1 = max, every MTP rollback) make dsv4_make_row_range run the
    // range out to layer.n_comp. Only rows below the written high-water actually hold data; the
    // empty tail [high-water, n_comp) needs no zeroing. Without this clamp, seq_rm zeros the full
    // allocated compressed cache every round — O(n_comp) per layer (~630 ms at 1M ctx) and the
    // dominant cost of MTP at large context. Clamping makes it O(rejected rows). Requires the
    // pre-trim high-water, hence dsv4_seq_rm runs before mem_attn->seq_rm (see seq_rm()).
    const uint32_t n_written = dsv4_n_state_rows(il, seq_id);
    const uint32_t row_end   = std::min<uint32_t>(range.begin + range.size(), n_written);
    if (row_end <= range.begin) {
        return;
    }
    const uint32_t n_rows = row_end - range.begin;

    // the raw-window mirror is NOT cleared here: a raw row is only ever visible to the attention
    // if the SWA cache holds a live token in that cell, and every such cell is (re)written by the
    // same ubatch that writes the SWA cache. Zeroing it per rollback would undo the high-water
    // clamp above (36 MB of pointless writes per seq_rm).
    dsv4_zero_cache_rows(layer.attn_k,  seq_id, dsv4_n_raw + range.begin, n_rows);
    dsv4_zero_cache_rows(layer.index_k, seq_id, range.begin, n_rows);
}

void llama_memory_hybrid_iswa::dsv4_copy_rows(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, int32_t il, llama_pos p0, llama_pos p1) {
    GGML_ASSERT(seq_id_src >= 0 && (uint32_t) seq_id_src < dsv4_n_seq_max);
    GGML_ASSERT(seq_id_dst >= 0 && (uint32_t) seq_id_dst < dsv4_n_seq_max);
    GGML_ASSERT(il >= 0 && il < (int32_t) dsv4_cache_layers.size());

    const uint32_t ratio = hparams.attn_compress_ratio[il];
    if (ratio == 0) {
        return;
    }

    const auto & layer = dsv4_cache_layers[il];
    const auto range = dsv4_make_row_range(layer.n_comp, ratio, p0, p1);

    dsv4_copy_cache_rows(layer.attn_k,  seq_id_src, seq_id_dst, dsv4_n_raw + range.begin, range.size());
    dsv4_copy_cache_rows(layer.index_k, seq_id_src, seq_id_dst, range.begin, range.size());

    // the SWA cache shares cells between the two sequences after a seq_cp, so the destination
    // plane's raw-window mirror must hold them as well — the mask will make them visible to dst
    // queries, but nothing will rewrite them.
    dsv4_copy_cache_rows(layer.attn_k, seq_id_src, seq_id_dst, 0, dsv4_n_raw);
}

uint32_t llama_memory_hybrid_iswa::dsv4_n_state_rows(int32_t il, llama_seq_id seq_id) const {
    GGML_ASSERT(seq_id >= 0 && (uint32_t) seq_id < dsv4_n_seq_max);
    GGML_ASSERT(il >= 0 && il < (int32_t) dsv4_cache_layers.size());

    const uint32_t ratio = hparams.attn_compress_ratio[il];
    if (ratio == 0) {
        return 0;
    }

    const llama_pos pos_max = mem_attn->seq_pos_max(seq_id);
    if (pos_max < 0) {
        return 0;
    }

    const uint64_t n_rows = ((uint64_t) pos_max + 1) / ratio;
    return (uint32_t) std::min<uint64_t>(n_rows, dsv4_cache_layers[il].n_comp);
}

void llama_memory_hybrid_iswa::dsv4_state_write(llama_io_write_i & io, llama_seq_id seq_id) const {
    if (!has_dsv4_compressed_kv()) {
        return;
    }

    GGML_ASSERT(seq_id == -1 || (seq_id >= 0 && (uint32_t) seq_id < dsv4_n_seq_max));

    std::vector<llama_seq_id> seq_ids;
    auto seq_has_rows = [&](llama_seq_id seq) {
        // a live sequence with no compressed row yet (context < compress_ratio) still owns raw
        // SWA cells, and its raw-window mirror must be part of the state
        if (dsv4_n_raw > 0 && mem_attn->seq_pos_max(seq) >= 0) {
            return true;
        }
        for (int32_t il = 0; il < (int32_t) dsv4_cache_layers.size(); ++il) {
            if (dsv4_n_state_rows(il, seq) > 0) {
                return true;
            }
        }
        return false;
    };

    if (seq_id >= 0) {
        if (seq_has_rows(seq_id)) {
            seq_ids.push_back(seq_id);
        }
    } else {
        for (uint32_t seq = 0; seq < dsv4_n_seq_max; ++seq) {
            if (seq_has_rows(seq)) {
                seq_ids.push_back(seq);
            }
        }
    }

    const uint32_t magic   = DSV4_COMPRESSED_KV_STATE_MAGIC;
    const uint32_t version = DSV4_COMPRESSED_KV_STATE_VERSION;
    const uint32_t n_layer = hparams.n_layer;
    const uint32_t n_seq   = seq_ids.size();
    const uint32_t n_raw   = dsv4_n_raw;

    io.write(&magic,   sizeof(magic));
    io.write(&version, sizeof(version));
    io.write(&n_layer, sizeof(n_layer));
    io.write(&n_seq,   sizeof(n_seq));
    io.write(&n_raw,   sizeof(n_raw));

    for (uint32_t il = 0; il < n_layer; ++il) {
        const auto & layer = dsv4_cache_layers[il];

        const uint32_t n_comp = layer.n_comp;
        io.write(&n_comp, sizeof(n_comp));

        const uint32_t has_attn = layer.attn_k != nullptr;
        io.write(&has_attn, sizeof(has_attn));
        if (has_attn) {
            const int32_t  type_i   = (int32_t) layer.attn_k->type;
            const uint64_t row_size = dsv4_cache_row_size(layer.attn_k);
            io.write(&type_i,   sizeof(type_i));
            io.write(&row_size, sizeof(row_size));
        }

        const uint32_t has_index = layer.index_k != nullptr;
        io.write(&has_index, sizeof(has_index));
        if (has_index) {
            const int32_t  type_i   = (int32_t) layer.index_k->type;
            const uint64_t row_size = dsv4_cache_row_size(layer.index_k);
            io.write(&type_i,   sizeof(type_i));
            io.write(&row_size, sizeof(row_size));
        }
    }

    for (llama_seq_id seq : seq_ids) {
        io.write(&seq, sizeof(seq));

        for (uint32_t il = 0; il < n_layer; ++il) {
            const auto & layer = dsv4_cache_layers[il];
            const uint32_t n_rows = dsv4_n_state_rows(il, seq);

            if (layer.attn_k != nullptr) {
                const uint64_t row_size = dsv4_cache_row_size(layer.attn_k);
                io.write(&n_rows, sizeof(n_rows));
                if (n_rows > 0) {
                    io.write_tensor(layer.attn_k,
                            dsv4_cache_offset(layer.attn_k, seq, dsv4_n_raw), (size_t) n_rows*row_size);
                }
                // the raw-window mirror: the SWA cache's own state does not restore it
                if (dsv4_n_raw > 0) {
                    io.write_tensor(layer.attn_k,
                            dsv4_cache_offset(layer.attn_k, seq, 0), (size_t) dsv4_n_raw*row_size);
                }
            }

            if (layer.index_k != nullptr) {
                const uint64_t row_size = dsv4_cache_row_size(layer.index_k);
                io.write(&n_rows, sizeof(n_rows));
                if (n_rows > 0) {
                    io.write_tensor(layer.index_k, dsv4_cache_offset(layer.index_k, seq, 0), (size_t) n_rows*row_size);
                }
            }
        }
    }
}

void llama_memory_hybrid_iswa::dsv4_state_read(llama_io_read_i & io, llama_seq_id seq_id) {
    if (!has_dsv4_compressed_kv()) {
        return;
    }

    GGML_ASSERT(seq_id == -1 || (seq_id >= 0 && (uint32_t) seq_id < dsv4_n_seq_max));

    uint32_t magic;
    uint32_t version;
    uint32_t n_layer;
    uint32_t n_seq;
    uint32_t n_raw;

    io.read(&magic,   sizeof(magic));
    io.read(&version, sizeof(version));
    io.read(&n_layer, sizeof(n_layer));
    io.read(&n_seq,   sizeof(n_seq));
    io.read(&n_raw,   sizeof(n_raw));

    if (magic != DSV4_COMPRESSED_KV_STATE_MAGIC) {
        throw std::runtime_error("failed to restore DeepSeek V4 compressed KV cache: bad magic");
    }
    if (version != DSV4_COMPRESSED_KV_STATE_VERSION) {
        throw std::runtime_error("failed to restore DeepSeek V4 compressed KV cache: bad version");
    }
    if (n_layer != hparams.n_layer || n_layer != dsv4_cache_layers.size()) {
        throw std::runtime_error("failed to restore DeepSeek V4 compressed KV cache: mismatched layer count");
    }
    if (n_raw != dsv4_n_raw) {
        throw std::runtime_error("failed to restore DeepSeek V4 compressed KV cache: mismatched raw window size");
    }

    struct layer_meta {
        uint32_t n_comp = 0;
        bool has_attn = false;
        int32_t attn_type = -1;
        uint64_t attn_row_size = 0;
        bool has_index = false;
        int32_t index_type = -1;
        uint64_t index_row_size = 0;
    };

    std::vector<layer_meta> meta(n_layer);
    for (uint32_t il = 0; il < n_layer; ++il) {
        auto & m = meta[il];
        const auto & layer = dsv4_cache_layers[il];

        io.read(&m.n_comp, sizeof(m.n_comp));

        uint32_t has_attn;
        io.read(&has_attn, sizeof(has_attn));
        m.has_attn = has_attn != 0;
        if (m.has_attn) {
            io.read(&m.attn_type,     sizeof(m.attn_type));
            io.read(&m.attn_row_size, sizeof(m.attn_row_size));
        }

        uint32_t has_index;
        io.read(&has_index, sizeof(has_index));
        m.has_index = has_index != 0;
        if (m.has_index) {
            io.read(&m.index_type,     sizeof(m.index_type));
            io.read(&m.index_row_size, sizeof(m.index_row_size));
        }

        const bool local_has_attn  = layer.attn_k  != nullptr;
        const bool local_has_index = layer.index_k != nullptr;

        if (m.n_comp != layer.n_comp || m.has_attn != local_has_attn || m.has_index != local_has_index) {
            throw std::runtime_error("failed to restore DeepSeek V4 compressed KV cache: mismatched layer layout");
        }
        if (local_has_attn) {
            const int32_t  type_i   = (int32_t) layer.attn_k->type;
            const uint64_t row_size = dsv4_cache_row_size(layer.attn_k);
            if (m.attn_type != type_i || m.attn_row_size != row_size) {
                throw std::runtime_error("failed to restore DeepSeek V4 compressed KV cache: mismatched attention cache type");
            }
        }
        if (local_has_index) {
            const int32_t  type_i   = (int32_t) layer.index_k->type;
            const uint64_t row_size = dsv4_cache_row_size(layer.index_k);
            if (m.index_type != type_i || m.index_row_size != row_size) {
                throw std::runtime_error("failed to restore DeepSeek V4 compressed KV cache: mismatched index cache type");
            }
        }
    }

    if (seq_id == -1) {
        for (auto & [_, buf] : dsv4_ctxs_bufs) {
            ggml_backend_buffer_clear(buf.get(), 0);
        }
    } else {
        dsv4_clear_seq(seq_id);
    }

    for (uint32_t is = 0; is < n_seq; ++is) {
        llama_seq_id src_seq_id;
        io.read(&src_seq_id, sizeof(src_seq_id));

        const llama_seq_id dst_seq_id = seq_id == -1 ? src_seq_id : seq_id;
        if (dst_seq_id < 0 || (uint32_t) dst_seq_id >= dsv4_n_seq_max) {
            throw std::runtime_error("failed to restore DeepSeek V4 compressed KV cache: invalid sequence id");
        }

        for (uint32_t il = 0; il < n_layer; ++il) {
            const auto & layer = dsv4_cache_layers[il];

            // NOTE: must mirror dsv4_state_write's io.write_tensor — a raw io.read
            // here breaks ON_DEVICE checkpoints (the tensor bytes live in device
            // storage, not the host stream); read_tensor handles both io kinds.
            if (layer.attn_k != nullptr) {
                const size_t row_size = dsv4_cache_row_size(layer.attn_k);

                uint32_t n_rows;
                io.read(&n_rows, sizeof(n_rows));
                if (n_rows > layer.n_comp) {
                    throw std::runtime_error("failed to restore DeepSeek V4 compressed KV cache: too many attention rows");
                }
                if (n_rows > 0) {
                    io.read_tensor(layer.attn_k,
                            dsv4_cache_offset(layer.attn_k, dst_seq_id, dsv4_n_raw), (size_t) n_rows*row_size);
                }
                if (dsv4_n_raw > 0) {
                    io.read_tensor(layer.attn_k,
                            dsv4_cache_offset(layer.attn_k, dst_seq_id, 0), (size_t) dsv4_n_raw*row_size);
                }
            }

            if (layer.index_k != nullptr) {
                uint32_t n_rows;
                io.read(&n_rows, sizeof(n_rows));
                if (n_rows > layer.n_comp) {
                    throw std::runtime_error("failed to restore DeepSeek V4 compressed KV cache: too many index rows");
                }
                if (n_rows > 0) {
                    const size_t row_size = dsv4_cache_row_size(layer.index_k);
                    io.read_tensor(layer.index_k,
                            dsv4_cache_offset(layer.index_k, dst_seq_id, 0), (size_t) n_rows*row_size);
                }
            }
        }
    }
}

llama_kv_cache_iswa * llama_memory_hybrid_iswa::get_mem_attn() const {
    return mem_attn.get();
}

llama_memory_recurrent * llama_memory_hybrid_iswa::get_mem_recr() const {
    return mem_recr.get();
}

bool llama_memory_hybrid_iswa::has_dsv4_compressed_kv() const {
    for (const auto & layer : dsv4_cache_layers) {
        if (layer.n_comp != 0) {
            return true;
        }
    }

    return false;
}

uint32_t llama_memory_hybrid_iswa::get_dsv4_n_comp(int32_t il) const {
    GGML_ASSERT(il >= 0 && il < (int32_t) dsv4_cache_layers.size());
    return dsv4_cache_layers[il].n_comp;
}

uint32_t llama_memory_hybrid_iswa::get_dsv4_n_raw() const {
    return dsv4_n_raw;
}

// the COMPRESSED region of the plane: [n_comp + 1 (scratch)] rows, row 0 == compressed row 0.
ggml_tensor * llama_memory_hybrid_iswa::get_dsv4_attn_k(ggml_context * ctx, int32_t il, llama_seq_id seq_id) const {
    GGML_ASSERT(il >= 0 && il < (int32_t) dsv4_cache_layers.size());
    GGML_ASSERT(seq_id >= 0 && (uint32_t) seq_id < dsv4_n_seq_max);

    ggml_tensor * t = dsv4_cache_layers[il].attn_k;
    GGML_ASSERT(t != nullptr);

    return ggml_view_2d(ctx, t, t->ne[0], t->ne[1] - dsv4_n_raw, t->nb[1],
            seq_id*t->nb[2] + (size_t) dsv4_n_raw*t->nb[1]);
}

// the RAW region of the plane: [dsv4_n_raw] rows, indexed by SWA cell (set_rows destination).
ggml_tensor * llama_memory_hybrid_iswa::get_dsv4_attn_raw(ggml_context * ctx, int32_t il, llama_seq_id seq_id) const {
    GGML_ASSERT(il >= 0 && il < (int32_t) dsv4_cache_layers.size());
    GGML_ASSERT(seq_id >= 0 && (uint32_t) seq_id < dsv4_n_seq_max);

    ggml_tensor * t = dsv4_cache_layers[il].attn_k;
    GGML_ASSERT(t != nullptr);

    return ggml_view_2d(ctx, t, t->ne[0], dsv4_n_raw, t->nb[1], seq_id*t->nb[2]);
}

// raw ++ compressed as ONE tensor — the whole point of the layout: no concat, no copy.
ggml_tensor * llama_memory_hybrid_iswa::get_dsv4_attn_kall(ggml_context * ctx, int32_t il, llama_seq_id seq_id, int64_t n_comp_rows) const {
    GGML_ASSERT(il >= 0 && il < (int32_t) dsv4_cache_layers.size());
    GGML_ASSERT(seq_id >= 0 && (uint32_t) seq_id < dsv4_n_seq_max);

    ggml_tensor * t = dsv4_cache_layers[il].attn_k;
    GGML_ASSERT(t != nullptr);
    GGML_ASSERT(n_comp_rows >= 0 && (int64_t) dsv4_n_raw + n_comp_rows <= t->ne[1]);

    const int64_t n_rows = (int64_t) dsv4_n_raw + n_comp_rows;

    ggml_tensor * v = ggml_view_2d(ctx, t, t->ne[0], n_rows, t->nb[1], seq_id*t->nb[2]);

    return ggml_reshape_3d(ctx, v, t->ne[0], 1, n_rows);
}

ggml_tensor * llama_memory_hybrid_iswa::get_dsv4_index_k(ggml_context * ctx, int32_t il, llama_seq_id seq_id) const {
    GGML_ASSERT(il >= 0 && il < (int32_t) dsv4_cache_layers.size());
    GGML_ASSERT(seq_id >= 0 && (uint32_t) seq_id < dsv4_n_seq_max);

    ggml_tensor * t = dsv4_cache_layers[il].index_k;
    GGML_ASSERT(t != nullptr);

    return ggml_view_2d(ctx, t, t->ne[0], t->ne[1], t->nb[1], seq_id*t->nb[2]);
}

//
// llama_memory_hybrid_iswa_context
//

llama_memory_hybrid_iswa_context::llama_memory_hybrid_iswa_context(llama_memory_status status) : status(status) {}

llama_memory_hybrid_iswa_context::llama_memory_hybrid_iswa_context(llama_memory_hybrid_iswa * mem) :
    mem(mem),
    ctx_attn(mem->get_mem_attn()->init_full()),
    ctx_recr(mem->get_mem_recr()->init_full()),
    status(llama_memory_status_combine(ctx_attn->get_status(), ctx_recr->get_status())) {
}

llama_memory_hybrid_iswa_context::llama_memory_hybrid_iswa_context(
        llama_memory_hybrid_iswa * mem,
                   llama_context * lctx,
                            bool   optimize) :
    mem(mem),
    ctx_attn(mem->get_mem_attn()->init_update(lctx, optimize)),
    ctx_recr(mem->get_mem_recr()->init_update(lctx, optimize)),
    status(llama_memory_status_combine(ctx_attn->get_status(), ctx_recr->get_status())) {
}

llama_memory_hybrid_iswa_context::llama_memory_hybrid_iswa_context(
           llama_memory_hybrid_iswa * mem,
                    slot_info_vec_t   sinfos_base,
                    slot_info_vec_t   sinfos_swa,
          std::vector<llama_ubatch>   ubatches) :
    mem(mem),
    ubatches(std::move(ubatches)),
    // note: here we copy the ubatches. not sure if this is ideal
    ctx_attn(new llama_kv_cache_iswa_context(mem->get_mem_attn(), std::move(sinfos_base), std::move(sinfos_swa), this->ubatches)),
    ctx_recr(new llama_memory_recurrent_context(mem->get_mem_recr(), this->ubatches)),
    status(llama_memory_status_combine(ctx_attn->get_status(), ctx_recr->get_status())) {
}

bool llama_memory_hybrid_iswa_context::next() {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    ctx_attn->next();
    ctx_recr->next();

    if (++i_next >= ubatches.size()) {
        return false;
    }

    return true;
}

bool llama_memory_hybrid_iswa_context::apply() {
    assert(!llama_memory_status_is_fail(status));

    bool res = true;

    res = res & ctx_attn->apply();
    res = res & ctx_recr->apply();

    return res;
}

llama_memory_status llama_memory_hybrid_iswa_context::get_status() const {
    return status;
}

const llama_ubatch & llama_memory_hybrid_iswa_context::get_ubatch() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);
    return ubatches[i_next];
}

const llama_kv_cache_iswa_context * llama_memory_hybrid_iswa_context::get_attn() const {
    return static_cast<const llama_kv_cache_iswa_context *>(ctx_attn.get());
}

const llama_memory_recurrent_context * llama_memory_hybrid_iswa_context::get_recr() const {
    return static_cast<const llama_memory_recurrent_context *>(ctx_recr.get());
}

bool llama_memory_hybrid_iswa_context::has_dsv4_compressed_kv() const {
    return mem != nullptr && mem->has_dsv4_compressed_kv();
}

uint32_t llama_memory_hybrid_iswa_context::get_dsv4_n_comp(int32_t il) const {
    GGML_ASSERT(mem != nullptr);
    return mem->get_dsv4_n_comp(il);
}

uint32_t llama_memory_hybrid_iswa_context::get_dsv4_n_raw() const {
    GGML_ASSERT(mem != nullptr);
    return mem->get_dsv4_n_raw();
}

ggml_tensor * llama_memory_hybrid_iswa_context::get_dsv4_attn_k(ggml_context * ctx, int32_t il, llama_seq_id seq_id) const {
    GGML_ASSERT(mem != nullptr);
    return mem->get_dsv4_attn_k(ctx, il, seq_id);
}

ggml_tensor * llama_memory_hybrid_iswa_context::get_dsv4_attn_raw(ggml_context * ctx, int32_t il, llama_seq_id seq_id) const {
    GGML_ASSERT(mem != nullptr);
    return mem->get_dsv4_attn_raw(ctx, il, seq_id);
}

ggml_tensor * llama_memory_hybrid_iswa_context::get_dsv4_attn_kall(ggml_context * ctx, int32_t il, llama_seq_id seq_id, int64_t n_comp_rows) const {
    GGML_ASSERT(mem != nullptr);
    return mem->get_dsv4_attn_kall(ctx, il, seq_id, n_comp_rows);
}

ggml_tensor * llama_memory_hybrid_iswa_context::get_dsv4_index_k(ggml_context * ctx, int32_t il, llama_seq_id seq_id) const {
    GGML_ASSERT(mem != nullptr);
    return mem->get_dsv4_index_k(ctx, il, seq_id);
}
