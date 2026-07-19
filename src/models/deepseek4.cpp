#include "models.h"

#include "ggml-backend.h"
#include "llama-kv-cache-iswa.h"
#include "llama-memory-hybrid-iswa.h"
#include "llama-memory-recurrent.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

// [DSV4_MOE_SIDECAR] defined in llama-model.cpp / ggml-backend-meta.cpp (same libs).
bool dsv4_sidecar_read_layer_set(const char * dir, int rank, std::set<int> & out);
extern "C" int ggml_backend_meta_tp_rank_public(void);

namespace {

struct dsv4_hc_mix {
    ggml_tensor * x;
    ggml_tensor * mixes;
    ggml_tensor * pre;
    ggml_tensor * post;
    ggml_tensor * comb;
};

struct dsv4_state_pair {
    ggml_tensor * kv;
    ggml_tensor * score;
};

struct dsv4_decode_compressor {
    ggml_tensor * kv_state;
    ggml_tensor * score_state;
    ggml_tensor * kv_comp;
};

struct dsv4_state_layout {
    int64_t width;
    int64_t rows;
    int64_t elems;
};

enum class dsv4_mask_kind {
    RAW_WINDOW,
    COMPRESS_CAUSAL,
    ATTN_STATIC,
    COMPRESS_CAUSAL_BLOCKDIAG,  // multi-slot: query token t (sequence s) sees ONLY its own
                                // sequence's compressed block [s*block, s*block+block) with
                                // compress-causal visibility; every other sequence's block is -inf.
};

// position-derived i32 graph inputs for the phase-uniform decode path (n_tokens == 1).
// Content changes per token but shape/pointer stay fixed, so CUDA-graph node properties
// remain stable (no per-token re-capture).
enum class dsv4_ivec_kind {
    ROW_IDX,    // [1]    state write row:   ratio==4 ? 4 + pos%4 : pos%ratio
    COMP_POS,   // [1]    rope position of the compressed row: pos + 1 - ratio
    STATE_PERM, // [rows] state column permutation: boundary ? shift : identity
    CACHE_ROW,  // [1]    compressed-KV cache row: boundary ? (pos+1)/ratio - 1 : scratch
    APE_PHASE,  // [1]    ape column: pos % ratio
};

struct dsv4_mask_entry {
    ggml_tensor   * tensor = nullptr;
    dsv4_mask_kind kind;
    int64_t         n_raw = 0;
    int64_t         n_comp = 0;
    int64_t         window = 0;
    int64_t         ratio = 0;
};

class dsv4_graph_inputs : public llm_graph_input_i {
public:
    ggml_tensor * add_mask(
            ggml_context  * ctx,
            dsv4_mask_kind kind,
            int64_t        n0,
            int64_t        n1,
            int64_t        n_raw,
            int64_t        n_comp,
            int64_t        window,
            int64_t        ratio,
            const char   * name) {
        // mask contents are fully determined by (kind, shape, n_raw, n_comp, window, ratio) and
        // the ubatch — share one tensor across layers with identical parameters so set_input()
        // fills and uploads each distinct mask once per ubatch instead of once per layer
        for (const auto & m : masks) {
            if (m.kind == kind && m.n_raw == n_raw && m.n_comp == n_comp &&
                m.window == window && m.ratio == ratio &&
                m.tensor->ne[0] == n0 && m.tensor->ne[1] == n1) {
                return m.tensor;
            }
        }
        ggml_tensor * t = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, n0, n1, 1, 1);
        ggml_set_input(t);
        ggml_set_name(t, name);
        masks.push_back({ t, kind, n_raw, n_comp, window, ratio });
        return t;
    }

    ggml_tensor * add_ivec(
            ggml_context * ctx,
            dsv4_ivec_kind kind,
            int64_t        ratio,
            int64_t        length,
            int64_t        scratch_row,
            const char   * name) {
        for (const auto & v : ivecs) {
            if (v.kind == kind && v.ratio == ratio) {
                // a mismatching scratch_row would silently write off-boundary compress results
                // into a VALID row of the layer with the larger cache
                GGML_ASSERT(v.scratch_row == scratch_row && v.tensor->ne[0] == length &&
                            "dsv4 ivec dedup: same (kind,ratio) with different scratch_row/length");
                return v.tensor;  // shared across layers with the same ratio
            }
        }
        ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, length);
        ggml_set_input(t);
        ggml_set_name(t, name);
        ivecs.push_back({ t, kind, ratio, scratch_row });
        return t;
    }

    void set_input(const llama_ubatch * ubatch) override {
        // phase-uniform path generalized to width K (K==1 decode, K>1 MTP verify): each ivec is
        // K*per_step long, step s deriving its values from pos[s] via the identical single-step
        // formula. K==1 is byte-identical to the original decode-only fill (base=0, per_step=len).
        //
        // STEP-MAJOR layout: a token at ubatch index s = (seq q = s/nst, step r = s%nst) is written
        // to logical token (r*n_seqs + q). For the multi-slot+MTP case (n_seqs>1 AND nst>1) this lets
        // dsv4_build_compressor_decode_chunk_multislot take each step's n_seqs values as a CONTIGUOUS
        // offset view of the input ivec — no transpose/cont, which the TP/meta graph splitter chokes
        // on. When n_seqs==1 OR nst==1 (every non-MTP path) step-major == seq-major -> byte-identical.
        const int64_t K     = ubatch->n_tokens;
        const int64_t nst   = ubatch->n_seq_tokens > 0 ? (int64_t) ubatch->n_seq_tokens : 1;
        const int64_t nseqs = ubatch->n_seqs > 0 ? (int64_t) ubatch->n_seqs : 1;
        for (const auto & v : ivecs) {
            if (v.tensor->buffer == nullptr) {
                continue;
            }
            const int64_t len      = v.tensor->ne[0];
            const int64_t per_step = K > 0 ? len / K : len;
            GGML_ASSERT(K > 0 && per_step * K == len &&
                        "dsv4 ivec length must be n_tokens * per-step width");

            ivec_buf.assign(len, 0);
            std::vector<int32_t> & data = ivec_buf;
            for (int64_t s = 0; s < K; ++s) {
                const llama_pos pos      = ubatch->pos ? ubatch->pos[s] : (llama_pos) s;
                const bool      boundary = ((pos + 1) % v.ratio) == 0;
                const int64_t   tok_out  = (s % nst) * nseqs + (s / nst); // step-major (== s when n_seqs==1 || nst==1)
                const int64_t   base     = tok_out * per_step;
                switch (v.kind) {
                    case dsv4_ivec_kind::ROW_IDX:
                        data[base] = (int32_t) (v.ratio == 4 ? v.ratio + pos % v.ratio : pos % v.ratio);
                        break;
                    case dsv4_ivec_kind::COMP_POS:
                        data[base] = (int32_t) (pos + 1 - v.ratio);
                        break;
                    case dsv4_ivec_kind::STATE_PERM:
                        for (int64_t i = 0; i < per_step; ++i) {
                            data[base + i] = (int32_t) i;
                        }
                        if (boundary && v.ratio == 4) {
                            // shift current window -> previous; duplicated current rows are
                            // fully overwritten before the next boundary
                            for (int64_t i = 0; i < per_step; ++i) {
                                data[base + i] = (int32_t) (v.ratio + i % v.ratio);
                            }
                        }
                        break;
                    case dsv4_ivec_kind::CACHE_ROW:
                        data[base] = (int32_t) (boundary ? (pos + 1) / v.ratio - 1 : v.scratch_row);
                        break;
                    case dsv4_ivec_kind::APE_PHASE:
                        data[base] = (int32_t) (pos % v.ratio);
                        break;
                }
            }
            ggml_backend_tensor_set(v.tensor, data.data(), 0, data.size()*sizeof(int32_t));
        }

        for (const auto & mask : masks) {
            GGML_ASSERT(mask.tensor != nullptr);
            if (mask.tensor->buffer == nullptr) {
                continue;
            }

            const int64_t n0 = mask.tensor->ne[0];
            const int64_t n1 = mask.tensor->ne[1];

            mask_buf.assign(n0*n1, -INFINITY);
            std::vector<float> & data = mask_buf;

            switch (mask.kind) {
                case dsv4_mask_kind::RAW_WINDOW:
                    fill_raw_window(data, n0, n1, mask.window, ubatch);
                    break;
                case dsv4_mask_kind::COMPRESS_CAUSAL:
                    fill_compress_causal(data, n0, n1, mask.ratio, 0, ubatch);
                    break;
                case dsv4_mask_kind::ATTN_STATIC:
                    fill_raw_window(data, n0, n1, mask.window, ubatch);
                    fill_compress_causal(data, n0, n1, mask.ratio, mask.n_raw, ubatch);
                    break;
                case dsv4_mask_kind::COMPRESS_CAUSAL_BLOCKDIAG:
                    // multi-slot: n_comp holds the per-sequence block stride (n_comp_view)
                    fill_compress_causal_blockdiag(data, n0, n1, mask.n_comp, mask.ratio, ubatch);
                    break;
            }

            ggml_backend_tensor_set(mask.tensor, data.data(), 0, data.size()*sizeof(float));
        }
    }

private:
    // query token a and key token b may attend only if they share a sequence id.
    // For n_seqs==1 (every token in one sequence) this is always true -> no-op,
    // so single-sequence output is byte-identical to before this gate was added.
    static bool dsv4_tokens_share_seq(const llama_ubatch * ubatch, int64_t a, int64_t b) {
        if (ubatch->seq_id == nullptr) {
            return true;
        }
        for (int32_t ia = 0; ia < ubatch->n_seq_id[a]; ++ia) {
            for (int32_t ib = 0; ib < ubatch->n_seq_id[b]; ++ib) {
                if (ubatch->seq_id[a][ia] == ubatch->seq_id[b][ib]) {
                    return true;
                }
            }
        }
        return false;
    }

    static void fill_raw_window(
            std::vector<float> & data,
            int64_t              n0,
            int64_t              n1,
            int64_t              window,
            const llama_ubatch * ubatch) {
        GGML_ASSERT((int64_t) ubatch->n_tokens == n1);

        for (int64_t iq = 0; iq < n1; ++iq) {
            const llama_pos p1 = ubatch->pos ? ubatch->pos[iq] : (llama_pos) iq;

            for (int64_t ik = 0; ik < std::min<int64_t>(n0, ubatch->n_tokens); ++ik) {
                const llama_pos p0 = ubatch->pos ? ubatch->pos[ik] : (llama_pos) ik;

                if (p0 > p1) {
                    continue;
                }

                if (window > 0 && p1 - p0 >= window) {
                    continue;
                }

                // block-diagonal across sequences: a raw-window key is visible to a
                // query only when they belong to the same sequence (multi-slot correctness).
                if (!dsv4_tokens_share_seq(ubatch, iq, ik)) {
                    continue;
                }

                data[iq*n0 + ik] = 0.0f;
            }
        }
    }

    static void fill_compress_causal(
            std::vector<float> & data,
            int64_t              n0,
            int64_t              n1,
            int64_t              ratio,
            int64_t              offset,
            const llama_ubatch * ubatch) {
        GGML_ASSERT(ratio > 0);

        const int64_t n_comp = n0 - offset;
        for (int64_t iq = 0; iq < n1; ++iq) {
            const llama_pos p1 = ubatch->pos ? ubatch->pos[iq] : (llama_pos) iq;
            const int64_t n_visible = (p1 + 1) / ratio;

            for (int64_t ic = 0; ic < std::min<int64_t>(n_comp, n_visible); ++ic) {
                data[iq*n0 + offset + ic] = 0.0f;
            }
        }
    }

    // Multi-slot block-diagonal compress-causal mask: the compressed-KV columns are the per-sequence
    // caches concatenated in ubatch-sequence order, each `block` rows wide (block == n_comp_view).
    // split_equal lays the ubatch out sequence-major ([seq0 tokens..][seq1 tokens..]..), so token iq
    // belongs to sequence ordinal (iq / n_seq_tokens), whose cache occupies block [ord*block, +block).
    // The query may attend only that block, and only its compress-causal-visible rows; every other
    // sequence's block stays -inf — this is what separates concurrent requests at attention time.
    // (n_seq_tokens==1 today => ord==iq; the /n_seq_tokens keeps it correct for future K-token-per-seq
    // batched speculative decode, where the cache concat is still one block per sequence ordinal.)
    static void fill_compress_causal_blockdiag(
            std::vector<float> & data,
            int64_t              n0,      // n_seqs * block
            int64_t              n1,      // n_tokens (== n_seqs * n_seq_tokens)
            int64_t              block,   // per-sequence compressed-cache stride (n_comp_view)
            int64_t              ratio,
            const llama_ubatch * ubatch) {
        GGML_ASSERT(ratio > 0 && block > 0);
        const int64_t nst = ubatch->n_seq_tokens > 0 ? (int64_t) ubatch->n_seq_tokens : 1;

        for (int64_t iq = 0; iq < n1; ++iq) {
            const llama_pos p1 = ubatch->pos ? ubatch->pos[iq] : (llama_pos) iq;
            const int64_t n_visible = (p1 + 1) / ratio;
            const int64_t base = (iq / nst) * block;   // this query's own sequence block

            for (int64_t ic = 0; ic < std::min<int64_t>(block, n_visible); ++ic) {
                data[iq*n0 + base + ic] = 0.0f;
            }
        }
    }

    std::vector<dsv4_mask_entry> masks;
    struct dsv4_ivec_entry {
        ggml_tensor *  tensor;
        dsv4_ivec_kind kind;
        int64_t        ratio;
        int64_t        scratch_row;
    };
    std::vector<dsv4_ivec_entry> ivecs;

    // Reused staging buffers for set_input — this object is shared across decode tokens of a
    // reused graph, so keeping these as members avoids a per-token heap alloc per ivec/mask
    // (the decode path runs once per generated token over every compress layer).
    std::vector<int32_t> ivec_buf;
    std::vector<float>   mask_buf;

    // Topology fingerprint for graph reuse: one record per compress layer, capturing every
    // pos-dependent value that decides graph SHAPE in the phase-uniform (n_tokens==1) decode
    // path. Mask/ivec CONTENTS are pos-dependent by design and refreshed in set_input(); only
    // these recorded values change topology: the 256-padded compressed-KV view size, the
    // visible<=top_k indexer branch and the padded-causal-mask branch.
    struct dsv4_reuse_rec {
        int64_t ratio;
        int64_t top_k;
        int64_t n_comp_cache;
        int64_t n_comp_view;
        bool    le_topk;
        bool    pad_mask;
    };
    std::vector<dsv4_reuse_rec> reuse_recs;
    uint32_t reuse_n_tokens = 0;

public:
    void add_reuse_key(int64_t ratio, int64_t top_k, int64_t n_comp_cache, int64_t n_comp_view,
                       bool le_topk, bool pad_mask, uint32_t n_tokens) {
        reuse_recs.push_back({ ratio, top_k, n_comp_cache, n_comp_view, le_topk, pad_mask });
        reuse_n_tokens = n_tokens;
    }

    bool can_reuse(const llm_graph_params & params) override {
        // The phase-uniform path records a reuse key for both decode (n_tokens==1) and the MTP verify
        // width K (n_tokens==K, only when DSV4_VERIFY_REUSE admitted it). A recorded graph is reusable
        // iff the new ubatch has the SAME width and the same pos-derived topology drivers (256-padded
        // view, visible<=top_k, padded-mask) evaluated at its LAST position. chunk/prefill builds never
        // call add_reuse_key, so reuse_recs stays empty for them and they never qualify.
        if (reuse_recs.empty() || reuse_n_tokens == 0 ||
            params.ubatch.n_tokens != reuse_n_tokens) {
            return false;
        }

        // [tp-2node-dsv4] FIX#2: derive the topology drivers from the MAX pos over the
        // whole ubatch, not the LAST token's pos. The graph BUILD path sizes the
        // compressed-cache view from the maximum position in the ubatch; under
        // multi-slot / concurrent batching the last token is not necessarily the
        // highest-pos token, so keying reuse on pos[last] could admit a graph whose
        // view/mask shape differs from what the build path would produce -> shape
        // mismatch -> abort. Taking the max makes can_reuse match the build path.
        const int64_t last = (int64_t) params.ubatch.n_tokens - 1;
        llama_pos pos = params.ubatch.pos ? params.ubatch.pos[0] : (llama_pos) last;
        if (params.ubatch.pos) {
            for (int64_t i = 1; i < (int64_t) params.ubatch.n_tokens; i++) {
                if (params.ubatch.pos[i] > pos) { pos = params.ubatch.pos[i]; }
            }
        } else {
            pos = (llama_pos) last;
        }

        for (const auto & r : reuse_recs) {
            const int64_t visible = (pos + 1) / r.ratio;
            const int64_t view    = std::min<int64_t>(r.n_comp_cache,
                                                      GGML_PAD(std::max<int64_t>(visible, 1), 256));
            if (view != r.n_comp_view ||
                (visible <= r.top_k) != r.le_topk ||
                (view > visible)     != r.pad_mask) {
                return false;
            }
        }

        return true;
    }
};

struct dsv4_rope_cfg {
    int32_t n_ctx_orig;
    float   freq_base;
    float   freq_scale;
    float   ext_factor;
    float   attn_factor;
    float   beta_fast;
    float   beta_slow;
};

static ggml_tensor * dsv4_view_scale(ggml_context * ctx, ggml_tensor * scale, int64_t idx) {
    return ggml_view_2d(ctx, scale, 1, 1, scale->nb[0], idx * scale->nb[0]);
}

static ggml_tensor * dsv4_add_scalar(ggml_context * ctx, ggml_tensor * x, float value) {
    ggml_tensor * shape = x;
    x = ggml_cont(ctx, x);
    x = ggml_reshape_1d(ctx, x, ggml_nelements(x));
    x = ggml_scale_bias(ctx, x, 1.0f, value);
    return ggml_reshape(ctx, x, shape);
}

static ggml_tensor * dsv4_mul_scalar(ggml_context * ctx, ggml_tensor * x, float value) {
    ggml_tensor * shape = x;
    x = ggml_cont(ctx, x);
    x = ggml_reshape_1d(ctx, x, ggml_nelements(x));
    x = ggml_scale(ctx, x, value);
    return ggml_reshape(ctx, x, shape);
}

static ggml_tensor * dsv4_arange_i32(ggml_context * ctx, int64_t begin, int64_t end) {
    ggml_tensor * t = ggml_arange(ctx, (float) begin, (float) end, 1.0f);
    return ggml_cast(ctx, t, GGML_TYPE_I32);
}

static ggml_tensor * dsv4_new_filled_2d(ggml_context * ctx, int64_t n0, int64_t n1, float value) {
    return ggml_fill(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n0, n1), value);
}

static ggml_tensor * dsv4_new_filled_3d(ggml_context * ctx, int64_t n0, int64_t n1, int64_t n2, float value) {
    return ggml_fill(ctx, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n0, n1, n2), value);
}

static dsv4_state_layout dsv4_make_state_layout(int64_t compress_ratio, int64_t head_dim) {
    const int64_t coff = compress_ratio == 4 ? 2 : 1;
    const int64_t width = coff * head_dim;
    const int64_t rows  = coff * compress_ratio;
    return { width, rows, width * rows };
}

static ggml_tensor * dsv4_view_cols(
        ggml_context * ctx,
        ggml_tensor  * x,
        int64_t        n0,
        int64_t        n1,
        int64_t        off0,
        int64_t        off1) {
    return ggml_view_2d(ctx, x, n0, n1, x->nb[1], off1*x->nb[1] + off0*x->nb[0]);
}

static ggml_tensor * dsv4_view_state_segment(
        ggml_context * ctx,
        ggml_tensor  * state,
        int64_t        offset,
        int64_t        width,
        int64_t        rows) {
    return ggml_view_2d(ctx, state, width, rows, width*state->nb[0], offset*state->nb[0]);
}

// ggml_cont() on an already-contiguous tensor is a full memcpy kernel that changes nothing.
// The DSV4 decode graph emits ~400 of them per token (of 527 CONTs total) -- every one a launch,
// in a step whose whole problem is kernel COUNT, not bytes (measured: halving the bytes of the
// small ops bought +1.2%, because they are latency-bound). Materialize only when the layout
// actually needs it. DSV4_KEEP_NOOP_CONT restores the old behaviour for an A/B; it changes the
// GRAPH, so it must be set on BOTH ranks or SPMD desynchronizes.
static ggml_tensor * dsv4_cont_if_needed(ggml_context * ctx, ggml_tensor * t) {
    static const bool keep = getenv("DSV4_KEEP_NOOP_CONT") != nullptr;
    return (!keep && ggml_is_contiguous(t)) ? t : ggml_cont(ctx, t);
}

static void dsv4_store_state_segment(
        ggml_context * ctx,
        ggml_cgraph  * gf,
        ggml_tensor  * src,
        ggml_tensor  * dst,
        int64_t        state_size,
        int64_t        head,
        int64_t        offset,
        int64_t        mem_size   = 0,   // # of recurrent cells (plane stride); only needed if cache_slot>0
        int64_t        cache_slot = 0) { // recurrent rollback snapshot plane (0 = current state)
    const int64_t n = ggml_nelements(src);
    src = dsv4_cont_if_needed(ctx, src);
    src = ggml_reshape_1d(ctx, src, n);

    const int64_t row = cache_slot * mem_size + head;
    ggml_tensor * view = ggml_view_1d(ctx, dst, n, (row*state_size + offset)*ggml_element_size(dst));
    ggml_build_forward_expand(gf, ggml_cpy(ctx, src, view));
}

// Multi-slot (n_seqs>1) view of ONE state segment across the per-sequence recurrent cells.
// build_rs returns [state_size, n_seqs] with one contiguous cell per sequence (cells laid out
// at stride state_size, in ubatch-seq order — same layout mamba's contiguous store relies on).
// This views the [width, rows] segment at `offset` within each cell as a batched [width, rows, n_seqs]
// tensor (column stride = state->nb[1] = state_size). For n_seqs==1 it is the 2D segment with a
// trailing dim of 1 — the single-slot path keeps using dsv4_view_state_segment (2D) untouched.
static ggml_tensor * dsv4_view_state_segment_3d(
        ggml_context * ctx,
        ggml_tensor  * state,
        int64_t        offset,
        int64_t        width,
        int64_t        rows,
        int64_t        n_seqs) {
    return ggml_view_3d(ctx, state, width, rows, n_seqs,
            width*state->nb[0], state->nb[1], offset*state->nb[0]);
}

// Multi-slot store: write src [width, rows, n_seqs] back so plane s lands in the recurrent cell
// (head + s) at byte `offset` within that cell. DSV4 packs two segments (attn, index) per cell,
// so a single segment across sequences is a strided [width*rows, n_seqs] region (cell stride =
// state_size) — the strided generalization of mamba's contiguous y_ssm store. With MTP (n_rs_seq>0)
// rollback snapshot planes are written per-seq via dsv4_store_rollback_planes_multi (cache_slot>0).
static void dsv4_store_state_segment_multi(
        ggml_context * ctx,
        ggml_cgraph  * gf,
        ggml_tensor  * src,   // [width, rows, n_seqs]
        ggml_tensor  * dst,
        int64_t        state_size,
        int64_t        head,
        int64_t        offset,
        int64_t        n_seqs,
        int64_t        mem_size   = 0,   // # of recurrent cells (rollback plane stride); only if cache_slot>0
        int64_t        cache_slot = 0) { // rollback snapshot plane (0 = current state)
    const int64_t seg = ggml_nelements(src) / n_seqs;   // width*rows per sequence
    src = ggml_cont(ctx, src);
    src = ggml_reshape_2d(ctx, src, seg, n_seqs);
    // plane `cache_slot` of sequence s lives at cell (cache_slot*mem_size + head + s); the n_seqs cells
    // are contiguous so one strided [seg, n_seqs] view (stride = state_size) covers them all.
    const int64_t row = cache_slot*mem_size + head;
    ggml_tensor * view = ggml_view_2d(ctx, dst, seg, n_seqs,
            state_size*ggml_element_size(dst),
            (row*state_size + offset)*ggml_element_size(dst));
    ggml_build_forward_expand(gf, ggml_cpy(ctx, src, view));
}

// Multi-slot rollback planes: like dsv4_store_rollback_planes but each snapshot is [width,rows,n_seqs]
// and is scattered across the n_seqs cells per plane. n_seq_tokens (K) per sequence; plane r holds the
// per-seq state "as of (K-r) tokens".
static void dsv4_store_rollback_planes_multi(
        ggml_context * ctx, ggml_cgraph * gf,
        ggml_tensor  * r_dst, ggml_tensor * s_dst,
        const std::vector<dsv4_state_pair> & snaps,  // K entries, each {kv,score} = [width,rows,n_seqs]
        int64_t state_size, int64_t head, int64_t offset,
        int64_t mem_size, int64_t n_rs_seq, int64_t n_seq_tokens, int64_t n_seqs) {
    const int64_t r_max = std::min<int64_t>(n_rs_seq, n_seq_tokens - 1);
    for (int64_t r = 1; r <= r_max; ++r) {
        const dsv4_state_pair & s = snaps[(n_seq_tokens - r) - 1]; // s_{K-r}
        dsv4_store_state_segment_multi(ctx, gf, s.kv,    r_dst, state_size, head, offset, n_seqs, mem_size, r);
        dsv4_store_state_segment_multi(ctx, gf, s.score, s_dst, state_size, head, offset, n_seqs, mem_size, r);
    }
}

// Store recurrent ROLLBACK SNAPSHOT planes 1..min(n_rs_seq, n_tokens-1) for one state segment.
// snaps = the per-token intermediate states s_1..s_{n_tokens} captured by the chunk compressor.
// Plane r holds the compress-state "as of (n_tokens - r) tokens" = snaps[n_tokens-r-1], so that after
// a partial draft acceptance, seq_rm(rollback=r) reads a valid older state instead of a 2nd target
// verify. Plane 0 (current/final) is written by the normal dsv4_store_state_segment call. A rollback
// can never exceed n_tokens-1 (>=1 token is always accepted), so k=n_tokens-r is always >=1 and every
// readable plane is covered by this batch's intermediates — no cross-batch shifting needed.
static void dsv4_store_rollback_planes(
        ggml_context * ctx, ggml_cgraph * gf,
        ggml_tensor  * r_dst, ggml_tensor * s_dst,
        const std::vector<dsv4_state_pair> & snaps,
        int64_t state_size, int64_t head, int64_t offset,
        int64_t mem_size, int64_t n_rs_seq, int64_t n_tokens) {
    const int64_t r_max = std::min<int64_t>(n_rs_seq, n_tokens - 1);
    static const bool rbprobe = [](){ const char* e = getenv("DFLASH_RBPROBE"); return e && atoi(e); }();
    if (rbprobe && head == 0 && offset == 0) {
        fprintf(stderr, "[RB store] n_tokens=%lld n_rs_seq=%lld r_max=%lld\n",
                (long long) n_tokens, (long long) n_rs_seq, (long long) r_max);
    }
    for (int64_t r = 1; r <= r_max; ++r) {
        const dsv4_state_pair & s = snaps[(n_tokens - r) - 1]; // s_{n_tokens-r}
        dsv4_store_state_segment(ctx, gf, s.kv,    r_dst, state_size, head, offset, mem_size, r);
        dsv4_store_state_segment(ctx, gf, s.score, s_dst, state_size, head, offset, mem_size, r);
    }
}

static void dsv4_store_cache_rows(
        ggml_context * ctx,
        ggml_cgraph  * gf,
        ggml_tensor  * cache,
        ggml_tensor  * src,
        int64_t        row_start,
        int64_t        n_rows) {
    if (n_rows <= 0) {
        return;
    }

    src = ggml_cont(ctx, src);
    src = ggml_reshape_2d(ctx, src, cache->ne[0], n_rows);

    ggml_tensor * rows = dsv4_arange_i32(ctx, row_start, row_start + n_rows);
    ggml_build_forward_expand(gf, ggml_set_rows(ctx, cache, src, rows));
}

// phase-uniform variant: the destination row comes from an i32 graph INPUT
// (CACHE_ROW), so the node properties stay stable across tokens
static void dsv4_store_cache_rows_idx(
        ggml_context * ctx,
        ggml_cgraph  * gf,
        ggml_tensor  * cache,
        ggml_tensor  * src,
        ggml_tensor  * rows) {
    src = dsv4_cont_if_needed(ctx, src);
    src = ggml_reshape_2d(ctx, src, cache->ne[0], rows->ne[0]);
    ggml_build_forward_expand(gf, ggml_set_rows(ctx, cache, src, rows));
}

static dsv4_rope_cfg dsv4_make_rope_cfg(
        const llama_hparams & hparams,
        const llama_cparams  & cparams,
        uint32_t              compress_ratio) {
    if (compress_ratio == 0) {
        return {
            0,
            hparams.rope_freq_base_train,
            1.0f,
            0.0f,
            1.0f,
            cparams.yarn_beta_fast,
            cparams.yarn_beta_slow,
        };
    }

    float attn_factor = 1.0f;
    if (cparams.yarn_ext_factor != 0.0f && cparams.rope_freq_scale > 0.0f) {
        // DeepSeek V4 uses YaRN-style frequency interpolation for compressed RoPE,
        // but the reference implementation does not apply YaRN's magnitude scale.
        attn_factor /= 1.0f + 0.1f * std::log(1.0f / cparams.rope_freq_scale);
    }

    return {
        (int32_t) cparams.n_ctx_orig_yarn,
        hparams.compress_rope_freq_base > 0.0f ? hparams.compress_rope_freq_base : cparams.rope_freq_base,
        cparams.rope_freq_scale,
        cparams.yarn_ext_factor,
        attn_factor,
        cparams.yarn_beta_fast,
        cparams.yarn_beta_slow,
    };
}

static ggml_tensor * dsv4_view_base(ggml_context * ctx, ggml_tensor * base, int64_t n, int64_t off) {
    return ggml_view_2d(ctx, base, n, 1, base->nb[0], off * base->nb[0]);
}

static ggml_tensor * dsv4_apply_rope_tail(
        ggml_context * ctx,
        ggml_tensor  * x,
        ggml_tensor  * inp_pos,
        int64_t        n_embd_head,
        int64_t        n_head,
        int64_t        n_tokens,
        int64_t        n_rot,
        int            rope_type,
        int32_t        n_ctx_orig,
        float          freq_base,
        float          freq_scale,
        float          ext_factor,
        float          attn_factor,
        float          beta_fast,
        float          beta_slow,
        bool           inverse) {
    GGML_ASSERT(x->ne[0] == n_embd_head);
    GGML_ASSERT(x->ne[1] == n_head);
    GGML_ASSERT(x->ne[2] == n_tokens);

    if (n_rot == n_embd_head) {
        return inverse
            ? ggml_rope_ext_back(ctx, x, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow)
            : ggml_rope_ext     (ctx, x, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
    }

    const int64_t n_nope = n_embd_head - n_rot;
    GGML_ASSERT(n_nope > 0);

    return ggml_dsv4_rope_tail(ctx, x, inp_pos, nullptr, n_rot, rope_type,
            n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor,
            beta_fast, beta_slow, inverse);
}

static dsv4_hc_mix dsv4_hc_pre(
        ggml_context * ctx,
        ggml_tensor  * x,
        ggml_tensor  * hc_fn,
        ggml_tensor  * hc_scale,
        ggml_tensor  * hc_base,
        int64_t        n_embd,
        int64_t        n_hc,
        int64_t        n_tokens,
        float          norm_eps,
        int            sinkhorn_iters,
        float          hc_eps) {
    const int64_t hc_dim = n_embd * n_hc;
    ggml_tensor * flat = dsv4_cont_if_needed(ctx, ggml_reshape_2d(ctx, x, hc_dim, n_tokens));
    flat = ggml_rms_norm(ctx, flat, norm_eps);
    // [DSV4_HC_BF16] The hyper-connection mixing GEMM (hc_*_fn, [n_embd*n_hc=16384, 24]) is stored
    // F32 in the gguf -> cublasSgemm (CUDA cores, tensor cores idle), ~7.6% of prefill at -ub=2048
    // per DSV4_OPPROF. Casting the weight to BF16 routes it to cublasGemmEx CUDA_R_16BF +
    // CUBLAS_COMPUTE_32F (BF16 tensor cores, FP32 accumulate) and halves the weight wire volume.
    // The product feeds a Sinkhorn/sigmoid mixing-weight (precision-tolerant, F32 downstream).
    // O(ub) not O(ub^2). Default OFF = byte-identical F32. Gate: DSV4_HC_BF16.
    static const bool hc_bf16 = getenv("DSV4_HC_BF16") != nullptr;
    ggml_tensor * hc_fn_g = hc_bf16 ? ggml_cast(ctx, hc_fn, GGML_TYPE_BF16) : hc_fn;
    ggml_tensor * mixes = ggml_mul_mat(ctx, hc_fn_g, flat); // [mix_hc, n_tokens]
    ggml_tensor * split = ggml_dsv4_hc_split_sinkhorn(ctx, mixes, hc_scale, hc_base, n_hc, sinkhorn_iters, hc_eps);
    ggml_tensor * pre = ggml_view_2d(ctx, split, n_hc, n_tokens, split->nb[1], 0);
    ggml_tensor * post = ggml_view_2d(ctx, split, n_hc, n_tokens, split->nb[1], n_hc * split->nb[0]);
    ggml_tensor * comb = ggml_view_2d(ctx, split, n_hc * n_hc, n_tokens, split->nb[1], 2 * n_hc * split->nb[0]);
    if (n_tokens != 1) {
        pre = ggml_cont(ctx, pre);
        post = ggml_cont(ctx, post);
        comb = ggml_cont(ctx, comb);
    }
    comb = ggml_reshape_3d(ctx, comb, n_hc, n_hc, n_tokens); // [src_hc, dst_hc, n_tokens]
    ggml_tensor * y = ggml_dsv4_hc_weighted_sum(ctx, x, pre);
    return { y, mixes, pre, post, comb };
}

static ggml_tensor * dsv4_hc_post(
        ggml_context * ctx,
        ggml_tensor  * x,
        ggml_tensor  * residual,
        ggml_tensor  * post,
        ggml_tensor  * comb,
        int64_t        n_embd,
        int64_t        n_hc,
        int64_t        n_tokens) {
    GGML_ASSERT(x->ne[0] == n_embd);
    GGML_ASSERT(x->ne[1] == n_tokens);
    GGML_ASSERT(residual->ne[0] == n_embd);
    GGML_ASSERT(residual->ne[1] == n_hc);
    GGML_ASSERT(residual->ne[2] == n_tokens);
    GGML_ASSERT(post->ne[0] == n_hc);
    GGML_ASSERT(post->ne[1] == n_tokens);
    GGML_ASSERT(comb->ne[0] == n_hc);
    GGML_ASSERT(comb->ne[1] == n_hc);
    GGML_ASSERT(comb->ne[2] == n_tokens);

    return ggml_dsv4_hc_expand(ctx, x, residual, post, comb);
}

static ggml_tensor * dsv4_hc_head(
        ggml_context * ctx,
        ggml_tensor  * x,
        ggml_tensor  * hc_fn,
        ggml_tensor  * hc_scale,
        ggml_tensor  * hc_base,
        int64_t        n_embd,
        int64_t        n_hc,
        int64_t        n_tokens,
        float          norm_eps,
        float          hc_eps) {
    const int64_t hc_dim = n_embd * n_hc;

    ggml_tensor * flat = dsv4_cont_if_needed(ctx, ggml_reshape_2d(ctx, x, hc_dim, n_tokens));
    flat = ggml_rms_norm(ctx, flat, norm_eps);

    // [DSV4_HC_BF16] same lever as dsv4_hc_pre — BF16 tensor-core the hc mixing GEMM. Default OFF.
    static const bool hc_bf16 = getenv("DSV4_HC_BF16") != nullptr;
    ggml_tensor * hc_fn_g = hc_bf16 ? ggml_cast(ctx, hc_fn, GGML_TYPE_BF16) : hc_fn;
    ggml_tensor * pre = ggml_mul_mat(ctx, hc_fn_g, flat); // [hc, n_tokens]
    pre = ggml_mul(ctx, pre, dsv4_view_scale(ctx, hc_scale, 0));
    pre = ggml_add(ctx, pre, dsv4_view_base(ctx, hc_base, n_hc, 0));
    pre = dsv4_add_scalar(ctx, ggml_sigmoid(ctx, pre), hc_eps);

    return ggml_dsv4_hc_weighted_sum(ctx, x, pre);
}

static ggml_tensor * dsv4_grouped_out(
        ggml_context * ctx,
        ggml_tensor  * o,
        ggml_tensor  * wo_a,
        ggml_tensor  * wo_b,
        int64_t        n_embd_head,
        int64_t        n_head,
        int64_t        n_groups,
        int64_t        o_lora_rank,
        int64_t        n_tokens) {
    GGML_ASSERT(n_head % n_groups == 0);

    const int64_t group_heads = n_head / n_groups;
    const int64_t group_dim   = n_embd_head * group_heads;

    o = dsv4_cont_if_needed(ctx, o);

    // [dsv4-attn-split] The group routing is the IDENTITY (ids = arange), so the mul_mat_id
    // is mathematically a batched MUL_MAT over the group dim. Under the TP head-split the
    // batched form is required: it gives the meta backend clean aligned-AXIS_2 semantics
    // (weight and per-group activations split on the same batch dim -> each rank multiplies
    // only its local groups; no ids remap machinery). Kept gated to leave the long-verified
    // mul_mat_id path byte-identical when the split is off.
    static const bool attn_split = getenv("DSV4_ATTN_SPLIT") != nullptr;
    // [prefill] DSV4_GROUPED_OUT_BMM=1 uses the batched form WITHOUT the TP split too:
    // measured 13k prefill spends 8.2% in the identity mul_mat_id (2138ms) — the batched
    // MUL_MAT is a proper GEMM at prefill M and removes the ids machinery entirely.
    static const bool bmm_out = attn_split || getenv("DSV4_GROUPED_OUT_BMM") != nullptr;
    if (bmm_out) {
        ggml_tensor * o4 = ggml_reshape_4d(ctx, o, group_dim, 1, n_groups, n_tokens);
        ggml_tensor * wo_a_g = ggml_reshape_3d(ctx, wo_a, group_dim, o_lora_rank, n_groups);
        ggml_tensor * low = ggml_mul_mat(ctx, wo_a_g, o4); // [o_lora_rank, 1, n_groups, n_tokens]
        low = ggml_reshape_2d(ctx, low, o_lora_rank * n_groups, n_tokens);
        return ggml_mul_mat(ctx, wo_b, low);
    }

    o = ggml_reshape_3d(ctx, o, group_dim, n_groups, n_tokens);

    ggml_tensor * wo_a_g = ggml_reshape_3d(ctx, wo_a, group_dim, o_lora_rank, n_groups);
    ggml_tensor * ids = ggml_arange(ctx, 0.0f, float(n_groups), 1.0f);
    ids = ggml_cast(ctx, ids, GGML_TYPE_I32);
    ids = ggml_repeat_4d(ctx, ids, n_groups, n_tokens, 1, 1);

    ggml_tensor * low = ggml_mul_mat_id(ctx, wo_a_g, o, ids); // [o_lora_rank, n_groups, n_tokens]
    low = ggml_reshape_2d(ctx, low, o_lora_rank * n_groups, n_tokens);

    return ggml_mul_mat(ctx, wo_b, low);
}

static ggml_tensor * dsv4_softmax_pool_ratio(
        ggml_context * ctx,
        ggml_tensor  * kv,
        ggml_tensor  * score) {
    score = ggml_soft_max(ctx, score);
    // [DSV4_COMPRESSOR_BF16] kv may arrive BF16 (the compressor's transpose chain ran in BF16 to cut
    // the CONT byte traffic). The softmax weights are F32; ggml_mul needs matching types, so upcast kv
    // back to F32 here (the BF16 win is the upstream transpose; this is a cheap final copy). F32 kv is a
    // no-op cast.
    if (kv->type != GGML_TYPE_F32) {
        kv = ggml_cast(ctx, kv, GGML_TYPE_F32);
    }
    ggml_tensor * pooled = ggml_mul(ctx, kv, score);
    pooled = ggml_sum_rows(ctx, pooled);
    return ggml_reshape_2d(ctx, pooled, kv->ne[1], kv->ne[2]);
}

static ggml_tensor * dsv4_shift_overlap_state(
        ggml_context * ctx,
        ggml_tensor  * x,
        float          pad_value) {
    const int64_t n_embd  = x->ne[0];
    const int64_t ratio   = x->ne[1];
    const int64_t n_comp  = x->ne[2];

    ggml_tensor * first = ggml_view_3d(ctx, x, n_embd, ratio, 1,
            x->nb[1], x->nb[2], 0);
    ggml_tensor * pad = ggml_fill(ctx, ggml_cont(ctx, first), pad_value);

    if (n_comp == 1) {
        return pad;
    }

    ggml_tensor * prev = ggml_view_3d(ctx, x, n_embd, ratio, n_comp - 1,
            x->nb[1], x->nb[2], 0);
    return ggml_concat(ctx, pad, prev, 2);
}

static ggml_tensor * dsv4_build_compressor_prefill(
        ggml_context       * ctx,
        ggml_tensor        * x,
        ggml_tensor        * wkv,
        ggml_tensor        * wgate,
        ggml_tensor        * ape,
        ggml_tensor        * norm,
        ggml_tensor        * pos,
        int64_t              n_embd_head,
        int64_t              n_rot,
        int64_t              n_tokens,
        int64_t              compress_ratio,
        int                  rope_type,
        const dsv4_rope_cfg & rope_cfg,
        float                norm_eps) {
    GGML_ASSERT(compress_ratio > 0);
    const int64_t n_comp = n_tokens / compress_ratio;
    GGML_ASSERT(n_comp > 0);

    const int64_t coff = compress_ratio == 4 ? 2 : 1;
    const int64_t n_kv = coff * n_embd_head;
    const int64_t cutoff = n_comp * compress_ratio;

    ggml_tensor * kv = ggml_mul_mat(ctx, wkv, x);       // [coff*head_dim, n_tokens]
    ggml_tensor * score = ggml_mul_mat(ctx, wgate, x);  // [coff*head_dim, n_tokens]

    // [DSV4_COMPRESSOR_BF16] The compressor's post-mul_mat activation chain (the view->permute->CONT
    // transposes that feed the softmax-pool) runs in F32 -> the DSV4_ROPE_TAIL/CONT/RMS_NORM op-classes
    // that the prefill OPPROF ranks at ~9% total (3.2+2.8+2.8). The CONT transposes are pure data
    // movement; casting kv to BF16 here halves their byte traffic. score (the softmax logits) stays
    // F32 — soft_max needs F32 and the pool weights are precision-sensitive. kv is only the VALUE that
    // gets softmax-weighted then F16-stored in the compressed cache, so BF16 on it is tolerant. The
    // final rms_norm/rope re-emit F32 for the cache store. Default OFF = byte-identical. Gate:
    // DSV4_COMPRESSOR_BF16=1.
    static const bool compressor_bf16 = getenv("DSV4_COMPRESSOR_BF16") != nullptr;
    if (compressor_bf16) {
        kv = ggml_cast(ctx, kv, GGML_TYPE_BF16);
    }

    kv = ggml_view_3d(ctx, kv, n_kv, compress_ratio, n_comp,
            kv->nb[1],
            kv->nb[1] * compress_ratio,
            0);
    score = ggml_view_3d(ctx, score, n_kv, compress_ratio, n_comp,
            score->nb[1],
            score->nb[1] * compress_ratio,
            0);
    GGML_ASSERT(cutoff <= n_tokens);

    ggml_tensor * ape_f = ape->type == GGML_TYPE_F32 ? ape : ggml_cast(ctx, ape, GGML_TYPE_F32);
    score = ggml_add(ctx, score, ggml_repeat(ctx, ape_f, score));

    if (coff == 1) {
        kv = ggml_cont(ctx, ggml_permute(ctx, kv, 1, 0, 2, 3));       // [ratio, head_dim, n_comp]
        score = ggml_cont(ctx, ggml_permute(ctx, score, 1, 0, 2, 3)); // [ratio, head_dim, n_comp]
        kv = dsv4_softmax_pool_ratio(ctx, kv, score);                // [head_dim, n_comp]
    } else {
        ggml_tensor * kv_prev = ggml_view_3d(ctx, kv, n_embd_head, compress_ratio, n_comp,
                kv->nb[1], kv->nb[2], 0);
        ggml_tensor * kv_curr = ggml_view_3d(ctx, kv, n_embd_head, compress_ratio, n_comp,
                kv->nb[1], kv->nb[2], n_embd_head * kv->nb[0]);
        ggml_tensor * score_prev = ggml_view_3d(ctx, score, n_embd_head, compress_ratio, n_comp,
                score->nb[1], score->nb[2], 0);
        ggml_tensor * score_curr = ggml_view_3d(ctx, score, n_embd_head, compress_ratio, n_comp,
                score->nb[1], score->nb[2], n_embd_head * score->nb[0]);

        kv_prev    = dsv4_shift_overlap_state(ctx, kv_prev,    0.0f);
        score_prev = dsv4_shift_overlap_state(ctx, score_prev, -INFINITY);

        kv_prev    = ggml_cont(ctx, ggml_permute(ctx, kv_prev,    1, 0, 2, 3)); // [ratio, head_dim, n_comp]
        kv_curr    = ggml_cont(ctx, ggml_permute(ctx, kv_curr,    1, 0, 2, 3));
        score_prev = ggml_cont(ctx, ggml_permute(ctx, score_prev, 1, 0, 2, 3));
        score_curr = ggml_cont(ctx, ggml_permute(ctx, score_curr, 1, 0, 2, 3));

        kv    = ggml_concat(ctx, kv_prev,    kv_curr,    0); // [2*ratio, head_dim, n_comp]
        score = ggml_concat(ctx, score_prev, score_curr, 0);
        kv = dsv4_softmax_pool_ratio(ctx, kv, score);        // [head_dim, n_comp]
    }

    kv = ggml_rms_norm(ctx, kv, norm_eps);
    kv = ggml_mul(ctx, kv, norm);
    kv = ggml_reshape_3d(ctx, kv, n_embd_head, 1, n_comp);

    kv = dsv4_apply_rope_tail(ctx, kv, pos,
            n_embd_head, 1, n_comp, n_rot, rope_type,
            rope_cfg.n_ctx_orig, rope_cfg.freq_base, rope_cfg.freq_scale,
            rope_cfg.ext_factor, rope_cfg.attn_factor, rope_cfg.beta_fast, rope_cfg.beta_slow, false);

    return kv;
}

static dsv4_state_pair dsv4_build_compressor_prefill_state(
        ggml_context * ctx,
        ggml_tensor  * x,
        ggml_tensor  * wkv,
        ggml_tensor  * wgate,
        ggml_tensor  * ape,
        int64_t        head_dim,
        int64_t        n_tokens,
        int64_t        compress_ratio) {
    const dsv4_state_layout layout = dsv4_make_state_layout(compress_ratio, head_dim);

    const int64_t cutoff    = (n_tokens / compress_ratio) * compress_ratio;
    const int64_t remainder = n_tokens - cutoff;

    ggml_tensor * kv    = ggml_mul_mat(ctx, wkv,    x); // [width, n_tokens]
    ggml_tensor * score = ggml_mul_mat(ctx, wgate,  x);
    ggml_tensor * ape_f = ape->type == GGML_TYPE_F32 ? ape : ggml_cast(ctx, ape, GGML_TYPE_F32);

    if (compress_ratio == 4) {
        ggml_tensor * kv_prev    = dsv4_new_filled_2d(ctx, layout.width, compress_ratio, 0.0f);
        ggml_tensor * score_prev = dsv4_new_filled_2d(ctx, layout.width, compress_ratio, -INFINITY);

        if (cutoff >= compress_ratio) {
            kv_prev = ggml_view_2d(ctx, kv, layout.width, compress_ratio, kv->nb[1], (cutoff - compress_ratio)*kv->nb[1]);
            score_prev = ggml_view_2d(ctx, score, layout.width, compress_ratio, score->nb[1], (cutoff - compress_ratio)*score->nb[1]);
            score_prev = ggml_add(ctx, score_prev, ape_f);
        }

        ggml_tensor * kv_curr    = dsv4_new_filled_2d(ctx, layout.width, compress_ratio, 0.0f);
        ggml_tensor * score_curr = dsv4_new_filled_2d(ctx, layout.width, compress_ratio, -INFINITY);

        if (remainder > 0) {
            ggml_tensor * kv_rem = ggml_view_2d(ctx, kv, layout.width, remainder, kv->nb[1], cutoff*kv->nb[1]);
            ggml_tensor * sc_rem = ggml_view_2d(ctx, score, layout.width, remainder, score->nb[1], cutoff*score->nb[1]);
            sc_rem = ggml_add(ctx, sc_rem, ggml_view_2d(ctx, ape_f, layout.width, remainder, ape_f->nb[1], 0));

            if (remainder == compress_ratio) {
                kv_curr = kv_rem;
                score_curr = sc_rem;
            } else {
                kv_curr = ggml_concat(ctx, kv_rem,
                        dsv4_new_filled_2d(ctx, layout.width, compress_ratio - remainder, 0.0f), 1);
                score_curr = ggml_concat(ctx, sc_rem,
                        dsv4_new_filled_2d(ctx, layout.width, compress_ratio - remainder, -INFINITY), 1);
            }
        }

        return {
            ggml_concat(ctx, kv_prev,    kv_curr,    1),
            ggml_concat(ctx, score_prev, score_curr, 1),
        };
    }

    ggml_tensor * kv_state    = dsv4_new_filled_2d(ctx, layout.width, compress_ratio, 0.0f);
    ggml_tensor * score_state = dsv4_new_filled_2d(ctx, layout.width, compress_ratio, -INFINITY);

    if (remainder > 0) {
        ggml_tensor * kv_rem = ggml_view_2d(ctx, kv, layout.width, remainder, kv->nb[1], cutoff*kv->nb[1]);
        ggml_tensor * sc_rem = ggml_view_2d(ctx, score, layout.width, remainder, score->nb[1], cutoff*score->nb[1]);
        sc_rem = ggml_add(ctx, sc_rem, ggml_view_2d(ctx, ape_f, layout.width, remainder, ape_f->nb[1], 0));

        if (remainder == compress_ratio) {
            kv_state = kv_rem;
            score_state = sc_rem;
        } else {
            kv_state = ggml_concat(ctx, kv_rem,
                    dsv4_new_filled_2d(ctx, layout.width, compress_ratio - remainder, 0.0f), 1);
            score_state = ggml_concat(ctx, sc_rem,
                    dsv4_new_filled_2d(ctx, layout.width, compress_ratio - remainder, -INFINITY), 1);
        }
    }

    return { kv_state, score_state };
}

static ggml_tensor * dsv4_pool_decode_state(
        ggml_context * ctx,
        ggml_tensor  * kv,
        ggml_tensor  * score,
        ggml_tensor  * norm,
        ggml_tensor  * pos,
        int64_t        head_dim,
        int64_t        n_rot,
        int            rope_type,
        const dsv4_rope_cfg & rope_cfg,
        float          norm_eps) {
    const int64_t n_rows = kv->ne[1];
    // batch dim (trailing): n_seqs==1 for single-slot decode (kv 2D -> ne[2]==1, byte-identical);
    // multi-slot batched decode passes kv [head_dim, n_rows, n_seqs] so each sequence pools its own
    // window independently. softmax/sum_rows/rms_norm all run per-column, broadcasting over the batch.
    const int64_t n_seqs = kv->ne[2];
    kv    = ggml_reshape_3d(ctx, ggml_cont(ctx, ggml_transpose(ctx, kv)),    n_rows, head_dim, n_seqs);
    score = ggml_reshape_3d(ctx, ggml_cont(ctx, ggml_transpose(ctx, score)), n_rows, head_dim, n_seqs);

    ggml_tensor * pooled = dsv4_softmax_pool_ratio(ctx, kv, score);
    pooled = ggml_rms_norm(ctx, pooled, norm_eps);
    pooled = ggml_mul(ctx, pooled, norm);
    pooled = ggml_reshape_3d(ctx, pooled, head_dim, 1, n_seqs);

    return dsv4_apply_rope_tail(ctx, pooled, pos,
            head_dim, 1, n_seqs, n_rot, rope_type,
            rope_cfg.n_ctx_orig, rope_cfg.freq_base, rope_cfg.freq_scale,
            rope_cfg.ext_factor, rope_cfg.attn_factor, rope_cfg.beta_fast, rope_cfg.beta_slow, false);
}

static dsv4_decode_compressor dsv4_build_compressor_decode_projected(
        ggml_context       * ctx,
        ggml_tensor        * kv_cur,
        ggml_tensor        * sc_cur,
        ggml_tensor        * prev_kv_state,
        ggml_tensor        * prev_score_state,
        ggml_tensor        * norm,
        int64_t              head_dim,
        int64_t              n_rot,
        int64_t              pos,
        int64_t              compress_ratio,
        int                  rope_type,
        const dsv4_rope_cfg & rope_cfg,
        float                norm_eps);

// phase-uniform decode compressor (n_tokens == 1): identical math to the pos-based
// version, but every position-dependent construct comes from i32 graph INPUTS so the
// graph topology and node properties are token-invariant (CUDA-graph capturable):
//  - state write row / ape column / rope position are input tensors
//  - pooling runs UNCONDITIONALLY (the ratio==4 pool reads fixed-offset views; off-boundary
//    results land in the cache scratch row via CACHE_ROW and are never visible)
//  - the boundary state shift is a column permutation gather (identity off-boundary)
static dsv4_decode_compressor dsv4_build_compressor_decode_uniform(
        ggml_context       * ctx,
        ggml_tensor        * x,
        ggml_tensor        * prev_kv_state,
        ggml_tensor        * prev_score_state,
        ggml_tensor        * wkv,
        ggml_tensor        * wgate,
        ggml_tensor        * ape,
        ggml_tensor        * norm,
        ggml_tensor        * row_idx,
        ggml_tensor        * state_perm,  // nullptr for ratio != 4
        ggml_tensor        * comp_pos,
        ggml_tensor        * ape_phase,
        int64_t              head_dim,
        int64_t              n_rot,
        int64_t              compress_ratio,
        int                  rope_type,
        const dsv4_rope_cfg & rope_cfg,
        float                norm_eps) {
    ggml_tensor * kv_cur = ggml_mul_mat(ctx, wkv, x);       // [width, 1]
    ggml_tensor * sc_cur = ggml_mul_mat(ctx, wgate, x);
    ggml_tensor * ape_f  = ape->type == GGML_TYPE_F32 ? ape : ggml_cast(ctx, ape, GGML_TYPE_F32);
    sc_cur = ggml_add(ctx, sc_cur, ggml_get_rows(ctx, ape_f, ape_phase));

    ggml_tensor * kv_state    = ggml_set_rows(ctx, prev_kv_state,    kv_cur, row_idx);
    ggml_tensor * score_state = ggml_set_rows(ctx, prev_score_state, sc_cur, row_idx);

    ggml_tensor * kv_pool;
    ggml_tensor * score_pool;
    if (compress_ratio == 4) {
        ggml_tensor * kv_prev = dsv4_view_cols(ctx, kv_state,    head_dim, compress_ratio, 0,        0);
        ggml_tensor * kv_curr = dsv4_view_cols(ctx, kv_state,    head_dim, compress_ratio, head_dim, compress_ratio);
        ggml_tensor * sc_prev = dsv4_view_cols(ctx, score_state, head_dim, compress_ratio, 0,        0);
        ggml_tensor * sc_curr = dsv4_view_cols(ctx, score_state, head_dim, compress_ratio, head_dim, compress_ratio);

        kv_pool    = ggml_concat(ctx, kv_prev, kv_curr, 1);
        score_pool = ggml_concat(ctx, sc_prev, sc_curr, 1);
    } else {
        kv_pool    = kv_state;
        score_pool = score_state;
    }

    ggml_tensor * kv_comp = dsv4_pool_decode_state(ctx, kv_pool, score_pool, norm, comp_pos,
            head_dim, n_rot, rope_type, rope_cfg, norm_eps);

    if (state_perm != nullptr) {
        kv_state    = ggml_get_rows(ctx, kv_state,    state_perm);
        score_state = ggml_get_rows(ctx, score_state, state_perm);
    }

    return { kv_state, score_state, kv_comp };
}

// Multi-slot batched decode compressor (n_seqs>1, one token per sequence). Identical math to
// dsv4_build_compressor_decode_uniform, lifted into a trailing n_seqs batch dim so N independent
// sequences advance their OWN recurrent state in one graph (no chaining between sequences — unlike
// the MTP chunk path which chains K tokens of ONE sequence through a single 2D state).
//   x              [n_embd, n_seqs]          one token per sequence
//   prev_*_state   [width,  rows, n_seqs]    per-sequence recurrent state (build_rs -> 3D segment)
//   row_idx        [n_seqs]                  per-seq state write row (set_rows index ne1 = n_seqs)
//   state_perm     [rows*n_seqs] or nullptr  per-seq boundary column permutation (ratio==4)
//   comp_pos       [n_seqs]                  per-seq compressed-row rope position
//   ape_phase      [n_seqs]                  per-seq ape column
// Every op is a stock ggml primitive whose batch semantics are defined for the trailing dim:
// ggml_set_rows broadcasts the index over ne2 (ne2 % ne11 == 0), ggml_get_rows indexes per ne2
// plane, soft_max/sum_rows/rms_norm/rope all run per-column. No custom kernel, no architecture
// change — the single-slot uniform path above is left byte-identical and still used for n_seqs==1.
static dsv4_decode_compressor dsv4_build_compressor_decode_multislot(
        ggml_context       * ctx,
        ggml_tensor        * x,
        ggml_tensor        * prev_kv_state,
        ggml_tensor        * prev_score_state,
        ggml_tensor        * wkv,
        ggml_tensor        * wgate,
        ggml_tensor        * ape,
        ggml_tensor        * norm,
        ggml_tensor        * row_idx,
        ggml_tensor        * state_perm,  // nullptr for ratio != 4
        ggml_tensor        * comp_pos,
        ggml_tensor        * ape_phase,
        int64_t              head_dim,
        int64_t              n_rot,
        int64_t              compress_ratio,
        int64_t              n_seqs,
        int                  rope_type,
        const dsv4_rope_cfg & rope_cfg,
        float                norm_eps) {
    const int64_t width = prev_kv_state->ne[0];
    const int64_t rows  = prev_kv_state->ne[1];

    ggml_tensor * kv_cur = ggml_mul_mat(ctx, wkv,   x);    // [width, n_seqs]
    ggml_tensor * sc_cur = ggml_mul_mat(ctx, wgate, x);
    ggml_tensor * ape_f  = ape->type == GGML_TYPE_F32 ? ape : ggml_cast(ctx, ape, GGML_TYPE_F32);
    sc_cur = ggml_add(ctx, sc_cur, ggml_get_rows(ctx, ape_f, ape_phase));   // [width, n_seqs]

    // per-seq state write: src plane s -> dst[:, row_idx[s], s]
    ggml_tensor * kv_cur_b   = ggml_reshape_3d(ctx, kv_cur, width, 1, n_seqs);
    ggml_tensor * sc_cur_b   = ggml_reshape_3d(ctx, sc_cur, width, 1, n_seqs);
    ggml_tensor * row_idx_b  = ggml_reshape_2d(ctx, row_idx, 1, n_seqs);
    ggml_tensor * kv_state    = ggml_set_rows(ctx, prev_kv_state,    kv_cur_b, row_idx_b);  // [width, rows, n_seqs]
    ggml_tensor * score_state = ggml_set_rows(ctx, prev_score_state, sc_cur_b, row_idx_b);

    ggml_tensor * kv_pool;
    ggml_tensor * score_pool;
    if (compress_ratio == 4) {
        const int64_t nb0 = kv_state->nb[0], nb1 = kv_state->nb[1], nb2 = kv_state->nb[2];
        const int64_t off_curr = compress_ratio*nb1 + head_dim*nb0;
        ggml_tensor * kv_prev = ggml_view_3d(ctx, kv_state,    head_dim, compress_ratio, n_seqs, nb1, nb2, 0);
        ggml_tensor * kv_curr = ggml_view_3d(ctx, kv_state,    head_dim, compress_ratio, n_seqs, nb1, nb2, off_curr);
        ggml_tensor * sc_prev = ggml_view_3d(ctx, score_state, head_dim, compress_ratio, n_seqs, nb1, nb2, 0);
        ggml_tensor * sc_curr = ggml_view_3d(ctx, score_state, head_dim, compress_ratio, n_seqs, nb1, nb2, off_curr);

        kv_pool    = ggml_concat(ctx, kv_prev, kv_curr, 1);   // [head_dim, 2*ratio, n_seqs]
        score_pool = ggml_concat(ctx, sc_prev, sc_curr, 1);
    } else {
        kv_pool    = kv_state;       // [width(=head_dim), rows, n_seqs]
        score_pool = score_state;
    }

    ggml_tensor * kv_comp = dsv4_pool_decode_state(ctx, kv_pool, score_pool, norm, comp_pos,
            head_dim, n_rot, rope_type, rope_cfg, norm_eps);   // [head_dim, 1, n_seqs]

    if (state_perm != nullptr) {
        ggml_tensor * perm_b = ggml_reshape_2d(ctx, state_perm, rows, n_seqs);  // per-seq row gather
        kv_state    = ggml_get_rows(ctx, kv_state,    perm_b);
        score_state = ggml_get_rows(ctx, score_state, perm_b);
    }

    return { kv_state, score_state, kv_comp };
}

// Multi-slot + MTP: N sequences (n_seqs) each verifying K tokens (n_seq_tokens). The split_equal
// ubatch is sequence-major ([s0t0..s0t(K-1)][s1t0..]), so token (s,k) is column s*K+k -> the k-th
// token of every sequence is a stride-K view. Chain K steps through the per-seq 3D state (like
// chunk_uniform chains K steps through a 2D state), reusing the batched single-step multislot build.
// kv_comp is concatenated step-major [head_dim,1,K*n_seqs]; the caller routes each (step,seq)
// compressed row to its sequence's cache via the matching stride-K CACHE_ROW slice.
static dsv4_decode_compressor dsv4_build_compressor_decode_chunk_multislot(
        ggml_context       * ctx,
        ggml_tensor        * x,             // [n_embd, n_seqs*n_seq_tokens] sequence-major
        ggml_tensor        * prev_kv_state, // [width, rows, n_seqs]
        ggml_tensor        * prev_score_state,
        ggml_tensor        * wkv,
        ggml_tensor        * wgate,
        ggml_tensor        * ape,
        ggml_tensor        * norm,
        ggml_tensor        * row_idx,    // [n_seqs*n_seq_tokens]
        ggml_tensor        * state_perm, // [n_seqs*n_seq_tokens * rows] or nullptr
        ggml_tensor        * comp_pos,   // [n_seqs*n_seq_tokens]
        ggml_tensor        * ape_phase,  // [n_seqs*n_seq_tokens]
        int64_t              head_dim,
        int64_t              n_rot,
        int64_t              compress_ratio,
        int64_t              n_seqs,
        int64_t              n_seq_tokens,
        int                  rope_type,
        const dsv4_rope_cfg & rope_cfg,
        float                norm_eps,
        std::vector<dsv4_state_pair> * out_snaps = nullptr) {
    const int64_t K    = n_seq_tokens;
    const int64_t rows = prev_kv_state->ne[1];
    ggml_tensor * kv_state    = prev_kv_state;
    ggml_tensor * score_state = prev_score_state;
    ggml_tensor * kv_comp     = nullptr;

    // The ivecs are filled STEP-MAJOR (see dsv4_graph_inputs::set_input): step k's n_seqs values occupy
    // the contiguous block [k*n_seqs, (k+1)*n_seqs). So each step is a CONTIGUOUS offset view of the input
    // ivec directly -- no transpose/cont. This matters for TP/meta: (1) contiguous -> set_tensor across a
    // split is fine; (2) view_src is the host NONE-leaf input -> the meta splitter's host-leaf-view special
    // case handles it (a cont/transpose would instead add compute nodes the splitter's ring logic drops).
    // x is the hidden state (compute buffer, ubatch-order/seq-major columns) -> strided view + cont is ok.
    const size_t es = ggml_element_size(row_idx);
    for (int64_t k = 0; k < K; ++k) {
        ggml_tensor * x_k    = ggml_cont(ctx, ggml_view_2d(ctx, x, x->ne[0], n_seqs, K*x->nb[1], k*x->nb[1]));
        ggml_tensor * row_k  = ggml_view_1d(ctx, row_idx,   n_seqs, k*n_seqs*es);
        ggml_tensor * cpos_k = ggml_view_1d(ctx, comp_pos,  n_seqs, k*n_seqs*es);
        ggml_tensor * aph_k  = ggml_view_1d(ctx, ape_phase, n_seqs, k*n_seqs*es);
        ggml_tensor * perm_k = state_perm == nullptr ? nullptr
            : ggml_view_1d(ctx, state_perm, rows*n_seqs, k*rows*n_seqs*es);

        dsv4_decode_compressor dec = dsv4_build_compressor_decode_multislot(ctx, x_k,
                kv_state, score_state, wkv, wgate, ape, norm,
                row_k, perm_k, cpos_k, aph_k,
                head_dim, n_rot, compress_ratio, n_seqs, rope_type, rope_cfg, norm_eps);
        kv_state    = dec.kv_state;
        score_state = dec.score_state;
        kv_comp = kv_comp == nullptr ? dec.kv_comp : ggml_concat(ctx, kv_comp, dec.kv_comp, 2);
        if (out_snaps) out_snaps->push_back({ kv_state, score_state }); // s_{k+1} per sequence (3D) for rollback
    }

    return { kv_state, score_state, kv_comp };
}

static dsv4_decode_compressor dsv4_build_compressor_decode_projected(
        ggml_context       * ctx,
        ggml_tensor        * kv_cur,
        ggml_tensor        * sc_cur,
        ggml_tensor        * prev_kv_state,
        ggml_tensor        * prev_score_state,
        ggml_tensor        * norm,
        int64_t              head_dim,
        int64_t              n_rot,
        int64_t              pos,
        int64_t              compress_ratio,
        int                  rope_type,
        const dsv4_rope_cfg & rope_cfg,
        float                norm_eps) {
    const dsv4_state_layout layout = dsv4_make_state_layout(compress_ratio, head_dim);
    const int64_t pos_mod = pos % compress_ratio;
    const int64_t row = compress_ratio == 4 ? compress_ratio + pos_mod : pos_mod;
    const bool should_compress = (pos + 1) % compress_ratio == 0;

    ggml_tensor * row_idx = dsv4_arange_i32(ctx, row, row + 1);
    ggml_tensor * kv_state    = ggml_set_rows(ctx, prev_kv_state,    kv_cur, row_idx);
    ggml_tensor * score_state = ggml_set_rows(ctx, prev_score_state, sc_cur, row_idx);
    ggml_tensor * kv_comp = nullptr;

    if (should_compress) {
        ggml_tensor * kv_pool;
        ggml_tensor * score_pool;

        if (compress_ratio == 4) {
            ggml_tensor * kv_prev = dsv4_view_cols(ctx, kv_state,    head_dim, compress_ratio, 0,        0);
            ggml_tensor * kv_curr = dsv4_view_cols(ctx, kv_state,    head_dim, compress_ratio, head_dim, compress_ratio);
            ggml_tensor * sc_prev = dsv4_view_cols(ctx, score_state, head_dim, compress_ratio, 0,        0);
            ggml_tensor * sc_curr = dsv4_view_cols(ctx, score_state, head_dim, compress_ratio, head_dim, compress_ratio);

            kv_pool    = ggml_concat(ctx, kv_prev, kv_curr, 1);
            score_pool = ggml_concat(ctx, sc_prev, sc_curr, 1);

            ggml_tensor * shifted_kv    = dsv4_view_cols(ctx, kv_state,    layout.width, compress_ratio, 0, compress_ratio);
            ggml_tensor * shifted_score = dsv4_view_cols(ctx, score_state, layout.width, compress_ratio, 0, compress_ratio);
            kv_state    = ggml_concat(ctx, shifted_kv,    shifted_kv,    1);
            score_state = ggml_concat(ctx, shifted_score, shifted_score, 1);
        } else {
            kv_pool    = kv_state;
            score_pool = score_state;
        }

        ggml_tensor * comp_pos = dsv4_arange_i32(ctx, pos + 1 - compress_ratio, pos + 2 - compress_ratio);
        kv_comp = dsv4_pool_decode_state(ctx, kv_pool, score_pool, norm, comp_pos,
                head_dim, n_rot, rope_type, rope_cfg, norm_eps);
    }

    return { kv_state, score_state, kv_comp };
}

static dsv4_decode_compressor dsv4_build_compressor_decode_chunk(
        ggml_context       * ctx,
        ggml_tensor        * x,
        ggml_tensor        * prev_kv_state,
        ggml_tensor        * prev_score_state,
        ggml_tensor        * wkv,
        ggml_tensor        * wgate,
        ggml_tensor        * ape,
        ggml_tensor        * norm,
        const llama_ubatch & ubatch,
        int64_t              head_dim,
        int64_t              n_rot,
        int64_t              n_tokens,
        int64_t              compress_ratio,
        int                  rope_type,
        const dsv4_rope_cfg & rope_cfg,
        float                norm_eps,
        std::vector<dsv4_state_pair> * out_snaps = nullptr) {
    const dsv4_state_layout layout = dsv4_make_state_layout(compress_ratio, head_dim);

    ggml_tensor * kv_all = ggml_mul_mat(ctx, wkv,   x); // [width, n_tokens]
    ggml_tensor * sc_all = ggml_mul_mat(ctx, wgate, x);
    ggml_tensor * ape_f  = ape->type == GGML_TYPE_F32 ? ape : ggml_cast(ctx, ape, GGML_TYPE_F32);

    ggml_tensor * kv_state    = prev_kv_state;
    ggml_tensor * score_state = prev_score_state;
    ggml_tensor * kv_comp     = nullptr;

    for (int64_t i = 0; i < n_tokens; ++i) {
        const llama_pos pos = ubatch.pos ? ubatch.pos[i] : (llama_pos) i;
        const int64_t pos_mod = pos % compress_ratio;

        ggml_tensor * kv_cur = ggml_view_2d(ctx, kv_all, layout.width, 1, kv_all->nb[1], i*kv_all->nb[1]);
        ggml_tensor * sc_cur = ggml_view_2d(ctx, sc_all, layout.width, 1, sc_all->nb[1], i*sc_all->nb[1]);
        sc_cur = ggml_add(ctx, sc_cur, ggml_view_2d(ctx, ape_f, layout.width, 1, ape_f->nb[1], pos_mod*ape_f->nb[1]));

        dsv4_decode_compressor dec = dsv4_build_compressor_decode_projected(ctx,
                kv_cur,
                sc_cur,
                kv_state,
                score_state,
                norm,
                head_dim,
                n_rot,
                pos,
                compress_ratio,
                rope_type,
                rope_cfg,
                norm_eps);

        kv_state    = dec.kv_state;
        score_state = dec.score_state;
        if (dec.kv_comp != nullptr) {
            kv_comp = kv_comp == nullptr ? dec.kv_comp : ggml_concat(ctx, kv_comp, dec.kv_comp, 2);
        }
        // capture the post-token-i state (s_{i+1}) for recurrent rollback snapshots
        if (out_snaps) out_snaps->push_back({ kv_state, score_state });
    }

    return { kv_state, score_state, kv_comp };
}

// ============================================================================================
// BATCHED chunk compressor (roadmap item ③).  Same signature/return as
// dsv4_build_compressor_decode_chunk, producing NUMERICALLY IDENTICAL kv_comp and carry-out
// {kv_state, score_state} for the out_snaps==null case (the crashing long-prefill-chunk case).
//
// The unrolled _chunk loops n_tokens times, each step building a full windowed-state update
// (~tens of ggml objects) -> for a 512-wide non-prefill chunk over 43 layers this is ~1M graph
// objects -> ggml_new_object arena exhaustion (ggml.c assert).  This builds the SAME compression
// in O(1) graph objects (independent of n_tokens), reusing the proven batched math of
// dsv4_build_compressor_prefill.
//
// THE RECURRENCE IS A FIXED-SIZE SLIDING WINDOW (dsv4_make_state_layout): each token writes one
// state row at residue pos%ratio; every `ratio` tokens the window is pooled -> one kv_comp.  So
// compression = strided pooling of consecutive ratio-token blocks.  Positions in a chunk are
// CONTIGUOUS (first_pos..first_pos+n_tokens-1), so the block structure is:
//   r0  = first_pos % ratio                         (carry-in phase)
//   b0  = (r0==0) ? 0 : ratio - r0                  (#tokens completing the carry-in block)
//   the carry-in block (output #0, only if r0!=0) pools: carry-IN state rows [0,r0) (already
//     ape'd when stored by a previous chunk) ++ this chunk's tokens [0,b0);
//   the BULK is the block-aligned region [b0, b0 + n_full*ratio) pooled exactly like _prefill;
//   the trailing tokens after the last full block are written to the carry-OUT state, no output.
//
// ape phase alignment (matches _chunk line ~1285 which adds ape_f[:, pos%ratio]): within the bulk
// the residue equals the in-block row j (block-aligned) so we reuse _prefill's repeat-ape; the
// carry-in tokens [0,b0) sit at residues r0..ratio-1 so we add ape_f[:, r0..ratio-1]; trailing
// tokens [.., n_tokens) sit at residues 0..rem-1 so we add ape_f[:, 0..rem-1].
//
// comp_pos (matches _chunk's dsv4_arange at line ~1243: comp_pos = pos+1-ratio per output): the
// k-th output (k=0..n_out-1) is block n_comp_before+k, whose first token is at global position
// (n_comp_before+k)*ratio, i.e. comp_pos = (n_comp_before+k)*ratio.
//
// Only handles out_snaps==null.  The caller keeps the unrolled path for out_snaps!=null (MTP
// verify, small K, no explosion) and for irregular/non-contiguous positions.
static dsv4_decode_compressor dsv4_build_compressor_decode_chunk_batched(
        ggml_context       * ctx,
        ggml_tensor        * x,
        ggml_tensor        * prev_kv_state,
        ggml_tensor        * prev_score_state,
        ggml_tensor        * wkv,
        ggml_tensor        * wgate,
        ggml_tensor        * ape,
        ggml_tensor        * norm,
        const llama_ubatch & ubatch,
        int64_t              head_dim,
        int64_t              n_rot,
        int64_t              n_tokens,
        int64_t              compress_ratio,
        int                  rope_type,
        const dsv4_rope_cfg & rope_cfg,
        float                norm_eps) {
    const dsv4_state_layout layout = dsv4_make_state_layout(compress_ratio, head_dim);
    const int64_t width = layout.width;   // ratio!=4: head_dim ; ratio==4: 2*head_dim
    (void) layout;

    const llama_pos first_pos = ubatch.pos ? ubatch.pos[0]            : (llama_pos) 0;
    const llama_pos last_pos  = ubatch.pos ? ubatch.pos[n_tokens - 1] : (llama_pos) (n_tokens - 1);

    const int64_t r0 = first_pos % compress_ratio;                 // carry-in phase
    // tokens completing the carry-in block. CAP at n_tokens: a short chunk may not even reach the
    // first boundary (n_tokens < compress_ratio - r0), in which case ALL tokens stay in the carry-in
    // block (no boundary crossed, n_out==0). Without the cap, n_after_b0 = n_tokens - b0 goes negative,
    // n_full/n_trail truncate to 0, and the carry-in ape run over-counts -> ape_tok->ne[1] (== b0) != n_tokens.
    const int64_t b0 = (r0 == 0) ? 0 : std::min<int64_t>(compress_ratio - r0, n_tokens);

    const int64_t n_comp_before  = first_pos / compress_ratio;
    const int64_t n_comp_visible = (last_pos + 1) / compress_ratio;
    const int64_t n_out          = n_comp_visible - n_comp_before;  // == #boundaries crossed

    // project + ape (ape added per-token at residue (first_pos+i)%ratio, matching _chunk).
    ggml_tensor * kv_all = ggml_mul_mat(ctx, wkv,   x);   // [width, n_tokens]
    ggml_tensor * sc_all = ggml_mul_mat(ctx, wgate, x);
    ggml_tensor * ape_f  = ape->type == GGML_TYPE_F32 ? ape : ggml_cast(ctx, ape, GGML_TYPE_F32);

    // Build a per-token ape tensor [width, n_tokens] phase-aligned to first_pos by assembling the
    // three contiguous residue runs (carry-in run r0.., full repeated blocks, trailing run ..rem).
    const int64_t n_after_b0  = n_tokens - b0;
    const int64_t n_full      = n_after_b0 / compress_ratio;
    const int64_t cutoff      = b0 + n_full * compress_ratio;       // first index past last full block
    const int64_t n_trail     = n_tokens - cutoff;                  // trailing partial block (carry-out)

    auto ape_cols = [&](int64_t col0, int64_t ncol) -> ggml_tensor * {
        return ggml_view_2d(ctx, ape_f, width, ncol, ape_f->nb[1], col0 * ape_f->nb[1]);
    };

    ggml_tensor * ape_tok = nullptr;
    if (b0 > 0) {
        ape_tok = ggml_cont(ctx, ape_cols(r0, b0));                 // residues r0..ratio-1
    }
    if (n_full > 0) {
        // repeat the full [width, ratio] ape n_full times -> [width, n_full*ratio]
        ggml_tensor * full = ggml_repeat(ctx, ape_f,
                ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, n_full * compress_ratio));
        ape_tok = ape_tok == nullptr ? full : ggml_concat(ctx, ape_tok, full, 1);
    }
    if (n_trail > 0) {
        ggml_tensor * tail = ape_cols(0, n_trail);                 // residues 0..rem-1
        ape_tok = ape_tok == nullptr ? ggml_cont(ctx, tail) : ggml_concat(ctx, ape_tok, tail, 1);
    }
    GGML_ASSERT(ape_tok != nullptr && ape_tok->ne[1] == n_tokens);
    sc_all = ggml_add(ctx, sc_all, ape_tok);                       // [width, n_tokens], post-ape

    // ---- carry-out state: rows the trailing tokens [cutoff, n_tokens) write, in residue order ----
    // The unrolled path writes token i to state row (ratio==4 ? ratio + pos%ratio : pos%ratio).
    // Trailing tokens cover residues 0..n_trail-1 (cutoff is block-aligned, so residue==col index).
    // Their state form must match prev_*_state layout exactly so the next chunk reads it correctly.
    // We reconstruct the same final state the unrolled recurrence leaves.
    ggml_tensor * out_kv_state;
    ggml_tensor * out_sc_state;

    if (compress_ratio == 4) {
        // ---- ratio==4 double-window, mirroring dsv4_build_compressor_prefill (lines ~838-863) ----
        // n_kv = 2*head_dim. Each output c pools 8 rows = block c's curr (upper-half proj) +
        // block (c-1)'s prev (lower-half proj). The decode state reconstructs exactly this: the
        // "prev" of bulk block 0 is the carry-in block (this chunk's tokens [0,b0) completing it,
        // or, if r0==0, the previous chunk's last block held in prev_*_state).
        const int64_t n_kv = 2 * head_dim;

        // Build the projection stream that feeds the pool, INCLUDING the carry-in block so block 0's
        // "prev" picks up real data (not the zero-pad _prefill uses for a fresh sequence).
        //   carry-in block lower-half "prev" data lives in prev_kv_state rows [0,4) dim0[0,head_dim).
        //   We assemble an augmented [n_kv, (n_out)*ratio] curr stream + a matching prev stream.
        //
        // curr stream (upper half, dim0 [head_dim,2hd)) for output blocks 0..n_out-1:
        //   block k's curr = the ratio tokens at chunk indices [b0 + (k-?)...]. Output block k is
        //   global block n_comp_before+k; its tokens are chunk indices [k*ratio + b0 - 0 ...]. Since
        //   output 0 is the carry-in block (chunk tokens [0,b0)) plus carry-in state when r0!=0, we
        //   special-case it; outputs >=1 (or all, when r0==0) are bulk blocks of full ratio tokens.
        //
        // To keep this provably equal to the unrolled recurrence we build each output's two windows
        // explicitly from contiguous views and pool them in ONE batched dsv4_pool_decode_state.

        // kv_all/sc_all viewed as [n_kv, ratio, *] blocks. Bulk blocks start at chunk index b0.
        // We materialize, for every output k, the curr window [head_dim, ratio] (upper proj) and the
        // prev window [head_dim, ratio] (lower proj of the previous block).
        //
        // Assemble a "block tokens" tensor of shape [n_kv, ratio, n_out] where slice k holds the
        // ratio tokens of output block k (carry-in block for k==0 when r0!=0, else bulk blocks).
        ggml_tensor * blk_kv;   // [n_kv, ratio, n_out]
        ggml_tensor * blk_sc;

        // bulk blocks (chunk indices [b0, cutoff)) reshape directly.
        ggml_tensor * bulk_kv = n_full > 0
            ? ggml_view_3d(ctx, kv_all, n_kv, compress_ratio, n_full,
                    kv_all->nb[1], kv_all->nb[1]*compress_ratio, b0*kv_all->nb[1])
            : nullptr;
        ggml_tensor * bulk_sc = n_full > 0
            ? ggml_view_3d(ctx, sc_all, n_kv, compress_ratio, n_full,
                    sc_all->nb[1], sc_all->nb[1]*compress_ratio, b0*sc_all->nb[1])
            : nullptr;

        if (r0 != 0) {
            // output #0 = carry-in block: its ratio rows = carry-in state rows [0,r0) ++ chunk
            // tokens [0,b0). The carry-in state rows hold the (already-ape'd, full-width) projections
            // of the block's first r0 tokens, stored at prev_kv_state dim1[ratio + (0..r0-1)] = the
            // "curr" region rows the unrolled path wrote them to. Read them as full width [n_kv, r0].
            ggml_tensor * ci_kv = ggml_view_2d(ctx, prev_kv_state, n_kv, r0,
                    prev_kv_state->nb[1], compress_ratio*prev_kv_state->nb[1]);
            ggml_tensor * ci_sc = ggml_view_2d(ctx, prev_score_state, n_kv, r0,
                    prev_score_state->nb[1], compress_ratio*prev_score_state->nb[1]);
            ggml_tensor * t0_kv = ggml_view_2d(ctx, kv_all, n_kv, b0, kv_all->nb[1], 0);
            ggml_tensor * t0_sc = ggml_view_2d(ctx, sc_all, n_kv, b0, sc_all->nb[1], 0);
            ggml_tensor * c0_kv = ggml_reshape_3d(ctx, ggml_concat(ctx, ci_kv, t0_kv, 1), n_kv, compress_ratio, 1);
            ggml_tensor * c0_sc = ggml_reshape_3d(ctx, ggml_concat(ctx, ci_sc, t0_sc, 1), n_kv, compress_ratio, 1);
            blk_kv = bulk_kv ? ggml_concat(ctx, c0_kv, bulk_kv, 2) : c0_kv;
            blk_sc = bulk_sc ? ggml_concat(ctx, c0_sc, bulk_sc, 2) : c0_sc;
        } else {
            blk_kv = bulk_kv;   // r0==0: all outputs are bulk blocks
            blk_sc = bulk_sc;
        }
        GGML_ASSERT(blk_kv && blk_kv->ne[2] == n_out);

        // curr window = upper half (dim0 [head_dim, 2hd)) of each block.
        ggml_tensor * kv_curr = ggml_view_3d(ctx, blk_kv, head_dim, compress_ratio, n_out,
                blk_kv->nb[1], blk_kv->nb[2], head_dim*blk_kv->nb[0]);
        ggml_tensor * sc_curr = ggml_view_3d(ctx, blk_sc, head_dim, compress_ratio, n_out,
                blk_sc->nb[1], blk_sc->nb[2], head_dim*blk_sc->nb[0]);

        // prev window = lower half (dim0 [0, head_dim)) of the PREVIOUS block, shifted by one with a
        // pad block first. For block 0 the "previous block" is the carry-in held in prev_*_state's
        // prev region (dim1[0,ratio), dim0[0,head_dim)) when r0==0 we still must seed it from state;
        // when r0!=0 block 0's prev is the previous-previous block, also living in that same state
        // region (the unrolled shift copies curr->prev each boundary). So the pad/seed for block 0 is
        // ALWAYS prev_*_state's lower-half prev region.
        ggml_tensor * kv_lower = ggml_view_3d(ctx, blk_kv, head_dim, compress_ratio, n_out,
                blk_kv->nb[1], blk_kv->nb[2], 0);                  // lower half of each block
        ggml_tensor * sc_lower = ggml_view_3d(ctx, blk_sc, head_dim, compress_ratio, n_out,
                blk_sc->nb[1], blk_sc->nb[2], 0);

        // seed (block 0's prev) = prev_kv_state lower-half prev region [head_dim, ratio].
        ggml_tensor * seed_kv = ggml_cont(ctx, ggml_view_2d(ctx, prev_kv_state, head_dim, compress_ratio,
                prev_kv_state->nb[1], 0));
        ggml_tensor * seed_sc = ggml_cont(ctx, ggml_view_2d(ctx, prev_score_state, head_dim, compress_ratio,
                prev_score_state->nb[1], 0));
        seed_kv = ggml_reshape_3d(ctx, seed_kv, head_dim, compress_ratio, 1);
        seed_sc = ggml_reshape_3d(ctx, seed_sc, head_dim, compress_ratio, 1);

        // prev stream = [seed, lower[0..n_out-1)] -> [head_dim, ratio, n_out]
        ggml_tensor * kv_prev = (n_out > 1)
            ? ggml_concat(ctx, seed_kv, ggml_view_3d(ctx, kv_lower, head_dim, compress_ratio, n_out-1,
                    kv_lower->nb[1], kv_lower->nb[2], 0), 2)
            : seed_kv;
        ggml_tensor * sc_prev = (n_out > 1)
            ? ggml_concat(ctx, seed_sc, ggml_view_3d(ctx, sc_lower, head_dim, compress_ratio, n_out-1,
                    sc_lower->nb[1], sc_lower->nb[2], 0), 2)
            : seed_sc;

        // permute to [ratio, head_dim, n_out] and concat prev||curr on dim0 -> [2*ratio, head_dim, n_out]
        kv_prev = ggml_cont(ctx, ggml_permute(ctx, kv_prev, 1, 0, 2, 3));
        sc_prev = ggml_cont(ctx, ggml_permute(ctx, sc_prev, 1, 0, 2, 3));
        ggml_tensor * kv_currp = ggml_cont(ctx, ggml_permute(ctx, kv_curr, 1, 0, 2, 3));
        ggml_tensor * sc_currp = ggml_cont(ctx, ggml_permute(ctx, sc_curr, 1, 0, 2, 3));

        ggml_tensor * kv_pool = ggml_concat(ctx, kv_prev, kv_currp, 0);   // [2*ratio, head_dim, n_out]
        ggml_tensor * sc_pool = ggml_concat(ctx, sc_prev, sc_currp, 0);

        // comp_pos[k] = (n_comp_before + k) * ratio  (step == ratio)
        ggml_tensor * comp_pos = ggml_cast(ctx, ggml_arange(ctx, (float)(n_comp_before*compress_ratio),
                (float)((n_comp_before + n_out)*compress_ratio), (float) compress_ratio), GGML_TYPE_I32);

        // pool: dsv4_pool_decode_state expects kv [head_dim_arg, n_rows, n_seqs]; here we have the
        // transposed [2*ratio, head_dim, n_out] form already matching _prefill's call into
        // dsv4_softmax_pool_ratio -> reshape. Reuse the same body as _prefill: softmax-pool then norm+rope.
        ggml_tensor * pooled = dsv4_softmax_pool_ratio(ctx, kv_pool, sc_pool);  // [head_dim, n_out]
        pooled = ggml_rms_norm(ctx, pooled, norm_eps);
        pooled = ggml_mul(ctx, pooled, norm);
        pooled = ggml_reshape_3d(ctx, pooled, head_dim, 1, n_out);
        ggml_tensor * kv_comp = dsv4_apply_rope_tail(ctx, pooled, comp_pos,
                head_dim, 1, n_out, n_rot, rope_type,
                rope_cfg.n_ctx_orig, rope_cfg.freq_base, rope_cfg.freq_scale,
                rope_cfg.ext_factor, rope_cfg.attn_factor, rope_cfg.beta_fast, rope_cfg.beta_slow, false);

        // ---- carry-out state ----
        // Replay the unrolled recurrence's FINAL state. After the last boundary, the unrolled shift set
        // BOTH dim1 halves of the state to the last completed block's "curr" region (dim1[ratio,2ratio)).
        // Then the n_trail trailing tokens were written full-width into rows ratio + (0..n_trail-1).
        // Equivalent reconstruction:
        //   last completed block's curr region = blk's last slice curr = upper proj of last full block.
        //   For the "prev" half after shift we need the full-width (n_kv) curr region of the last block.
        // We get the last completed block tokens (full width n_kv): if n_out>0 it's blk_kv slice (n_out-1).
        if (n_out > 0) {
            ggml_tensor * last_blk_kv = ggml_view_2d(ctx, blk_kv, n_kv, compress_ratio,
                    blk_kv->nb[1], (n_out-1)*blk_kv->nb[2]);   // [n_kv, ratio]
            ggml_tensor * last_blk_sc = ggml_view_2d(ctx, blk_sc, n_kv, compress_ratio,
                    blk_sc->nb[1], (n_out-1)*blk_sc->nb[2]);
            // shifted state = concat(last_blk, last_blk) on dim1 -> [n_kv, 2*ratio] (matches _projected)
            ggml_tensor * st_kv = ggml_concat(ctx, last_blk_kv, last_blk_kv, 1);
            ggml_tensor * st_sc = ggml_concat(ctx, last_blk_sc, last_blk_sc, 1);
            // write trailing tokens (residues 0..n_trail-1) into rows ratio + (0..n_trail-1).
            if (n_trail > 0) {
                ggml_tensor * tr_kv = ggml_view_2d(ctx, kv_all, n_kv, n_trail, kv_all->nb[1], cutoff*kv_all->nb[1]);
                ggml_tensor * tr_sc = ggml_view_2d(ctx, sc_all, n_kv, n_trail, sc_all->nb[1], cutoff*sc_all->nb[1]);
                ggml_tensor * tr_rows = dsv4_arange_i32(ctx, compress_ratio, compress_ratio + n_trail);
                st_kv = ggml_set_rows(ctx, st_kv, tr_kv, tr_rows);
                st_sc = ggml_set_rows(ctx, st_sc, tr_sc, tr_rows);
            }
            out_kv_state = st_kv;
            out_sc_state = st_sc;
        } else {
            // No boundary crossed in this chunk: just write all tokens into the carry-in state at their
            // residue rows ratio + (r0 .. r0+n_tokens-1). residues are r0..r0+n_tokens-1 (< ratio here).
            ggml_tensor * rws = dsv4_arange_i32(ctx, compress_ratio + r0, compress_ratio + r0 + n_tokens);
            out_kv_state = ggml_set_rows(ctx, prev_kv_state,    kv_all, rws);
            out_sc_state = ggml_set_rows(ctx, prev_score_state, sc_all, rws);
        }

        return { out_kv_state, out_sc_state, kv_comp };
    }

    // ---- ratio != 4 (==128): single window [head_dim(=width), ratio] ----
    // Each output pools one block of `ratio` consecutive tokens, in residue (row) order. comp_pos =
    // block_start. Carry-in: output 0 (if r0!=0) = carry-in state rows [0,r0) ++ tokens [0,b0).
    {
        ggml_tensor * blk_kv;   // [width, ratio, n_out]
        ggml_tensor * blk_sc;

        ggml_tensor * bulk_kv = n_full > 0
            ? ggml_view_3d(ctx, kv_all, width, compress_ratio, n_full,
                    kv_all->nb[1], kv_all->nb[1]*compress_ratio, b0*kv_all->nb[1])
            : nullptr;
        ggml_tensor * bulk_sc = n_full > 0
            ? ggml_view_3d(ctx, sc_all, width, compress_ratio, n_full,
                    sc_all->nb[1], sc_all->nb[1]*compress_ratio, b0*sc_all->nb[1])
            : nullptr;

        if (r0 != 0) {
            ggml_tensor * ci_kv = ggml_view_2d(ctx, prev_kv_state, width, r0, prev_kv_state->nb[1], 0);
            ggml_tensor * ci_sc = ggml_view_2d(ctx, prev_score_state, width, r0, prev_score_state->nb[1], 0);
            ggml_tensor * t0_kv = ggml_view_2d(ctx, kv_all, width, b0, kv_all->nb[1], 0);
            ggml_tensor * t0_sc = ggml_view_2d(ctx, sc_all, width, b0, sc_all->nb[1], 0);
            ggml_tensor * c0_kv = ggml_reshape_3d(ctx, ggml_concat(ctx, ci_kv, t0_kv, 1), width, compress_ratio, 1);
            ggml_tensor * c0_sc = ggml_reshape_3d(ctx, ggml_concat(ctx, ci_sc, t0_sc, 1), width, compress_ratio, 1);
            blk_kv = bulk_kv ? ggml_concat(ctx, c0_kv, bulk_kv, 2) : c0_kv;
            blk_sc = bulk_sc ? ggml_concat(ctx, c0_sc, bulk_sc, 2) : c0_sc;
        } else {
            blk_kv = bulk_kv;
            blk_sc = bulk_sc;
        }

        ggml_tensor * kv_comp = nullptr;
        if (n_out > 0) {
            GGML_ASSERT(blk_kv && blk_kv->ne[2] == n_out);
            // permute to [ratio, head_dim, n_out] (width==head_dim here) -> softmax-pool over ratio rows.
            ggml_tensor * kv_pool = ggml_cont(ctx, ggml_permute(ctx, blk_kv, 1, 0, 2, 3));
            ggml_tensor * sc_pool = ggml_cont(ctx, ggml_permute(ctx, blk_sc, 1, 0, 2, 3));
            ggml_tensor * pooled  = dsv4_softmax_pool_ratio(ctx, kv_pool, sc_pool);  // [head_dim, n_out]
            pooled = ggml_rms_norm(ctx, pooled, norm_eps);
            pooled = ggml_mul(ctx, pooled, norm);
            pooled = ggml_reshape_3d(ctx, pooled, head_dim, 1, n_out);

            ggml_tensor * comp_pos = ggml_cast(ctx, ggml_arange(ctx,
                    (float)(n_comp_before*compress_ratio),
                    (float)((n_comp_before + n_out)*compress_ratio), (float) compress_ratio), GGML_TYPE_I32);
            kv_comp = dsv4_apply_rope_tail(ctx, pooled, comp_pos,
                    head_dim, 1, n_out, n_rot, rope_type,
                    rope_cfg.n_ctx_orig, rope_cfg.freq_base, rope_cfg.freq_scale,
                    rope_cfg.ext_factor, rope_cfg.attn_factor, rope_cfg.beta_fast, rope_cfg.beta_slow, false);
        }

        // ---- carry-out state: write trailing tokens [cutoff, n_tokens) into rows 0..n_trail-1; if no
        // boundary crossed, write all tokens into rows r0..r0+n_tokens-1 of the carry-in state. The
        // single-window state's other rows are stale-but-unused (next pool overwrites all `ratio` rows
        // before the next boundary) — exactly the unrolled recurrence's invariant.
        if (n_out > 0) {
            // After a boundary the unrolled path leaves the window rows as-is (no shift for ratio!=4);
            // the trailing tokens then overwrite rows 0..n_trail-1. Reconstruct: start from prev_state
            // (its rows will be overwritten as the recurrence advanced), write the LAST full block's
            // rows 0..ratio-1, then the trailing rows. Equivalent: take prev_state, set rows for the
            // tokens of the last full block at residues 0..ratio-1, then set trailing rows 0..n_trail-1.
            // Simpler exact form: the final state rows 0..ratio-1 hold the most-recent token written at
            // each residue. After the last boundary (block-aligned), residues fill 0,1,.. as tokens
            // arrive; trailing tokens [cutoff,n_tokens) cover residues 0..n_trail-1. Residues
            // n_trail..ratio-1 still hold the LAST full block's tokens (written at cutoff-ratio..cutoff-1).
            ggml_tensor * st_kv = prev_kv_state;
            ggml_tensor * st_sc = prev_score_state;
            // last completed block tokens -> residues 0..ratio-1.
            // [dsv4-fp4 AUDIT FIX] n_full>=1: the last full block is the contiguous slice
            // kv_all[cutoff-ratio, cutoff). n_full==0 (carry-in block is the only/last completed
            // block, e.g. ratio==128, non-aligned first_pos, short chunk): cutoff-ratio would be
            // NEGATIVE (OOB view). The carry-in block's NEW tokens are kv_all[0,b0) at residues
            // r0..ratio-1; residues 0..r0-1 already hold prev_state. Don't touch the latter.
            if (n_full >= 1) {
                ggml_tensor * lb_kv = ggml_view_2d(ctx, kv_all, width, compress_ratio, kv_all->nb[1], (cutoff-compress_ratio)*kv_all->nb[1]);
                ggml_tensor * lb_sc = ggml_view_2d(ctx, sc_all, width, compress_ratio, sc_all->nb[1], (cutoff-compress_ratio)*sc_all->nb[1]);
                ggml_tensor * lb_rows = dsv4_arange_i32(ctx, 0, compress_ratio);
                st_kv = ggml_set_rows(ctx, st_kv, lb_kv, lb_rows);
                st_sc = ggml_set_rows(ctx, st_sc, lb_sc, lb_rows);
            } else {
                ggml_tensor * ci_kv = ggml_view_2d(ctx, kv_all, width, b0, kv_all->nb[1], 0);
                ggml_tensor * ci_sc = ggml_view_2d(ctx, sc_all, width, b0, sc_all->nb[1], 0);
                ggml_tensor * ci_rows = dsv4_arange_i32(ctx, r0, r0 + b0);   // r0..ratio-1
                st_kv = ggml_set_rows(ctx, st_kv, ci_kv, ci_rows);
                st_sc = ggml_set_rows(ctx, st_sc, ci_sc, ci_rows);
            }
            if (n_trail > 0) {
                ggml_tensor * tr_kv = ggml_view_2d(ctx, kv_all, width, n_trail, kv_all->nb[1], cutoff*kv_all->nb[1]);
                ggml_tensor * tr_sc = ggml_view_2d(ctx, sc_all, width, n_trail, sc_all->nb[1], cutoff*sc_all->nb[1]);
                ggml_tensor * tr_rows = dsv4_arange_i32(ctx, 0, n_trail);
                st_kv = ggml_set_rows(ctx, st_kv, tr_kv, tr_rows);
                st_sc = ggml_set_rows(ctx, st_sc, tr_sc, tr_rows);
            }
            out_kv_state = st_kv;
            out_sc_state = st_sc;
        } else {
            ggml_tensor * rws = dsv4_arange_i32(ctx, r0, r0 + n_tokens);
            out_kv_state = ggml_set_rows(ctx, prev_kv_state,    kv_all, rws);
            out_sc_state = ggml_set_rows(ctx, prev_score_state, sc_all, rws);
        }

        return { out_kv_state, out_sc_state, kv_comp };
    }
}

// phase-uniform multi-token (MTP verify, n_tokens == K) compressor: the K-step state recurrence of
// dsv4_build_compressor_decode_chunk, but each step calls the input-driven _uniform builder (not the
// pos-baked _projected) on the i-th slice of the K-wide i32 ivecs. Pooling runs unconditionally every
// step and off-boundary results route to the cache scratch row (via CACHE_ROW), so the graph topology
// is position-invariant and the whole verify graph becomes reusable across MTP rounds. kv_comp holds
// all K pooled rows (concat on dim 2) for a single input-indexed cache scatter. K==1 reduces exactly
// to _uniform. Used only when DSV4_VERIFY_REUSE is set and n_comp_visible <= indexer_top_k.
static dsv4_decode_compressor dsv4_build_compressor_decode_chunk_uniform(
        ggml_context       * ctx,
        ggml_tensor        * x,
        ggml_tensor        * prev_kv_state,
        ggml_tensor        * prev_score_state,
        ggml_tensor        * wkv,
        ggml_tensor        * wgate,
        ggml_tensor        * ape,
        ggml_tensor        * norm,
        ggml_tensor        * row_idx,     // [K]
        ggml_tensor        * state_perm,  // [K * 2*ratio], nullptr for ratio != 4
        ggml_tensor        * comp_pos,    // [K]
        ggml_tensor        * ape_phase,   // [K]
        int64_t              head_dim,
        int64_t              n_rot,
        int64_t              n_tokens,
        int64_t              compress_ratio,
        int                  rope_type,
        const dsv4_rope_cfg & rope_cfg,
        float                norm_eps,
        std::vector<dsv4_state_pair> * out_snaps = nullptr) {
    ggml_tensor * kv_state    = prev_kv_state;
    ggml_tensor * score_state = prev_score_state;
    ggml_tensor * kv_comp     = nullptr;
    const int64_t perm_step   = state_perm ? state_perm->ne[0] / n_tokens : 0;

    for (int64_t i = 0; i < n_tokens; ++i) {
        ggml_tensor * x_i    = ggml_view_2d(ctx, x, x->ne[0], 1, x->nb[1], i*x->nb[1]);
        ggml_tensor * row_i  = ggml_view_1d(ctx, row_idx,   1, i*row_idx->nb[0]);
        ggml_tensor * cpos_i = ggml_view_1d(ctx, comp_pos,  1, i*comp_pos->nb[0]);
        ggml_tensor * aph_i  = ggml_view_1d(ctx, ape_phase, 1, i*ape_phase->nb[0]);
        ggml_tensor * perm_i = state_perm
            ? ggml_view_1d(ctx, state_perm, perm_step, i*perm_step*state_perm->nb[0])
            : nullptr;

        dsv4_decode_compressor dec = dsv4_build_compressor_decode_uniform(ctx, x_i,
                kv_state, score_state, wkv, wgate, ape, norm,
                row_i, perm_i, cpos_i, aph_i,
                head_dim, n_rot, compress_ratio, rope_type, rope_cfg, norm_eps);

        kv_state    = dec.kv_state;
        score_state = dec.score_state;
        kv_comp = kv_comp == nullptr ? dec.kv_comp : ggml_concat(ctx, kv_comp, dec.kv_comp, 2);
        // capture the post-token-i state (s_{i+1}) for recurrent rollback snapshots
        if (out_snaps) out_snaps->push_back({ kv_state, score_state });
    }

    return { kv_state, score_state, kv_comp };
}

static ggml_tensor * dsv4_build_indexer_scores_prefill(
        ggml_context       * ctx,
        ggml_tensor        * x,
        ggml_tensor        * qr,
        ggml_tensor        * index_kv,
        ggml_tensor        * wq_b,
        ggml_tensor        * wproj,
        ggml_tensor        * pos,
        ggml_tensor        * causal_mask,
        int64_t              n_index_head,
        int64_t              n_index_head_size,
        int64_t              n_tokens,
        int64_t              n_rot,
        int                  rope_type,
        const dsv4_rope_cfg & rope_cfg) {
    ggml_tensor * q = ggml_mul_mat(ctx, wq_b, qr);
    q = ggml_reshape_3d(ctx, q, n_index_head_size, n_index_head, n_tokens);
    q = dsv4_apply_rope_tail(ctx, q, pos,
            n_index_head_size, n_index_head, n_tokens, n_rot, rope_type,
            rope_cfg.n_ctx_orig, rope_cfg.freq_base, rope_cfg.freq_scale,
            rope_cfg.ext_factor, rope_cfg.attn_factor, rope_cfg.beta_fast, rope_cfg.beta_slow, false);

    ggml_tensor * k3 = ggml_permute(ctx, index_kv, 0, 2, 1, 3); // [head_dim, n_comp, 1]
    ggml_tensor * q3 = ggml_permute(ctx, q, 0, 2, 1, 3);        // [head_dim, n_tokens, n_heads]

    // weights = scaled wproj(x) -> [n_heads, n_tokens]
    ggml_tensor * weights = ggml_mul_mat(ctx, wproj, x);      // [n_heads, n_tokens]
    const float scale = 1.0f / std::sqrt(float(n_index_head_size) * float(n_index_head));
    weights = dsv4_mul_scalar(ctx, weights, scale);

    // [DSV4_INDEXER_FUSED] Aiden's lightning-indexer fusion: compute the head-summed logits
    //   logits[c,t] = sum_h weights[h,t] * relu(dot_d(q[d,t,h], k[d,c]))
    // in ONE kernel, so the O(n_comp*ub*n_head) score tensor AND its O(ub^2) cont-transpose are
    // never materialized (the prefill memory+compute wall). Replaces mul_mat+relu+mul+cont+sum_rows.
    // Default ON (same gate as the decode path, which is where it is worth +31%; prefill measured
    // 227.6 -> 232 t/s). DSV4_INDEXER_FUSED=0 restores the explicit chain. The fused path requires
    // F32/F16 k/q (head_dim==128); it falls back to the chain otherwise via supports_op.
    static const bool indexer_fused = []{
        const char * e = getenv("DSV4_INDEXER_FUSED"); return e == nullptr || atoi(e) != 0;
    }();
    if (indexer_fused) {
        ggml_tensor * logits = ggml_dsv4_indexer_logits(ctx, k3, q3, weights); // [n_comp, n_tokens]
        return ggml_add(ctx, logits, causal_mask);
    }

    // [DSV4_INDEXER_BF16] Aiden's +29% prefill lever (hazyumps sm12x_deep_gemm_fallbacks.py): the
    // lightning-indexer logits GEMM is the prefill wall and it is O(ub*n_comp*n_head). vLLM/DeepGEMM
    // runs it on FP8 tensor cores (Hopper) or a tf32 Triton kernel (GB10); ours runs it in plain F32,
    // which on sm120/121 dispatches to cublas SGEMM (CUDA cores, tensor cores idle). Casting k and q
    // to BF16 routes the same GEMM to ggml_cuda_mul_mat_batched_cublas -> BF16 tensor cores with FP32
    // accumulation. The score feeds a top-k SELECTION so reduced precision is tolerable. Gate:
    // DSV4_INDEXER_BF16=1; default OFF = byte-identical F32.
    static const bool indexer_bf16 = getenv("DSV4_INDEXER_BF16") != nullptr;
    if (indexer_bf16) {
        k3 = ggml_cast(ctx, k3, GGML_TYPE_BF16);
        q3 = ggml_cast(ctx, q3, GGML_TYPE_BF16);
    }

    ggml_tensor * score = ggml_mul_mat(ctx, k3, q3);         // [n_comp, n_tokens, n_heads]
    score = ggml_relu(ctx, score);

    ggml_tensor * w3 = ggml_reshape_3d(ctx, weights, 1, n_index_head, n_tokens);
    w3 = ggml_permute(ctx, w3, 0, 2, 1, 3);                  // [1, n_tokens, n_heads]

    score = ggml_mul(ctx, score, w3);
    score = ggml_cont(ctx, ggml_permute(ctx, score, 1, 2, 0, 3)); // [n_heads, n_comp, n_tokens]
    score = ggml_sum_rows(ctx, score);                            // [1, n_comp, n_tokens]
    score = ggml_reshape_2d(ctx, score, index_kv->ne[2], n_tokens);

    return ggml_add(ctx, score, causal_mask);
}

static ggml_tensor * dsv4_build_indexer_scores_decode(
        ggml_context       * ctx,
        ggml_tensor        * x,
        ggml_tensor        * qr,
        ggml_tensor        * index_kv,
        ggml_tensor        * wq_b,
        ggml_tensor        * wproj,
        ggml_tensor        * pos,
        int64_t              n_index_head,
        int64_t              n_index_head_size,
        int64_t              n_comp,
        int64_t              n_tokens,
        int64_t              n_rot,
        int                  rope_type,
        const dsv4_rope_cfg & rope_cfg) {
    // n_tokens > 1 under multi-slot decode (one query token per concurrent slot). Hard-coding 1
    // here made ggml_reshape_3d abort the moment two requests decoded together. [TAG_MULTISLOT_INDEXER]
    ggml_tensor * q = ggml_mul_mat(ctx, wq_b, qr);
    q = ggml_reshape_3d(ctx, q, n_index_head_size, n_index_head, n_tokens);
    q = dsv4_apply_rope_tail(ctx, q, pos,
            n_index_head_size, n_index_head, n_tokens, n_rot, rope_type,
            rope_cfg.n_ctx_orig, rope_cfg.freq_base, rope_cfg.freq_scale,
            rope_cfg.ext_factor, rope_cfg.attn_factor, rope_cfg.beta_fast, rope_cfg.beta_slow, false);

    ggml_tensor * k = ggml_reshape_3d(ctx, index_kv, n_index_head_size, 1, n_comp);
    k = ggml_permute(ctx, k, 0, 2, 1, 3); // [head_dim, n_comp, 1]
    q = ggml_permute(ctx, q, 0, 2, 1, 3); // [head_dim, n_tokens, n_heads]

    ggml_tensor * weights = ggml_mul_mat(ctx, wproj, x); // [n_heads, n_tokens]
    const float scale = 1.0f / std::sqrt(float(n_index_head_size) * float(n_index_head));
    weights = dsv4_mul_scalar(ctx, weights, scale);

    // [DSV4_INDEXER_FUSED] The decode indexer logits are THE long-context hog. `ggml_mul_mat(k, q)`
    // broadcasts the head-shared K over q's 64 head channels, so the [128 x n_comp] index cache is
    // re-read ONCE PER HEAD: at n_comp=6400 that is 64 x 1.6 MB = 102 MB per layer per token, 2.1 GB
    // across the 21 c4a layers — measured 12.1 ms/token at 24.7k ctx (15% of the step, and it grows
    // linearly with context). The kernel is not slow (170 GB/s); it is reading 64x too much.
    //
    // The fused op streams K once and dots it against all 64 query heads, folding relu, the
    // per-head weight and the head-sum into the same kernel — the same math, 1/64 of the traffic.
    //
    // Default ON: measured 7.73 -> 10.11 t/s (+31%) at 24.7k ctx, greedy output bit-identical to the
    // explicit chain. DSV4_INDEXER_FUSED=0 restores the chain.
    static const bool indexer_fused = []{
        const char * e = getenv("DSV4_INDEXER_FUSED"); return e == nullptr || atoi(e) != 0;
    }();
    if (indexer_fused) {
        return ggml_dsv4_indexer_logits(ctx, k, q, weights); // [n_comp, n_tokens]
    }

    // [DSV4_INDEXER_BF16] same lever as the prefill path (see dsv4_build_indexer_scores_prefill):
    // route the indexer logits GEMM through BF16 tensor cores. Decode n_comp is large at long ctx,
    // so this also helps the MTP/multi-slot decode indexer. Default OFF = byte-identical F32.
    static const bool indexer_bf16 = getenv("DSV4_INDEXER_BF16") != nullptr;
    if (indexer_bf16) {
        k = ggml_cast(ctx, k, GGML_TYPE_BF16);
        q = ggml_cast(ctx, q, GGML_TYPE_BF16);
    }

    ggml_tensor * score = ggml_mul_mat(ctx, k, q); // [n_comp, n_tokens, n_heads]
    score = ggml_relu(ctx, score);

    ggml_tensor * w3 = ggml_reshape_3d(ctx, weights, 1, n_index_head, n_tokens);
    w3 = ggml_permute(ctx, w3, 0, 2, 1, 3); // [1, n_tokens, n_heads]

    score = ggml_mul(ctx, score, w3);
    score = ggml_cont(ctx, ggml_permute(ctx, score, 1, 2, 0, 3)); // [n_heads, n_comp, n_tokens]
    score = ggml_sum_rows(ctx, score);
    return ggml_reshape_2d(ctx, score, n_comp, n_tokens);
}

static ggml_tensor * dsv4_build_compressed_mask_from_topk(
        ggml_context * ctx,
        ggml_tensor  * scores,
        ggml_tensor  * topk) {
    const int64_t n_comp   = scores->ne[0];
    const int64_t n_tokens = scores->ne[1];

    ggml_tensor * scores_rows = ggml_reshape_3d(ctx, scores, 1, scores->ne[0], scores->ne[1]);
    ggml_tensor * selected_scores = ggml_get_rows(ctx, scores_rows, topk); // [1, top_k, n_tokens]
    ggml_tensor * valid = ggml_step(ctx, dsv4_add_scalar(ctx, selected_scores, 1.0e30f));
    ggml_tensor * values = dsv4_mul_scalar(ctx, dsv4_add_scalar(ctx, valid, -1.0f), 1.0e9f);

    ggml_tensor * mask = dsv4_new_filled_3d(ctx, 1, n_comp, n_tokens, -INFINITY);
    mask = ggml_set_rows(ctx, mask, values, topk);
    return ggml_reshape_2d(ctx, mask, n_comp, n_tokens);
}

// [DSV4_INDEXER_QTILE] Tiled prefill indexer mask builder — breaks the O(ub^2) memory wall that
// blocks large ubatch (the structural lever for tokens-per-expert -> compute-bound MoE).
//
// The non-tiled path (dsv4_build_indexer_scores_prefill + argsort_top_k + mask) materializes, for
// the WHOLE ubatch at once, the score tensor [n_comp, ub, n_heads] and its [n_heads, n_comp, ub]
// transpose (ggml_cont) = O(n_heads * (ub/ratio) * ub) F32 per layer. At ub=8192, n_comp=2048,
// n_heads=64 that is ~4 GB/layer of TRANSIENT compute buffer -> OOM well before the GEMMs are
// compute-bound.
//
// The top-k selection is INDEPENDENT PER QUERY (no cross-query dependence in the indexer score /
// argsort / mask), so we tile the QUERY (ub) dimension: for each tile of `qtile` queries we slice
// qr/cur/pos/causal_mask, run the exact same score->topk->mask pipeline, and CONCAT the per-tile
// [n_comp, qtile] masks along the token dim into the full [n_comp, ub] mask. Peak indexer memory
// drops from O(n_heads*n_comp*ub) to O(n_heads*n_comp*qtile) -> bounded by qtile, not ub.
// Numerically identical to the non-tiled path (same per-query math, just batched in slices).
// index_kv (the full compressed cache, [head_dim,1,n_comp]) and the weights are shared across tiles.
static ggml_tensor * dsv4_build_indexer_mask_tiled_prefill(
        ggml_context        * ctx,
        ggml_tensor         * x,        // [n_embd, n_tokens]
        ggml_tensor         * qr,       // [q_lora, n_tokens]
        ggml_tensor         * index_kv, // [head_dim, 1, n_comp] (shared, full)
        ggml_tensor         * wq_b,
        ggml_tensor         * wproj,
        ggml_tensor         * pos,      // [n_tokens] i32
        ggml_tensor         * causal_mask, // [n_comp, n_tokens]
        int64_t               n_index_head,
        int64_t               n_index_head_size,
        int64_t               n_tokens,
        int64_t               n_rot,
        int                   rope_type,
        const dsv4_rope_cfg & rope_cfg,
        int64_t               n_comp,
        int64_t               top_k,
        int64_t               qtile) {
    // [DSV4_INDEXER_FUSED] When the fused indexer op is active, _scores_prefill emits the
    // head-summed [n_comp, ub] logits in ONE kernel WITHOUT the O(n_comp*ub*n_head) score tensor /
    // cont-transpose. The whole point of the qtile loop was to bound THAT materialization; with the
    // fused op the score path is already O(n_comp*ub) (no n_head factor, no transpose), so tiling only
    // MULTIPLIES the graph (per-tile fused-op + argsort + get_rows/set_rows + concat) -> the 3x node
    // blowup the VRAM probe saw on the resumed chunk. Force the whole-ub (untiled) path when fused so
    // the resumed chunk builds ONE fused op + one argsort + one mask, same as the is_prefill chunk.
    static const bool indexer_fused_q = []{
        const char * e = getenv("DSV4_INDEXER_FUSED"); return e == nullptr || atoi(e) != 0;
    }();
    if (indexer_fused_q || qtile <= 0 || qtile >= n_tokens) {
        // No tiling: whole-ubatch path. With DSV4_INDEXER_FUSED the score chain inside
        // _scores_prefill is the single fused op -> O(n_comp*ub) memory, no qtile needed.
        ggml_tensor * scores = dsv4_build_indexer_scores_prefill(ctx, x, qr, index_kv, wq_b, wproj,
                pos, causal_mask, n_index_head, n_index_head_size, n_tokens, n_rot, rope_type, rope_cfg);
        ggml_tensor * topk = ggml_argsort_top_k(ctx, scores, top_k);
        return dsv4_build_compressed_mask_from_topk(ctx, scores, topk);
    }

    ggml_tensor * full_mask = nullptr;
    for (int64_t t0 = 0; t0 < n_tokens; t0 += qtile) {
        const int64_t tn = std::min<int64_t>(qtile, n_tokens - t0);

        // Slice the per-query inputs to [.., tn]. All are contiguous along the token (last) dim.
        ggml_tensor * x_t  = ggml_view_2d(ctx, x,  x->ne[0],  tn, x->nb[1],  t0 * x->nb[1]);
        ggml_tensor * qr_t = ggml_view_2d(ctx, qr, qr->ne[0], tn, qr->nb[1], t0 * qr->nb[1]);
        ggml_tensor * pos_t = ggml_view_1d(ctx, pos, tn, t0 * pos->nb[0]);
        // causal_mask is [n_comp, n_tokens]; slice its query columns.
        ggml_tensor * cm_t = ggml_view_2d(ctx, causal_mask, causal_mask->ne[0], tn,
                                          causal_mask->nb[1], t0 * causal_mask->nb[1]);

        ggml_tensor * scores_t = dsv4_build_indexer_scores_prefill(ctx, x_t, qr_t, index_kv,
                wq_b, wproj, pos_t, cm_t, n_index_head, n_index_head_size, tn, n_rot, rope_type, rope_cfg);
        ggml_tensor * topk_t = ggml_argsort_top_k(ctx, scores_t, top_k);
        ggml_tensor * mask_t = dsv4_build_compressed_mask_from_topk(ctx, scores_t, topk_t); // [n_comp, tn]

        full_mask = full_mask ? ggml_concat(ctx, full_mask, mask_t, 1) : mask_t;
    }
    GGML_ASSERT(full_mask && full_mask->ne[0] == n_comp && full_mask->ne[1] == n_tokens);
    return full_mask;
}

static ggml_tensor * dsv4_cache_view_3d(ggml_context * ctx, ggml_tensor * cache, int64_t n_rows) {
    ggml_tensor * view = ggml_view_2d(ctx, cache, cache->ne[0], n_rows, cache->nb[1], 0);
    return ggml_reshape_3d(ctx, view, cache->ne[0], 1, n_rows);
}

} // namespace

llama_model_deepseek4::graph::graph(const llama_model & model, const llm_graph_params & params) :
	dsv4_graph_base(params) {

    const int64_t n_hc        = hparams.n_hc;
    const int64_t n_lora_q    = hparams.n_lora_q;
    const int64_t n_lora_o    = hparams.n_lora_o;
    const int64_t n_out_group = hparams.n_attn_out_groups;

    GGML_ASSERT(n_hc > 0);
    GGML_ASSERT(n_lora_q > 0);
    GGML_ASSERT(n_lora_o > 0);
    GGML_ASSERT(n_out_group > 0);
    GGML_ASSERT(n_embd_head_k == n_embd_head_v);
    ggml_tensor * inpL = build_inp_embd(model.tok_embd);
    ggml_tensor * inp_tokens = res->t_inp_tokens;
    ggml_tensor * inp_pos = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    auto * inp_mem  = build_inp_mem_hybrid_iswa();
    auto * inp_attn = inp_mem->get_attn();
    auto * inp_rs   = inp_mem->get_recr();
    const auto * mctx_dsv4 = inp_mem->mctx;

    // [DSV4_KV_ADJACENT] unified page layout: the raw SWA window is mirrored into the front of each
    // compressed-cache plane, so k_all = [raw | comp] is a VIEW, not a per-token concat copy.
    // Requires a unified KV cache (n_stream==1): with per-stream KV the SWA row indices are global
    // across streams and would not address a single plane's raw region. Set DSV4_KV_ADJACENT=0 to
    // fall back to the concat path (the mirror is still written, so the two paths stay in sync).
    static const bool dsv4_kv_adjacent_env = []{
        const char * e = getenv("DSV4_KV_ADJACENT");
        return e == nullptr || atoi(e) != 0;
    }();
    const bool dsv4_kv_adjacent = dsv4_kv_adjacent_env && cparams.kv_unified;

    dsv4_graph_inputs * inp_dsv4 = nullptr;
    auto get_dsv4_inputs = [&]() {
        if (inp_dsv4 == nullptr) {
            auto inputs = std::make_unique<dsv4_graph_inputs>();
            inp_dsv4 = inputs.get();
            res->add_input(std::move(inputs));
        }
        return inp_dsv4;
    };

    inpL = ggml_reshape_3d(ctx0, inpL, n_embd, 1, n_tokens);
    inpL = ggml_repeat_4d(ctx0, inpL, n_embd, n_hc, n_tokens, 1);
    inpL = ggml_reshape_3d(ctx0, inpL, n_embd, n_hc, n_tokens);

    const float kq_scale = 1.0f / std::sqrt(float(n_embd_head_k));

    // the trailing NextN/MTP layer(s) are not part of the main decoder graph
    const int n_main_layers = n_layer - (int) hparams.nextn_predict_layers;

    // DFlash in-graph feature capture: the set of target layers whose n_hc-collapsed hc_ffn_post is
    // stacked into res->t_dflash_feat (avoids a per-op eval callback that would disable CUDA graphs).
    static const std::set<int> dflash_layers = []{
        std::set<int> s; const char * e = getenv("DFLASH_TARGET_LAYERS");
        std::string str = e ? e : "2,12,22,32,40"; size_t p = 0;
        while (p < str.size()) { size_t c = str.find(',', p);
            s.insert(std::atoi(str.substr(p, c == std::string::npos ? c : c - p).c_str()));
            if (c == std::string::npos) break; p = c + 1; }
        return s; }();

    for (int il = 0; il < n_main_layers; ++il) {
        const auto & layer = model.layers[il];
        const uint32_t compress_ratio = hparams.attn_compress_ratio[il];
        const dsv4_rope_cfg rope_cfg = dsv4_make_rope_cfg(hparams, cparams, compress_ratio);
        const bool is_prefill = ubatch.pos == nullptr || ubatch.pos[0] == 0;
        // Multi-slot batched decode (DSV4_MULTISLOT): the splitter emits n_seqs>1 decode ubatches
        // (one token per sequence). The compressed-layer build below carries a per-sequence batch
        // dimension through the recurrent compressor and a block-diagonal compressed-cache attention.
        // n_seqs==1 (single-slot / prefill chunks) keeps the original byte-identical path.
        const int64_t n_seqs    = ubatch.n_seqs;
        const bool    multislot = n_seqs > 1;

        if (compress_ratio != 0) {
            if (compress_ratio != 4 && compress_ratio != 128) {
                throw std::runtime_error("DeepSeek V4 unsupported attention compression ratio " + std::to_string(compress_ratio));
            }
            // Single sequence per ubatch, OR a multi-slot decode batch (n_seqs>1, each seq carrying
            // K==n_seq_tokens new tokens — K==1 plain multi-slot, K>1 multi-slot+MTP verify). Mixed
            // prefill+decode multi-sequence ubatches are never emitted for compressed DSV4; the
            // splitter guarantees !is_prefill and uniform-K via split_equal (llama-memory-hybrid-iswa.cpp).
            const bool ms_ok = ubatch.n_seqs == 1 || (multislot && !is_prefill);
            static const bool ms_dbg = getenv("DSV4_MS_DBG") != nullptr;
            if (!ms_ok || ms_dbg) {
                fprintf(stderr, "[DSV4_MS] il=%d n_seqs=%d n_seq_tokens=%d n_tokens=%d is_prefill=%d pos0=%d pos_last=%d\n",
                        il, (int) ubatch.n_seqs, (int) ubatch.n_seq_tokens, (int) ubatch.n_tokens, (int) is_prefill,
                        ubatch.pos ? (int) ubatch.pos[0] : -1, ubatch.pos ? (int) ubatch.pos[ubatch.n_tokens-1] : -1);
            }
            GGML_ASSERT(ms_ok);
        }

        ggml_tensor * residual = inpL;
        dsv4_hc_mix mix = dsv4_hc_pre(ctx0, inpL,
                layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base,
                n_embd, n_hc, n_tokens, norm_rms_eps, hparams.hc_sinkhorn_iters, hparams.hc_eps);
        ggml_tensor * cur = mix.x;
        cb(cur, "hc_attn_pre", il);
        cb(mix.mixes, "hc_attn_pre_mixes", il);
        cb(mix.pre, "hc_attn_pre_weights", il);
        cb(mix.post, "hc_attn_pre_post_weights", il);
        cb(mix.comb, "hc_attn_pre_comb", il);
        cur = build_norm(cur, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);
        ggml_tensor * qr = ggml_mul_mat(ctx0, layer.wq_a, cur);
        cb(qr, "q_lora", il);
        qr = build_norm(qr, layer.attn_q_a_norm, nullptr, LLM_NORM_RMS, il);
        cb(qr, "q_lora_norm", il);

        // [DSV4_NORM_ROPE] Fusion 3 (the vLLM recipe's "fused Q norm + KV RoPE + K insert").
        // The Q path was rms_norm -> rope_tail (2 launches) and the KV path was rms_norm -> mul(w)
        // -> rope_tail -> fp8_quantize (4 launches). Every one of them is a per-ROW kernel over a
        // 576-wide row — for KV that is a SINGLE BLOCK, i.e. one of 48 SMs, launched four times, in
        // all 41 layers. ggml_dsv4_norm_rope does the whole chain in one block-per-row kernel out of
        // shared memory. Same math, 6 launches -> 2, and no intermediate row ever reaches HBM.
        // DSV4_NORM_ROPE=0 restores the explicit chain.
        static const bool norm_rope_fused = []{
            const char * e = getenv("DSV4_NORM_ROPE"); return e == nullptr || atoi(e) != 0;
        }();

        ggml_tensor * q = ggml_mul_mat(ctx0, layer.wq_b, qr);
        q = ggml_reshape_3d(ctx0, q, n_embd_head_k, n_head, n_tokens);
        ggml_tensor * kv = ggml_mul_mat(ctx0, layer.attn_kv, cur);

        if (norm_rope_fused) {
            q = ggml_dsv4_norm_rope(ctx0, q, inp_pos, nullptr, norm_rms_eps,
                    n_rot, rope_type, rope_cfg.n_ctx_orig, rope_cfg.freq_base, rope_cfg.freq_scale,
                    rope_cfg.ext_factor, rope_cfg.attn_factor, rope_cfg.beta_fast, rope_cfg.beta_slow,
                    /*fp8_kv =*/ false);
            cb(q, "Qcur", il);

            kv = ggml_reshape_3d(ctx0, kv, n_embd_head_k, 1, n_tokens);
            kv = ggml_dsv4_norm_rope(ctx0, kv, inp_pos, layer.attn_kv_a_norm, norm_rms_eps,
                    n_rot, rope_type, rope_cfg.n_ctx_orig, rope_cfg.freq_base, rope_cfg.freq_scale,
                    rope_cfg.ext_factor, rope_cfg.attn_factor, rope_cfg.beta_fast, rope_cfg.beta_slow,
                    /*fp8_kv =*/ true);
            cb(kv, "KVcur", il);
        } else {
            q = ggml_rms_norm(ctx0, q, norm_rms_eps);
            cb(q, "Qnorm", il);
            q = dsv4_apply_rope_tail(ctx0, q, inp_pos,
                    n_embd_head_k, n_head, n_tokens, n_rot, rope_type,
                    rope_cfg.n_ctx_orig, rope_cfg.freq_base, rope_cfg.freq_scale,
                    rope_cfg.ext_factor, rope_cfg.attn_factor, rope_cfg.beta_fast, rope_cfg.beta_slow, false);
            cb(q, "Qcur", il);

            kv = build_norm(kv, layer.attn_kv_a_norm, nullptr, LLM_NORM_RMS, il);
            kv = ggml_reshape_3d(ctx0, kv, n_embd_head_k, 1, n_tokens);
            cb(kv, "KVnorm", il);
            kv = dsv4_apply_rope_tail(ctx0, kv, inp_pos,
                    n_embd_head_k, 1, n_tokens, n_rot, rope_type,
                    rope_cfg.n_ctx_orig, rope_cfg.freq_base, rope_cfg.freq_scale,
                    rope_cfg.ext_factor, rope_cfg.attn_factor, rope_cfg.beta_fast, rope_cfg.beta_slow, false);
            cb(kv, "KVrope", il);
            kv = ggml_dsv4_fp8_kv_quantize(ctx0, kv, n_rot);
            cb(kv, "KVcur", il);
        }

        const auto * mctx_swa = inp_attn->mctx->get_swa();
        ggml_build_forward_expand(gf, q);
        ggml_build_forward_expand(gf, kv);
        ggml_build_forward_expand(gf, mctx_swa->cpy_k(ctx0, kv, inp_attn->get_k_idxs_swa(), il));

        // [DSV4_KV_ADJACENT] mirror the same raw K row into the front (raw) region of this layer's
        // compressed-cache plane, at the SAME SWA cell index. That makes the attention's
        //     k_all = [raw window | compressed rows]
        // one CONTIGUOUS region, so k_all is a ggml VIEW instead of a ggml_concat that physically
        // copies (n_raw + n_comp_view) x 1152 B per layer per token — a copy that grows with context
        // (at 128k ctx it is ~790 MB/token). Cost of the mirror: one extra 1152-byte set_rows per
        // compressed layer per token. See llama-memory-hybrid-iswa.h (dsv4_n_raw).
        if (compress_ratio != 0 && dsv4_kv_adjacent && mctx_dsv4 != nullptr && mctx_dsv4->has_dsv4_compressed_kv()) {
            ggml_tensor * k_idxs = inp_attn->get_k_idxs_swa();
            // kv is [n_embd_head_k, 1, n_tokens] -> the set_rows source is [n_embd_head_k, n_tokens]
            ggml_tensor * k_src = ggml_view_2d(ctx0, kv, kv->ne[0]*kv->ne[1], n_tokens, kv->nb[2], 0);

            const int64_t n_seq_tok = ubatch.n_seq_tokens;
            for (int64_t s = 0; s < n_seqs; ++s) {
                // tokens of sequence s are contiguous: [s*n_seq_tokens, (s+1)*n_seq_tokens)
                ggml_tensor * src_s  = ggml_view_2d(ctx0, k_src, k_src->ne[0], n_seq_tok,
                        k_src->nb[1], s*n_seq_tok*k_src->nb[1]);
                ggml_tensor * idxs_s = ggml_view_1d(ctx0, k_idxs, n_seq_tok,
                        s*n_seq_tok*ggml_element_size(k_idxs));

                for (int32_t is = 0; is < ubatch.n_seq_id[s*n_seq_tok]; ++is) {
                    const llama_seq_id sid = ubatch.seq_id[s*n_seq_tok][is];
                    ggml_tensor * raw = mctx_dsv4->get_dsv4_attn_raw(ctx0, il, sid);
                    ggml_build_forward_expand(gf, ggml_set_rows(ctx0, raw, src_s, idxs_s));
                }
            }
        }

        if (compress_ratio == 0) {
            ggml_tensor * k_cache = mctx_swa->get_k(ctx0, il);
            // get_k returns [n_embd_head_k, n_head_kv(=1 for MLA), n_kv, n_stream]. Preserve the
            // per-stream dim (ne[3]) so build_attn_mha (which reads n_stream = k->ne[3] and splits
            // q accordingly) attends each sequence against its own KV stream. Collapsing to 3D
            // discards n_stream and only works for a single stream; for n_stream==1 reshape_4d
            // here is byte-identical to the previous reshape_3d (trailing dim == 1).
            k_cache = ggml_reshape_4d(ctx0, k_cache, n_embd_head_k, 1, k_cache->ne[2], k_cache->ne[3]);
            cur = build_attn_mha(q, k_cache, k_cache, nullptr, inp_attn->get_kq_mask_swa(),
                    layer.attn_sinks, nullptr, kq_scale, il);
            cb(cur, "kqv_out", il);
        } else {
            ggml_tensor * k_all = kv;
            ggml_tensor * v_all = kv;
            ggml_tensor * attn_mask = nullptr;
            // DSV4 sparse-attn (DSV4_SPARSE_ATTN=1): per-query top-k comp-row indices + raw-window
            // row count, captured in the chunk path, consumed at build_attn_mha (src[6]). Null=dense.
            ggml_tensor * sparse_topk  = nullptr;
            int64_t       sparse_n_raw = 0;
            const llama_seq_id seq_id = ubatch.seq_id[0][0];
            auto store_attn_cache_rows = [&](ggml_tensor * src, int64_t row_start, int64_t n_rows) {
                for (int32_t is = 0; is < ubatch.n_seq_id[0]; ++is) {
                    const llama_seq_id dst_seq_id = ubatch.seq_id[0][is];
                    dsv4_store_cache_rows(ctx0, gf, mctx_dsv4->get_dsv4_attn_k(ctx0, il, dst_seq_id), src, row_start, n_rows);
                }
            };
            auto store_index_cache_rows = [&](ggml_tensor * src, int64_t row_start, int64_t n_rows) {
                for (int32_t is = 0; is < ubatch.n_seq_id[0]; ++is) {
                    const llama_seq_id dst_seq_id = ubatch.seq_id[0][is];
                    dsv4_store_cache_rows(ctx0, gf, mctx_dsv4->get_dsv4_index_k(ctx0, il, dst_seq_id), src, row_start, n_rows);
                }
            };
            auto store_attn_cache_rows_idx = [&](ggml_tensor * src, ggml_tensor * rows) {
                for (int32_t is = 0; is < ubatch.n_seq_id[0]; ++is) {
                    const llama_seq_id dst_seq_id = ubatch.seq_id[0][is];
                    dsv4_store_cache_rows_idx(ctx0, gf, mctx_dsv4->get_dsv4_attn_k(ctx0, il, dst_seq_id), src, rows);
                }
            };
            auto store_index_cache_rows_idx = [&](ggml_tensor * src, ggml_tensor * rows) {
                for (int32_t is = 0; is < ubatch.n_seq_id[0]; ++is) {
                    const llama_seq_id dst_seq_id = ubatch.seq_id[0][is];
                    dsv4_store_cache_rows_idx(ctx0, gf, mctx_dsv4->get_dsv4_index_k(ctx0, il, dst_seq_id), src, rows);
                }
            };
            const int64_t state_size = hparams.n_embd_r();
            const dsv4_state_layout attn_state_layout = dsv4_make_state_layout(compress_ratio, n_embd_head_k);

            ggml_tensor * prev_kv_state_all = build_rs(inp_rs, inp_rs->mctx->get_r_l(il), state_size, ubatch.n_seqs);
            ggml_tensor * prev_sc_state_all = build_rs(inp_rs, inp_rs->mctx->get_s_l(il), state_size, ubatch.n_seqs);
            ggml_tensor * prev_attn_kv_state = multislot
                ? dsv4_view_state_segment_3d(ctx0, prev_kv_state_all, 0, attn_state_layout.width, attn_state_layout.rows, n_seqs)
                : dsv4_view_state_segment(ctx0, prev_kv_state_all, 0, attn_state_layout.width, attn_state_layout.rows);
            ggml_tensor * prev_attn_sc_state = multislot
                ? dsv4_view_state_segment_3d(ctx0, prev_sc_state_all, 0, attn_state_layout.width, attn_state_layout.rows, n_seqs)
                : dsv4_view_state_segment(ctx0, prev_sc_state_all, 0, attn_state_layout.width, attn_state_layout.rows);

            const int64_t n_comp = n_tokens / compress_ratio;
            if (is_prefill) {
                dsv4_state_pair state = dsv4_build_compressor_prefill_state(ctx0, cur,
                        layer.attn_compressor_kv,
                        layer.attn_compressor_gate,
                        layer.attn_compressor_ape,
                        n_embd_head_k,
                        n_tokens,
                        compress_ratio);
                dsv4_store_state_segment(ctx0, gf, state.kv,    inp_rs->mctx->get_r_l(il), state_size, inp_rs->head, 0);
                dsv4_store_state_segment(ctx0, gf, state.score, inp_rs->mctx->get_s_l(il), state_size, inp_rs->head, 0);

                if (compress_ratio == 4) {
                    const dsv4_state_layout index_state_layout = dsv4_make_state_layout(compress_ratio, hparams.indexer_head_size);
                    dsv4_state_pair index_state = dsv4_build_compressor_prefill_state(ctx0, cur,
                            layer.indexer_compressor_kv,
                            layer.indexer_compressor_gate,
                            layer.indexer_compressor_ape,
                            hparams.indexer_head_size,
                            n_tokens,
                            compress_ratio);
                    dsv4_store_state_segment(ctx0, gf, index_state.kv,    inp_rs->mctx->get_r_l(il), state_size, inp_rs->head, attn_state_layout.elems);
                    dsv4_store_state_segment(ctx0, gf, index_state.score, inp_rs->mctx->get_s_l(il), state_size, inp_rs->head, attn_state_layout.elems);
                    GGML_ASSERT(attn_state_layout.elems + index_state_layout.elems <= state_size);
                }
            }

            if (is_prefill && n_comp > 0) {
                ggml_tensor * comp_pos = ggml_arange(ctx0, 0.0f, float(n_comp * compress_ratio), float(compress_ratio));
                comp_pos = ggml_cast(ctx0, comp_pos, GGML_TYPE_I32);

                ggml_tensor * kv_comp = dsv4_build_compressor_prefill(ctx0, cur,
                        layer.attn_compressor_kv,
                        layer.attn_compressor_gate,
                        layer.attn_compressor_ape,
                        layer.attn_compressor_norm,
                        comp_pos,
                        n_embd_head_k, n_rot, n_tokens, compress_ratio, rope_type, rope_cfg, norm_rms_eps);
                kv_comp = ggml_dsv4_fp8_kv_quantize(ctx0, kv_comp, n_rot);
                cb(kv_comp, "KVcompress", il);

                store_attn_cache_rows(kv_comp, 0, n_comp);

                k_all = ggml_concat(ctx0, kv, kv_comp, 2);
                v_all = k_all;

                if (compress_ratio == 4) {
                    ggml_tensor * raw_mask = get_dsv4_inputs()->add_mask(ctx0,
                            dsv4_mask_kind::RAW_WINDOW,
                            n_tokens, n_tokens,
                            n_tokens, n_comp, hparams.n_swa, compress_ratio,
                            "dsv4_attn_raw_window_mask");
                    ggml_tensor * index_mask = get_dsv4_inputs()->add_mask(ctx0,
                            dsv4_mask_kind::COMPRESS_CAUSAL,
                            n_comp, n_tokens,
                            0, n_comp, 0, compress_ratio,
                            "dsv4_indexer_causal_mask");

                    ggml_tensor * index_kv = dsv4_build_compressor_prefill(ctx0, cur,
                            layer.indexer_compressor_kv,
                            layer.indexer_compressor_gate,
                            layer.indexer_compressor_ape,
                            layer.indexer_compressor_norm,
                            comp_pos,
                            hparams.indexer_head_size, n_rot, n_tokens, compress_ratio, rope_type, rope_cfg, norm_rms_eps);
                    cb(index_kv, "indexer_KVcompress", il);

                    store_index_cache_rows(index_kv, 0, n_comp);

                    const int top_k = std::min<int64_t>(hparams.indexer_top_k, n_comp);

                    // [DSV4_INDEXER_QTILE] Tile the indexer's query dim to bound peak memory at
                    // O(n_heads*n_comp*qtile) instead of O(n_heads*n_comp*ub) -> unblocks large UB
                    // (the tokens-per-expert lever). Default 0 = OFF (whole-ub, byte-identical to the
                    // prior path). Set DSV4_INDEXER_QTILE=2048 (or so) for large UB. The top-k is
                    // per-query independent, so tiling is numerically exact.
                    static const int64_t indexer_qtile = []{
                        const char * e = getenv("DSV4_INDEXER_QTILE"); return e ? (int64_t) atoll(e) : 0;
                    }();
                    ggml_tensor * comp_mask = dsv4_build_indexer_mask_tiled_prefill(ctx0,
                            cur, qr, index_kv,
                            layer.indexer_attn_q_b,
                            layer.indexer_proj,
                            inp_pos,
                            index_mask,
                            hparams.indexer_n_head,
                            hparams.indexer_head_size,
                            n_tokens,
                            n_rot,
                            rope_type,
                            rope_cfg,
                            n_comp,
                            top_k,
                            indexer_qtile);
                    cb(comp_mask, "dsv4_attn_compress_mask", il);

                    attn_mask = ggml_concat(ctx0, raw_mask, comp_mask, 0);
                } else {
                    attn_mask = get_dsv4_inputs()->add_mask(ctx0,
                            dsv4_mask_kind::ATTN_STATIC,
                            n_tokens + n_comp, n_tokens,
                            n_tokens, n_comp, hparams.n_swa, compress_ratio,
                            "dsv4_attn_static_mask");
                }
            } else if (is_prefill) {
                // first chunk shorter than compress_ratio (n_comp == 0): raw window only.
                // The !is_prefill case needs no mask here — the decode path unconditionally
                // takes attn_mask from inp_attn->self_kq_mask_swa below.
                attn_mask = get_dsv4_inputs()->add_mask(ctx0,
                        dsv4_mask_kind::RAW_WINDOW,
                        n_tokens, n_tokens,
                        n_tokens, 0, hparams.n_swa, compress_ratio,
                        "dsv4_attn_raw_window_mask");
            }

            if (!is_prefill) {
                const llama_pos first_pos = ubatch.pos ? ubatch.pos[0] : 0;
                const llama_pos last_pos  = ubatch.pos ? ubatch.pos[n_tokens - 1] : n_tokens - 1;
                // DSV4_NT_PROF: dump per-build (decode/verify) compressed-layer widths so the
                // MTP verify-graph reuse work (#1) can measure the n_tokens distribution and the
                // graph rebuild rate. Off by default; one getenv per process.
                static const bool dsv4_nt_prof = getenv("DSV4_NT_PROF") != nullptr;
                if (dsv4_nt_prof) {
                    fprintf(stderr, "DSV4_NT il=%d n_tokens=%d first_pos=%d last_pos=%d ratio=%u\n",
                            il, (int) n_tokens, (int) first_pos, (int) last_pos, compress_ratio);
                }
                const int64_t n_comp_before  = first_pos / compress_ratio;
                int64_t n_comp_visible = (last_pos + 1) / compress_ratio;
                if (multislot && ubatch.pos) {
                    // sequences sit at independent positions; the shared compressed-cache block width
                    // must cover the furthest-along sequence. Each query is still limited to its OWN
                    // visible count by the block-diagonal mask, so over-wide blocks add only -inf cols.
                    llama_pos max_pos = 0;
                    for (int64_t s = 0; s < n_tokens; ++s) {
                        max_pos = std::max<llama_pos>(max_pos, ubatch.pos[s]);
                    }
                    n_comp_visible = (max_pos + 1) / compress_ratio;
                }
                const int64_t n_comp_cache = mctx_dsv4->get_dsv4_n_comp(il);
                GGML_ASSERT(n_comp_visible <= n_comp_cache);

                // #1 MTP verify-graph reuse: route the K-token verify build through the phase-uniform
                // path (input-driven ivecs, 256-padded view, scratch-routed cache) so the graph is
                // reusable across MTP rounds. Gated by DSV4_VERIFY_REUSE and restricted to the regime
                // where the indexer top-k scores path is unused (n_comp_visible <= indexer_top_k) — the
                // long-context case falls back to the non-reusable chunk path (correct, just rebuilds).
                // n_tokens==1 keeps uniform==true → behaviour byte-identical to the validated decode path.
                // Cap the unrolled uniform path to real MTP verify widths: _chunk_uniform builds K
                // _uniform steps, each pooling UNCONDITIONALLY (more nodes/step than _chunk's boundary-
                // only _projected). Wide non-decode batches (prefill chunks, the n_tokens=512 graph
                // reservation) must stay on _chunk or they overflow the graph context (GGML_ASSERT
                // obj_new). Measured verify widths are {1..4}; 16 leaves headroom and excludes 512.
                static const int64_t DSV4_VERIFY_MAX = 16;
                // ⚠ DO NOT enable DSV4_VERIFY_REUSE in production without a PERPLEXITY gate. The uniform-K
                // verify path is MATHEMATICALLY CORRECT (proven bit-identical to the chunk path under a raw
                // compressed-view) but the 256-padded view it REQUIRES for reuse changes the flash-attn
                // online-softmax fp-accumulation ORDER, which flips a near-tie greedy argmax ~8 tokens in.
                // It is NOT greedy-bit-identical to the default path. Measurement also showed graph reuse is
                // off the critical path (build overlaps async GPU), so enabling this gains ~0 t/s alone — it
                // needs the (abandoned) multi-slot cache to matter. Kept as validated infra, default OFF.
                // See turboquant/DSV4_PERF_STATUS.md #1 + PIVOT.
                static const bool dsv4_verify_reuse = getenv("DSV4_VERIFY_REUSE") != nullptr;
                const bool    uniform  = (n_tokens == 1) || multislot ||
                        (dsv4_verify_reuse && n_tokens > 1 && n_tokens <= DSV4_VERIFY_MAX &&
                         n_comp_visible <= (int64_t) hparams.indexer_top_k);
                const int64_t ivec_len = uniform ? n_tokens : 1;

                // phase-uniform decode (n_tokens == 1): position-derived values come from
                // i32 graph inputs (shared per ratio), the compressed-KV view is padded to
                // 256-row steps, and off-boundary compress results go to the cache scratch
                // row — so the graph topology/properties are token-invariant.
                const std::string rsuf = "_r" + std::to_string(compress_ratio);
                ggml_tensor * in_row_idx    = nullptr;
                ggml_tensor * in_state_perm = nullptr;
                ggml_tensor * in_comp_pos   = nullptr;
                ggml_tensor * in_ape_phase  = nullptr;
                ggml_tensor * in_cache_row  = nullptr;
                if (uniform) {
                    auto * di = get_dsv4_inputs();
                    in_row_idx   = di->add_ivec(ctx0, dsv4_ivec_kind::ROW_IDX,   compress_ratio, ivec_len, 0, ("dsv4_row_idx"   + rsuf).c_str());
                    in_comp_pos  = di->add_ivec(ctx0, dsv4_ivec_kind::COMP_POS,  compress_ratio, ivec_len, 0, ("dsv4_comp_pos"  + rsuf).c_str());
                    in_ape_phase = di->add_ivec(ctx0, dsv4_ivec_kind::APE_PHASE, compress_ratio, ivec_len, 0, ("dsv4_ape_phase" + rsuf).c_str());
                    in_cache_row = di->add_ivec(ctx0, dsv4_ivec_kind::CACHE_ROW, compress_ratio, ivec_len, n_comp_cache, ("dsv4_cache_row" + rsuf).c_str());
                    if (compress_ratio == 4) {
                        in_state_perm = di->add_ivec(ctx0, dsv4_ivec_kind::STATE_PERM, compress_ratio, ivec_len*2*compress_ratio, 0, ("dsv4_state_perm" + rsuf).c_str());
                    }
                }
                const int64_t n_comp_view = uniform
                    ? std::min<int64_t>(n_comp_cache, GGML_PAD(std::max<int64_t>(n_comp_visible, 1), 256))
                    : n_comp_visible;

                if (uniform) {
                    // record the pos-dependent topology drivers so can_reuse() can tell whether a
                    // new ubatch would rebuild this exact graph shape (see dsv4_graph_inputs)
                    get_dsv4_inputs()->add_reuse_key(compress_ratio, hparams.indexer_top_k,
                            n_comp_cache, n_comp_view,
                            n_comp_visible <= hparams.indexer_top_k,
                            n_comp_view > n_comp_visible,
                            n_tokens);
                }

                std::vector<dsv4_state_pair> attn_snaps; // per-token states for rollback planes
                // multi-slot + MTP: n_seqs sequences each verifying K>1 tokens -> chunk the K steps
                // through the batched single-step multislot build. K==1 keeps the plain multislot path.
                const bool chunk_ms = multislot && ubatch.n_seq_tokens > 1;
                // roadmap ③: O(1)-graph batched chunk compressor, replacing the unrolled per-token
                // _chunk recurrence that explodes the graph arena on long multi-turn prefill chunks
                // (the "괭"-after-N-turns crash). It is numerically equal to the unrolled recurrence,
                // so it is the DEFAULT — a correctness fix, not an opt-in. Only applies when the plain
                // (non-multislot, non-uniform, n_tokens>1) _chunk would run AND out_snaps are unused
                // (cparams.n_rs_seq==0, the crashing prefill-chunk case) — MTP rollback (n_rs_seq>0)
                // keeps the unrolled snaps path. DSV4_DISABLE_BATCHED_COMPRESSOR forces the old unrolled
                // path for debugging only.
                static const bool dsv4_batched_comp = getenv("DSV4_DISABLE_BATCHED_COMPRESSOR") == nullptr;
                // Batched path only pays off (and is only worth its edge-case surface) for genuinely
                // large chunks where the unrolled per-token graph-object explosion bites. Small chunks
                // (< 64 tokens) stay on the proven unrolled _chunk: cheap, no explosion, and it already
                // handles every short/edge phase (incl. n_tokens < compress_ratio - r0) directly.
                const bool use_batched_chunk = dsv4_batched_comp && !multislot && !uniform &&
                        n_tokens >= 64 && cparams.n_rs_seq == 0;
                dsv4_decode_compressor dec =
                      chunk_ms ? dsv4_build_compressor_decode_chunk_multislot(ctx0, cur,
                            prev_attn_kv_state,
                            prev_attn_sc_state,
                            layer.attn_compressor_kv,
                            layer.attn_compressor_gate,
                            layer.attn_compressor_ape,
                            layer.attn_compressor_norm,
                            in_row_idx,
                            in_state_perm,
                            in_comp_pos,
                            in_ape_phase,
                            n_embd_head_k,
                            n_rot,
                            compress_ratio,
                            n_seqs,
                            ubatch.n_seq_tokens,
                            rope_type,
                            rope_cfg,
                            norm_rms_eps, &attn_snaps)
                    : multislot ? dsv4_build_compressor_decode_multislot(ctx0, cur,
                            prev_attn_kv_state,
                            prev_attn_sc_state,
                            layer.attn_compressor_kv,
                            layer.attn_compressor_gate,
                            layer.attn_compressor_ape,
                            layer.attn_compressor_norm,
                            in_row_idx,
                            in_state_perm,
                            in_comp_pos,
                            in_ape_phase,
                            n_embd_head_k,
                            n_rot,
                            compress_ratio,
                            n_seqs,
                            rope_type,
                            rope_cfg,
                            norm_rms_eps)
                    : n_tokens == 1 ? dsv4_build_compressor_decode_uniform(ctx0, cur,
                            prev_attn_kv_state,
                            prev_attn_sc_state,
                            layer.attn_compressor_kv,
                            layer.attn_compressor_gate,
                            layer.attn_compressor_ape,
                            layer.attn_compressor_norm,
                            in_row_idx,
                            in_state_perm,
                            in_comp_pos,
                            in_ape_phase,
                            n_embd_head_k,
                            n_rot,
                            compress_ratio,
                            rope_type,
                            rope_cfg,
                            norm_rms_eps)
                    : uniform ? dsv4_build_compressor_decode_chunk_uniform(ctx0, cur,
                            prev_attn_kv_state,
                            prev_attn_sc_state,
                            layer.attn_compressor_kv,
                            layer.attn_compressor_gate,
                            layer.attn_compressor_ape,
                            layer.attn_compressor_norm,
                            in_row_idx,
                            in_state_perm,
                            in_comp_pos,
                            in_ape_phase,
                            n_embd_head_k,
                            n_rot,
                            n_tokens,
                            compress_ratio,
                            rope_type,
                            rope_cfg,
                            norm_rms_eps, &attn_snaps)
                    : use_batched_chunk ? dsv4_build_compressor_decode_chunk_batched(ctx0, cur,
                            prev_attn_kv_state,
                            prev_attn_sc_state,
                            layer.attn_compressor_kv,
                            layer.attn_compressor_gate,
                            layer.attn_compressor_ape,
                            layer.attn_compressor_norm,
                            ubatch,
                            n_embd_head_k,
                            n_rot,
                            n_tokens,
                            compress_ratio,
                            rope_type,
                            rope_cfg,
                            norm_rms_eps)
                    : dsv4_build_compressor_decode_chunk(ctx0, cur,
                            prev_attn_kv_state,
                            prev_attn_sc_state,
                            layer.attn_compressor_kv,
                            layer.attn_compressor_gate,
                            layer.attn_compressor_ape,
                            layer.attn_compressor_norm,
                            ubatch,
                            n_embd_head_k,
                            n_rot,
                            n_tokens,
                            compress_ratio,
                            rope_type,
                            rope_cfg,
                            norm_rms_eps, &attn_snaps);

                if (multislot) {
                    dsv4_store_state_segment_multi(ctx0, gf, dec.kv_state,    inp_rs->mctx->get_r_l(il), state_size, inp_rs->head, 0, n_seqs);
                    dsv4_store_state_segment_multi(ctx0, gf, dec.score_state, inp_rs->mctx->get_s_l(il), state_size, inp_rs->head, 0, n_seqs);
                    if (chunk_ms && cparams.n_rs_seq > 0) {
                        dsv4_store_rollback_planes_multi(ctx0, gf, inp_rs->mctx->get_r_l(il), inp_rs->mctx->get_s_l(il),
                                attn_snaps, state_size, inp_rs->head, 0,
                                inp_rs->mctx->get_size(), cparams.n_rs_seq, ubatch.n_seq_tokens, n_seqs);
                    }
                } else {
                    dsv4_store_state_segment(ctx0, gf, dec.kv_state,    inp_rs->mctx->get_r_l(il), state_size, inp_rs->head, 0);
                    dsv4_store_state_segment(ctx0, gf, dec.score_state, inp_rs->mctx->get_s_l(il), state_size, inp_rs->head, 0);
                    if (cparams.n_rs_seq > 0) {
                        dsv4_store_rollback_planes(ctx0, gf, inp_rs->mctx->get_r_l(il), inp_rs->mctx->get_s_l(il),
                                attn_snaps, state_size, inp_rs->head, 0,
                                inp_rs->mctx->get_size(), cparams.n_rs_seq, n_tokens);
                    }
                }

                if (multislot) {
                    // each (step k, seq s) compressed attn row -> seq s's cache at its own CACHE_ROW.
                    // both kv_comp (plane) and the step-major CACHE_ROW ivec index are k*n_seqs+s.
                    // Off-boundary steps route to the scratch row (K==1 byte-identical to plain multislot).
                    ggml_tensor * kv_comp_q = ggml_dsv4_fp8_kv_quantize(ctx0, dec.kv_comp, n_rot);
                    const int64_t K = ubatch.n_seq_tokens;
                    for (int64_t k = 0; k < K; ++k) {
                        for (int64_t s = 0; s < n_seqs; ++s) {
                            const int64_t      plane  = k*n_seqs + s;
                            const int64_t      cr_idx = k*n_seqs + s;
                            const llama_seq_id sid    = ubatch.seq_id[s*K][0];
                            ggml_tensor * row_s  = ggml_view_1d(ctx0, in_cache_row, 1, cr_idx*ggml_element_size(in_cache_row));
                            ggml_tensor * comp_s = ggml_view_2d(ctx0, kv_comp_q, kv_comp_q->ne[0], 1, kv_comp_q->nb[1], plane*kv_comp_q->nb[2]);
                            dsv4_store_cache_rows_idx(ctx0, gf, mctx_dsv4->get_dsv4_attn_k(ctx0, il, sid), comp_s, row_s);
                        }
                    }
                } else if (uniform) {
                    ggml_tensor * kv_comp_q = ggml_dsv4_fp8_kv_quantize(ctx0, dec.kv_comp, n_rot);
                    store_attn_cache_rows_idx(kv_comp_q, in_cache_row);
                } else if (dec.kv_comp != nullptr) {
                    dec.kv_comp = ggml_dsv4_fp8_kv_quantize(ctx0, dec.kv_comp, n_rot);
                    store_attn_cache_rows(dec.kv_comp, n_comp_before, n_comp_visible - n_comp_before);
                }

                ggml_tensor * k_raw_c = mctx_swa->get_k(ctx0, il);
                ggml_tensor * k_raw = ggml_reshape_3d(ctx0, k_raw_c, n_embd_head_k, 1, k_raw_c->ne[2]);
                k_all = k_raw;
                v_all = k_raw;
                attn_mask = inp_attn->self_kq_mask_swa;

                // [DSV4_KV_ADJACENT] the raw window can be taken from the mirror inside the
                // compressed-cache plane (making k_all a view) ONLY when the SWA cache's live row
                // count covers the WHOLE mirrored region — the mirror is addressed by cell index, so
                // a partial view [0, n_kv) would not sit adjacent to the compressed rows. n_kv is the
                // 256-padded used-cell count and equals the cache size in steady state (the ring
                // wraps within the first ~768 tokens); until then we take the concat path, which is
                // correct and identical to the pre-existing behaviour.
                const bool kv_adjacent = dsv4_kv_adjacent &&
                        k_raw_c->ne[3] == 1 &&
                        k_raw->ne[2] == (int64_t) mctx_dsv4->get_dsv4_n_raw();

                // DSV4 sparse-attention (DSV4_SPARSE_ATTN=1): GATHER only the top-k selected comp
                // rows per query instead of -inf-masking the rest and scanning the full compressed
                // cache. n_raw = dense raw-window row count (gather offset). Captured below in the
                // single-slot topk branch; consumed at build_attn_mha (src[6]).
                static const bool dsv4_sparse_attn = (getenv("DSV4_SPARSE_ATTN") != nullptr);
                sparse_n_raw = k_raw->ne[2];

                if (n_comp_visible > 0 || uniform) {
                    ggml_tensor * comp_mask = nullptr;

                    if (multislot) {
                        // ===== multi-slot: concat each sequence's OWN compressed cache; separate the
                        // concurrent requests with a block-diagonal compress-causal mask. =====
                        ggml_tensor * kv_comp_all = nullptr;
                        for (int64_t s = 0; s < n_seqs; ++s) {
                            const llama_seq_id sid = ubatch.seq_id[s*ubatch.n_seq_tokens][0];
                            ggml_tensor * cache_s = dsv4_cache_view_3d(ctx0, mctx_dsv4->get_dsv4_attn_k(ctx0, il, sid), n_comp_view);
                            kv_comp_all = (s == 0) ? cache_s : ggml_concat(ctx0, kv_comp_all, cache_s, 2);
                        }
                        k_all = ggml_concat(ctx0, k_raw, kv_comp_all, 2); // [hd,1, n_raw + n_seqs*n_comp_view]
                        v_all = k_all;

                        ggml_tensor * block_mask = get_dsv4_inputs()->add_mask(ctx0,
                                dsv4_mask_kind::COMPRESS_CAUSAL_BLOCKDIAG,
                                n_seqs*n_comp_view, n_tokens,
                                0, n_comp_view, 0, compress_ratio,
                                "dsv4_attn_compress_blockdiag_mask");

                        if (compress_ratio == 4) {
                            const dsv4_state_layout index_state_layout = dsv4_make_state_layout(compress_ratio, hparams.indexer_head_size);
                            ggml_tensor * prev_index_kv_state = dsv4_view_state_segment_3d(ctx0, prev_kv_state_all,
                                    attn_state_layout.elems, index_state_layout.width, index_state_layout.rows, n_seqs);
                            ggml_tensor * prev_index_sc_state = dsv4_view_state_segment_3d(ctx0, prev_sc_state_all,
                                    attn_state_layout.elems, index_state_layout.width, index_state_layout.rows, n_seqs);

                            std::vector<dsv4_state_pair> index_snaps; // per-step index states for rollback
                            dsv4_decode_compressor index_dec = chunk_ms
                                ? dsv4_build_compressor_decode_chunk_multislot(ctx0, cur,
                                    prev_index_kv_state, prev_index_sc_state,
                                    layer.indexer_compressor_kv, layer.indexer_compressor_gate,
                                    layer.indexer_compressor_ape, layer.indexer_compressor_norm,
                                    in_row_idx, in_state_perm, in_comp_pos, in_ape_phase,
                                    hparams.indexer_head_size, n_rot, compress_ratio, n_seqs, ubatch.n_seq_tokens,
                                    rope_type, rope_cfg, norm_rms_eps, &index_snaps)
                                : dsv4_build_compressor_decode_multislot(ctx0, cur,
                                    prev_index_kv_state, prev_index_sc_state,
                                    layer.indexer_compressor_kv, layer.indexer_compressor_gate,
                                    layer.indexer_compressor_ape, layer.indexer_compressor_norm,
                                    in_row_idx, in_state_perm, in_comp_pos, in_ape_phase,
                                    hparams.indexer_head_size, n_rot, compress_ratio, n_seqs,
                                    rope_type, rope_cfg, norm_rms_eps);

                            dsv4_store_state_segment_multi(ctx0, gf, index_dec.kv_state,    inp_rs->mctx->get_r_l(il), state_size, inp_rs->head, attn_state_layout.elems, n_seqs);
                            dsv4_store_state_segment_multi(ctx0, gf, index_dec.score_state, inp_rs->mctx->get_s_l(il), state_size, inp_rs->head, attn_state_layout.elems, n_seqs);
                            if (chunk_ms && cparams.n_rs_seq > 0) {
                                dsv4_store_rollback_planes_multi(ctx0, gf, inp_rs->mctx->get_r_l(il), inp_rs->mctx->get_s_l(il),
                                        index_snaps, state_size, inp_rs->head, attn_state_layout.elems,
                                        inp_rs->mctx->get_size(), cparams.n_rs_seq, ubatch.n_seq_tokens, n_seqs);
                            }

                            // each (step k, seq s) compressed index row -> seq s's index cache at its CACHE_ROW
                            // (both kv_comp plane and step-major CACHE_ROW index are k*n_seqs+s; same as attn)
                            ggml_tensor * index_comp_q = index_dec.kv_comp;
                            const int64_t K = ubatch.n_seq_tokens;
                            for (int64_t k = 0; k < K; ++k) {
                                for (int64_t s = 0; s < n_seqs; ++s) {
                                    const int64_t      plane  = k*n_seqs + s;
                                    const int64_t      cr_idx = k*n_seqs + s;
                                    const llama_seq_id sid    = ubatch.seq_id[s*K][0];
                                    ggml_tensor * row_s  = ggml_view_1d(ctx0, in_cache_row, 1, cr_idx*ggml_element_size(in_cache_row));
                                    ggml_tensor * comp_s = ggml_view_2d(ctx0, index_comp_q, index_comp_q->ne[0], 1, index_comp_q->nb[1], plane*index_comp_q->nb[2]);
                                    dsv4_store_cache_rows_idx(ctx0, gf, mctx_dsv4->get_dsv4_index_k(ctx0, il, sid), comp_s, row_s);
                                }
                            }

                            if (n_comp_visible <= (int64_t) hparams.indexer_top_k) {
                                // every sequence within indexer budget: attend all visible comps (no top-k)
                                comp_mask = block_mask;
                            } else {
                                // per-query top-k over its OWN block: force off-block (and pad) to -inf first,
                                // then top-k naturally selects only within the query's sequence block.
                                ggml_tensor * index_cache_all = nullptr;
                                for (int64_t s = 0; s < n_seqs; ++s) {
                                    const llama_seq_id sid = ubatch.seq_id[s*ubatch.n_seq_tokens][0];
                                    ggml_tensor * ic = dsv4_cache_view_3d(ctx0, mctx_dsv4->get_dsv4_index_k(ctx0, il, sid), n_comp_view);
                                    ic = ggml_reshape_2d(ctx0, ic, hparams.indexer_head_size, n_comp_view);
                                    index_cache_all = (s == 0) ? ic : ggml_concat(ctx0, index_cache_all, ic, 1);
                                }
                                ggml_tensor * index_scores = dsv4_build_indexer_scores_decode(ctx0,
                                        cur, qr, index_cache_all,
                                        layer.indexer_attn_q_b, layer.indexer_proj, inp_pos,
                                        hparams.indexer_n_head, hparams.indexer_head_size,
                                        n_seqs*n_comp_view, n_tokens, n_rot, rope_type, rope_cfg);
                                index_scores = ggml_add(ctx0, index_scores, block_mask);
                                const int top_k = std::min<int64_t>(hparams.indexer_top_k, n_comp_view);
                                ggml_tensor * topk = ggml_argsort_top_k(ctx0, index_scores, top_k);
                                comp_mask = dsv4_build_compressed_mask_from_topk(ctx0, index_scores, topk);
                            }
                        } else {
                            comp_mask = block_mask;
                        }
                    } else {
                    if (kv_adjacent) {
                        // one contiguous [raw | comp] region -> a VIEW. No copy, no concat kernel.
                        k_all = mctx_dsv4->get_dsv4_attn_kall(ctx0, il, seq_id, n_comp_view);
                    } else {
                        ggml_tensor * kv_comp_cache = dsv4_cache_view_3d(ctx0, mctx_dsv4->get_dsv4_attn_k(ctx0, il, seq_id), n_comp_view);
                        k_all = ggml_concat(ctx0, k_raw, kv_comp_cache, 2);
                    }
                    v_all = k_all;

                    if (compress_ratio == 4) {
                        const dsv4_state_layout index_state_layout = dsv4_make_state_layout(compress_ratio, hparams.indexer_head_size);
                        ggml_tensor * prev_index_kv_state = dsv4_view_state_segment(ctx0, prev_kv_state_all,
                                attn_state_layout.elems, index_state_layout.width, index_state_layout.rows);
                        ggml_tensor * prev_index_sc_state = dsv4_view_state_segment(ctx0, prev_sc_state_all,
                                attn_state_layout.elems, index_state_layout.width, index_state_layout.rows);

                        std::vector<dsv4_state_pair> index_snaps; // per-token states for rollback planes
                        dsv4_decode_compressor index_dec =
                              n_tokens == 1 ? dsv4_build_compressor_decode_uniform(ctx0, cur,
                                    prev_index_kv_state,
                                    prev_index_sc_state,
                                    layer.indexer_compressor_kv,
                                    layer.indexer_compressor_gate,
                                    layer.indexer_compressor_ape,
                                    layer.indexer_compressor_norm,
                                    in_row_idx,
                                    in_state_perm,
                                    in_comp_pos,
                                    in_ape_phase,
                                    hparams.indexer_head_size,
                                    n_rot,
                                    compress_ratio,
                                    rope_type,
                                    rope_cfg,
                                    norm_rms_eps)
                            : uniform ? dsv4_build_compressor_decode_chunk_uniform(ctx0, cur,
                                    prev_index_kv_state,
                                    prev_index_sc_state,
                                    layer.indexer_compressor_kv,
                                    layer.indexer_compressor_gate,
                                    layer.indexer_compressor_ape,
                                    layer.indexer_compressor_norm,
                                    in_row_idx,
                                    in_state_perm,
                                    in_comp_pos,
                                    in_ape_phase,
                                    hparams.indexer_head_size,
                                    n_rot,
                                    n_tokens,
                                    compress_ratio,
                                    rope_type,
                                    rope_cfg,
                                    norm_rms_eps, &index_snaps)
                            : use_batched_chunk ? dsv4_build_compressor_decode_chunk_batched(ctx0, cur,
                                    prev_index_kv_state,
                                    prev_index_sc_state,
                                    layer.indexer_compressor_kv,
                                    layer.indexer_compressor_gate,
                                    layer.indexer_compressor_ape,
                                    layer.indexer_compressor_norm,
                                    ubatch,
                                    hparams.indexer_head_size,
                                    n_rot,
                                    n_tokens,
                                    compress_ratio,
                                    rope_type,
                                    rope_cfg,
                                    norm_rms_eps)
                            : dsv4_build_compressor_decode_chunk(ctx0, cur,
                                    prev_index_kv_state,
                                    prev_index_sc_state,
                                    layer.indexer_compressor_kv,
                                    layer.indexer_compressor_gate,
                                    layer.indexer_compressor_ape,
                                    layer.indexer_compressor_norm,
                                    ubatch,
                                    hparams.indexer_head_size,
                                    n_rot,
                                    n_tokens,
                                    compress_ratio,
                                    rope_type,
                                    rope_cfg,
                                    norm_rms_eps, &index_snaps);

                        dsv4_store_state_segment(ctx0, gf, index_dec.kv_state,    inp_rs->mctx->get_r_l(il), state_size, inp_rs->head, attn_state_layout.elems);
                        dsv4_store_state_segment(ctx0, gf, index_dec.score_state, inp_rs->mctx->get_s_l(il), state_size, inp_rs->head, attn_state_layout.elems);
                        if (cparams.n_rs_seq > 0) {
                            dsv4_store_rollback_planes(ctx0, gf, inp_rs->mctx->get_r_l(il), inp_rs->mctx->get_s_l(il),
                                    index_snaps, state_size, inp_rs->head, attn_state_layout.elems,
                                    inp_rs->mctx->get_size(), cparams.n_rs_seq, n_tokens);
                        }

                        if (uniform) {
                            store_index_cache_rows_idx(index_dec.kv_comp, in_cache_row);
                        } else if (index_dec.kv_comp != nullptr) {
                            store_index_cache_rows(index_dec.kv_comp, n_comp_before, n_comp_visible - n_comp_before);
                        }

                        // uniform-K verify is admitted only when n_comp_visible <= indexer_top_k, so it
                        // always lands here (the simple causal mask) and never reaches the per-query
                        // indexer-scores top-k path below — which stays n_tokens==1 (decode) / chunk.
                        if (uniform && n_comp_visible <= (int64_t) hparams.indexer_top_k) {
                            comp_mask = get_dsv4_inputs()->add_mask(ctx0,
                                    dsv4_mask_kind::COMPRESS_CAUSAL,
                                    n_comp_view, n_tokens,
                                    0, n_comp_view, 0, compress_ratio,
                                    "dsv4_attn_compress_mask");
                        } else {
                            ggml_tensor * index_cache = dsv4_cache_view_3d(ctx0, mctx_dsv4->get_dsv4_index_k(ctx0, il, seq_id), n_comp_view);
                            index_cache = ggml_reshape_2d(ctx0, index_cache, hparams.indexer_head_size, n_comp_view);
                            const int top_k = std::min<int64_t>(hparams.indexer_top_k, n_comp_view);

                            // [DSV4_INDEXER_QTILE] RESUMED-CHUNK path (n_tokens>1, !is_prefill): this is the
                            // graph the DSV4 worst-case RESERVE builds at reserve_pos0 = max(n_batch, 8*ub)
                            // (llama-context.cpp:632) -> n_comp_view ~ pos0/ratio is LARGE, so the untiled
                            // indexer score [n_comp_view, ub, n_head] (built by _scores_prefill below) is the
                            // ~2 GiB@ub2048 / ~8 GiB@ub4096 ub^2 driver that SIZES the gallocr compute buffer
                            // (6278 MiB / OOM). It is per-query independent exactly like the is_prefill indexer,
                            // so tile the QUERY dim with the SAME builder -> peak O(n_head*n_comp_view*qtile).
                            // Only the dense (non-sparse) mask branch can use the mask-returning tiled builder;
                            // when sparse-attn needs the raw top-k we keep the whole-ub score (sparse mode does
                            // not materialize the [n_comp,ub] mask, but its score is still whole-ub — acceptable
                            // since DSV4_SPARSE_ATTN is off in the EP/fused deploy). qtile<=0 or >=ub = OFF
                            // (whole-ub, byte-identical). Same DSV4_INDEXER_QTILE knob as the is_prefill path.
                            static const int64_t indexer_qtile_resumed = []{
                                const char * e = getenv("DSV4_INDEXER_QTILE"); return e ? (int64_t) atoll(e) : 0;
                            }();
                            const bool tile_resumed = n_tokens > 1 && indexer_qtile_resumed > 0 && indexer_qtile_resumed < n_tokens &&
                                                      !(dsv4_sparse_attn && cparams.flash_attn && n_comp_view > top_k);
                            if (tile_resumed) {
                                ggml_tensor * resumed_causal = get_dsv4_inputs()->add_mask(ctx0,
                                        dsv4_mask_kind::COMPRESS_CAUSAL,
                                        n_comp_visible, n_tokens,
                                        0, n_comp_visible, 0, compress_ratio,
                                        "dsv4_indexer_decode_causal_mask");
                                // The tiled builder slices qr/cur/pos/causal_mask per query tile, runs the
                                // exact _scores_prefill -> argsort_top_k -> mask pipeline per tile, and concats
                                // the [n_comp_view, qtile] masks -> identical to the whole-ub path.
                                comp_mask = dsv4_build_indexer_mask_tiled_prefill(ctx0,
                                        cur, qr,
                                        dsv4_cache_view_3d(ctx0, mctx_dsv4->get_dsv4_index_k(ctx0, il, seq_id), n_comp_view),
                                        layer.indexer_attn_q_b, layer.indexer_proj, inp_pos,
                                        resumed_causal,
                                        hparams.indexer_n_head, hparams.indexer_head_size,
                                        n_tokens, n_rot, rope_type, rope_cfg,
                                        n_comp_view, top_k, indexer_qtile_resumed);
                                cb(comp_mask, "indexer_scores", il);
                            } else {
                            ggml_tensor * index_scores = n_tokens == 1
                                ? dsv4_build_indexer_scores_decode(ctx0,
                                        cur, qr, index_cache,
                                        layer.indexer_attn_q_b,
                                        layer.indexer_proj,
                                        inp_pos,
                                        hparams.indexer_n_head,
                                        hparams.indexer_head_size,
                                        n_comp_view,
                                        n_tokens,
                                        n_rot,
                                        rope_type,
                                        rope_cfg)
                                : dsv4_build_indexer_scores_prefill(ctx0,
                                        cur, qr, dsv4_cache_view_3d(ctx0, mctx_dsv4->get_dsv4_index_k(ctx0, il, seq_id), n_comp_view),
                                        layer.indexer_attn_q_b,
                                        layer.indexer_proj,
                                        inp_pos,
                                        get_dsv4_inputs()->add_mask(ctx0,
                                                dsv4_mask_kind::COMPRESS_CAUSAL,
                                                n_comp_visible, n_tokens,
                                                0, n_comp_visible, 0, compress_ratio,
                                                "dsv4_indexer_decode_causal_mask"),
                                        hparams.indexer_n_head,
                                        hparams.indexer_head_size,
                                        n_tokens,
                                        n_rot,
                                        rope_type,
                                        rope_cfg);
                            cb(index_scores, "indexer_scores", il);

                            if (n_tokens == 1 && n_comp_view > n_comp_visible) {
                                // padded cache rows hold scratch/garbage keys: force their
                                // scores to -inf so top-k can never select them as visible
                                ggml_tensor * index_causal = get_dsv4_inputs()->add_mask(ctx0,
                                        dsv4_mask_kind::COMPRESS_CAUSAL,
                                        n_comp_view, n_tokens,
                                        0, n_comp_view, 0, compress_ratio,
                                        "dsv4_indexer_pad_causal_mask");
                                index_scores = ggml_add(ctx0, index_scores, index_causal);
                            }

                            ggml_tensor * topk = ggml_argsort_top_k(ctx0, index_scores, top_k);
                            cb(topk, "indexer_topk", il);

                            // DSV4 sparse-attn: gather the selected comp rows instead of -inf masking
                            // them. Only when flash-attn is on AND there is something to gain
                            // (n_comp_view > top_k); otherwise fall back to the dense -inf mask.
                            // The comp mask becomes ALL-ZERO (every gathered comp row is valid; the
                            // gather already restricts to the selected set), keeping the K-layout mask
                            // contract intact: raw = causal, comp = 0, pad = -inf. Only the K/V row
                            // ADDRESSING changes in-kernel (gathered vs scanned).
                            if (dsv4_sparse_attn && cparams.flash_attn && n_comp_view > top_k) {
                                sparse_topk = topk;          // consumed at build_attn_mha (src[6]) below
                                comp_mask   = dsv4_new_filled_3d(ctx0, n_comp_view, n_tokens, 1, 0.0f);
                                comp_mask   = ggml_reshape_2d(ctx0, comp_mask, n_comp_view, n_tokens);
                            } else {
                                comp_mask = dsv4_build_compressed_mask_from_topk(ctx0, index_scores, topk);
                            }
                            }
                        }
                    } else {
                        comp_mask = get_dsv4_inputs()->add_mask(ctx0,
                                dsv4_mask_kind::COMPRESS_CAUSAL,
                                n_comp_view, n_tokens,
                                0, n_comp_view, 0, compress_ratio,
                                "dsv4_attn_compress_mask");
                    }
                    } // end single-slot branch (else of multislot)

                    // newer upstream returns the standard kq_mask as F16 when flash-attn is on;
                    // the freshly built comp_mask is F32 — normalize before the concat.
                    // Sparse path: comp_mask is all-zero (gathered comp rows are all valid), so the
                    // concat keeps the K-layout mask contract (raw=causal, comp=0); only the in-kernel
                    // K/V addressing differs.
                    if (comp_mask->type != attn_mask->type) {
                        comp_mask = ggml_cast(ctx0, comp_mask, attn_mask->type);
                    }
                    attn_mask = ggml_concat(ctx0, attn_mask, comp_mask, 0);
                }
            }

            // CUDA FA kernels for D=512 require n_kv % FATTN_KQ_STRIDE (256) == 0 (and 16B-aligned
            // mask rows). DSV4's raw-window + compressed-rows concat yields arbitrary n_kv, which
            // silently kicked the whole attention to the CPU. Pad K/V with zero rows and the mask
            // with -inf columns so flash attention stays on the GPU.
            if (cparams.flash_attn) {
                const int64_t n_kv     = k_all->ne[2];
                const int64_t n_kv_pad = GGML_PAD(n_kv, 256);
                if (n_kv_pad != n_kv) {
                    const int64_t n_extra = n_kv_pad - n_kv;

                    ggml_tensor * k_pad = dsv4_new_filled_3d(ctx0, k_all->ne[0], k_all->ne[1], n_extra, 0.0f);
                    if (k_pad->type != k_all->type) {
                        k_pad = ggml_cast(ctx0, k_pad, k_all->type);
                    }
                    const bool v_is_k = (v_all == k_all);
                    k_all = ggml_concat(ctx0, k_all, k_pad, 2);
                    if (v_is_k) {
                        v_all = k_all;
                    } else {
                        ggml_tensor * v_pad = dsv4_new_filled_3d(ctx0, v_all->ne[0], v_all->ne[1], n_extra, 0.0f);
                        if (v_pad->type != v_all->type) {
                            v_pad = ggml_cast(ctx0, v_pad, v_all->type);
                        }
                        v_all = ggml_concat(ctx0, v_all, v_pad, 2);
                    }

                    ggml_tensor * m_pad = dsv4_new_filled_3d(ctx0, n_extra, attn_mask->ne[1], attn_mask->ne[2], -INFINITY);
                    if (m_pad->type != attn_mask->type) {
                        m_pad = ggml_cast(ctx0, m_pad, attn_mask->type);
                    }
                    attn_mask = ggml_concat(ctx0, attn_mask, m_pad, 0);
                }
            }

            ggml_tensor * attn_mask_cnv = (cparams.flash_attn && attn_mask->type != GGML_TYPE_F16) ? ggml_cast(ctx0, attn_mask, GGML_TYPE_F16) : attn_mask;
            // DSV4 sparse-attn: when sparse_topk is set, pass it (+ raw-window row count) so the
            // flash-attn vec gather computes only the selected comp rows. Null = dense path.
            //
            // [DSV4_ATTN_QTILE] Tile the attention over the QUERY (token) dim to bound the per-ubatch
            // compute buffer to O(qtile) instead of O(ub). The attention is per-query INDEPENDENT: each
            // query token attends the SHARED, query-global K/V (k_all/v_all — already stored to cache
            // above) under its OWN mask column; tokens never interact. So slicing q + the mask's query
            // columns into tiles, running build_attn_mha per tile, and CONCATenating the per-tile output
            // along the token dim is NUMERICALLY IDENTICAL to the whole-ub call (same per-query math).
            //
            // This is the NEXT O(ub^2) lever after the indexer qtile: with non-flash attention the KQ /
            // softmax tensor is [n_kv, ub, n_head] (n_kv ~ 1.25*ub -> O(ub^2), ~1.25 GiB at ub=2048,
            // ~5 GiB at ub=4096 PER live copy); with flash attention the FA kq_mask [n_kv, ub] and the
            // per-query FA work scale with ub. Tiling bounds ALL of these to qtile. K/V are NOT sliced
            // (the cache is query-global) and are reused across tiles, so the only growth is the small
            // per-tile output that we concat. sparse_topk (DSV4_SPARSE_ATTN) is itself per-query indexed,
            // so it is sliced the same way. Default 0 = OFF (whole-ub call, byte-identical to prior path).
            // Same env-knob style as DSV4_INDEXER_QTILE; capture-safe (fixed node count per build given
            // a fixed ub+qtile -> identical graph topology on both SPMD ranks).
            static const int64_t attn_qtile = []{
                const char * e = getenv("DSV4_ATTN_QTILE"); return e ? (int64_t) atoll(e) : 0;
            }();
            if (attn_qtile > 0 && attn_qtile < n_tokens && !multislot) {
                ggml_tensor * out = nullptr;
                for (int64_t t0 = 0; t0 < n_tokens; t0 += attn_qtile) {
                    const int64_t tn = std::min<int64_t>(attn_qtile, n_tokens - t0);
                    // q: [n_embd_head_k, n_head, n_tokens] -> slice the token (last) dim.
                    ggml_tensor * q_t = ggml_view_3d(ctx0, q, q->ne[0], q->ne[1], tn,
                            q->nb[1], q->nb[2], t0 * q->nb[2]);
                    // mask: [n_kv, n_tokens(, 1, 1)] -> slice the query columns (dim 1).
                    ggml_tensor * m_t = ggml_view_2d(ctx0, attn_mask_cnv, attn_mask_cnv->ne[0], tn,
                            attn_mask_cnv->nb[1], t0 * attn_mask_cnv->nb[1]);
                    // sparse_topk (if any) is [*, n_tokens] keyed per query -> slice its query columns.
                    ggml_tensor * st_t = sparse_topk
                        ? ggml_view_2d(ctx0, sparse_topk, sparse_topk->ne[0], tn,
                                       sparse_topk->nb[1], t0 * sparse_topk->nb[1])
                        : nullptr;
                    ggml_tensor * cur_t = build_attn_mha(q_t, k_all, v_all, nullptr, m_t,
                            layer.attn_sinks, nullptr, kq_scale, il, st_t, (int) sparse_n_raw);
                    // cur_t: [n_embd_head_v*n_head, tn] -> concat along the token dim.
                    out = out ? ggml_concat(ctx0, out, cur_t, 1) : cur_t;
                }
                cur = out;
            } else {
                cur = build_attn_mha(q, k_all, v_all, nullptr, attn_mask_cnv, layer.attn_sinks, nullptr, kq_scale, il,
                                     sparse_topk, (int) sparse_n_raw);
            }
            cb(cur, "kqv_out", il);
        }
        cur = ggml_reshape_3d(ctx0, cur, n_embd_head_v, n_head, n_tokens);
        cur = dsv4_apply_rope_tail(ctx0, cur, inp_pos,
                n_embd_head_v, n_head, n_tokens, n_rot, rope_type,
                rope_cfg.n_ctx_orig, rope_cfg.freq_base, rope_cfg.freq_scale,
                rope_cfg.ext_factor, rope_cfg.attn_factor, rope_cfg.beta_fast, rope_cfg.beta_slow, true);
        cur = dsv4_grouped_out(ctx0, cur, layer.attn_wo_a, layer.attn_wo_b,
                n_embd_head_v, n_head, n_out_group, n_lora_o, n_tokens);
        cb(cur, "attn_out", il);
        inpL = dsv4_hc_post(ctx0, cur, residual, mix.post, mix.comb, n_embd, n_hc, n_tokens);
        cb(inpL, "hc_attn_post", il);

        residual = inpL;
        mix = dsv4_hc_pre(ctx0, inpL,
                layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base,
                n_embd, n_hc, n_tokens, norm_rms_eps, hparams.hc_sinkhorn_iters, hparams.hc_eps);
        cur = mix.x;
        cb(cur, "hc_ffn_pre", il);
        cb(mix.mixes, "hc_ffn_pre_mixes", il);
        cb(mix.pre, "hc_ffn_pre_weights", il);
        cb(mix.post, "hc_ffn_pre_post_weights", il);
        cb(mix.comb, "hc_ffn_pre_comb", il);
        cur = build_norm(cur, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);
        ggml_tensor * selected = nullptr;
        // hash routing is a function of the token id — embeddings-only batches (ubatch.token ==
        // nullptr) leave inp_tokens unset, so fall back to gate routing like the warmup case
        if ((uint32_t) il < hparams.n_hash_layers && !cparams.warmup && ubatch.token != nullptr) {
            selected = ggml_get_rows(ctx0, layer.ffn_gate_tid2eid, inp_tokens);
            cb(selected, "ffn_moe_hash_topk", il);
        }

        ggml_tensor * moe_out = build_moe_ffn(cur,
                layer.ffn_gate_inp,
                layer.ffn_up_exps,
                layer.ffn_gate_exps,
                layer.ffn_down_exps,
                layer.ffn_exp_probs_b,
                n_expert, n_expert_used,
                LLM_FFN_SILU, hparams.expert_weights_norm,
                hparams.expert_weights_scale,
                (llama_expert_gating_func_type) hparams.expert_gating_func,
                il,
                nullptr,
                nullptr,
                nullptr,
                nullptr,
                nullptr,
                selected);
        cb(moe_out, "ffn_moe_out", il);
        ggml_tensor * ffn_shexp = build_ffn(cur,
                layer.ffn_up_shexp,   nullptr, nullptr,
                layer.ffn_gate_shexp, nullptr, nullptr,
                layer.ffn_down_shexp, nullptr, nullptr,
                nullptr,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(ffn_shexp, "ffn_shexp", il);

        cur = ggml_add(ctx0, moe_out, ffn_shexp);
        cb(cur, "ffn_out", il);
        inpL = dsv4_hc_post(ctx0, cur, residual, mix.post, mix.comb, n_embd, n_hc, n_tokens);
        cb(inpL, "hc_ffn_post", il);

        if (dflash_layers.count(il)) {
            // DFlash capture (PR#22105 style): NO extra graph ops. Just mark the existing hc_ffn_post
            // tensor as a graph output and stash its pointer. The n_hc SUM-collapse + 5-layer stack is
            // done on the HOST after decode (llama-context.cpp). This keeps the target graph identical
            // to plain inference — no per-token kernel launches, no graph-reuse breakage.
            ggml_set_output(inpL);
            res->t_dflash_layers.push_back(inpL);
        }
    }
    if (cparams.embeddings_pre_norm && !cparams.embeddings_pre_norm_masked) {
        // expose the full hyper-connection state to the MTP draft head —
        // one row of n_hc*n_embd per token (unmasked: every batch position).
        // the view is a graph leaf — expand it so the scheduler allocates it
        ggml_tensor * h_flat = ggml_reshape_2d(ctx0, inpL, n_embd * n_hc, n_tokens);
        cb(h_flat, "h_pre_norm", -1);
        res->t_h_pre_norm = h_flat;
        ggml_build_forward_expand(gf, h_flat);
    }

    if (inp_out_ids) {
        inpL = ggml_reshape_2d(ctx0, inpL, n_embd * n_hc, n_tokens);
        inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
        inpL = ggml_reshape_3d(ctx0, inpL, n_embd, n_hc, n_outputs);
    }

    if (cparams.embeddings_pre_norm && cparams.embeddings_pre_norm_masked) {
        ggml_tensor * h_flat = ggml_reshape_2d(ctx0, inpL, n_embd * n_hc, inp_out_ids ? n_outputs : n_tokens);
        cb(h_flat, "h_pre_norm", -1);
        res->t_h_pre_norm = h_flat;
        ggml_build_forward_expand(gf, h_flat);
    }

    ggml_tensor * cur = dsv4_hc_head(ctx0, inpL,
            model.output_hc_fn, model.output_hc_scale, model.output_hc_base,
            n_embd, n_hc, inp_out_ids ? n_outputs : n_tokens,
            norm_rms_eps, hparams.hc_eps);
    cb(cur, "result_hc", -1);

    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = ggml_mul_mat(ctx0, model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;
    ggml_build_forward_expand(gf, cur);

    // [DSV4_MTP_FOLD] fold the NextN draft head into the trunk verify graph: append
    // the MTP head nodes here, reading the target's on-device hyper-connection state
    // (res->t_h_pre_norm, unmasked = [hc_dim, n_tokens]) instead of a separate ctx_dft
    // decode + D2H handoff. Emits res->t_mtp_logits; the trunk output above is untouched.
    // Increment 1: prove the attach is non-perturbing (trunk output identical) with a
    // simple unshifted tok_embd = emb(inp_tokens); the exact (h_p, x_{p+1}) token shift
    // + sampled-token feedback lands in the next increment. Gated OFF by default.
    static const bool dsv4_mtp_fold = getenv("DSV4_MTP_FOLD") != nullptr;
    // Only fold when the trunk actually exposed its unmasked pre-norm hidden state.
    // llama_context init forces embeddings_pre_norm on for the (non-MTP) trunk when
    // DSV4_MTP_FOLD is set, so this holds at both graph_reserve and decode; guard
    // gracefully (skip, don't abort) in case a caller builds the trunk without it.
    if (dsv4_mtp_fold && res->t_h_pre_norm && !cparams.embeddings_pre_norm_masked) {
        // draft-token embeddings for the folded head's e_proj branch (increment-1: unshifted)
        ggml_tensor * mtp_tok_embd = ggml_get_rows(ctx0, model.tok_embd, inp_tokens);
        cb(mtp_tok_embd, "mtp_fold_tok_embd", -1);
        build_mtp_head(model, res->t_h_pre_norm, mtp_tok_embd,
                       /*folded*/ true, inp_attn, inp_pos, inp_out_ids);
        if (res->t_mtp_logits) {
            ggml_build_forward_expand(gf, res->t_mtp_logits);
        }
    }
}

// LLM_GRAPH_TYPE_DECODER_MTP draft head for DeepSeek-V4 (ds4 numerics):
// input_hc = repeat(e_proj(enorm(emb)), n_hc) + h_proj(hnorm(prev_hc rows)),
// one plain-SWA MLA+MoE decoder layer (the NextN block), then the MTP head's
// own hyper-connection collapse in front of the shared output head.
llama_model_deepseek4::graph_mtp::graph_mtp(const llama_model & model, const llm_graph_params & params)
    : dsv4_graph_base(params) {
    GGML_ASSERT(hparams.nextn_predict_layers == 1 && "DSV4 MTP supports exactly one NextN layer");

    const int il = (int) hparams.n_layer - 1;
    const auto & layer = model.layers[il];

    if (!layer.nextn.e_proj || !layer.nextn.h_proj || !layer.nextn.hc_head_fn) {
        throw std::runtime_error("DeepSeek V4 MTP tensors missing from the GGUF "
                "(create the MTP shard with turboquant/ds4_mtp_to_shard.py)");
    }

    const int64_t n_hc   = hparams.n_hc;
    const int64_t hc_dim = n_hc * n_embd;

    auto inp = std::make_unique<llm_graph_input_embd_h>(hc_dim);

    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->tokens);

    inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hc_dim, n_tokens);
    ggml_set_input(inp->embd);

    inp->h = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hc_dim, n_tokens);
    ggml_set_input(inp->h);
    ggml_set_name(inp->h, "mtp_h_input");

    GGML_ASSERT(ubatch.token && "DSV4 MTP draft batch must carry tokens");
    ggml_tensor * tok_embd = ggml_get_rows(ctx0, model.tok_embd, inp->tokens);
    cb(tok_embd, "mtp_tok_embd", il);

    ggml_tensor * h_in = inp->h;
    res->add_input(std::move(inp));

    build_mtp_head(model, h_in, tok_embd);
}

// NextN MTP draft head, shared by graph_mtp (standalone draft graph) and the folded
// head inside the trunk graph (DSV4_MTP_FOLD). Reads h_in (target post-layer
// hyper-connection state, hc_dim x n_tokens) + tok_embd (draft token embeddings,
// n_embd x n_tokens). Builds one plain-SWA MLA+MoE decoder layer + the MTP head's
// hyper-connection collapse + shared output head; emits res->t_logits / t_embd /
// t_h_pre_norm. Body is behavior-identical to the pre-refactor graph_mtp inline code.
void llama_model_deepseek4::dsv4_graph_base::build_mtp_head(
        const llama_model & model, ggml_tensor * h_in, ggml_tensor * tok_embd,
        bool folded,
        llm_graph_input_attn_kv_iswa * inp_attn_ext,
        ggml_tensor * inp_pos_ext,
        ggml_tensor * inp_out_ids_ext) {
    const int il = (int) hparams.n_layer - 1;
    const auto & layer = model.layers[il];

    const int64_t n_hc        = hparams.n_hc;
    const int64_t hc_dim      = n_hc * n_embd;
    const int64_t n_out_group = hparams.n_attn_out_groups;
    const int64_t n_lora_o    = hparams.n_lora_o;

    const float kq_scale = 1.0f / std::sqrt(float(n_embd_head_k));
    const dsv4_rope_cfg rope_cfg = dsv4_make_rope_cfg(hparams, cparams, 0);

    // Folded: reuse the trunk's position / out_ids / hybrid-SWA attn inputs so the
    // NextN head rides the trunk verify decode (no separate ctx_dft, no new inputs).
    // Standalone (graph_mtp): build our own, as before.
    ggml_tensor * inp_pos     = folded ? inp_pos_ext     : build_inp_pos();
    ggml_tensor * inp_out_ids = folded ? inp_out_ids_ext : build_inp_out_ids();

    llm_graph_input_attn_kv_iswa * inp_attn = folded ? inp_attn_ext : build_attn_inp_kv_iswa();
    GGML_ASSERT(inp_attn && "MTP head needs an SWA attention input");

    // e_proj(enorm(emb)) broadcast over the n_hc rows
    ggml_tensor * e = build_norm(tok_embd, layer.nextn.enorm, nullptr, LLM_NORM_RMS, il);
    cb(e, "mtp_enorm", il);
    e = ggml_mul_mat(ctx0, layer.nextn.e_proj, e);
    cb(e, "mtp_e_proj", il);
    e = ggml_reshape_3d(ctx0, e, n_embd, 1, n_tokens);
    e = ggml_repeat_4d(ctx0, e, n_embd, n_hc, n_tokens, 1);
    e = ggml_reshape_3d(ctx0, e, n_embd, n_hc, n_tokens);

    // h_proj(hnorm(.)) applied per hyper-connection row of the target's state
    ggml_tensor * h = ggml_reshape_2d(ctx0, h_in, n_embd, n_hc * n_tokens);
    h = build_norm(h, layer.nextn.hnorm, nullptr, LLM_NORM_RMS, il);
    cb(h, "mtp_hnorm", il);
    h = ggml_mul_mat(ctx0, layer.nextn.h_proj, h);
    cb(h, "mtp_h_proj", il);
    h = ggml_reshape_3d(ctx0, h, n_embd, n_hc, n_tokens);

    ggml_tensor * inpL = ggml_add(ctx0, e, h);
    cb(inpL, "mtp_input_hc", il);

    // one plain-SWA decoder layer (= main-graph layer body with compress_ratio == 0)
    ggml_tensor * residual = inpL;
    dsv4_hc_mix mix = dsv4_hc_pre(ctx0, inpL,
            layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base,
            n_embd, n_hc, n_tokens, norm_rms_eps, hparams.hc_sinkhorn_iters, hparams.hc_eps);
    ggml_tensor * cur = mix.x;
    cb(cur, "mtp_hc_attn_pre", il);
    cur = build_norm(cur, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
    cb(cur, "mtp_attn_norm", il);

    ggml_tensor * qr = ggml_mul_mat(ctx0, layer.wq_a, cur);
    qr = build_norm(qr, layer.attn_q_a_norm, nullptr, LLM_NORM_RMS, il);

    ggml_tensor * q = ggml_mul_mat(ctx0, layer.wq_b, qr);
    q = ggml_reshape_3d(ctx0, q, n_embd_head_k, n_head, n_tokens);
    q = ggml_rms_norm(ctx0, q, norm_rms_eps);
    q = dsv4_apply_rope_tail(ctx0, q, inp_pos,
            n_embd_head_k, n_head, n_tokens, n_rot, rope_type,
            rope_cfg.n_ctx_orig, rope_cfg.freq_base, rope_cfg.freq_scale,
            rope_cfg.ext_factor, rope_cfg.attn_factor, rope_cfg.beta_fast, rope_cfg.beta_slow, false);
    cb(q, "mtp_Qcur", il);

    ggml_tensor * kv = ggml_mul_mat(ctx0, layer.attn_kv, cur);
    kv = build_norm(kv, layer.attn_kv_a_norm, nullptr, LLM_NORM_RMS, il);
    kv = ggml_reshape_3d(ctx0, kv, n_embd_head_k, 1, n_tokens);
    kv = dsv4_apply_rope_tail(ctx0, kv, inp_pos,
            n_embd_head_k, 1, n_tokens, n_rot, rope_type,
            rope_cfg.n_ctx_orig, rope_cfg.freq_base, rope_cfg.freq_scale,
            rope_cfg.ext_factor, rope_cfg.attn_factor, rope_cfg.beta_fast, rope_cfg.beta_slow, false);
    kv = ggml_dsv4_fp8_kv_quantize(ctx0, kv, n_rot);
    cb(kv, "mtp_KVcur", il);

    const auto * mctx_swa = inp_attn->mctx->get_swa();
    ggml_build_forward_expand(gf, q);
    ggml_build_forward_expand(gf, kv);
    ggml_build_forward_expand(gf, mctx_swa->cpy_k(ctx0, kv, inp_attn->get_k_idxs_swa(), il));

    ggml_tensor * k_cache = mctx_swa->get_k(ctx0, il);
    k_cache = ggml_reshape_3d(ctx0, k_cache, n_embd_head_k, 1, k_cache->ne[2]);
    cur = build_attn_mha(q, k_cache, k_cache, nullptr, inp_attn->get_kq_mask_swa(),
            layer.attn_sinks, nullptr, kq_scale, il);
    cb(cur, "mtp_kqv_out", il);

    cur = ggml_reshape_3d(ctx0, cur, n_embd_head_v, n_head, n_tokens);
    cur = dsv4_apply_rope_tail(ctx0, cur, inp_pos,
            n_embd_head_v, n_head, n_tokens, n_rot, rope_type,
            rope_cfg.n_ctx_orig, rope_cfg.freq_base, rope_cfg.freq_scale,
            rope_cfg.ext_factor, rope_cfg.attn_factor, rope_cfg.beta_fast, rope_cfg.beta_slow, true);
    cur = dsv4_grouped_out(ctx0, cur, layer.attn_wo_a, layer.attn_wo_b,
            n_embd_head_v, n_head, n_out_group, n_lora_o, n_tokens);
    cb(cur, "mtp_attn_out", il);
    inpL = dsv4_hc_post(ctx0, cur, residual, mix.post, mix.comb, n_embd, n_hc, n_tokens);
    cb(inpL, "mtp_hc_attn_post", il);

    residual = inpL;
    mix = dsv4_hc_pre(ctx0, inpL,
            layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base,
            n_embd, n_hc, n_tokens, norm_rms_eps, hparams.hc_sinkhorn_iters, hparams.hc_eps);
    cur = mix.x;
    cb(cur, "mtp_hc_ffn_pre", il);
    cur = build_norm(cur, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);
    cb(cur, "mtp_ffn_norm", il);

    ggml_tensor * moe_out = build_moe_ffn(cur,
            layer.ffn_gate_inp,
            layer.ffn_up_exps,
            layer.ffn_gate_exps,
            layer.ffn_down_exps,
            layer.ffn_exp_probs_b,
            n_expert, n_expert_used,
            LLM_FFN_SILU, hparams.expert_weights_norm,
            hparams.expert_weights_scale,
            (llama_expert_gating_func_type) hparams.expert_gating_func,
            il,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr);
    cb(moe_out, "mtp_ffn_moe_out", il);
    ggml_tensor * ffn_shexp = build_ffn(cur,
            layer.ffn_up_shexp,   nullptr, nullptr,
            layer.ffn_gate_shexp, nullptr, nullptr,
            layer.ffn_down_shexp, nullptr, nullptr,
            nullptr,
            LLM_FFN_SILU, LLM_FFN_PAR, il);
    cb(ffn_shexp, "mtp_ffn_shexp", il);

    cur = ggml_add(ctx0, moe_out, ffn_shexp);
    inpL = dsv4_hc_post(ctx0, cur, residual, mix.post, mix.comb, n_embd, n_hc, n_tokens);
    cb(inpL, "mtp_hc_ffn_post", il);

    // expose the post-layer hyper-connection state so the AR draft loop can
    // seed the next MTP step (same slot as the trunk graph's t_h_pre_norm).
    // Folded: do NOT touch res->t_h_pre_norm — that slot holds the TRUNK's h_flat
    // which is this head's INPUT; overwriting it would corrupt the trunk output.
    if (!folded) {
        ggml_tensor * h_flat = ggml_reshape_2d(ctx0, inpL, hc_dim, n_tokens);
        cb(h_flat, "h_pre_norm", -1);
        res->t_h_pre_norm = h_flat;
        ggml_build_forward_expand(gf, h_flat);
    }

    if (inp_out_ids) {
        inpL = ggml_reshape_2d(ctx0, inpL, hc_dim, n_tokens);
        inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
        inpL = ggml_reshape_3d(ctx0, inpL, n_embd, n_hc, n_outputs);
    }

    cur = dsv4_hc_head(ctx0, inpL,
            layer.nextn.hc_head_fn, layer.nextn.hc_head_scale, layer.nextn.hc_head_base,
            n_embd, n_hc, inp_out_ids ? n_outputs : n_tokens,
            norm_rms_eps, hparams.hc_eps);
    cb(cur, "mtp_result_hc", -1);

    ggml_tensor * head_norm_w = layer.nextn.shared_head_norm
            ? layer.nextn.shared_head_norm
            : model.output_norm;
    cur = build_norm(cur, head_norm_w, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "mtp_result_norm", -1);
    if (!folded) {
        res->t_embd = cur;
    }

    cur = ggml_mul_mat(ctx0, model.output, cur);
    cb(cur, "mtp_result_output", -1);
    // Folded: emit into the dedicated MTP slot so the trunk's t_logits (already set
    // by the trunk output head) stays intact; the server reads t_mtp_logits to draft.
    if (folded) {
        res->t_mtp_logits = cur;
    } else {
        res->t_logits = cur;
    }
    ggml_build_forward_expand(gf, cur);
}

void llama_model_deepseek4::load_arch_hparams(llama_model_loader & ml) {
    // [dsv4-fp4] The nsparks DeepSeek-V4 native F8_E4M3/MXFP4 gguf (ftype 41) is the SAME
    // architecture as our converter's output but omits several metadata keys (and renames ~21
    // tensors, handled in the loader). These are DeepSeek-V4-Flash architectural constants; supply
    // them as defaults when the key is absent so the proven graph loads unchanged. Gated on the F8
    // ftype => our own gguf files (which always carry these keys, required=true) are untouched.
    const bool nsparks_f8 = (ml.ftype == LLAMA_FTYPE_MOSTLY_F8_E4M3_MXFP4);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_ATTENTION_Q_LORA_RANK,       hparams.n_lora_q);
    if (!ml.get_key(LLM_KV_ATTENTION_OUTPUT_LORA_RANK,   hparams.n_lora_o,           !nsparks_f8)) { hparams.n_lora_o = 1024; }
    if (!ml.get_key(LLM_KV_ATTENTION_OUTPUT_GROUP_COUNT, hparams.n_attn_out_groups,  !nsparks_f8)) { hparams.n_attn_out_groups = 8; }
    ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,  hparams.n_ff_exp);
    ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,         hparams.n_expert_shared);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,        hparams.expert_weights_scale, false);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,         hparams.expert_weights_norm, false);
    ml.get_key(LLM_KV_EXPERT_GATING_FUNC,          hparams.expert_gating_func, false);
    if (hparams.expert_gating_func == LLAMA_EXPERT_GATING_FUNC_TYPE_NONE) {
        hparams.expert_gating_func = LLAMA_EXPERT_GATING_FUNC_TYPE_SQRTSOFTPLUS;
    }

    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW,          hparams.n_swa, false);
    if (hparams.n_swa > 0) {
        hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
        hparams.set_swa_pattern(0, false);
        hparams.rope_freq_base_train_swa  = hparams.rope_freq_base_train;
        hparams.rope_freq_scale_train_swa = hparams.rope_freq_scale_train;
    }
    ml.get_key(LLM_KV_ATTENTION_COMPRESS_ROPE_FREQ_BASE, hparams.compress_rope_freq_base, false);
    if (nsparks_f8 && hparams.compress_rope_freq_base == 0.0f) { hparams.compress_rope_freq_base = 160000.0f; }
    ml.get_key(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT,      hparams.indexer_n_head, false);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH,      hparams.indexer_head_size, false);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_TOP_K,           hparams.indexer_top_k, false);
    if (!ml.get_key(LLM_KV_HASH_LAYER_COUNT,             hparams.n_hash_layers,      !nsparks_f8)) { hparams.n_hash_layers = 3; }
    ml.get_key(LLM_KV_NEXTN_PREDICT_LAYERS,              hparams.nextn_predict_layers, false);
    if (!ml.get_key(LLM_KV_HYPER_CONNECTION_COUNT,          hparams.n_hc,             !nsparks_f8)) { hparams.n_hc = 4; }
    if (!ml.get_key(LLM_KV_HYPER_CONNECTION_SINKHORN_ITERS, hparams.hc_sinkhorn_iters,!nsparks_f8)) { hparams.hc_sinkhorn_iters = 20; }
    if (!ml.get_key(LLM_KV_HYPER_CONNECTION_EPS,            hparams.hc_eps,           !nsparks_f8)) { hparams.hc_eps = 1e-6f; }
    ml.get_key_or_arr(LLM_KV_SWIGLU_CLAMP_EXP,           hparams.swiglu_clamp_exp, hparams.n_layer, false);

    std::vector<uint32_t> compress_ratios;
    if (!ml.get_arr(LLM_KV_ATTENTION_COMPRESS_RATIOS, compress_ratios, !nsparks_f8) && nsparks_f8) {
        // DeepSeek-V4-Flash per-layer compress ratio (43 main layers): layers 0-1 dense (0),
        // then alternating 4 / 128 (indexer vs sliding-window compressors). Identical to our
        // converter's array for the same base model; the nsparks gguf simply omits the key.
        compress_ratios = {0,0,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,
                           4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4};
    }
    if (compress_ratios.size() < hparams.n_layer) {
        throw std::runtime_error(format("DeepSeek V4 compress ratio count mismatch: got %zu, expected %u",
                    compress_ratios.size(), hparams.n_layer));
    }
    std::copy_n(compress_ratios.begin(), hparams.n_layer, hparams.attn_compress_ratio.begin());

    for (uint32_t il = 0; il < hparams.n_layer; ++il) {
        const uint32_t ratio = hparams.attn_compress_ratio[il];
        if (ratio == 0) {
            continue;
        }

        const uint32_t coff = ratio == 4 ? 2 : 1;
        uint32_t state_size = coff * ratio * coff * hparams.n_embd_head_k(il);
        if (ratio == 4) {
            state_size += coff * ratio * coff * hparams.indexer_head_size;
        }
        hparams.dsv4_state_size = std::max(hparams.dsv4_state_size, state_size);
    }

    if (hparams.nextn_predict_layers > 0) {
        // Include the MTP/NextN draft layer(s) in n_layer (qwen35 convention) so
        // device mapping and model.layers[] cover them. The main context excludes
        // them via the create_memory layer filters; per-layer hparams mirror the
        // last main layer (plain SWA attention, no compressor, no indexer).
        // The MTP head consumes the full hyper-connection state as its hidden input.
        hparams.n_embd_h_mtp = hparams.n_hc * hparams.n_embd;
        const uint32_t n_main = hparams.n_layer;
        for (uint32_t k = 0; k < hparams.nextn_predict_layers; ++k) {
            const uint32_t il = n_main + k;
            hparams.attn_compress_ratio[il] = 0;
            hparams.swa_layers[il]          = 1;
            hparams.n_head_arr[il]          = hparams.n_head_arr[n_main - 1];
            hparams.n_head_kv_arr[il]       = hparams.n_head_kv_arr[n_main - 1];
            hparams.n_ff_arr[il]            = hparams.n_ff_arr[n_main - 1];
            hparams.swiglu_clamp_exp[il]    = hparams.swiglu_clamp_exp[n_main - 1];
        }
        hparams.n_layer += hparams.nextn_predict_layers;
    }

    type = LLM_TYPE_UNKNOWN;

    LLAMA_LOG_INFO("%s: n_lora_q              = %d\n",     __func__, hparams.n_lora_q);
    LLAMA_LOG_INFO("%s: n_lora_o              = %d\n",     __func__, hparams.n_lora_o);
    LLAMA_LOG_INFO("%s: n_attn_out_groups     = %d\n",     __func__, hparams.n_attn_out_groups);
    LLAMA_LOG_INFO("%s: n_ff_exp              = %d\n",     __func__, hparams.n_ff_exp);
    LLAMA_LOG_INFO("%s: n_expert_shared       = %d\n",     __func__, hparams.n_expert_shared);
    LLAMA_LOG_INFO("%s: n_swa                 = %d\n",     __func__, hparams.n_swa);
    LLAMA_LOG_INFO("%s: compress_rope_freq_base = %.1f\n", __func__, hparams.compress_rope_freq_base);
    LLAMA_LOG_INFO("%s: indexer_n_head        = %d\n",     __func__, hparams.indexer_n_head);
    LLAMA_LOG_INFO("%s: indexer_head_size     = %d\n",     __func__, hparams.indexer_head_size);
    LLAMA_LOG_INFO("%s: indexer_top_k         = %d\n",     __func__, hparams.indexer_top_k);
    LLAMA_LOG_INFO("%s: n_hash_layers         = %d\n",     __func__, hparams.n_hash_layers);
    LLAMA_LOG_INFO("%s: n_hc                  = %d\n",     __func__, hparams.n_hc);
    LLAMA_LOG_INFO("%s: hc_sinkhorn_iters     = %d\n",     __func__, hparams.hc_sinkhorn_iters);
    LLAMA_LOG_INFO("%s: hc_eps                = %.1e\n",   __func__, hparams.hc_eps);
    LLAMA_LOG_INFO("%s: nextn_predict_layers  = %d\n",     __func__, hparams.nextn_predict_layers);
    LLAMA_LOG_INFO("%s: expert_weights_scale  = %.1f\n",   __func__, hparams.expert_weights_scale);
    LLAMA_LOG_INFO("%s: expert_weights_norm   = %d\n",     __func__, hparams.expert_weights_norm);
}

void llama_model_deepseek4::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    const int64_t q_lora_rank       = hparams.n_lora_q;
    const int64_t o_lora_rank       = hparams.n_lora_o;
    const int64_t n_out_groups      = hparams.n_attn_out_groups;
    const int64_t n_ff_exp          = hparams.n_ff_exp;
    const int64_t n_expert_shared   = hparams.n_expert_shared;
    const int64_t n_hc              = hparams.n_hc;
    const int64_t hc_dim            = n_hc * n_embd;
    const int64_t hc_mix            = (2 + n_hc) * n_hc;

    // [DSV4_MOE_SIDECAR] When the NVFP4 grouped-GEMM MoE adapter loads its experts from a pre-converted
    // per-rank sidecar (DSV4_MOE_SIDECAR set), the routed-expert MXFP4 weights are NEVER referenced by
    // the graph and must NOT be device-resident (holding MXFP4 next to the NVFP4 registry is the ~2x
    // OOM). Mark them TENSOR_SKIP so the loader neither creates nor loads them (ffn_*_exps stay null);
    // the NVFP4 registry is filled post-load from the sidecar.
    //
    // PER-LAYER: TENSOR_SKIP applies ONLY to layers present in this rank's sidecar (the MXFP4 MoE
    // layers actually pre-converted). The MTP/nextn draft layer has Q4_K experts that are NOT in the
    // sidecar -> it must load its experts NORMALLY and run on the standard mul_mat_id path. We read the
    // sidecar header's layer table here (tiny, no blobs) to learn which il's are present. Env off ->
    // dsv4_sidecar_layers empty -> exps_flags_for(i) == 0 for all layers (default path, byte-identical).
    if (const char * sc_dir = getenv("DSV4_MOE_SIDECAR")) {
        const int rank = ggml_backend_meta_tp_rank_public();
        if (!dsv4_sidecar_read_layer_set(sc_dir, rank, dsv4_sidecar_layers)) {
            throw std::runtime_error("DSV4_MOE_SIDECAR set but sidecar_rank" + std::to_string(rank) +
                ".bin is missing/invalid in " + sc_dir);
        }
    }
    auto exps_flags_for = [&](int il) -> int {
        return dsv4_sidecar_layers.count(il) ? llama_model_loader::TENSOR_SKIP : 0;
    };

    if (n_out_groups == 0) {
        throw std::runtime_error("DeepSeek V4 requires attention output groups");
    }

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    output_norm     = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM,     "weight"), {n_embd}, 0);
    output          = create_tensor(tn(LLM_TENSOR_OUTPUT,          "weight"), {n_embd, n_vocab}, 0);
    output_hc_base  = create_tensor(tn(LLM_TENSOR_OUTPUT_HC_BASE,  "weight"), {n_hc}, 0);
    output_hc_fn    = create_tensor(tn(LLM_TENSOR_OUTPUT_HC_FN,    "weight"), {hc_dim, n_hc}, 0);
    output_hc_scale = create_tensor(tn(LLM_TENSOR_OUTPUT_HC_SCALE, "weight"), {1}, 0);

    auto create_deepseek4_compressor = [&](llama_layer & layer, int bid, int64_t compress_ratio, int64_t head_size, bool indexer) {
        const int64_t coff = compress_ratio == 4 ? 2 : 1;
        ggml_tensor *& ape  = indexer ? layer.indexer_compressor_ape  : layer.attn_compressor_ape;
        ggml_tensor *& kv   = indexer ? layer.indexer_compressor_kv   : layer.attn_compressor_kv;
        ggml_tensor *& gate = indexer ? layer.indexer_compressor_gate : layer.attn_compressor_gate;
        ggml_tensor *& norm = indexer ? layer.indexer_compressor_norm : layer.attn_compressor_norm;

        ape  = create_tensor(tn(indexer ? LLM_TENSOR_INDEXER_COMPRESSOR_APE  : LLM_TENSOR_ATTN_COMPRESSOR_APE,  "weight", bid), {coff * head_size, compress_ratio}, 0);
        kv   = create_tensor(tn(indexer ? LLM_TENSOR_INDEXER_COMPRESSOR_KV   : LLM_TENSOR_ATTN_COMPRESSOR_KV,   "weight", bid), {n_embd, coff * head_size}, 0);
        gate = create_tensor(tn(indexer ? LLM_TENSOR_INDEXER_COMPRESSOR_GATE : LLM_TENSOR_ATTN_COMPRESSOR_GATE, "weight", bid), {n_embd, coff * head_size}, 0);
        norm = create_tensor(tn(indexer ? LLM_TENSOR_INDEXER_COMPRESSOR_NORM : LLM_TENSOR_ATTN_COMPRESSOR_NORM, "weight", bid), {head_size}, 0);
    };

    const int n_main_layers = n_layer - (int) hparams.nextn_predict_layers;

    for (int i = 0; i < n_main_layers; ++i) {
        auto & layer = layers[i];

        const int64_t compress_ratio = hparams.attn_compress_ratio[i];

        layer.hc_attn_base  = create_tensor(tn(LLM_TENSOR_HC_ATTN_BASE,  "weight", i), {hc_mix}, 0);
        layer.hc_attn_fn    = create_tensor(tn(LLM_TENSOR_HC_ATTN_FN,    "weight", i), {hc_dim, hc_mix}, 0);
        layer.hc_attn_scale = create_tensor(tn(LLM_TENSOR_HC_ATTN_SCALE, "weight", i), {3}, 0);
        layer.hc_ffn_base   = create_tensor(tn(LLM_TENSOR_HC_FFN_BASE,   "weight", i), {hc_mix}, 0);
        layer.hc_ffn_fn     = create_tensor(tn(LLM_TENSOR_HC_FFN_FN,     "weight", i), {hc_dim, hc_mix}, 0);
        layer.hc_ffn_scale  = create_tensor(tn(LLM_TENSOR_HC_FFN_SCALE,  "weight", i), {3}, 0);

        layer.attn_norm      = create_tensor(tn(LLM_TENSOR_ATTN_NORM,      "weight", i), {n_embd}, 0);
        layer.ffn_norm       = create_tensor(tn(LLM_TENSOR_FFN_NORM,       "weight", i), {n_embd}, 0);
        layer.attn_sinks     = create_tensor(tn(LLM_TENSOR_ATTN_SINKS,     "weight", i), {n_head}, 0);
        layer.attn_q_a_norm  = create_tensor(tn(LLM_TENSOR_ATTN_Q_A_NORM,  "weight", i), {q_lora_rank}, 0);
        layer.attn_kv_a_norm = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_NORM, "weight", i), {n_embd_head_k}, 0);

        layer.wq_a    = create_tensor(tn(LLM_TENSOR_ATTN_Q_A,    "weight", i), {n_embd, q_lora_rank}, 0);
        layer.wq_b    = create_tensor(tn(LLM_TENSOR_ATTN_Q_B,    "weight", i), {q_lora_rank, n_head * n_embd_head_k}, 0);
        layer.attn_kv = create_tensor(tn(LLM_TENSOR_ATTN_KV,     "weight", i), {n_embd, n_embd_head_k}, 0);
        layer.attn_wo_a = create_tensor(tn(LLM_TENSOR_ATTN_OUT_A, "weight", i), {n_head * n_embd_head_v / n_out_groups, n_out_groups * o_lora_rank}, 0);
        layer.attn_wo_b = create_tensor(tn(LLM_TENSOR_ATTN_OUT_B, "weight", i), {n_out_groups * o_lora_rank, n_embd}, 0);

        if (compress_ratio > 0) {
            create_deepseek4_compressor(layer, i, compress_ratio, n_embd_head_k, false);
        }
        if (compress_ratio == 4) {
            layer.indexer_attn_q_b = create_tensor(tn(LLM_TENSOR_INDEXER_ATTN_Q_B, "weight", i), {q_lora_rank, hparams.indexer_n_head * hparams.indexer_head_size}, 0);
            layer.indexer_proj     = create_tensor(tn(LLM_TENSOR_INDEXER_PROJ,     "weight", i), {n_embd, hparams.indexer_n_head}, 0);
            create_deepseek4_compressor(layer, i, compress_ratio, hparams.indexer_head_size, true);
        }

        layer.ffn_gate_inp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), {n_embd, n_expert}, 0);
        if (static_cast<uint32_t>(i) < hparams.n_hash_layers) {
            layer.ffn_gate_tid2eid = create_tensor(tn(LLM_TENSOR_FFN_GATE_TID2EID, "weight", i), {n_expert_used, n_vocab}, 0);
            layer.ffn_exp_probs_b  = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B,  "bias",   i), {n_expert}, TENSOR_NOT_REQUIRED);
        } else {
            layer.ffn_exp_probs_b  = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B,  "bias",   i), {n_expert}, 0);
            layer.ffn_gate_tid2eid = create_tensor(tn(LLM_TENSOR_FFN_GATE_TID2EID, "weight", i), {n_expert_used, n_vocab}, TENSOR_NOT_REQUIRED);
        }

        layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd,   n_ff_exp, n_expert}, exps_flags_for(i));
        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp, n_embd,   n_expert}, exps_flags_for(i));
        layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {n_embd,   n_ff_exp, n_expert}, exps_flags_for(i));

        layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd,   n_ff_exp * n_expert_shared}, 0);
        layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {n_ff_exp * n_expert_shared, n_embd}, 0);
        layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd,   n_ff_exp * n_expert_shared}, 0);
    }

    // DeepSeek-V4 MTP/NextN draft layer(s): a full MLA+MoE decoder layer (no
    // compressor, no indexer, no hash routing) plus the split eh projections
    // and the head's own hyper-connection collapse. All optional: ds4 GGUFs
    // advertise nextn_predict_layers in metadata even when the converter did
    // not preserve the MTP tensors.
    for (int i = n_main_layers; i < n_layer; ++i) {
        auto & layer = layers[i];
        const int f = TENSOR_NOT_REQUIRED;

        layer.hc_attn_base  = create_tensor(tn(LLM_TENSOR_HC_ATTN_BASE,  "weight", i), {hc_mix}, f);
        layer.hc_attn_fn    = create_tensor(tn(LLM_TENSOR_HC_ATTN_FN,    "weight", i), {hc_dim, hc_mix}, f);
        layer.hc_attn_scale = create_tensor(tn(LLM_TENSOR_HC_ATTN_SCALE, "weight", i), {3}, f);
        layer.hc_ffn_base   = create_tensor(tn(LLM_TENSOR_HC_FFN_BASE,   "weight", i), {hc_mix}, f);
        layer.hc_ffn_fn     = create_tensor(tn(LLM_TENSOR_HC_FFN_FN,     "weight", i), {hc_dim, hc_mix}, f);
        layer.hc_ffn_scale  = create_tensor(tn(LLM_TENSOR_HC_FFN_SCALE,  "weight", i), {3}, f);

        layer.attn_norm      = create_tensor(tn(LLM_TENSOR_ATTN_NORM,      "weight", i), {n_embd}, f);
        layer.ffn_norm       = create_tensor(tn(LLM_TENSOR_FFN_NORM,       "weight", i), {n_embd}, f);
        layer.attn_sinks     = create_tensor(tn(LLM_TENSOR_ATTN_SINKS,     "weight", i), {n_head}, f);
        layer.attn_q_a_norm  = create_tensor(tn(LLM_TENSOR_ATTN_Q_A_NORM,  "weight", i), {q_lora_rank}, f);
        layer.attn_kv_a_norm = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_NORM, "weight", i), {n_embd_head_k}, f);

        layer.wq_a      = create_tensor(tn(LLM_TENSOR_ATTN_Q_A,    "weight", i), {n_embd, q_lora_rank}, f);
        layer.wq_b      = create_tensor(tn(LLM_TENSOR_ATTN_Q_B,    "weight", i), {q_lora_rank, n_head * n_embd_head_k}, f);
        layer.attn_kv   = create_tensor(tn(LLM_TENSOR_ATTN_KV,     "weight", i), {n_embd, n_embd_head_k}, f);
        layer.attn_wo_a = create_tensor(tn(LLM_TENSOR_ATTN_OUT_A,  "weight", i), {n_head * n_embd_head_v / n_out_groups, n_out_groups * o_lora_rank}, f);
        layer.attn_wo_b = create_tensor(tn(LLM_TENSOR_ATTN_OUT_B,  "weight", i), {n_out_groups * o_lora_rank, n_embd}, f);

        layer.ffn_gate_inp    = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,    "weight", i), {n_embd, n_expert}, f);
        layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias",   i), {n_expert}, f);

        layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd,   n_ff_exp, n_expert}, f | exps_flags_for(i));
        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp, n_embd,   n_expert}, f | exps_flags_for(i));
        layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {n_embd,   n_ff_exp, n_expert}, f | exps_flags_for(i));

        layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd,   n_ff_exp * n_expert_shared}, f);
        layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {n_ff_exp * n_expert_shared, n_embd}, f);
        layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd,   n_ff_exp * n_expert_shared}, f);

        layer.nextn.e_proj           = create_tensor(tn(LLM_TENSOR_NEXTN_E_PROJ,           "weight", i), {n_embd, n_embd}, f);
        layer.nextn.h_proj           = create_tensor(tn(LLM_TENSOR_NEXTN_H_PROJ,           "weight", i), {n_embd, n_embd}, f);
        layer.nextn.enorm            = create_tensor(tn(LLM_TENSOR_NEXTN_ENORM,            "weight", i), {n_embd}, f);
        layer.nextn.hnorm            = create_tensor(tn(LLM_TENSOR_NEXTN_HNORM,            "weight", i), {n_embd}, f);
        layer.nextn.shared_head_norm = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_NORM, "weight", i), {n_embd}, f);
        layer.nextn.hc_head_base     = create_tensor(tn(LLM_TENSOR_NEXTN_HC_HEAD_BASE,     "weight", i), {n_hc}, f);
        layer.nextn.hc_head_fn       = create_tensor(tn(LLM_TENSOR_NEXTN_HC_HEAD_FN,       "weight", i), {hc_dim, n_hc}, f);
        layer.nextn.hc_head_scale    = create_tensor(tn(LLM_TENSOR_NEXTN_HC_HEAD_SCALE,    "weight", i), {1}, f);
    }
}

std::unique_ptr<llm_graph_context> llama_model_deepseek4::build_arch_graph(const llm_graph_params & params) const {
    if (params.gtype == LLM_GRAPH_TYPE_DECODER_MTP) {
        return std::make_unique<graph_mtp>(*this, params);
    }
    return std::make_unique<graph>(*this, params);
}
