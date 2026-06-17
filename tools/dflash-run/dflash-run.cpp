// llama-dflash-run — standalone DFlash block-diffusion speculative decoding for DeepSeek-V4-Flash.
//
// This is a ground-up DFlash generation loop that OWNS its own verify/commit cycle, instead of
// bolting the DFlash drafter onto the server's generic speculative dispatcher. The two structural
// wins that the server's generic loop cannot express are baked in here:
//
//   1. CONSTANT-SHAPE VERIFY. Every round verifies a full G-token block on the target (anchor +
//      G-1 drafts), so the target decode batch shape never changes -> the CUDA graph stays warm
//      across rounds (the server verifies a variable number of accepted drafts, re-capturing the
//      graph every round).
//
//   2. SNAPSHOT ROLLBACK, NO 2ND VERIFY. DSV4 carries a RECURRENT compress-state, so a partial
//      accept cannot be rolled back by dropping KV rows alone. We size the target context with
//      n_rs_seq = G-1 recurrent rollback planes; the DSV4 graph snapshots the compress-state after
//      each verified token, and a single seq_rm restores the state as-of the last accepted token.
//      The server instead re-decodes the accepted prefix (a 2nd full target decode per round).
//
// Profiling on the server showed a DFlash round ~= 234ms split as verify#1 ~117ms (cuda-cold from
// the variable batch) + verify#2 ~94ms (the 2nd verify) + drafter ~4ms + host ~19ms. Those two
// target decodes (211ms) are the whole cost; this binary attacks both.
//
// Usage:
//   llama-dflash-run -m <dsv4.gguf> -md <dflash-drafter.gguf> -p "<prompt>" -n 128 [common flags]
// Env knobs:
//   DFLASH_TARGET_LAYERS=2,12,22,32,40   target layers the drafter conditions on (count -> featdim)
//   DFLASH_GREEDY=1                       reference mode: pure target greedy (no drafter), for losslessness diff
//   DFLASH_NO_ROLLBACK=1                  use a 2nd-verify rollback instead of snapshots (A/B the snapshot path)
//   DFLASH_SELFCHECK=1                    after each round, recompute the confirmed-prefix state and warn on drift
//   DFLASH_PROF=1                         per-phase timing summary

#include "arg.h"
#include "common.h"
#include "llama.h"
#include "log.h"
#include "ggml.h"

#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include <set>
#include <map>
#include <algorithm>

static int env_int(const char * k, int dflt) {
    const char * e = getenv(k);
    return e ? std::atoi(e) : dflt;
}

// ---- per-layer row-0 bisection probe (DFLASH_PROBE): capture the anchor's residual hidden at every
// layer in a multi-token block vs a single-token decode; the first divergent layer is the leak. ----
struct probe_state {
    int phase = 0; // 0 = off, 1 = block, 2 = single
    std::map<int, std::vector<float>> block, single;
    std::map<int, std::vector<float>> kqv_blk, kqv_sgl; // layer -> row0 attention output
};
static probe_state g_probe;

static bool probe_cb(struct ggml_tensor * t, bool ask, void * /*ud*/) {
    if (g_probe.phase == 0) return false;
    // KQVDBG: dump layer-0 attention output for EVERY token row (block phase) so an FA-on vs FA-off
    // run can be matched row-by-row -> reveals whether FA permutes/corrupts per-query output.
    // capture per-layer attention output (kqv_out-{il}) ROW 0 in block(phase1) and single(phase2)
    // -> find the first layer where the block anchor's attention diverges from the single decode.
    if ((g_probe.phase == 1 || g_probe.phase == 2) && std::strncmp(t->name, "kqv_out-", 8) == 0) {
        if (ask) return true;
        const int il = std::atoi(t->name + 8);
        const int64_t dim = t->ne[0];                  // row 0 = first dim elements
        std::vector<float> buf(dim);
        if (t->type == GGML_TYPE_F32) ggml_backend_tensor_get(t, buf.data(), 0, dim * sizeof(float));
        else if (t->type == GGML_TYPE_F16) { std::vector<uint16_t> h(dim); ggml_backend_tensor_get(t, h.data(), 0, dim*sizeof(uint16_t)); for (int64_t i=0;i<dim;++i) buf[i]=ggml_fp16_to_fp32(h[i]); }
        else return true;
        auto & dst = (g_probe.phase == 1 ? g_probe.kqv_blk : g_probe.kqv_sgl);
        if (dst.find(il) == dst.end()) dst[il] = buf; // first firing per layer = the real verify/single decode
        return true;
    }
    static const char * want = "hc_ffn_post-";
    const size_t pl = std::strlen(want);
    if (std::strncmp(t->name, want, pl) != 0) return false;
    if (ask) return true;
    const int il = std::atoi(t->name + pl);
    const int64_t ne0 = t->ne[0];                 // n_embd
    const bool    hc  = (t->ne[2] > 1);
    const int64_t nhc = hc ? t->ne[1] : 1;
    const size_t  n   = (size_t) ne0 * nhc;        // token 0 plane (contiguous prefix)
    std::vector<float> buf(n);
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, buf.data(), 0, n * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        std::vector<uint16_t> h(n);
        ggml_backend_tensor_get(t, h.data(), 0, n * sizeof(uint16_t));
        for (size_t i = 0; i < n; ++i) buf[i] = ggml_fp16_to_fp32(h[i]);
    } else return true;
    std::vector<float> row(ne0, 0.0f);             // collapse n_hc by sum -> [n_embd]
    for (int64_t h = 0; h < nhc; ++h)
        for (int64_t e = 0; e < ne0; ++e) row[e] += buf[(size_t) h * ne0 + e];
    (g_probe.phase == 1 ? g_probe.block : g_probe.single)[il] = std::move(row);
    return true;
}

static int argmax_logits(const float * lg, int n_vocab) {
    int best = 0; float bv = lg[0];
    for (int v = 1; v < n_vocab; ++v) if (lg[v] > bv) { bv = lg[v]; best = v; }
    return best;
}

// argmax + softmax confidence (max prob) of that token — used by the denoising unmask schedule.
static int argmax_conf(const float * lg, int n_vocab, float * conf) {
    int best = 0; float mx = lg[0];
    for (int v = 1; v < n_vocab; ++v) if (lg[v] > mx) { mx = lg[v]; best = v; }
    double se = 0.0; for (int v = 0; v < n_vocab; ++v) se += std::exp((double) lg[v] - mx);
    *conf = (float) (1.0 / se);
    return best;
}

// Medusa-style typical acceptance: accept draft if p_target(draft) > min(0.3, delta*exp(-H)), where
// H is the target entropy at this position. High-entropy (uncertain) -> permissive; sharp -> strict.
// delta=0 reduces to exact match. mx must be the argmax logit (= max).
static bool typical_accept(const float * lg, int n_vocab, llama_token draft, float mx, float delta) {
    if (delta <= 0.0f) return false;
    double Z = 0.0, Hs = 0.0;
    for (int v = 0; v < n_vocab; ++v) Z += std::exp((double) lg[v] - mx);
    for (int v = 0; v < n_vocab; ++v) { double p = std::exp((double) lg[v] - mx) / Z; if (p > 0) Hs -= p * std::log(p); }
    const double p_draft = std::exp((double) lg[draft] - mx) / Z;
    const double thresh  = std::min(0.3, (double) delta * std::exp(-Hs));
    return p_draft > thresh;
}

int main(int argc, char ** argv) {
    common_params params;
    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SPECULATIVE)) {
        return 1;
    }
    if (params.speculative.draft.mparams.path.empty() && params.speculative.draft.mparams.hf_repo.empty()) {
        LOG_ERR("%s: a DFlash drafter is required (-md / --model-draft)\n", __func__);
        return 1;
    }

    const bool greedy_ref  = env_int("DFLASH_GREEDY", 0) != 0;
    const bool no_rollback = env_int("DFLASH_NO_ROLLBACK", 0) != 0;
    const bool prof        = env_int("DFLASH_PROF", 0) != 0;
    const bool do_probe    = env_int("DFLASH_PROBE", 0) != 0;
    if (do_probe) { params.cb_eval = probe_cb; params.cb_eval_user_data = nullptr; params.warmup = false; }

    llama_backend_init();
    llama_numa_init(params.numa);

    // ---- drafter metadata: block_size (G), mask_id, featdim. Load the drafter model first so we
    //      can size the target context's rollback planes (n_rs_seq = G-1). ----
    int32_t n_target = 5;
    {
        const char * env = getenv("DFLASH_TARGET_LAYERS");
        std::string s = env ? env : "2,12,22,32,40";
        std::set<int> ids; size_t p = 0;
        while (p < s.size()) {
            size_t c = s.find(',', p);
            ids.insert(std::atoi(s.substr(p, c == std::string::npos ? c : c - p).c_str()));
            if (c == std::string::npos) break;
            p = c + 1;
        }
        n_target = (int32_t) ids.size();
    }

    // drafter model load (mirrors the server's params_dft derivation)
    llama_model_ptr   model_dft;
    llama_context_ptr ctx_dft;
    int32_t block_size = 16, mask_id = 2;
    {
        const auto & ps = params.speculative.draft;
        auto params_dft = params;
        params_dft.devices      = ps.devices;
        params_dft.model        = ps.mparams;
        params_dft.n_gpu_layers = ps.n_gpu_layers;
        if (ps.cpuparams.n_threads > 0) {
            params_dft.cpuparams.n_threads       = ps.cpuparams.n_threads;
            params_dft.cpuparams_batch.n_threads = ps.cpuparams_batch.n_threads;
        }
        params_dft.tensor_buft_overrides = ps.tensor_buft_overrides;

        auto mparams_dft = common_model_params_to_llama(params_dft);
        model_dft.reset(llama_model_load_from_file(params_dft.model.path.c_str(), mparams_dft));
        if (!model_dft) { LOG_ERR("failed to load drafter '%s'\n", params_dft.model.path.c_str()); return 1; }

        char buf[64];
        if (llama_model_meta_val_str(model_dft.get(), "dflash.block_size", buf, sizeof(buf)) > 0)    block_size = std::atoi(buf);
        if (llama_model_meta_val_str(model_dft.get(), "dflash.mask_token_id", buf, sizeof(buf)) > 0) mask_id    = std::atoi(buf);

        auto cparams_dft = common_context_params_to_llama(params_dft);
        cparams_dft.n_rs_seq = 0;
        ctx_dft.reset(llama_init_from_model(model_dft.get(), cparams_dft));
        if (!ctx_dft) { LOG_ERR("failed to create drafter context\n"); return 1; }
    }
    const int32_t G       = block_size;            // verify block width
    const int32_t n_embd  = llama_model_n_embd(model_dft.get());
    const int32_t featdim = n_target * n_embd;

    // ---- target (DSV4): build the context manually so we can set n_rs_seq = G-1 (the server's
    //      need_n_rs_seq path is MTP-only; DFlash needs the rollback planes here). ----
    llama_model_ptr   model_tgt;
    llama_context_ptr ctx_tgt;
    {
        auto mparams_tgt = common_model_params_to_llama(params);
        model_tgt.reset(llama_model_load_from_file(params.model.path.c_str(), mparams_tgt));
        if (!model_tgt) { LOG_ERR("failed to load target '%s'\n", params.model.path.c_str()); return 1; }

        auto cparams_tgt = common_context_params_to_llama(params);
        // rollback planes; DFLASH_NRS overrides (set 0 to disable the plane store for isolation tests)
        cparams_tgt.n_rs_seq = (uint32_t) env_int("DFLASH_NRS", G - 1);
        ctx_tgt.reset(llama_init_from_model(model_tgt.get(), cparams_tgt));
        if (!ctx_tgt) { LOG_ERR("failed to create target context\n"); return 1; }
    }

    const llama_vocab * vocab   = llama_model_get_vocab(model_tgt.get());
    const int32_t       n_vocab = llama_vocab_n_tokens(vocab);
    llama_memory_t      mem_tgt = llama_get_memory(ctx_tgt.get());

    // wire drafter <- target (shared embd + lm_head) and enable in-graph feature capture on target
    llama_set_dflash_target(ctx_dft.get(), ctx_tgt.get());
    llama_set_dflash_capture(ctx_tgt.get(), true);

    // ---- tokenize once (shared across modes) ----
    std::vector<llama_token> inp = common_tokenize(ctx_tgt.get(), params.prompt, true, true);
    if (inp.empty()) { LOG_ERR("empty prompt\n"); return 1; }
    if ((int) inp.size() >= llama_n_ctx(ctx_tgt.get())) { LOG_ERR("prompt too long\n"); return 1; }
    const int n_predict = params.n_predict > 0 ? params.n_predict : 128;
    llama_memory_t mem_dft = llama_get_memory(ctx_dft.get());

    // ---- DFLASH_PROBE: per-layer row-0 bisection. Decode the prompt, then capture the anchor's
    // residual hidden at every layer (a) inside a G-token block and (b) as a lone single token.
    // The first layer where they diverge is the attention that leaks future block tokens into row 0.
    if (do_probe) {
        llama_set_dflash_capture(ctx_tgt.get(), false); // probe uses the eval cb, not in-graph capture
        auto prefill = [&](llama_token & anchor, int & n_past) {
            llama_memory_clear(mem_tgt, true);
            llama_batch pb = llama_batch_init((int) inp.size(), 0, 1);
            pb.n_tokens = (int) inp.size();
            for (int i = 0; i < (int) inp.size(); ++i) {
                pb.token[i] = inp[i]; pb.pos[i] = i; pb.n_seq_id[i] = 1; pb.seq_id[i][0] = 0;
                pb.logits[i] = (i == (int) inp.size() - 1);
            }
            llama_decode(ctx_tgt.get(), pb);
            llama_batch_free(pb);
            anchor = argmax_logits(llama_get_logits_ith(ctx_tgt.get(), (int) inp.size() - 1), n_vocab);
            n_past = (int) inp.size();
        };
        llama_token anchor = 0; int n_past = 0;

        // (a) block: [anchor, mask x (G-1)]
        prefill(anchor, n_past);
        g_probe.phase = 1;
        {
            llama_batch b = llama_batch_init(G, 0, 1);
            b.n_tokens = G;
            for (int j = 0; j < G; ++j) {
                b.token[j] = (j == 0) ? anchor : mask_id;
                b.pos[j] = n_past + j; b.n_seq_id[j] = 1; b.seq_id[j][0] = 0; b.logits[j] = 1;
            }
            llama_decode(ctx_tgt.get(), b);
            llama_batch_free(b);
        }
        std::vector<float> bl(n_vocab);
        std::memcpy(bl.data(), llama_get_logits_ith(ctx_tgt.get(), 0), (size_t) n_vocab * sizeof(float));
        const llama_token blk_tok0 = argmax_logits(bl.data(), n_vocab);

        // (b) single: [anchor] alone, from the same post-prompt state
        prefill(anchor, n_past);
        g_probe.phase = 2;
        {
            llama_batch b = llama_batch_init(1, 0, 1);
            b.n_tokens = 1;
            b.token[0] = anchor; b.pos[0] = n_past; b.n_seq_id[0] = 1; b.seq_id[0][0] = 0; b.logits[0] = 1;
            llama_decode(ctx_tgt.get(), b);
            llama_batch_free(b);
        }
        std::vector<float> sg(n_vocab);
        std::memcpy(sg.data(), llama_get_logits_ith(ctx_tgt.get(), 0), (size_t) n_vocab * sizeof(float));
        const llama_token sgl_tok0 = argmax_logits(sg.data(), n_vocab);
        g_probe.phase = 0;

        LOG_INF("PROBE: anchor=%d  block_argmax=%d  single_argmax=%d  %s\n", anchor, blk_tok0, sgl_tok0,
                blk_tok0 == sgl_tok0 ? "(MATCH - no leak?!)" : "(LEAK confirmed at logits)");
        // direct logit gap: is block_argmax vs single_argmax a near-tie or a systematic margin?
        auto top5 = [&](const char * tag, const std::vector<float> & L) {
            std::vector<int> idx(n_vocab); for (int i=0;i<n_vocab;++i) idx[i]=i;
            std::partial_sort(idx.begin(), idx.begin()+5, idx.end(), [&](int a,int b){return L[a]>L[b];});
            char buf[256]; int o=0; for (int k=0;k<5;++k) o+=std::snprintf(buf+o,sizeof(buf)-o,"%d=%.3f ",idx[k],L[idx[k]]);
            LOG_INF("PROBE %s top5: %s\n", tag, buf);
        };
        top5("BLK row0", bl);
        top5("SGL row0", sg);
        LOG_INF("PROBE GAP  in BLK: L[%d]=%.4f  L[%d]=%.4f  diff=%.4f\n", blk_tok0, bl[blk_tok0], sgl_tok0, bl[sgl_tok0], bl[blk_tok0]-bl[sgl_tok0]);
        LOG_INF("PROBE GAP  in SGL: L[%d]=%.4f  L[%d]=%.4f  diff=%.4f\n", blk_tok0, sg[blk_tok0], sgl_tok0, sg[sgl_tok0], sg[blk_tok0]-sg[sgl_tok0]);
        int first = -1;
        for (auto & kv : g_probe.single) {
            const int il = kv.first;
            auto it = g_probe.block.find(il);
            if (it == g_probe.block.end()) continue;
            const auto & a = it->second; const auto & b = kv.second;
            double num = 0, da = 0, db = 0, maxd = 0;
            for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
                num += (double) a[i] * b[i]; da += (double) a[i] * a[i]; db += (double) b[i] * b[i];
                maxd = std::max(maxd, (double) std::fabs(a[i] - b[i]));
            }
            const double cos = (da > 0 && db > 0) ? num / std::sqrt(da * db) : 1.0;
            const bool diverged = cos < 0.99999 || maxd > 1e-2;
            LOG_INF("PROBE layer %2d: cos=%.6f maxabs=%.4g %s\n", il, cos, maxd, diverged ? "<-- DIVERGES" : "");
            if (diverged && first < 0) first = il;
        }
        LOG_INF("PROBE: first diverging layer = %d\n", first);

        // per-layer ATTENTION-output (kqv_out) row0: block vs single. Correct layout, the clean test.
        int kfirst = -1;
        for (auto & kv : g_probe.kqv_sgl) {
            const int il = kv.first;
            auto it = g_probe.kqv_blk.find(il);
            if (it == g_probe.kqv_blk.end()) continue;
            const auto & a = it->second; const auto & b = kv.second;
            double num=0, da=0, db=0, maxd=0;
            for (size_t i=0;i<a.size()&&i<b.size();++i){ num+=(double)a[i]*b[i]; da+=(double)a[i]*a[i]; db+=(double)b[i]*b[i]; maxd=std::max(maxd,(double)std::fabs(a[i]-b[i])); }
            const double cos = (da>0&&db>0)? num/std::sqrt(da*db) : 1.0;
            const bool diverged = cos < 0.999 || maxd > 0.5;
            LOG_INF("KQVCMP layer %2d: cos=%.6f maxabs=%.4g %s\n", il, cos, maxd, diverged?"<-- DIVERGES":"");
            if (diverged && kfirst<0) kfirst=il;
        }
        LOG_INF("KQVCMP: first attention-diverging layer = %d\n", kfirst);
        llama_backend_free();
        return 0;
    }

    enum Mode { SNAPSHOT = 0, REVERIFY = 1, GREEDY = 2 };
    struct Result {
        std::vector<llama_token> out;
        double wall = 0, t_dft = 0, t_vrf = 0, t_roll = 0, t_cross = 0, t_misc = 0;
        int rounds = 0, acc_total = 0;
    };

    // The first token (BOS) is an attention SINK -> its hc_ffn_post norm is ~250x the rest, which
    // overflows the F16 drafter fc -> NaN logits -> degenerate BOS drafts. Cap each committed feature
    // row's norm ONCE (here, at commit time); the drafter's post-fc hidden_norm (RMS) is scale-
    // invariant, so capping the outlier only prevents overflow without changing the result.
    const float feat_cap = []{ const char* e=getenv("DFLASH_FEAT_CAP"); return e?(float)std::atof(e):4096.0f; }();
    auto cap_row = [feat_cap](std::vector<float> & row) {
        if (feat_cap <= 0.0f) return;
        double nrm = 0; for (float v : row) nrm += (double) v * v; nrm = std::sqrt(nrm);
        if (nrm > feat_cap) { const float s = feat_cap / (float) nrm; for (float & v : row) v *= s; }
    };

    // one full generation pass in the requested rollback mode; reuses the loaded contexts.
    // vg = verify block width (<= G); the drafter still decodes a full G-block, only the first
    // vg-1 drafts are verified/accepted. Lets a G-sweep trade verify cost vs accept length.
    auto run_mode = [&](Mode mode, int vg, int dsteps, float eps) -> Result {
        const bool greedy   = (mode == GREEDY);
        const bool reverify = (mode == REVERIFY);
        if (vg < 2) vg = 2;
        if (vg > G) vg = G;
        const int n_steps = std::max(1, dsteps);
        Result R;

        llama_memory_clear(mem_tgt, true);
        llama_memory_clear(mem_dft, true);

        // committed[i] = featdim feature row for confirmed position i (drafter cross-attention ctx)
        std::vector<std::vector<float>> committed;
        int n_past = 0;
        llama_token anchor = 0;

        // ---- prefill: decode the prompt, capture per-position features, pick the first anchor ----
        {
            llama_batch pb = llama_batch_init((int) inp.size(), 0, 1);
            pb.n_tokens = (int) inp.size();
            for (int i = 0; i < (int) inp.size(); ++i) {
                pb.token[i] = inp[i]; pb.pos[i] = i; pb.n_seq_id[i] = 1; pb.seq_id[i][0] = 0;
                pb.logits[i] = (i == (int) inp.size() - 1);
            }
            if (llama_decode(ctx_tgt.get(), pb) != 0) { LOG_ERR("prefill failed\n"); llama_batch_free(pb); return R; }
            llama_batch_free(pb);

            int32_t nt = 0, dim = 0;
            const float * feat = llama_get_dflash_feat(ctx_tgt.get(), &nt, &dim);
            if (!feat || dim != featdim || nt != (int) inp.size()) {
                LOG_ERR("prefill capture bad (dim=%d/%d nt=%d/%zu)\n", dim, featdim, nt, inp.size());
                return R;
            }
            committed.resize(inp.size());
            for (int i = 0; i < (int) inp.size(); ++i) {
                committed[i].assign(feat + (size_t) i * featdim, feat + (size_t)(i + 1) * featdim);
                cap_row(committed[i]);
            }
            anchor = argmax_logits(llama_get_logits_ith(ctx_tgt.get(), (int) inp.size() - 1), n_vocab);
            n_past = (int) inp.size();
        }

        llama_batch dft_batch = llama_batch_init(G, 0, 1);
        llama_batch vrf_batch = llama_batch_init(G, 0, 1);
        std::vector<float> cross;
        std::vector<llama_token> draft(G - 1), targ(G);
        int n_emit = 0;
        bool stop = false;
        const int64_t t_start = ggml_time_us();

        while (n_emit < n_predict && !stop) {
            int n_accepted = 0;

            // DFLASH_NODRAFT: skip the drafter, fill draft positions with mask_id. Isolates whether
            // the drafter decode (shared target tensors) corrupts the subsequent target verify.
            static const bool nodraft = env_int("DFLASH_NODRAFT", 0) != 0;
            if (!greedy && !nodraft) {
                // drafter cross-attends to committed features (already norm-capped at commit time).
                // The drafter cost is dominated by re-encoding K/V for every committed feature each
                // round, so cap to a recent window (DFLASH_MAX_CTX) -> bounds the drafter cost; recent
                // context dominates the next-token prediction so tau is largely preserved.
                static const int max_ctx = env_int("DFLASH_MAX_CTX", 0);
                const int total_ctx = (int) committed.size();
                const int beg   = (max_ctx > 0 && total_ctx > max_ctx) ? total_ctx - max_ctx : 0;
                const int n_ctx = total_ctx - beg;
                const int64_t c0 = ggml_time_us();
                cross.resize((size_t) featdim * n_ctx);
                for (int i = 0; i < n_ctx; ++i)
                    std::memcpy(cross.data() + (size_t) i * featdim, committed[beg + i].data(), (size_t) featdim * sizeof(float));
                llama_set_cross_embd(ctx_dft.get(), cross.data(), featdim, n_ctx);
                R.t_cross += (ggml_time_us() - c0) * 1e-3;

                // multi-step block-diffusion denoising. The drafter only needs to predict the vg-1
                // positions the target will verify, so decode a vg-wide block (not the full G) -> the
                // drafter cost scales with the verify width instead of being a fixed 16-token decode.
                const int dw = vg;   // draft width = verify width
                std::vector<llama_token> blk(dw, mask_id); blk[0] = anchor;
                std::vector<char> revealed(dw, 0); revealed[0] = 1;
                std::vector<llama_token> pred(dw, mask_id);
                std::vector<float> conf(dw, 0.0f);
                bool dft_fail = false;
                const int64_t d0 = ggml_time_us();
                for (int step = 0; step < n_steps; ++step) {
                    dft_batch.n_tokens = dw;
                    for (int j = 0; j < dw; ++j) {
                        dft_batch.token[j] = blk[j];
                        dft_batch.pos[j] = j; dft_batch.n_seq_id[j] = 1; dft_batch.seq_id[j][0] = 0; dft_batch.logits[j] = 1;
                    }
                    if (llama_decode(ctx_dft.get(), dft_batch) != 0) { LOG_ERR("drafter decode failed\n"); dft_fail = true; break; }
                    const bool need_conf = (step < n_steps - 1);
                    for (int off = 1; off < dw; ++off) {
                        const float * lg = llama_get_logits_ith(ctx_dft.get(), off);
                        pred[off] = need_conf ? argmax_conf(lg, n_vocab, &conf[off]) : argmax_logits(lg, n_vocab);
                    }
                    if (step == n_steps - 1) break;
                    const int target_rev = (int) std::lround((double) (step + 1) * (dw - 1) / n_steps);
                    int cur_rev = 0; for (int j = 1; j < dw; ++j) cur_rev += revealed[j];
                    for (int add = cur_rev; add < target_rev; ++add) {
                        int bj = -1; float bc = -1.0f;
                        for (int j = 1; j < dw; ++j) if (!revealed[j] && conf[j] > bc) { bc = conf[j]; bj = j; }
                        if (bj < 0) break;
                        revealed[bj] = 1; blk[bj] = pred[bj];
                    }
                }
                R.t_dft += (ggml_time_us() - d0) * 1e-3;
                if (dft_fail) break;
                for (int off = 1; off < dw; ++off) draft[off - 1] = pred[off];
            } else if (!greedy) {
                for (int j = 0; j < vg - 1; ++j) draft[j] = mask_id;
            }

            // target verify: constant-shape vg-block [anchor, draft...] at positions n_past..
            const int vb = greedy ? 1 : vg;
            vrf_batch.n_tokens = vb;
            for (int j = 0; j < vb; ++j) {
                vrf_batch.token[j] = (j == 0) ? anchor : draft[j - 1];
                vrf_batch.pos[j] = n_past + j; vrf_batch.n_seq_id[j] = 1; vrf_batch.seq_id[j][0] = 0; vrf_batch.logits[j] = 1;
            }
            const int64_t v0 = ggml_time_us();
            if (llama_decode(ctx_tgt.get(), vrf_batch) != 0) { LOG_ERR("verify decode failed\n"); break; }
            R.t_vrf += (ggml_time_us() - v0) * 1e-3;

            for (int j = 0; j < vb; ++j)
                targ[j] = argmax_logits(llama_get_logits_ith(ctx_tgt.get(), j), n_vocab);
            // Medusa-style typical acceptance: accept draft if it matches the argmax OR is "typical"
            // under the target distribution (eps here = delta). Entropy-adaptive -> better quality
            // than a flat logit-gap at the same tau. eps=0 -> exact match.
            // eps > 0 -> entropy-typical acceptance with delta=eps; eps < 0 -> raw logit-gap of |eps|
            // (the known-good criterion, for isolating drafter weakness vs a typical_accept bug).
            if (!greedy) {
                while (n_accepted < vg - 1) {
                    const llama_token d = draft[n_accepted];
                    if (d != targ[n_accepted]) {
                        const float * lg = llama_get_logits_ith(ctx_tgt.get(), n_accepted);
                        bool ok;
                        if (eps < 0.0f)      ok = lg[d] >= lg[targ[n_accepted]] + eps;   // gap = |eps|
                        else                 ok = typical_accept(lg, n_vocab, d, lg[targ[n_accepted]], eps);
                        if (!ok) break;
                    }
                    ++n_accepted;
                }
            }

            // DFLASH_SHOWDRAFT: dump draft vs target for the first few rounds -> bug (garbage drafts)
            // vs weakness (plausible-but-wrong drafts). Shows the per-position draft logit gap too.
            static const bool showdraft = env_int("DFLASH_SHOWDRAFT", 0) != 0;
            if (showdraft && !greedy && R.rounds < 4) {
                // feature sanity: norms of first/last committed rows (0 -> feature-supply bug) + the
                // drafter's own top-3 logits at offset 1 (always BOS with high logit -> wiring issue).
                double n0 = 0, nb = 0;
                if (!committed.empty()) {
                    for (float v : committed.front()) n0 += (double) v * v;
                    for (float v : committed.back())  nb += (double) v * v;
                }
                const float * dl = llama_get_logits_ith(ctx_dft.get(), 1);
                int t1 = argmax_logits(dl, n_vocab); float l1 = dl[t1];
                LOG_INF("SHOWDRAFT FEAT n_ctx=%d featdim=%zu ||feat[0]||=%.2f ||feat[last]||=%.2f | drafter off1 top='%s' L=%.2f\n",
                        (int) committed.size(), committed.empty()?0:committed.front().size(),
                        std::sqrt(n0), std::sqrt(nb), common_token_to_piece(ctx_dft.get(), t1).c_str(), l1);
                std::string s = "SHOWDRAFT r" + std::to_string(R.rounds) + " anchor='" +
                                common_token_to_piece(ctx_tgt.get(), anchor) + "':";
                for (int j = 0; j < std::min(6, vg - 1); ++j) {
                    const float * lg = llama_get_logits_ith(ctx_tgt.get(), j);
                    char b[160];
                    std::snprintf(b, sizeof(b), " [%d] draft='%s'(L%.1f) targ='%s'(L%.1f)%s", j,
                        common_token_to_piece(ctx_tgt.get(), draft[j]).c_str(), lg[draft[j]],
                        common_token_to_piece(ctx_tgt.get(), targ[j]).c_str(), lg[targ[j]],
                        draft[j] == targ[j] ? " MATCH" : "");
                    s += b;
                }
                LOG_INF("%s\n", s.c_str());
            }

            int32_t nt = 0, dim = 0;
            const float * feat = llama_get_dflash_feat(ctx_tgt.get(), &nt, &dim);
            const bool feat_ok = feat && dim == featdim && nt == vb;
            if (!feat_ok) LOG_WRN("verify capture bad (dim=%d/%d nt=%d/%d)\n", dim, featdim, nt, vb);

            // accepted positions emit the verified DRAFT (the token actually in the batch, so the
            // downstream targ stays valid); the bonus position emits the target's argmax correction.
            for (int j = 0; j <= n_accepted && !stop; ++j) {
                const llama_token tok = (j < n_accepted) ? draft[j] : targ[j];
                R.out.push_back(tok);
                printf("%s", common_token_to_piece(ctx_tgt.get(), tok).c_str()); fflush(stdout);
                ++n_emit;
                if (tok == llama_vocab_eos(vocab)) stop = true;
                if (feat_ok) {
                    committed.emplace_back(feat + (size_t) j * featdim, feat + (size_t)(j + 1) * featdim);
                    cap_row(committed.back());
                }
            }
            anchor = targ[n_accepted];
            const int new_past = n_past + n_accepted + 1;

            // rollback the rejected tail back to the confirmed boundary
            const int64_t g0 = ggml_time_us();
            if (greedy) {
                n_past = new_past;
            } else if (reverify) {
                // plane-INDEPENDENT oracle rollback: clear and re-decode the full confirmed prefix,
                // rebuilding committed from the fresh capture. O(n^2) but always correct and free of
                // the rollback-plane machinery. With DFLASH_NRS=0 this isolates the block-verify
                // forward pass: if reverify == greedy, the 16-token verify is causally correct and any
                // snapshot DRIFT is purely the plane store corrupting the forward/capture.
                llama_memory_clear(mem_tgt, true);
                const int total = (int) inp.size() + (int) R.out.size();   // == new_past
                llama_batch fb = llama_batch_init(total, 0, 1);
                fb.n_tokens = total;
                for (int i = 0; i < total; ++i) {
                    fb.token[i] = (i < (int) inp.size()) ? inp[i] : R.out[i - (int) inp.size()];
                    fb.pos[i] = i; fb.n_seq_id[i] = 1; fb.seq_id[i][0] = 0; fb.logits[i] = (i == total - 1);
                }
                if (llama_decode(ctx_tgt.get(), fb) != 0) LOG_ERR("reverify full redecode failed\n");
                int32_t fnt = 0, fdim = 0;
                const float * ff = llama_get_dflash_feat(ctx_tgt.get(), &fnt, &fdim);
                if (ff && fdim == featdim && fnt == total) {
                    committed.assign(total, {});
                    for (int i = 0; i < total; ++i) {
                        committed[i].assign(ff + (size_t) i * featdim, ff + (size_t)(i + 1) * featdim);
                        cap_row(committed[i]);
                    }
                }
                llama_batch_free(fb);
                n_past = new_past;
            } else {
                // snapshot rollback: seq_rm restores the recurrent compress-state snapshot as-of new_past-1
                llama_memory_seq_rm(mem_tgt, 0, new_past, -1);
                n_past = new_past;
            }
            R.t_roll += (ggml_time_us() - g0) * 1e-3;
            ++R.rounds; R.acc_total += n_accepted;

            if (n_past + G >= llama_n_ctx(ctx_tgt.get())) { LOG_WRN("context full\n"); break; }
        }
        R.wall = (ggml_time_us() - t_start) * 1e-3;
        llama_batch_free(dft_batch);
        llama_batch_free(vrf_batch);
        return R;
    };

    auto report = [&](const char * name, const Result & R) {
        printf("\n");
        LOG_INF("[%s] emitted=%zu rounds=%d tau=%.2f  %.1f tok/s  (%.0f ms)\n", name, R.out.size(),
                R.rounds, R.rounds ? (double) R.out.size() / R.rounds : 0.0,
                R.wall > 0 ? R.out.size() / (R.wall * 1e-3) : 0.0, R.wall);
        if (R.rounds) {
            const double misc = R.wall / R.rounds
                - (R.t_dft + R.t_vrf + R.t_roll + R.t_cross) / R.rounds;
            LOG_INF("[%s] per-round  drafter %.2f | verify %.2f | rollback %.2f | cross %.2f | misc %.2f ms  (accept/round %.2f)\n",
                    name, R.t_dft / R.rounds, R.t_vrf / R.rounds, R.t_roll / R.rounds,
                    R.t_cross / R.rounds, misc, (double) R.acc_total / R.rounds);
        }
    };
    auto diff = [&](const char * name, const Result & a, const Result & ref) {
        size_t n = std::min(a.out.size(), ref.out.size()), i = 0;
        while (i < n && a.out[i] == ref.out[i]) ++i;
        if (i == a.out.size() && a.out.size() == ref.out.size())
            LOG_INF("LOSSLESS: %s == greedy (%zu tokens identical)\n", name, a.out.size());
        else
            LOG_INF("DRIFT: %s diverges from greedy at token %zu / %zu\n", name, i, n);
    };

    const int   ds  = std::max(1, env_int("DFLASH_DENOISE_STEPS", 1));
    const float eps = []{ const char* e=getenv("DFLASH_ACCEPT_EPS"); return e?(float)std::atof(e):0.0f; }();

    // DFLASH_EPSWEEP: greedy ref, then snapshot at several typical-acceptance eps -> tau vs eps.
    if (env_int("DFLASH_EPSWEEP", 0)) {
        Result g = run_mode(GREEDY, 1, 1, 0.0f); report("greedy", g);
        for (float e : {0.0f, -2.0f, -4.0f, -8.0f, 0.3f, 1.0f}) {
            Result s = run_mode(SNAPSHOT, G, ds, e);
            char name[32]; std::snprintf(name, sizeof(name), e < 0 ? "snap gap=%.0f" : "snap typ=%.1f", e < 0 ? -e : e);
            report(name, s); diff(name, s, g);
        }
        llama_backend_free();
        return 0;
    }

    // DFLASH_DSWEEP: greedy ref, then snapshot at several denoising step counts -> tau vs steps.
    if (env_int("DFLASH_DSWEEP", 0)) {
        Result g = run_mode(GREEDY, 1, 1, 0.0f); report("greedy", g);
        for (int st : {1, 2, 4, 8}) {
            Result s = run_mode(SNAPSHOT, G, st, eps);
            char name[32]; std::snprintf(name, sizeof(name), "snap steps=%d", st);
            report(name, s); diff(name, s, g);
        }
        llama_backend_free();
        return 0;
    }

    // DFLASH_GSWEEP: greedy reference, then snapshot at several verify widths -> t/s vs G curve.
    if (env_int("DFLASH_GSWEEP", 0)) {
        Result g = run_mode(GREEDY, 1, ds, 0.0f); report("greedy", g);
        for (int vg : {2, 4, 8, 16}) {
            if (vg > G) break;
            Result s = run_mode(SNAPSHOT, vg, ds, eps);
            char name[32]; std::snprintf(name, sizeof(name), "snap G=%d", vg);
            report(name, s); diff(name, s, g);
        }
        llama_backend_free();
        return 0;
    }

    // single-load A/B: greedy reference, then each speculative rollback path, diffed for losslessness.
    const int nrs = env_int("DFLASH_NRS", G - 1);
    if (greedy_ref) { Result g = run_mode(GREEDY, G, ds, 0.0f); report("greedy", g); }
    else {
        Result g = run_mode(GREEDY, G, ds, 0.0f); report("greedy",   g);
        Result r = run_mode(REVERIFY, G, ds, eps); report("reverify", r); diff("reverify", r, g);
        if (nrs > 0 && !no_rollback) {
            Result s = run_mode(SNAPSHOT, G, ds, eps); report("snapshot", s); diff("snapshot", s, g);
        }
    }

    llama_backend_free();
    return 0;
}
