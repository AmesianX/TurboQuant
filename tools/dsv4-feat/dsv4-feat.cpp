// dsv4-feat — extract DeepSeek-V4-Flash teacher hidden states for DFlash drafter training.
//
// DFlash conditions its block-diffusion drafter on hidden features from a fixed set of target
// layers (paper: 5 layers uniformly sampled). This tool runs DSV4 as the teacher over each input
// sequence (single prefill forward) and captures the per-token layer-output hidden state
// ("ffn_out-{il}", shape [n_embd, n_tokens]) at the requested layers via the ggml eval callback —
// no graph-structure change needed (the layer outputs are already named by llm_graph_context::cb).
//
// Output: a simple binary feature file per the format below, consumable by the PyTorch drafter
// training (speculators-style). One file holds, for one input sequence:
//   header: magic"DSV4FEAT", u32 version=2, u32 n_layers, u32 n_embd, u32 n_tokens,
//           then n_layers * u32 layer_id, then n_tokens * u32 token_id,
//           then n_tokens * u32 greedy_id  (v2: DSV4's own argmax = target label for pos i+1)
//   data:   n_layers blocks, each n_tokens*n_embd float32 (row-major [token][embd])
//
// Usage:
//   dsv4-feat -m <dsv4.gguf> --layers 2,12,22,32,40 -f <prompts.txt> -o <out_dir> [common flags]
// Each non-empty line of prompts.txt becomes one sequence -> <out_dir>/feat_<N>.bin

#include "common.h"
#include "arg.h"
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <fstream>

// ---- capture state (single-threaded, one decode at a time) ----
struct feat_ctx {
    std::set<int>            target;     // layer ids to capture
    std::vector<int>         order;      // sorted layer ids -> output order
    std::map<int,int>        idx;        // layer id -> index in cur
    int                      n_embd = 0;
    int                      n_tokens = 0;
    // per-layer captured data for the CURRENT sequence: [layer_index][token*n_embd]
    std::vector<std::vector<float>> cur;
    void reset(int n_tok) {
        n_tokens = n_tok;
        cur.assign(order.size(), {});
    }
};
static feat_ctx g_feat;

// eval callback: capture the RESIDUAL-STREAM hidden the drafter must condition on.
// DSV4 uses hyper-connections (n_hc parallel residual streams), so the per-layer residual the model
// carries is "hc_ffn_post-{il}" with shape [n_embd, n_hc, n_tokens] — NOT "ffn_out" (the FFN sub-block
// output, which is just a SUMMAND; conditioning on it probes the next token at ~1.3% = useless).
// EAGLE/DFlash condition on the residual-stream hidden; here we collapse the n_hc streams to [n_embd]
// (sum = the effective residual of hyper-connections). Tensor name overridable via DSV4_FEAT_TENSOR.
static bool feat_cb(struct ggml_tensor * t, bool ask, void * /*user*/) {
    static const std::string want = []{ const char* e=getenv("DSV4_FEAT_TENSOR"); return std::string(e?e:"hc_ffn_post"); }();
    const char * nm = t->name;
    const size_t pl = want.size();
    if (std::strncmp(nm, want.c_str(), pl) != 0) {
        return false;
    }
    // tensors with il>=0 are named "name-<il>"; il<0 tensors (e.g. result_norm) have NO suffix.
    int il;
    if (nm[pl] == '\0')      il = -1;                  // exact match, no suffix -> il=-1
    else if (nm[pl] == '-')  il = std::atoi(nm + pl + 1);
    else                     return false;
    auto it = g_feat.idx.find(il);
    if (it == g_feat.idx.end()) {
        return false;
    }
    if (g_feat.cur.size() != g_feat.order.size()) {
        return false; // before reset() (warmup) -> skip
    }
    if (ask) {
        return true;
    }
    // residual stream: ne0=n_embd, ne1=n_hc, ne2=n_tokens (ffn_out fallback: ne1=n_tokens, no n_hc)
    const int64_t ne0  = t->ne[0];                       // n_embd
    const bool    hc   = (t->ne[2] > 1);                 // 3D = hyper-connection residual
    const int64_t nhc  = hc ? t->ne[1] : 1;              // n_hc
    const int64_t ntok = hc ? t->ne[2] : t->ne[1];       // n_tokens
    const size_t  n    = (size_t) ne0 * nhc * ntok;
    std::vector<float> buf(n);
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, buf.data(), 0, n * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        std::vector<uint16_t> h(n);
        ggml_backend_tensor_get(t, h.data(), 0, n * sizeof(uint16_t));
        for (size_t i = 0; i < n; ++i) buf[i] = ggml_fp16_to_fp32(h[i]);
    } else {
        fprintf(stderr, "dsv4-feat: WARN unexpected %s type %d at layer %d (skipping)\n", want.c_str(), (int) t->type, il);
        return true;
    }
    g_feat.n_embd = (int) ne0;
    // collapse n_hc by SUM -> [n_embd] per token, stored as [token][embd].
    // column-major element (e, h, tok) at e + h*ne0 + tok*ne0*nhc
    std::vector<float> & dst = g_feat.cur[it->second];
    dst.assign((size_t) ntok * ne0, 0.0f);
    for (int64_t tok = 0; tok < ntok; ++tok) {
        for (int64_t h = 0; h < nhc; ++h) {
            const size_t base = (size_t) h * ne0 + (size_t) tok * ne0 * nhc;
            float * out = &dst[(size_t) tok * ne0];
            for (int64_t e = 0; e < ne0; ++e) out[e] += buf[base + e];
        }
    }
    return true;
}

// format v2: adds a per-token "greedy_id" section (DSV4's own argmax = target label for training/
// tau eval), so the drafter is trained/scored against what the target actually predicts — no
// separate autoregressive generation needed. greedy[i] = argmax(logits at position i) = target's
// greedy token for position i+1. (v1 had no greedy section.)
static void write_feat(const std::string & path, const std::vector<llama_token> & toks,
                       const std::vector<llama_token> & greedy) {
    std::ofstream f(path, std::ios::binary);
    if (!f) { fprintf(stderr, "dsv4-feat: cannot open %s\n", path.c_str()); return; }
    const uint32_t ver = 2, nl = (uint32_t) g_feat.order.size(),
                   ne = (uint32_t) g_feat.n_embd, nt = (uint32_t) toks.size();
    f.write("DSV4FEAT", 8);
    f.write((char*)&ver, 4); f.write((char*)&nl, 4); f.write((char*)&ne, 4); f.write((char*)&nt, 4);
    for (int il : g_feat.order) { uint32_t v = (uint32_t) il; f.write((char*)&v, 4); }
    for (llama_token tk : toks)   { uint32_t v = (uint32_t) tk; f.write((char*)&v, 4); }
    for (llama_token tk : greedy) { uint32_t v = (uint32_t) tk; f.write((char*)&v, 4); }
    for (size_t l = 0; l < g_feat.order.size(); ++l) {
        const auto & d = g_feat.cur[l];
        f.write((const char*) d.data(), (std::streamsize)(d.size() * sizeof(float)));
    }
}

int main(int argc, char ** argv) {
    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_IMATRIX)) {
        return 1;
    }
    // --layers and -o are passed via params (kv-overrides / prompt-file) — parse from env for simplicity:
    // (kept minimal: layers via env DSV4_FEAT_LAYERS, out dir via env DSV4_FEAT_OUT, prompts = params.prompt file)
    const char * layers_env = getenv("DSV4_FEAT_LAYERS");
    const char * out_env    = getenv("DSV4_FEAT_OUT");
    std::string layers_str = layers_env ? layers_env : "";
    std::string out_dir    = out_env ? out_env : "./dsv4_feat";
    if (layers_str.empty()) {
        fprintf(stderr, "dsv4-feat: set DSV4_FEAT_LAYERS=2,12,22,32,40 (target layer ids) and DSV4_FEAT_OUT=<dir>\n");
        return 1;
    }
    { // parse layers
        std::set<int> s; size_t p = 0;
        while (p < layers_str.size()) {
            size_t c = layers_str.find(',', p);
            int il = std::atoi(layers_str.substr(p, c == std::string::npos ? std::string::npos : c - p).c_str());
            s.insert(il);
            if (c == std::string::npos) break; p = c + 1;
        }
        g_feat.target = s;
        g_feat.order.assign(s.begin(), s.end());
        for (size_t i = 0; i < g_feat.order.size(); ++i) g_feat.idx[g_feat.order[i]] = (int) i;
    }

    // wire the eval callback BEFORE context creation
    params.cb_eval = feat_cb;
    params.cb_eval_user_data = nullptr;
    params.embedding = false;
    params.warmup = false; // no empty warmup run (cb fires before reset() otherwise)
    params.n_batch = std::max<int>(params.n_batch, 512);
    // NOTE: do NOT crank n_ubatch — the physical compute buffer scales with it (n_ubatch=2048 tried
    // to alloc ~136GB). We instead cap each sequence to n_ubatch so it is a SINGLE ubatch: this both
    // bounds memory AND keeps the feature callback correct (a multi-ubatch split would fire feat_cb
    // once per ubatch and the per-sequence buffer would be overwritten by the tail ubatch).

    common_init();
    llama_backend_init();
    llama_numa_init(params.numa);

    auto llama_init = common_init_from_params(params);
    llama_model   * model = llama_init->model();
    llama_context * ctx   = llama_init->context();
    if (!model || !ctx) { fprintf(stderr, "dsv4-feat: failed to load model\n"); return 1; }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    // sequences = prompt(s) split on a separator. params.prompt holds the file contents (-f).
    // Default separator is newline (one prompt per line). For chat-template data with embedded
    // newlines, set DSV4_FEAT_SEP to a sentinel (e.g. a record-separator line) the prep script
    // joins sequences with, so multi-line conversations survive intact.
    const char * sep_env = getenv("DSV4_FEAT_SEP");
    const std::string sep = (sep_env && *sep_env) ? std::string(sep_env) : std::string("\n");
    std::vector<std::string> seqs;
    {
        std::string all = params.prompt;
        size_t p = 0;
        while (p <= all.size()) {
            size_t nl = all.find(sep, p);
            std::string seg = all.substr(p, nl == std::string::npos ? std::string::npos : nl - p);
            // trim a single trailing '\r' (CRLF) only for the newline separator
            if (sep == "\n" && !seg.empty() && seg.back() == '\r') seg.pop_back();
            if (!seg.empty()) seqs.push_back(seg);
            if (nl == std::string::npos) break; p = nl + sep.size();
        }
    }
    if (seqs.empty()) { fprintf(stderr, "dsv4-feat: no input sequences (use -f prompts.txt)\n"); return 1; }

    std::string mk = "mkdir -p '" + out_dir + "'"; (void) system(mk.c_str());
    fprintf(stderr, "dsv4-feat: %zu sequences, layers=%s, out=%s\n", seqs.size(), layers_str.c_str(), out_dir.c_str());

    const int n_vocab = llama_vocab_n_tokens(vocab);
    int done = 0;
    for (size_t si = 0; si < seqs.size(); ++si) {
        std::vector<llama_token> toks = common_tokenize(ctx, seqs[si], true, true);
        if (toks.empty()) continue;
        const int cap = std::min<int>(params.n_ctx, params.n_ubatch);

        // GREEDY GENERATION (DSV4_FEAT_GEN=n): DFlash trains on the TARGET's OWN rollout, not
        // teacher-forced corpus text. So first let DSV4 greedily continue the prompt for n tokens
        // (pass 1), then extract features over prompt+rollout (pass 2 = the existing code below).
        // On a greedy rollout, greedy[i]==toks[i+1] by construction -> block offsets >=2 become
        // learnable (teacher-forced text made them unlearnable; see speculators/z-lab DFlash).
        static const int n_gen = []{ const char* e=getenv("DSV4_FEAT_GEN"); return e?atoi(e):0; }();
        if (n_gen > 0) {
            // truncate prompt so prompt+gen fits one ubatch for the pass-2 single-shot extraction
            const int max_prompt = std::max<int>(8, cap - n_gen);
            if ((int) toks.size() > max_prompt) toks.resize(max_prompt);
            llama_memory_clear(llama_get_memory(ctx), true);
            // prefill prompt (logits on last token)
            llama_batch pb = llama_batch_get_one(toks.data(), (int32_t) toks.size());
            if (llama_decode(ctx, pb) != 0) { fprintf(stderr, "dsv4-feat: gen prefill failed seq %zu\n", si); continue; }
            for (int g = 0; g < n_gen && (int) toks.size() < cap; ++g) {
                const float * lg = llama_get_logits_ith(ctx, -1); // last token's logits
                if (!lg) break;
                int best = 0; float bv = lg[0];
                for (int v = 1; v < n_vocab; ++v) { if (lg[v] > bv) { bv = lg[v]; best = v; } }
                if (best == llama_vocab_eos(vocab)) break;
                toks.push_back((llama_token) best);
                llama_batch gb = llama_batch_get_one(&toks.back(), 1);
                if (llama_decode(ctx, gb) != 0) break;
            }
        }
        // cap to a single ubatch (and n_ctx): keeps all-position logits + feature capture in one pass
        if ((int) toks.size() > cap) toks.resize(cap);
        const int nt = (int) toks.size();

        llama_memory_clear(llama_get_memory(ctx), true);
        g_feat.reset(nt);

        // manual batch with logits requested at EVERY position so we can read the target's
        // argmax (greedy next-token) per position in the same prefill.
        llama_batch batch = llama_batch_init(nt, 0, 1);
        batch.n_tokens = nt;
        for (int i = 0; i < nt; ++i) {
            batch.token[i] = toks[i];
            batch.pos[i] = i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = 1;
        }
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "dsv4-feat: decode failed on seq %zu\n", si);
            llama_batch_free(batch);
            continue;
        }
        // greedy[i] = argmax over vocab of logits at position i (= target's greedy token for i+1)
        std::vector<llama_token> greedy(nt, 0);
        for (int i = 0; i < nt; ++i) {
            const float * lg = llama_get_logits_ith(ctx, i);
            if (!lg) { greedy[i] = toks[i]; continue; }
            int best = 0; float bv = lg[0];
            for (int v = 1; v < n_vocab; ++v) { if (lg[v] > bv) { bv = lg[v]; best = v; } }
            greedy[i] = best;
        }
        llama_batch_free(batch);

        char path[1024];
        snprintf(path, sizeof(path), "%s/feat_%06zu.bin", out_dir.c_str(), si);
        write_feat(path, toks, greedy);
        if (++done % 50 == 0) fprintf(stderr, "  %d/%zu done\n", done, seqs.size());
    }
    fprintf(stderr, "dsv4-feat: wrote %d feature files to %s\n", done, out_dir.c_str());

    llama_backend_free();
    return 0;
}
