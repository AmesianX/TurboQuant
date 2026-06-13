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
//   header: magic"DSV4FEAT", u32 version=1, u32 n_layers, u32 n_embd, u32 n_tokens,
//           then n_layers * u32 layer_id, then n_tokens * u32 token_id
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

// eval callback: capture "ffn_out-{il}" at target layers
static bool feat_cb(struct ggml_tensor * t, bool ask, void * /*user*/) {
    const char * nm = t->name;
    if (std::strncmp(nm, "ffn_out-", 8) != 0) {
        return false; // not ours -> not interested (ask) / nothing to do (data)
    }
    const int il = std::atoi(nm + 8);
    auto it = g_feat.idx.find(il);
    if (it == g_feat.idx.end()) {
        return false;
    }
    if (ask) {
        return true; // yes, keep this node's data alive for the follow-up call
    }
    // data ready: t shape = [n_embd, n_tokens] (per-token layer hidden), expect F32
    const int64_t ne0 = t->ne[0]; // n_embd
    const int64_t ne1 = t->ne[1]; // n_tokens
    const size_t  n   = (size_t) ne0 * ne1;
    std::vector<float> buf(n);
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, buf.data(), 0, n * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        std::vector<uint16_t> h(n);
        ggml_backend_tensor_get(t, h.data(), 0, n * sizeof(uint16_t));
        for (size_t i = 0; i < n; ++i) buf[i] = ggml_fp16_to_fp32(h[i]);
    } else {
        fprintf(stderr, "dsv4-feat: WARN unexpected ffn_out type %d at layer %d (skipping)\n", (int) t->type, il);
        return true;
    }
    g_feat.n_embd = (int) ne0;
    // store transposed to [token][embd] (ggml is column-major: element (e, tok) at e + tok*ne0)
    std::vector<float> & dst = g_feat.cur[it->second];
    dst.resize((size_t) ne1 * ne0);
    for (int64_t tok = 0; tok < ne1; ++tok) {
        for (int64_t e = 0; e < ne0; ++e) {
            dst[(size_t) tok * ne0 + e] = buf[(size_t) e + (size_t) tok * ne0];
        }
    }
    return true;
}

static void write_feat(const std::string & path, const std::vector<llama_token> & toks) {
    std::ofstream f(path, std::ios::binary);
    if (!f) { fprintf(stderr, "dsv4-feat: cannot open %s\n", path.c_str()); return; }
    const uint32_t ver = 1, nl = (uint32_t) g_feat.order.size(),
                   ne = (uint32_t) g_feat.n_embd, nt = (uint32_t) toks.size();
    f.write("DSV4FEAT", 8);
    f.write((char*)&ver, 4); f.write((char*)&nl, 4); f.write((char*)&ne, 4); f.write((char*)&nt, 4);
    for (int il : g_feat.order) { uint32_t v = (uint32_t) il; f.write((char*)&v, 4); }
    for (llama_token tk : toks) { uint32_t v = (uint32_t) tk; f.write((char*)&v, 4); }
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
    params.n_batch = std::max<int>(params.n_batch, 512);

    common_init();
    llama_backend_init();
    llama_numa_init(params.numa);

    auto llama_init = common_init_from_params(params);
    llama_model   * model = llama_init->model();
    llama_context * ctx   = llama_init->context();
    if (!model || !ctx) { fprintf(stderr, "dsv4-feat: failed to load model\n"); return 1; }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    // sequences = newline-split prompt(s). params.prompt holds the file contents when -f used.
    std::vector<std::string> seqs;
    {
        std::string all = params.prompt;
        size_t p = 0;
        while (p <= all.size()) {
            size_t nl = all.find('\n', p);
            std::string line = all.substr(p, nl == std::string::npos ? std::string::npos : nl - p);
            if (!line.empty()) seqs.push_back(line);
            if (nl == std::string::npos) break; p = nl + 1;
        }
    }
    if (seqs.empty()) { fprintf(stderr, "dsv4-feat: no input sequences (use -f prompts.txt)\n"); return 1; }

    std::string mk = "mkdir -p '" + out_dir + "'"; (void) system(mk.c_str());
    fprintf(stderr, "dsv4-feat: %zu sequences, layers=%s, out=%s\n", seqs.size(), layers_str.c_str(), out_dir.c_str());

    int done = 0;
    for (size_t si = 0; si < seqs.size(); ++si) {
        std::vector<llama_token> toks = common_tokenize(ctx, seqs[si], true, true);
        if (toks.empty()) continue;
        if ((int) toks.size() > params.n_ctx) toks.resize(params.n_ctx);

        llama_memory_clear(llama_get_memory(ctx), true);
        g_feat.reset((int) toks.size());

        llama_batch batch = llama_batch_get_one(toks.data(), (int32_t) toks.size());
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "dsv4-feat: decode failed on seq %zu\n", si);
            continue;
        }
        char path[1024];
        snprintf(path, sizeof(path), "%s/feat_%06zu.bin", out_dir.c_str(), si);
        write_feat(path, toks);
        if (++done % 50 == 0) fprintf(stderr, "  %d/%zu done\n", done, seqs.size());
    }
    fprintf(stderr, "dsv4-feat: wrote %d feature files to %s\n", done, out_dir.c_str());

    llama_backend_free();
    return 0;
}
