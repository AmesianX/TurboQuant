// OFFLINE MXFP4 -> NVFP4 pre-converter for the DSV4 grouped-GEMM MoE path.
//
// Reads a DeepSeek-V4 GGUF, and for EACH MoE layer's 3 routed-expert tensors
// (ffn_{gate,up,down}_exps.weight, MXFP4) and EACH rank r in [0, n_ranks):
//   1. slices THIS rank's local n_ff half EXACTLY as the meta backend tensor-split
//      does at load (gate/up = AXIS_1 split of dim1=n_ff; down = AXIS_0 split of
//      dim0=n_ff), baking the split into the sidecar so the engine never re-splits;
//   2. runs the SAME MXFP4->NVFP4 conversion the device registry uses
//      (dsv4_moe_grouped_convert_layer) to get the exact registry blob bytes;
//   3. appends a [blob_header][blob] record to sidecar_rank{r}.bin.
//
// The engine (DSV4_MOE_SIDECAR=<dir>, with DSV4_MOE_GROUPED) then memcpy-uploads
// each blob straight into the device registry -- NO conversion, NO MXFP4 on device.
//
// Build: target llama-dsv4-nvfp4-preconvert (links libllama -> libggml + libggml-cuda).
//
// Usage:
//   llama-dsv4-nvfp4-preconvert --model <gguf> --out <dir> [--n-ranks 2]
//
// Produces <dir>/sidecar_rank0.bin ... sidecar_rank{n_ranks-1}.bin.

#include "ggml.h"
#include "gguf.h"
#include "ggml-cuda/dsv4-moe-grouped-blob.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cerrno>
#include <string>
#include <vector>
#include <regex>
#include <algorithm>

#include <fcntl.h>
#include <unistd.h>

// MXFP4 block: 1 byte E8M0 exponent + 16 bytes (32 e2m1 nibbles). 17 bytes / 32 elems.
static const int    MX_QK    = 32;
static const size_t MX_BYTES = 17;

// Streamed expert tensor: shape from the no_alloc gguf metadata, raw MXFP4 bytes
// pread() straight from the file region (NEVER load the whole 160GB model at once).
struct ExpertTensor {
    std::string name;
    std::vector<uint8_t> data;       // raw MXFP4 bytes for THIS tensor only
    int64_t ne0 = 0, ne1 = 0, ne2 = 0;
};

// Is this layer's gate-expert tensor MXFP4 (i.e. a convertible MoE layer)? The MTP/nextn draft
// layer (deepseek4 nextn_predict_layers) ships Q4_K experts and must be SKIPPED, not converted.
// Returns: 1 = MXFP4 (convert), 0 = present but other type (skip), -1 = missing.
static int layer_is_mxfp4(gguf_context * gc, const std::string & gate_name) {
    int64_t tid = gguf_find_tensor(gc, gate_name.c_str());
    if (tid < 0) return -1;
    return gguf_get_tensor_type(gc, tid) == GGML_TYPE_MXFP4 ? 1 : 0;
}

// pread the full raw bytes of one gguf tensor into out.data using its file offset.
// fd: open fd of the gguf file. base_off: gguf_get_data_offset(gc).
static bool stream_tensor(int fd, gguf_context * gc, ggml_context * mc, size_t base_off,
                          const std::string & name, ExpertTensor & out) {
    int64_t tid = gguf_find_tensor(gc, name.c_str());
    if (tid < 0) { fprintf(stderr, "ERROR: tensor %s not found\n", name.c_str()); return false; }
    if (gguf_get_tensor_type(gc, tid) != GGML_TYPE_MXFP4) {
        fprintf(stderr, "ERROR: %s is not MXFP4 (type=%d)\n", name.c_str(), (int) gguf_get_tensor_type(gc, tid));
        return false;
    }
    ggml_tensor * t = ggml_get_tensor(mc, name.c_str()); // no_alloc -> ne valid, data null
    if (!t) { fprintf(stderr, "ERROR: %s missing in ctx\n", name.c_str()); return false; }

    const size_t nbytes   = gguf_get_tensor_size(gc, tid);
    const size_t file_off = base_off + gguf_get_tensor_offset(gc, tid);
    out.name = name;
    out.ne0 = t->ne[0]; out.ne1 = t->ne[1]; out.ne2 = t->ne[2];
    out.data.resize(nbytes);

    size_t done = 0;
    while (done < nbytes) {
        ssize_t r = pread(fd, out.data.data() + done, nbytes - done, (off_t)(file_off + done));
        if (r < 0)  { fprintf(stderr, "ERROR: pread %s: %s\n", name.c_str(), strerror(errno)); return false; }
        if (r == 0) { fprintf(stderr, "ERROR: short read on %s (%zu/%zu)\n", name.c_str(), done, nbytes); return false; }
        done += (size_t) r;
    }
    return true;
}

// elem (i,j,e) -> byte span helpers: MXFP4 is contiguous row-major [ne0, ne1, ne2].
// A contiguous run of N elements starting on a 32-aligned boundary is (N/32)*17 bytes.
static inline size_t mx_bytes_for(int64_t n_elem) {
    if (n_elem % MX_QK != 0) { fprintf(stderr, "ERROR: %lld not %d-aligned\n", (long long) n_elem, MX_QK); abort(); }
    return (size_t)(n_elem / MX_QK) * MX_BYTES;
}

int main(int argc, char ** argv) {
    std::string model, outdir;
    int n_ranks = 2;
    int max_layers = -1; // debug: process only the first N MoE layers (-1 = all)
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : std::string(); };
        if (a == "--model" || a == "-m") model = next();
        else if (a == "--out" || a == "-o") outdir = next();
        else if (a == "--n-ranks") n_ranks = atoi(next().c_str());
        else if (a == "--max-layers") max_layers = atoi(next().c_str());
        else { fprintf(stderr, "unknown arg: %s\n", a.c_str()); return 1; }
    }
    if (model.empty() || outdir.empty()) {
        fprintf(stderr, "usage: %s --model <gguf> --out <dir> [--n-ranks 2] [--max-layers N]\n", argv[0]);
        return 1;
    }

    // Load ONLY the GGUF metadata + tensor infos (no_alloc=true -> tiny ctx, NO 160GB blob).
    // Tensor shapes (ne) are valid; raw bytes are streamed per-tensor via pread() below.
    ggml_context * mc = nullptr;
    gguf_init_params gp{};
    gp.no_alloc = true;
    gp.ctx = &mc;
    gguf_context * gc = gguf_init_from_file(model.c_str(), gp);
    if (!gc || !mc) { fprintf(stderr, "ERROR: failed to load %s\n", model.c_str()); return 1; }

    const int fd = open(model.c_str(), O_RDONLY);
    if (fd < 0) { fprintf(stderr, "ERROR: open %s: %s\n", model.c_str(), strerror(errno)); return 1; }
    const size_t base_off = gguf_get_data_offset(gc);

    // Discover MoE layers: any blk.N.ffn_gate_exps.weight present (+ up + down). KEEP ONLY layers
    // whose experts are MXFP4 -- the MTP/nextn draft layer (deepseek4 nextn_predict_layers) ships
    // Q4_K experts and is SKIPPED (it stays on the engine's normal mul_mat_id path, not the sidecar).
    std::regex re_gate("blk\\.(\\d+)\\.ffn_gate_exps\\.weight");
    std::vector<int> moe_layers;
    int n_skipped_non_mxfp4 = 0;
    int64_t ntensors = gguf_get_n_tensors(gc);
    for (int64_t t = 0; t < ntensors; t++) {
        std::string nm = gguf_get_tensor_name(gc, t);
        std::smatch m;
        if (!std::regex_match(nm, m, re_gate)) continue;
        int il = atoi(m[1].str().c_str());
        int kind = layer_is_mxfp4(gc, nm);
        if (kind == 1) {
            moe_layers.push_back(il);
        } else {
            n_skipped_non_mxfp4++;
            int64_t tid = gguf_find_tensor(gc, nm.c_str());
            fprintf(stderr, "[preconvert] SKIP layer %d: %s is type %d (not MXFP4) -- stays on normal path\n",
                il, nm.c_str(), (int) gguf_get_tensor_type(gc, tid));
        }
    }
    std::sort(moe_layers.begin(), moe_layers.end());
    if (moe_layers.empty()) { fprintf(stderr, "ERROR: no MXFP4 ffn_gate_exps tensors found\n"); return 1; }

    // Sanity from the first layer: only the gate + down SHAPES are needed (stream just those two,
    // tiny vs a full layer; the main loop re-streams per layer anyway).
    ExpertTensor g0, d0;
    {
        int il = moe_layers.front();
        char nb[256];
        snprintf(nb, sizeof(nb), "blk.%d.ffn_gate_exps.weight", il);
        if (!stream_tensor(fd, gc, mc, base_off, nb, g0)) { return 1; }
        snprintf(nb, sizeof(nb), "blk.%d.ffn_down_exps.weight", il);
        if (!stream_tensor(fd, gc, mc, base_off, nb, d0)) { return 1; }
    }
    // gate/up: [n_embd, n_ff_exp, n_expert]; down: [n_ff_exp, n_embd, n_expert].
    const int n_embd   = (int) g0.ne0;
    const int n_ff_exp = (int) g0.ne1;
    const int n_expert = (int) g0.ne2;
    if ((int) d0.ne0 != n_ff_exp || (int) d0.ne1 != n_embd || (int) d0.ne2 != n_expert) {
        fprintf(stderr, "ERROR: down dims [%lld,%lld,%lld] != expected [%d,%d,%d]\n",
            (long long) d0.ne0, (long long) d0.ne1, (long long) d0.ne2, n_ff_exp, n_embd, n_expert);
        return 1;
    }
    if (n_ff_exp % n_ranks != 0) { fprintf(stderr, "ERROR: n_ff_exp %d not divisible by n_ranks %d\n", n_ff_exp, n_ranks); return 1; }
    const int n_ff_half = n_ff_exp / n_ranks;
    if ((n_ff_half * n_embd) % MX_QK != 0) { fprintf(stderr, "ERROR: rank slice not 32-aligned\n"); return 1; }

    if (max_layers >= 0 && max_layers < (int) moe_layers.size()) {
        moe_layers.resize(max_layers);
    }
    const int n_layers = (int) moe_layers.size();

    fprintf(stderr, "[preconvert] model=%s  MXFP4 MoE layers=%d%s  (skipped %d non-MXFP4, e.g. MTP/nextn)  "
        "n_expert=%d n_embd=%d n_ff_exp=%d -> n_ff_half=%d  n_ranks=%d\n",
        model.c_str(), n_layers, (max_layers >= 0 ? " (capped by --max-layers)" : ""),
        n_skipped_non_mxfp4, n_expert, n_embd, n_ff_exp, n_ff_half, n_ranks);

    // ---- per-rank slicing helpers (must match ggml-backend-meta set_tensor exactly) ----
    // gate/up (AXIS_1, [n_embd, n_ff_exp, n_expert]): per expert, rank r owns the contiguous
    // n_ff rows [r*n_ff_half : (r+1)*n_ff_half] -> a contiguous byte block within the expert.
    auto slice_gu = [&](const ExpertTensor & T, int r, std::vector<uint8_t> & out) {
        const size_t expert_elems  = (size_t) n_embd * n_ff_exp;
        const size_t half_elems     = (size_t) n_embd * n_ff_half;       // contiguous per expert
        const size_t expert_bytes  = mx_bytes_for((int64_t) expert_elems);
        const size_t half_bytes     = mx_bytes_for((int64_t) half_elems);
        const size_t off_in_expert  = mx_bytes_for((int64_t) r * half_elems);
        out.resize((size_t) n_expert * half_bytes);
        for (int e = 0; e < n_expert; e++) {
            const uint8_t * src = T.data.data() + (size_t) e * expert_bytes + off_in_expert;
            memcpy(out.data() + (size_t) e * half_bytes, src, half_bytes);
        }
    };
    // down (AXIS_0, [n_ff_exp, n_embd, n_expert]): per expert there are n_embd rows of length n_ff_exp;
    // rank r owns elements [r*n_ff_half : (r+1)*n_ff_half] of EACH row. So per expert we copy n_embd
    // sub-slices of n_ff_half elems, producing a [n_ff_half, n_embd] expert block (== down with n_ff
    // replaced by n_ff_half) -- exactly the rank-local simple tensor layout.
    auto slice_down = [&](const ExpertTensor & T, int r, std::vector<uint8_t> & out) {
        const size_t row_bytes      = mx_bytes_for((int64_t) n_ff_exp);   // one full row
        const size_t half_row_bytes = mx_bytes_for((int64_t) n_ff_half);  // rank slice of a row
        const size_t off_in_row     = mx_bytes_for((int64_t) r * n_ff_half);
        const size_t expert_bytes   = (size_t) n_embd * row_bytes;
        const size_t out_expert     = (size_t) n_embd * half_row_bytes;
        out.resize((size_t) n_expert * out_expert);
        for (int e = 0; e < n_expert; e++) {
            const uint8_t * sbase = T.data.data() + (size_t) e * expert_bytes;
            uint8_t       * dbase = out.data() + (size_t) e * out_expert;
            for (int row = 0; row < n_embd; row++) {
                memcpy(dbase + (size_t) row * half_row_bytes,
                       sbase + (size_t) row * row_bytes + off_in_row,
                       half_row_bytes);
            }
        }
    };

    // Open ALL rank sidecars at once and write them in a SINGLE pass over the layers, so each
    // 160GB-model expert tensor is streamed from disk exactly ONCE (not once per rank).
    int rc = 0;
    std::vector<FILE *> fout(n_ranks, nullptr);
    std::vector<dsv4_sidecar_file_header> fh(n_ranks);
    std::vector<std::vector<dsv4_sidecar_layer_entry>> table(n_ranks);
    const long data_off = (long) sizeof(dsv4_sidecar_file_header)
                        + (long)(n_layers * sizeof(dsv4_sidecar_layer_entry));
    for (int r = 0; r < n_ranks; r++) {
        std::string path = outdir + "/sidecar_rank" + std::to_string(r) + ".bin";
        fout[r] = fopen(path.c_str(), "wb");
        if (!fout[r]) { fprintf(stderr, "ERROR: cannot open %s for write\n", path.c_str()); rc = 1; }
        fh[r] = dsv4_sidecar_file_header{};
        fh[r].magic = DSV4_SIDECAR_MAGIC; fh[r].version = DSV4_SIDECAR_VERSION;
        fh[r].rank = r; fh[r].n_ranks = n_ranks; fh[r].n_layers = n_layers;
        fh[r].n_expert = n_expert; fh[r].n_embd = n_embd; fh[r].n_ff_half = n_ff_half;
        table[r].assign(n_layers, dsv4_sidecar_layer_entry{});
        if (fout[r]) fseek(fout[r], data_off, SEEK_SET); // reserve header + table; backfilled at end
    }

    std::vector<uint8_t> sg, su, sd; // rank-local sliced MXFP4 (reused)
    for (int li = 0; li < n_layers && !rc; li++) {
        int il = moe_layers[li];
        char nb[256];
        ExpertTensor G, U, D;
        snprintf(nb, sizeof(nb), "blk.%d.ffn_gate_exps.weight", il); if (!stream_tensor(fd, gc, mc, base_off, nb, G)) { rc = 1; break; }
        snprintf(nb, sizeof(nb), "blk.%d.ffn_up_exps.weight",   il); if (!stream_tensor(fd, gc, mc, base_off, nb, U)) { rc = 1; break; }
        snprintf(nb, sizeof(nb), "blk.%d.ffn_down_exps.weight", il); if (!stream_tensor(fd, gc, mc, base_off, nb, D)) { rc = 1; break; }

        for (int r = 0; r < n_ranks; r++) {
            slice_gu(G, r, sg);
            slice_gu(U, r, su);
            slice_down(D, r, sd);

            dsv4_moe_grouped_blob_header bh{};
            std::vector<uint8_t> blob;
            dsv4_moe_grouped_convert_layer(sg.data(), su.data(), sd.data(),
                                           n_expert, n_embd, n_ff_half, &bh, blob);

            long rec_off = ftell(fout[r]);
            fwrite(&bh, sizeof(bh), 1, fout[r]);
            fwrite(blob.data(), 1, blob.size(), fout[r]);
            table[r][li].il = il; table[r][li]._pad = 0;
            table[r][li].offset = (uint64_t) rec_off;
            table[r][li].size   = sizeof(bh) + blob.size();
            if (r == 0 && li == 0) {
                fprintf(stderr, "[preconvert] layer %d: per-rank record = %zu bytes "
                    "(blob_header %zu + blob %zu)\n", il, sizeof(bh) + blob.size(), sizeof(bh), blob.size());
            }
        }
        fprintf(stderr, "[preconvert] layer %d done (%d/%d)\n", il, li + 1, n_layers);
    }

    for (int r = 0; r < n_ranks; r++) {
        if (!fout[r]) continue;
        if (!rc) {
            fh[r].total_bytes = (uint64_t) ftell(fout[r]);
            fseek(fout[r], 0, SEEK_SET);
            fwrite(&fh[r], sizeof(fh[r]), 1, fout[r]);
            fwrite(table[r].data(), sizeof(table[r][0]), table[r].size(), fout[r]);
            fprintf(stderr, "[preconvert] wrote %s/sidecar_rank%d.bin  (%llu bytes, %d layers, rank %d/%d)\n",
                outdir.c_str(), r, (unsigned long long) fh[r].total_bytes, n_layers, r, n_ranks);
        }
        fclose(fout[r]);
    }

    close(fd);
    gguf_free(gc);
    ggml_free(mc);
    return rc;
}
