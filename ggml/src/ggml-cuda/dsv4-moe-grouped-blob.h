#pragma once

// DSV4 NVFP4 sidecar blob format + host entry points. Deliberately CUTLASS-free and
// header-only POD so it can be included by:
//   - the offline converter tool (tools/dsv4-nvfp4-preconvert)
//   - the ggml-cuda implementation (dsv4-moe-grouped.cu)
//   - the engine load path (src/llama-model.cpp)
// without dragging in common.cuh / the CUTLASS template tree.

#include <cstdint>
#include <cstddef>

#ifdef __cplusplus
#include <vector>
extern "C++" {
#endif

// Blob in-memory layout (ONE layer, ONE rank), all sections back-to-back, exactly the
// concatenation the device registry pointers reference (see dsv4-moe-grouped.cuh):
//   [0] gate q (packed e2m1)  : n_expert * (n_ff_half*n_embd)/2 bytes
//   [1] up   q (packed e2m1)  : n_expert * (n_ff_half*n_embd)/2 bytes
//   [2] down q (packed e2m1)  : n_expert * (n_embd*n_ff_half)/2 bytes
//   [3] gate SFB (ue4m3 swiz) : n_expert * sfb_words_gu bytes
//   [4] up   SFB (ue4m3 swiz) : n_expert * sfb_words_gu bytes
//   [5] down SFB (ue4m3 swiz) : n_expert * sfb_words_d  bytes
//   [6] gate global (fp32)    : n_expert * 4 bytes
//   [7] up   global (fp32)    : n_expert * 4 bytes
//   [8] down global (fp32)    : n_expert * 4 bytes
struct dsv4_moe_grouped_blob_header {
    int32_t  n_expert;
    int32_t  n_embd;
    int32_t  n_ff_half;
    int32_t  sfb_words_gu;   // ue4m3 words per expert for gate/up SFB (swizzled)
    int32_t  sfb_words_d;    // ue4m3 words per expert for down SFB (swizzled)
    int32_t  _pad;
    uint64_t blob_bytes;     // total bytes of the 9 concatenated sections
};

// ---- sidecar file format (one file per rank: sidecar_rank{r}.bin) -----------
// [ dsv4_sidecar_file_header ]
// [ dsv4_sidecar_layer_entry  x n_layers ]   (table; il + file offset/size of each layer record)
// [ per-layer record x n_layers ]            (each = dsv4_moe_grouped_blob_header followed by blob)
// The rank-local AXIS_1(gate/up)/AXIS_0(down) n_ff split is BAKED into each blob offline, so the
// engine never re-splits: it just memcpy-uploads blob[il] to the registry for layer il.
#define DSV4_SIDECAR_MAGIC   0x344E5650u  /* "PVN4" little-endian-ish tag */
#define DSV4_SIDECAR_VERSION 1u

struct dsv4_sidecar_file_header {
    uint32_t magic;          // DSV4_SIDECAR_MAGIC
    uint32_t version;        // DSV4_SIDECAR_VERSION
    int32_t  rank;           // which rank this sidecar is for (0 or 1)
    int32_t  n_ranks;        // total ranks the split was computed for (TP world size)
    int32_t  n_layers;       // number of MoE layer records that follow
    int32_t  n_expert;
    int32_t  n_embd;
    int32_t  n_ff_half;      // rank-local n_ff half (== n_ff_exp / n_ranks)
    uint64_t total_bytes;    // total file size for a sanity check
};

struct dsv4_sidecar_layer_entry {
    int32_t  il;             // layer index in the model
    int32_t  _pad;
    uint64_t offset;         // byte offset (from file start) of this layer's record
    uint64_t size;           // record size = sizeof(blob_header) + blob_bytes
};

#ifdef __cplusplus
// Convert ONE layer's (already rank-local-sliced) MXFP4 expert bytes -> the exact NVFP4
// registry blob, on the HOST (no CUDA device needed). gate/up host blocks are
// [n_expert][n_ff_half * n_embd] MXFP4; down is [n_expert][n_embd * n_ff_half] MXFP4.
// Appends the raw blob bytes to `out` and fills `hdr`.
void dsv4_moe_grouped_convert_layer(const void * gate_mxfp4,
                                    const void * up_mxfp4,
                                    const void * down_mxfp4,
                                    int n_expert,
                                    int n_embd,
                                    int n_ff_half,
                                    dsv4_moe_grouped_blob_header * hdr,
                                    std::vector<uint8_t> & out);
#endif

// Upload a pre-converted NVFP4 blob (from a sidecar) into the device registry for
// layer il. NO conversion, NO MXFP4. blob points at the 9-section layout above.
void dsv4_moe_grouped_set_expert_weights_blob(int il,
                                              const struct dsv4_moe_grouped_blob_header * hdr,
                                              const void * blob,
                                              size_t blob_size);

#ifdef __cplusplus
} // extern "C++"
#endif
