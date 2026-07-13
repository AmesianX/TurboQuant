# DSV4 decode — the map at REAL context (2026-07-13). Everything before this was profiled at a toy context.

The `DSV4_THE_REAL_DECODE_MAP.md` 56 ms profile was taken at a context so short that **the indexer
top-k path was dormant** (it only turns on once `n_comp_visible > indexer_top_k = 512`, i.e. past
~2k tokens for a c4a layer). Every enemy list built on it was therefore built with the DSA pipeline
switched off. Real agent/coding contexts are 20k-100k+.

Measured here: 2x GB10, plain decode (no MTP), TP=2 over RDMA, single slot, **24 699-token prompt**.

    decode, 24.7k ctx, plain:   7.73 t/s  (129 ms/token)     <- NOT 16 t/s. Context halves it.
    warm eager GPU time/step:   81 ms over ~1365 nodes

## Where the 81 ms goes (shape-level, warm, `DSV4_OPPROF_SKIP` past the whole prefill)

| ms/step | % | op | what |
|---|---|---|---|
| **12.1** | **14.9** | `MUL_MAT(f16, 128x6400, n=1)` x21 | **indexer logits — the #1 enemy at real context** |
| 7.9 | 9.3 | `MUL_MAT_ID(f8, 4096x1024, n=8)` | MoE |
| 7.1 | 8.3 | `MUL_MAT(f8, 1024x32768, n=1)` | attn q_b |
| 6.9 | 8.2 | `MUL_MAT(f8, 8192x4096, n=1)` | attn out |
| 6.4 | 7.5 | `MUL_MAT_ID(mxfp4, 4096x1024, n=1)` | MoE gate/up |
| **4.5** | **5.3** | `MUL_MAT(bf16, 4096x129280, n=1)` | **lm_head — still BF16, 1.06 GB/token** |
| 4.2 | 5.0 | `FLASH_ATTN_EXT(D=512, nq=1)` | attention |
| 3.6 | 4.2 | `MUL_MAT(f8, 4096x2048, n=1)` | q_a/kv |
| 3.3 | 3.9 | `MUL_MAT_ID(mxfp4, 1024x4096, n=6)` | MoE down |
| **3.3** | **4.1** | `MUL_MAT(bf16, 4096x{1024,512,256})` | **compressor/indexer weights — still BF16** |
| 2.3 | 2.7 | `ARGSORT(6400)` + `ARGSORT(256)` | top-k |
| **2.3** | **2.8** | `MUL_MAT(f32, {16384x24, 4096x256})` | **hc + router — still F32** |
| ~13 | 16 | CONCAT/GET_ROWS/CONT/SET_ROWS/CPY/... | glue |

## The indexer logits: not a slow kernel, a 64x read amplification

`deepseek4.cpp:1929`
```c
k = [128, n_comp, 1]      // index KV cache — SHARED across heads (MQA)
q = [128, n_tokens, 64]   // 64 query heads
score = ggml_mul_mat(k, q);   // ggml BROADCASTS k over q's 64 head channels
```
ggml re-reads the whole `[128 x 6400]` index cache **once per head**: 64 x 1.6 MB = 102 MB per layer
per token, **2.1 GB across the 21 c4a layers**. At the measured 602 us/call that kernel is running at
**170 GB/s — it is not slow, it is reading 64x too much.** And it scales linearly with context.

**Fix (landed):** `ggml_dsv4_indexer_logits` already existed (streams K once, dots all 64 heads,
folds relu + per-head weight + head-sum into one kernel) but was wired into the PREFILL path only.
Wired into decode behind the same `DSV4_INDEXER_FUSED` gate.

    7.73 -> 8.26 t/s at 24.7k ctx (+6.9%), greedy output sha IDENTICAL.

## The measured hardware roof (do not use 273)

| | GB/s | % of the 273 GB/s spec |
|---|---|---|
| pure-read stream (measured on this box) | **243.8** | 89.3% |
| FP4 weight-only GEMV (measured on this box) | 218.5 | 80% |
| copy / triad, 128 MB | 229 / 230 | 84% |
| **our MoE grouped GEMV** | **131** | **54% of the 244 read roof** |
| our dense F8 GEMV | 214-244 | **at the roof — done, do not touch** |

FP4 dequant method is worth **0.5%** (hw `F2FP` vs lop3 bit-trick, both measured on this box). It is
not where the MoE loss is. The loss is structural: llama.cpp's `mul_mat_vec_q` launches **one CTA per
(output row x expert)** with a shared-mem reduction, giving each thread 1-2 outstanding loads. vLLM
wrote `moe_wna16.cu` precisely because of this and measured **+41% end-to-end at batch 1**.

## Corrections to earlier conclusions (all were wrong)

1. ~~"k_all concat is an O(context) copy and worth ~1 ms"~~ — **measured +0.5% = noise.** The copy is
   real (197 MB/token at 24.7k) but the step is 129 ms, so it is 1.3%. The view landed (correct,
   bit-identical, `DSV4_KV_ADJACENT`, keep it) but it is not a lever.
2. ~~"GGML_CUDA_GRAPH_OPT finds nothing because the DSV4 graph is a straight chain"~~ — the upstream
   optimizer is gated on `strstr(node->name, "attn_norm")` and a fan-out of **exactly 3**
   (`ggml-cuda.cu:4151/4163`). It never looked at our graph. Relaxing those gates is ~10 lines
   (take upstream PR #21897's `NO_ALLOC_FREE` with it, or bs>1 corrupts).
3. ~~"decode is 16 t/s"~~ — that is at a toy context. **At 24.7k it is 7.7 t/s.**
4. ~~"vLLM plain is 40-45 t/s"~~ — vLLM PR #41834's 37-42 t/s on 2xGB10 is **with MTP2 + fp8 KV**,
   i.e. speculative. The plain-vs-plain gap is not yet established. Do not chase a number nobody has
   measured plain.

## b12x: there is no C++ to port

`lukealonso/b12x` is **pure-Python CuTe DSL** kernels (SM120-only, Apache-2.0 per pyproject; no
LICENSE file in the repo). FlashMLA and DeepGEMM are real C++ but stop at **SM90/SM100** — DeepGEMM
issue #317 ("DeepSeek-V4 on SM120: paged_mqa_logits missing") was closed with no maintainer response.
**The sm_121a DSA kernels do not exist in C++ anywhere.** We were not failing to copy something; we
were trying to copy something that isn't there.

The one real route: CuTe DSL supports **AOT `export_to_c()`** → a `.h` + `.o` with an embedded fatbin,
linkable from C++ with no Python/torch at runtime. ggml-cuda already links a CUTLASS static lib
(`ggml/src/ggml-cuda/CMakeLists.txt:339`), so the seam exists. Nobody has done this; we would be first.

## Order of work (rewritten from the real profile)

1. **`DSV4_INDEXER_FUSED` on by default** — landed, +6.9%, free. (Also re-check the WMMA vs CUDA-core
   path: the WMMA kernel tiles 16x16 but decode has n_tokens=1, wasting 15/16 of the tile.)
2. **Quantize the aux weights that are still BF16/F32** — lm_head (`DSV4_LM_HEAD_F8`, exists, off),
   compressor/indexer BF16, hc/router F32. **~10 ms/step of the 81 = 12%**, all env-gated already.
3. **MoE GEMV restructure** — 131 -> ~215 GB/s. The single biggest item, a known reference design
   (`moe_wna16.cu`), and a measured 218.5 GB/s FP4 GEMV on this exact box proves the roof is reachable.
4. **The glue** (~16%) — `ggml_can_fuse_subgraph` (shape-agnostic) is the right API; PDL enrollment of
   our DSV4 kernels; group-aware `topk_moe` fusion (upstream's fires on deepseek2 for +7% but cannot
   match DeepSeek's group-limited routing).
