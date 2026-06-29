# DSV4 FLASH_ATTN_EXT D=512 lever (11.4% of prefill) — root cause + port paths (2026-06-29)

## Aiden's MLA kernel is NOT borrowable on sm121 (verified)
Searched local vllm + flashinfer-src. The vLLM `FlashMLASparseBackend` (platforms/cuda.py:273-278,
selected for DSV4) calls FlashMLA `sparse_prefill_fwd` (attention/ops/flashmla.py:211-238), whose
native kernel (.deps/flashmla-src/csrc/sm100/prefill/sparse/fwd.cu) is **guarded SM90||SM100 ONLY**
(pybind.cpp:404) and built only for arch 9.0a/10.0a (flashmla.cmake:38-43). It uses Blackwell-DC
**tcgen05/UTCMMA** (sm100_mma 2x1SM cluster split of the 512-V) — instructions that **do NOT exist on
sm120/GB10**. So on real GB10 this path is non-functional / falls back. The AIDEN_1600_STACK.md D.3
"indexed-split sparse-MLA Triton prefill" (env VLLM_DEEPSEEK_V4_INDEXED_D512_SPLIT_PREFILL) is a
TARGET PLAN — grep finds ZERO hits in the vllm tree. **There is no GB10 MLA prefill kernel to port.**
The only borrowable PATTERN is fp8 dequant-in-tile-loader (sm100/decode/sparse_fp8/dequant.h:24-39):
raw LDG + in-register cvt_fp8→bf16 × scale feeding the MMA — the analogue of a TBQ per-tile dequant.

## CORRECTED root cause — K is ALREADY F16, FA is ALREADY MMA. The lever is DENSE→SPARSE.
- The DSV4 SWA raw-window cache is **force-upgraded to F16** regardless of -ctk
  (llama-kv-cache-iswa.cpp:124-128: "SWA K+V upgraded to f16 for quality"). The compressed cache is
  also F16 (memory-hybrid-iswa.cpp:240/244). So `k_all = concat(k_raw[F16], kv_comp[F16])` is F16.
- => the FA dispatcher does NOT take the TBQ→VEC branch; with F16 K, D=576/512 it routes to the
  EFFICIENT GB10-tuned MMA-MLA kernel (fattn.cu:1044-1077, case 576 at :200-206). The 11.4% is the
  genuine cost of the **DENSE** MMA attention over ALL n_comp rows. (My earlier VEC-from-tbq3 theory
  was WRONG — SWA is F16.)

## THE LEVER (already built, just disabled): DSV4_SPARSE_ATTN=1
DeepSeek-V4 IS a SPARSE-attention model (DSA): the indexer selects top-512 comp rows, and attention
should compute ONLY those — exactly Aiden's sparse-MLA. We run DENSE (all n_comp) by default.
- Our tensor-core sparse-MLA kernel `fattn-sparse-mla.cu` (bf16 WMMA, D=512, 1 CTA/query, all 64
  heads share the MQA gather; proven cos 0.999999) is FULLY WIRED end-to-end: the fused indexer
  produces `topk`, bound as src[6] (deepseek4.cpp:2861), routed to BEST_FATTN_KERNEL_SPARSE_MLA
  (fattn.cu:827-830). The dispatch comment: **"3.6-13x faster than dense + flat with context."**
- Gated by env DSV4_SPARSE_ATTN (default OFF). Engages when flash_attn && n_comp_view > top_k
  (deepseek4.cpp:2860) — i.e. once context exceeds 512 comp rows (~2k tokens), which prefill quickly
  passes. Single-stream only (supported() line 332); non-multislot prefill qualifies.
- This is Aiden's exact technique (attend top-512, not all n_comp) on OUR validated kernel. It is
  arguably MORE faithful to the model (the model is trained sparse) AND much faster, flat with ctx.

## ✅ ALREADY PROVEN (DSV4_SPARSE_MLA_ROUND6.md, this same branch feat/dsv4-sparse-mla-mma)
The tensor-core sparse-MLA kernel (fattn-sparse-mla.cu, "k4" design) was MEASURED vs a matched dense
baseline on GB10 sm121: 3.6x @ 8.5k ctx, 5.5x @ 13k, >13x @ 32k+, FLAT with context, cos 0.999998.
It is THE production kernel; round-5's "occupancy blocker" was a false alarm (dense is also 1 CTA/SM).
Round-3's 0.61x slowdown was the OLD VEC substrate — the MMA kernel (round 6) fixed it. The crossover
is ~512 visible keys = exactly the DSV4 guard (n_comp_view > top_k=512), so it wins at every real
prefill context. It's wired end-to-end and just default-OFF.

## Decision
**Test DSV4_SPARSE_ATTN=1 at -ub=3072** — it's a deploy flag, no new code, validated kernel. Expected:
FA 11.4% (dense, grows with ctx) -> sparse (top-512, flat) = the big FA cut, the largest single lever,
and Aiden's actual sparse-MLA mechanism. Verify quality (it changes attention to top-512 selection —
the DSA design; the indexer must be selecting well, which the fused indexer already does). If quality
holds, this is the FA win with ZERO kernel risk. The MMA-TBQ / FlashMLA ports are NOT needed (K is
F16, kernel exists). Aiden's FlashMLA is SM100-only and NOT borrowable on GB10 (verified above).
