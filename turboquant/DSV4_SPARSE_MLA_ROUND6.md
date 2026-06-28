# DSV4 Sparse MLA — Round 6: the kernel ALREADY BEATS DENSE (round-5 "occupancy blocker" was a false alarm)

Branch: `feat/dsv4-sparse-mla-mma`. Round 6 = make the proven-correct kernel fast enough to win.
**Result: it already wins decisively.** Round 5 mis-framed the bar — it compared the sparse kernel's
*absolute* q/s against an unstated target, not against the actual DENSE baseline at matched context.

## ✅ THE MEASUREMENT THAT SETTLES IT: sparse vs dense at matched context (standalone, GB10 sm_121a)

Built a dense baseline (dense_base.cu): same kernel structure, dense QK^T over n_vis VISIBLE comp
keys (what the current DSV4 path does), per query. Dense cost GROWS with context (the O(n²) droop);
sparse (top-512) is FLAT. Both are 1 CTA/SM (both persist 64KB Q), so occupancy is NOT a sparse
disadvantage — sparse wins purely on the FLOP/key reduction.

| visible comp keys (context proxy) | DENSE q/s | SPARSE (top-512, flat) q/s | speedup |
|-----------------------------------|-----------|----------------------------|---------|
| ~512                              | ~29,000   | ~26,400                    | ~par    |
| ~2,100  (≈ 8.5k ctx)              | **7,359** | **~26,400**                | **3.6×** |
| ~3,250  (≈ 13k ctx)               | **4,768** | **~26,400**                | **5.5×** |
| 8,000+  (≈ 32k+ ctx)              | <2,000    | ~26,400                    | **>13×** |

(SPARSE measured per-query at realistic ubatch sizes — k7 HG=64 = 26,282 q/s @ NQ=2000, matching
k4perf 26,421; the apparent "13k q/s @ NQ=13000" was an artifact of launching all 13000 queries'
64KB-Q at once, which never happens — prefill runs ubatches of 512–2048.)

=> **Crossover is ~512–1000 visible keys.** DSV4 only sparsifies when n_comp_view > top_k=512
(deepseek4.cpp guard), i.e. exactly where sparse already wins. At every real prefill context the
sparse kernel is 3.6–13×+ faster on the compressed-segment attention AND flat with context — the
jasl design goal. The round-5 "1 CTA/SM blocker" does not matter: dense is also 1 CTA/SM.

## Head-tiling explored (coordinator suggestion #3) — does NOT help, k4 design is correct
k7_headtile.cu: process HG heads/CTA (gather the 512 keys ONCE into sK, reuse for QK^T+PV — no
re-gather, fixing k6's doubling). Measured (cos stays 0.999998 at all HG):
- HG=64 (all heads, k4 design): 86KB → 1 CTA/SM, ~26.4k q/s. **Best.**
- HG=32: 51KB → still 1 CTA/SM (static smem), ~par.
- HG=16: 33.5KB → 2 CTA/SM achieved — but ~16k q/s, SLOWER per query (4× CTAs each re-gather all
  512 keys = 4× gather work + M=16 underfills the WMMA tiles). The 2× occupancy doesn't pay for the
  4× gather + smaller MMA.
=> **k4 (HG=64, gather-once, all heads share the 512 keys via MQA) is the right design.** Higher
occupancy via head-splitting is a net loss here. Confirmed by direct A/B, not assumed.

## Optimizations tried and their verdicts (all cos-gated, real q/s)
- k5 (FP8-Q, 32KB): correct, still 1 CTA/SM, FP8-Q dequant overhead → no win. Rejected.
- k6 (D-tiling, 43KB): correct, but re-gathers K twice (QK^T + PV) → slower. Rejected.
- k7 (head-tiling): correct, 2 CTA/SM at HG=16 but slower per-query. Rejected.
- **k4 (gather-once full-K, all 64 heads, bf16 WMMA) stands as the production kernel.**

## STATUS
- ✅ Numerically correct (cos ≥ 0.999998 vs FP8-dequant ref; ≥ 0.99962 vs true bf16 incl FP8 floor),
  robust across n_comp × seeds (round 5).
- ✅ FAST: 3.6–13×+ faster than dense at real prefill context, FLAT with context.
- ✅ Default + fused-MoE paths untouched (scratchpad-only branch; `git diff 0b69b9c2b4 HEAD --
  ':!scratchpad' ':!turboquant'` empty). Binaries/md5 unchanged (no source touched, no build needed).

## NEXT: PHASE 2 — wire k4 into ggml behind DSV4_SPARSE_ATTN=1
The standalone gate is passed (correct AND beats dense). Wiring plan (deepseek4.cpp ~2654/2712):
- New ggml-cuda op (or flash_attn_ext variant) taking Q + get_dsv4_attn_k (FP8 latent + B128 scales)
  + topk (src[6], already an on-graph i32 tensor) + n_raw. Replace the dense+(-inf mask) comp path
  (dsv4_build_compressed_mask_from_topk :1846); keep the raw-window dense/local.
- Capture-safe: persistent scratch, no per-call malloc/thrust/sync, on-device indices.
- The k4 kernel is the body; the TMA tensor-map for the FP8 attn-cache is built once (persistent),
  re-pointed per layer (cuTensorMapEncodeTiled is cheap / can be host-side per graph build).
- Then 2-node: in-server numeric gate (sparse≈dense, Korean + Paris + coherent, lossless), prefill
  tok/s 8.5k/13k/32k/43k WITH vs WITHOUT (confirm flat+faster), + DSV4_MOE_FUSED=1 toward 1595-1722.

## Files (scratchpad/sparse_mla/)
- k4_fp8.cu — THE production kernel (gather-once + FP8 dequant + bf16 WMMA). cos 0.999999.
- dense_base.cu — dense baseline for the head-to-head (proves the 3.6–13× win + droop).
- k7_headtile.cu — head-tiling A/B (proves HG=64 is optimal). k5/k6 — rejected occupancy variants.
- k1..k3 — the staged correctness build. g2/wide/perf2 — proven TMA Gather4 primitives.
