# DSV4 Prefill Gap — Round 5 (tile the O(ub²) indexer to unblock large UB)

Branch: feat/dsv4-sparse-mla-mma. Build .66 only.
libllama.so.0.0.9732 md5 (.66 == .67): 2c3b0f366a0790f5188b858ed13032c4
(ggml-cuda unchanged this round: 60415819c832724655c9ddb9c36b954e)
Commit: 9a6d50204

## The lever (from R4, accepted): tokens-per-expert via larger UB
R4 measured the fused MoE GEMM: 18 TF/s @ 24 tok/exp (UB=1024) -> 57 @96 (UB=4096)
-> 91 @192 (UB=8192). front() optimal, no tactic lever. So the ~5x prefill win is
LARGER UB, gated only by the memory wall.

## The wall: the prefill DSA indexer is O(ub²)
deepseek4.cpp dsv4_build_indexer_scores_prefill (~1761): for the WHOLE ubatch it builds
  - score = mul_mat(k,q)        -> [n_comp, ub, n_heads]   (n_comp = ub/ratio)
  - ggml_cont(permute(score))   -> [n_heads, n_comp, ub]
both ~ n_heads * (ub/ratio) * ub * 4B TRANSIENT compute buffer PER LAYER. With the DSA
indexer n_head~64, ratio=4: at ub=8192 -> 64*2048*8192*4 = ~4.3 GB/layer. ggml reuses
buffers across layers so it's not x43, but a few GB of live transient + the rest of the
graph -> OOM before the GEMMs are compute-bound. (The fused MoE workspace is linear in UB
and tiny; the ggml compute buffer ~0.5 MiB/tok is fine. The indexer quadratic is the wall.)

## Fix: tile the indexer QUERY dim (DSV4_INDEXER_QTILE)
dsv4_build_indexer_mask_tiled_prefill (deepseek4.cpp ~1863), wired at the prefill mask site
(~2184). The top-k is INDEPENDENT PER QUERY, so we loop over query tiles of `qtile`:
slice qr/cur/pos/causal_mask to [.., qtile] -> run the SAME score->argsort_top_k->mask
pipeline -> CONCAT per-tile [n_comp, qtile] masks along tokens into the full [n_comp, ub].
Peak indexer memory: O(n_heads*n_comp*qtile) instead of O(n_heads*n_comp*ub) -> bounded by
qtile (e.g. 2048), NOT ub. index_kv (full compressed cache) + weights shared across tiles.
NUMERICALLY IDENTICAL to the non-tiled path (same per-query math, batched in slices).

GATED: DSV4_INDEXER_QTILE (default 0 = OFF = whole-ub, byte-identical to before). The
qtile>=n_tokens / qtile<=0 branch is the exact original path.

## DEPLOY / MEASURE (coordinator)
Add to the FUSED launch, both ranks, and RAISE UB:
```
DSV4_INDEXER_QTILE=2048   -ub 4096    (then try -ub 8192 with QTILE 2048)
```
Note DSV4_MOE_PREFILL_MAX should track the new ubatch (the fused workspace presize).
1. Confirm it FITS (no OOM at warmup). If a DIFFERENT buffer is now the wall, the OOM log
   names it -> report it (candidates: the attn flash scratch, the ggml compute buffer, the
   raw_window mask [ub,ub] which is ALSO O(ub²) — see "next wall" below).
2. Max UB that fits: 4096? 8192? (WATCH_MIN_GB=4 guards.)
3. Standalone MoE TF/s at the achieved tok/expert (R4 bench predicts 57 @UB4096, 91 @UB8192).
4. End-to-end 13k prefill t/s.

## EXPECTED prefill delta
The fused MoE is 57.5% of prefill at ~18 TF/s (UB=1024). At UB=4096 the MoE GEMM goes to
~57 TF/s (3.2x) -> that 57.5% slice shrinks ~3.2x; at UB=8192 ~91 TF/s (5x). Rough Amdahl:
prefill_new ≈ 0.425*T + 0.575*T/3.2 (UB=4096) ≈ 0.60*T -> ~1.66x; (UB=8192) 0.425+0.575/5
≈ 0.54*T -> ~1.85x. From ~330 t/s baseline that's ~550 (UB4096) / ~610 (UB8192) IF the MoE
were the only thing scaling and nothing else regressed. The non-MoE terms (dense FP8 GEMMs,
attention, indexer) ALSO benefit from larger UB (better GEMM amortization), so the real
number should be higher — measure it. This is the structural step toward vLLM's 1600; the
residual after this is the per-expert GEMM efficiency gap (still 91 vs vLLM's higher, but
much closer).

## ⚠ NEXT WALL candidate (report if UB=8192 still OOMs)
The RAW_WINDOW attn mask (add_mask RAW_WINDOW, n_tokens x n_tokens, ~2097) is also O(ub²):
[ub, ub] -> at ub=8192 that's 256 MB/distinct-shape (masks are dedup'd across same-shape
layers, so ~one copy, tolerable). The dense attention itself is flash (no [ub,ub] KQ
materialization), so it's O(ub) memory. If UB=8192 OOMs after the indexer tiling, the next
suspect is the ggml compute-graph reservation (scales with ub*n_nodes) or the raw mask —
the OOM log will name the buffer; tile/dedup that next.

## Code state
- DSV4_INDEXER_QTILE tiling (gated, default off). All R1-R4 work intact.
- ggml-cuda unchanged (R4 lib); only libllama rebuilt.
