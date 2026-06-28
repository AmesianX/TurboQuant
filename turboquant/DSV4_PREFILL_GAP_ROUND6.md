# DSV4 Prefill Gap — Round 6 (the batch≠ubatch lever: VERDICT + wall-finder)

Branch: feat/dsv4-sparse-mla-mma. Build .66 only.
libllama: build/bin/libllama.so.0.0.9734  md5 aa6e04152c7cac37eedc531d724cc51e
llama-server md5 aaf5a44e0d1d6a257df7f70e8c543fef
(ggml-cuda UNCHANGED this round — only llama-context.cpp rebuilt; copy libllama + llama-server to .67.)

## The user's insight, stated precisely
BATCH (-b/n_batch) is VRAM-free; UBATCH (-ub/n_ubatch) drives activation VRAM. The MoE's
tokens-per-expert is the only prefill lever (fused MoE = 57.5% of compute, tile-starved at
24 tok/expert @ ub=1024 → 18 TF/s; → ~91 TF/s @ 192 tok/expert = ub=8192). So get the MoE
compute-bound from a LARGE batch, NOT a large ubatch.

## OPTION B (cross-ubatch MoE staging) — VERDICT: FUNDAMENTALLY INCOMPATIBLE with ggml. Honest.
The design "buffer routed (expert-id, token-act) pairs from K consecutive ubatches into a
persistent arena, fire ONE grouped GEMM at K·ub tokens, scatter back" CANNOT work in this
engine, for a structural reason — not a difficulty:

  * Each ubatch is an INDEPENDENT build→alloc→set_inputs→graph_compute cycle
    (llama-context.cpp process_ubatch:1406; the do/while at :1984 pulls one ubatch at a time
    from mctx->get_ubatch()). The whole forward for ubatch_k — all 43 layers, attention,
    MoE, residual adds, final logits — RUNS TO COMPLETION before ubatch_{k+1} even has a graph.
  * Inside that one graph, layer L's MoE output feeds layer L+1's residual add IMMEDIATELY
    (src/llama-graph.cpp build_moe_ffn:1670 returns moe_out, consumed in-place). There is no
    way to suspend a graph between layer L and L+1 to wait for K FUTURE ubatches' tokens —
    the future ubatches don't exist yet (their graphs aren't built).
  * So the MoE op at layer L of ubatch_k can only ever SEE ubatch_k's tokens. To make the MoE
    see K·ub tokens in one GEMM, the forward itself must carry K·ub tokens — which is exactly
    raising -ub. There is no token set to "accumulate across" because the forward is strictly
    sequential per-ubatch with no cross-ubatch overlap window.

Deferring the scatter-back is equally impossible: layer L+1..42 of ubatch_k need the MoE
result in the SAME graph_compute. A deferred scatter would leave the residual stream of
those layers reading stale/zero MoE output → wrong logits. Not a correctness-tunable; the
dataflow forbids it.

⇒ Option B is rejected on architecture, not effort. (HARD RULE: 정석만, but the orthodox
answer here is to reject an unimplementable design and take the one ggml DOES allow.)

## The lever that IS reachable (and is what the user's insight actually unlocks): IN-GRAPH TILING
The correct reading of "batch≠ubatch, make the MoE M large without growing activation VRAM"
inside ggml is: **run -ub LARGE so the MoE M = ub is large (compute-bound), but TILE every
O(ub) / O(ub²) buffer in the rest of the forward so PEAK VRAM stays small.** The MoE GEMM is
naturally M-batched (one call, M=ub tokens → M·U/E tok/expert); its workspace is LINEAR in M
and tiny (getWorkspaceSize, dsv4-moe-fused-run.cu:485). What blocks large ub is the
ATTENTION-side quadratic transients, not the MoE.

This is the same thing vLLM does (large logical batch, tiled attention), adapted to ggml's
per-ubatch graph: the "large batch" the user wants IS a large -ub, made VRAM-safe by tiling.

### The walls, by size (per layer, prefill, ub=8192, n_comp=2048, indexer n_head≈64):
1. ★ DSA indexer score+transpose: O(n_head·n_comp·ub) ≈ 4.3 GB/layer F32 transient.
   → ALREADY TILED in R5 (DSV4_INDEXER_QTILE, deepseek4.cpp:1879 builder, :2194 call site):
   per-query-independent top-k tiled over the query dim → peak O(n_head·n_comp·qtile).
   This was THE multi-GB wall. With it tiled, the remaining transients are 1–2 orders of
   magnitude smaller (below).
2. raw_window mask [ub,ub] F16 ≈ 128 MB; indexer mask [n_comp,ub] F32 ≈ 64 MB;
   assembled attn_mask [ub+n_comp,ub] F16 ≈ 168 MB. These are MASK INPUTS to flash-attn
   (FA needs the full mask), dedup'd across same-shape layers → ~ONE live copy each →
   ~0.36 GB total live, NOT ×43. Tolerable in the ~40 GB headroom.
3. Flash-attention itself: ggml_flash_attn_ext (build_attn_mha, src/llama-graph.cpp:2195) is
   O(ub) memory — it never materializes the [ub, n_kv] KQ matrix. NOT a wall.
4. Fused MoE workspace: LINEAR in M (tok_cap = ub), tiny; auto-presized from -ub via the
   server publishing DSV4_MOE_PREFILL_MAX = n_ubatch at startup (tools/server/server.cpp:108).
   NOT a wall.

⇒ After R5's indexer tiling, NO single multi-GB wall remains. ub=4096 and very likely 8192
should FIT. The open question was a MEASUREMENT, not a missing structural fix — which is what
R6 adds.

## R6 deliverable: the wall-finder probe (so the coordinator gets a real number, not a guess)
Added DSV4_PREFILL_VRAM_PROBE (llama-context.cpp:1474, right after ggml_backend_sched_alloc_graph).
On each NEW high-water prefill ubatch width it prints the per-backend compute-buffer
reservation (ggml_backend_sched_get_buffer_size) + the TOTAL + KiB/token. This is THE buffer
that scales with -ub and is the OOM wall. So a single large prefill at ub=4096 then ub=8192
prints exactly how big the compute buffer got, naming the wall instead of guessing.
  * env-gated, fires once per new width, zero compute effect.
  * Pairs with DSV4_INDEXER_QTILE: run the probe with/without tiling to SEE the indexer
    quadratic collapse in the number.

## DEPLOY / MEASURE (coordinator) — graph the lever
Both ranks, FUSED launch, raise -ub with the indexer tiled and the probe on:
```
DSV4_MOE_FUSED=1 DSV4_INDEXER_QTILE=2048 DSV4_PREFILL_VRAM_PROBE=1 \
  -b 8192 -ub 4096        # then -ub 8192 (keep -b ≥ -ub; -b is VRAM-free)
```
(server auto-sets DSV4_MOE_PREFILL_MAX = n_ubatch; WATCH_MIN_GB=4 watchdog STAYS ON.)
Report from the probe line:
1. compute-buffer MiB @ ub=1024 vs 4096 vs 8192 (does it stay sub-headroom? KiB/token slope).
2. Max -ub that FITS (4096? 8192?) under the 4 GB watchdog floor.
3. Standalone fused MoE TF/s at the achieved tok/expert (R4 bench: 57 @ub4096, 91 @ub8192).
4. End-to-end 13k prefill t/s vs the ub=1024 baseline.

## EXPECTED prefill delta (Amdahl, MoE 57.5% @ 18 TF/s baseline)
ub=4096 → MoE ~57 TF/s (3.2×): 0.425·T + 0.575·T/3.2 ≈ 0.60·T → ~1.66×.
ub=8192 → MoE ~91 TF/s (5×):   0.425·T + 0.575·T/5   ≈ 0.54·T → ~1.85×.
Non-MoE terms (dense FP8 GEMMs, attention) also amortize better at large ub, so the real
number should beat these. From the ~330 t/s baseline: ~550 (ub4096) / ~610 (ub8192) floor.
Residual gap to vLLM's 1600 is then the per-expert GEMM efficiency (91 vs higher), a separate
kernel-tactic problem — not a structural one.

## FALLBACK if ub=8192 STILL OOMs after the indexer tiling (Option C / EP)
The probe will name the buffer. If it's the masks (item 2), tile the raw_window + final
attn_mask the same per-query way (FA is query-independent) — bounds them to O(qtile). If it's
the ggml graph reservation (n_nodes·ub), that's linear and just needs a slightly smaller ub.
EP (each node holds 128 experts → 2× tok/expert at the same ub) remains the no-tiling
structural ~2× ceiling, but it needs ncclAllGather plumbing and is strictly weaker than the
ub lever — only pursue it if tiling genuinely can't reach ub=4096 (it should).

## Code state
- llama-context.cpp: DSV4_PREFILL_VRAM_PROBE wall-finder (gated, default off, no compute effect).
- All R1–R5 work intact (indexer qtile, fused MoE, BF16 TP reduce, sparse-MLA).
- ggml-cuda UNCHANGED (R5 lib). Only libllama + llama-server rebuilt → copy both to .67.
