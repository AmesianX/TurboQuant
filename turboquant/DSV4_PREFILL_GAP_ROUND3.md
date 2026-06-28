# DSV4 Prefill Gap — Round 3 (profiler fix, BF16 TP reduce, overlap analysis)

Branch: feat/dsv4-sparse-mla-mma. Build .66 only.
libggml-cuda.so.0.13.1 md5 (.66 == .67): 2f5e1d79cae43df3efb406a1e4498aeb
Commit: d309dfc53

## (1) 256x128 tactic REVERTED
Crashes at load on sm121 (coordinator-confirmed). Default tactic = tactics.front() again
(the only stable config, ~330-370 t/s). DSV4_MOE_FUSED_TACTIC=<i> still available to sweep.
dsv4-moe-fused-run.cu:~290.

## (2) MoE phase profiler FIXED — why it didn't print before
dump() called cudaDeviceSynchronize(), which HANGS under the 2-node SPMD meta-backend (it
syncs the whole device while the executor is mid-async cross-rank dispatch -> the prefill
never completed, so the auto-dump line never flushed).
Fix (dsv4-moe-fused-run.cu, namespace fprof):
- force-drain now syncs ONLY our own cudaEvents (cudaEventSynchronize per pair), never a full
  device sync.
- tick_and_maybe_dump() holds the lock and calls dump_locked() (no re-lock, no device sync).
- prints `[DSV4_MOE_FUSED_PROF] active, will auto-dump after N calls` at STARTUP so you can
  confirm it engaged, and the dump shows the call count reached.

### CAPTURE the GEMM-vs-glue split (coordinator)
Add to the FUSED launch, both ranks:
```
DSV4_MOE_FUSED_PROF=1 DSV4_MOE_FUSED_PROF_AFTER=400
```
- At startup each rank prints `[DSV4_MOE_FUSED_PROF] active ...` (confirms engaged).
- One 13k prefill ~= 559 fused calls > 400 -> auto-dump fires mid-run:
  `[DSV4_MOE_FUSED_PROF] (auto) per-call phase totals ... ms over N calls:` then 4 lines:
  cvt_in(f32->bf16) / route(absmax+alpha) / runMoe(GEMM+sort+finalize) / cvt_out(bf16->f32).
- grep `\[DSV4_MOE_FUSED_PROF\]`. If runMoe is ~the whole cost, the GEMM is the floor and glue
  is not worth touching. If cvt_in+cvt_out+route are >~10% combined, they're the next target.

## (3) DSV4_TP_REDUCE_BF16 (gated, default OFF) — the real comm lever (VOLUME)
ggml-cuda.cu SPMD AllReduce branch (~1305): the cross-node reduce of DSV4's MoE-combine partial
runs in NATIVE type. The fused MoE writes an F32 partial -> the inter-node AllReduce ships F32 =
2x the bytes of BF16. At prefill this reduce is bandwidth-bound (7168 * UB * 4B per layer x ~45
layers, serialized). Compress F32->BF16, reduce in BF16, decompress -> HALVES wire volume.
vLLM reduces TP in BF16/FP16 by default.
GATED (default off) because BF16 reduce changes numerics (sum of 2 ranks' partials rounded to
bf16). Only engages for F32 partials >= 32768 elems (small reduces are latency-bound).

### A/B + verify (coordinator)
- Baseline: unset. Then `DSV4_TP_REDUCE_BF16=1`. Measure 13k prefill t/s both ways.
- VERIFY OUTPUT QUALITY with bf16 on: greedy completion must stay coherent (the partial is a
  residual; bf16 reduce is a small perturbation, but confirm before keeping it). If quality
  holds, this is a real prefill win on the bandwidth-bound reduce with zero overlap risk.
- Capture the reduce structure: `GGML_TP_DBG=1` prints one `[tp] ... allreduce node=... ne=[..]`
  line per reduce per forward -> tells us EXACTLY how many reduces/forward and each payload size
  (confirms the ~45 and the F32 ne). Grep `allreduce node`.

## TP COMPUTE/COMM OVERLAP — honest structural finding (NOT implemented, here's why)
The coordinator asked to overlap reduce(i) with layer i+1's independent compute. In DSV4's
topology that independent compute DOES NOT EXIST on the critical path:
- MLA KV is MIRRORED -> attention does NOT reduce. The ONLY split op per MoE layer is the MoE
  down-proj -> exactly ONE reduce/layer (confirm count with GGML_TP_DBG above).
- That reduce feeds `residual + reduced`, consumed by the NEXT layer's attn_norm -> q_a/kv_a.
  The next subgraph's FIRST op needs the reduced result. There is no mirrored/independent compute
  queued between reduce(i) and its consumer to hide it behind.
- Putting NCCL on a separate stream + event-wait before the consumer is FUNCTIONALLY IDENTICAL
  to the current same-stream ordering (the consumer still waits) — no gain, added desync risk.
Real overlap would require REORDERING the DSV4 layer graph (e.g. Megatron sequence-parallel:
reduce-scatter the MoE-down, overlap the all-gather against the next layer's attn QKV GEMM). That
is a deep change to the proven graph + the meta-backend split model — high risk, deferred unless
the BF16-volume + GEMM-floor numbers show it's the only remaining lever worth that risk.

So the contained, orthodox comm win is BF16 volume (above), not overlap. Reported honestly.

## Residual-gap framing (for after the numbers)
If, after BF16 reduce + confirming the GEMM is at its CUTLASS floor (256x128 dead on sm121,
24 tok/expert over 256 tiny experts), prefill is still far below ~1600: that remaining gap is the
STRUCTURAL 256-expert MoE GEMM efficiency vLLM's kernels achieve and ours don't on this
hardware/CUTLASS — report it as the residual with the real number, not as a bug to keep chasing.
