# DSV4 Prefill Gap — Round 2 (fused MoE is 57.5%; build contamination + tactic)

Branch: feat/dsv4-sparse-mla-mma. Build .66 only.
libggml-cuda.so.0.13.1 md5 (.66 == .67): 179816977bddfe7ae499a6c14f0fb663
Commit: 59eb1d19f

## Q1 (coordinator): is the repack per-forward? -> NO.
The sidecar->fc1/fc2 repack is guarded ONE-TIME per layer, keyed by `il` in `g_fcache`:
- dsv4-moe-fused-run.cu:352-362 `dsv4_moe_fused_run` -> `g_fcache.find(il)`; build only on miss.
- build_fused_layer (237-305): concat + 2 swizzle + mallocs + setTactic + a blocking
  `cudaStreamSynchronize` (line 293). Runs ONCE per layer lifetime, never per forward.

WHY the profile showed "layer 37..41 repacked" DURING the forward: lazy FIRST-TOUCH. The
eager (GGML_CUDA_NO_GRAPHS) profiled forward was the first to reach those layers, so their
build ran inside the timed op. The DSV4_OPPROF brackets the whole op -> those ~5 calls
fold in the full one-time build (incl. the blocking sync at :293), inflating the 76.6 ms/call
average. The other ~32 layers were already built (timed clean). So the 57.5% / "4 TF/s" is
partly (a) the eager no-graphs penalty and (b) build contamination on ~5/37 calls — NOT a
per-forward repack.

## Reconciliation with prior work (DSV4_CUTLASS_FUSED_MOE_PORT.md)
That doc (exhaustive UB sweep) already measured the fused MoE at UB=1024 -> ~340 t/s prefill
and concluded "fused MoE is DONE: correct, fast." Key prior results, DO NOT re-derive:
- UB=1024 ~340 t/s (sweet spot); UB=2048 322-327 (LOWER); UB=4096 DIED at ctx-init (the
  non-MoE ggml activation buffers hit the box mem wall, NOT the MoE workspace which is 0.56GB).
- => LARGER UB IS NOT A LEVER (explored; plateaus/regresses). The 24->96 tok/expert increase
  doesn't help because the per-expert problem stays tile-bound and the bottleneck shifts.
- Single tactic = tactics.front(), NOT autotuned. The 256x128 (356-TFLOPS) tile flagged as the
  one untested MoE lever.

So the real prefill story: the ~330 t/s baseline ~matches the prior ~340; the "5x gap vs vLLM
1595" is NOT a fused-MoE regression — it is that vLLM's per-expert GEMM packs the 256-expert
problem far more efficiently (and overlaps TP comm). The remaining MoE lever is the TACTIC.

## What changed this round
1. **DSV4_MOE_FUSED_PROF** (dsv4-moe-fused-run.cu, namespace fprof): deferred-cudaEvent timing
   of the 4 per-call phases — cvt_in (f32->bf16), route (absmax+alpha), runMoe (GEMM+sort+
   finalize), cvt_out (bf16->f32). NO per-call sync (safe under 2-node TP), auto-dump after
   DSV4_MOE_FUSED_PROF_AFTER calls (default 4000) + at exit. Build is OUTSIDE the timed scopes
   -> CLEAN steady-state GEMM-vs-glue split. This settles whether the 76.6 ms is the GEMM or the
   surrounding glue/sort/finalize.
2. **Default tactic -> CtaShape256x128x64B** (CutlassTileConfigSM120 enum=4) when getTactics()
   exposes it, else front(). DSV4_MOE_FUSED_TACTIC=<i> overrides; =-1 forces old front().

## DEPLOY / MEASURE (coordinator)
Steady-state fused sub-profile (graphs ON, real speed) — add to the FUSED launch, both ranks:
```
DSV4_MOE_FUSED_PROF=1 DSV4_MOE_FUSED_PROF_AFTER=4000
```
Send TWO ~13k prefills (first warms the repacks; the auto-dump after 4000 calls lands in the
steady region). Prints `[DSV4_MOE_FUSED_PROF] (auto) per-call phase totals` = ms/% for
cvt_in / route / runMoe / cvt_out. Expectation: runMoe (GEMM) should be ~the whole cost; if
route/cvt are >10% combined, they're a fusable target.

Tactic A/B (real prefill t/s, graphs ON, NO profiler):
- Default build now = 256x128x64B. Baseline (old front()): add `DSV4_MOE_FUSED_TACTIC=-1`.
- Sweep: `DSV4_MOE_FUSED_TACTIC=0..N` (layer-0 log prints the tactic list + index mapping).
Report prefill t/s per tactic; pick the best.

EXPECTED delta: tactic 256x128 is the one untested MoE lever (prior doc). If it helps it shows
directly in prefill t/s; if neutral, the fused MoE is confirmed at its practical ceiling and the
next lever is the serialized TP all-reduce overlap (Round 1 analysis, ggml-backend-meta.cpp:2529)
+ accepting that the per-expert GEMM efficiency gap vs vLLM is structural (256 tiny experts).

## Round 3 candidates (after numbers)
- If fprof shows route/cvt non-trivial: fuse the f32<->bf16 converts into the runMoe pro/epilogue
  (the runner can take/emit f32 via its InputType path) to drop 2 full-hidden passes/call/layer.
- TP all-reduce overlap (the other structural win vs vLLM).
- Decode->45: MTP graph-context sizing fix (ggml.c:1929) so MTP runs with fused (no sparse).
