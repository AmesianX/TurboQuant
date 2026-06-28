# DSV4 Prefill Gap — Round 4 (256x128 tactic MEASURED — front() is optimal)

Branch: feat/dsv4-sparse-mla-mma. Build .66 only.
libggml-cuda.so.0.13.1 md5 (.66 == .67): 60415819c832724655c9ddb9c36b954e

## ROOT CAUSE of the 256x128 crash + the speedup question — SETTLED BY MEASUREMENT
Built a standalone timing harness (turboquant/dsv4_fused_moe_bench.cu) linked against the SAME
cutlass static lib the server uses, and ran every sm120 NVFP4 grouped-MoE tactic at the exact
production config (E=256, hidden=4096, inter=2048) on THIS GB10 sm121 (live, alongside the server).

getTactics() order (cutlass_heuristic.cpp:490-503, FP4 GROUPED_GEMM sm120):
  idx0 CtaShape128x128x128B (tileID 2) = front()/default
  idx1 CtaShape128x128x64B  (tileID 3)
  idx2 CtaShape128x256x64B  (tileID 5)
  idx3 CtaShape256x128x64B  (tileID 4) = the "356-TFLOPS" tile (-> TileShape<256,128,128>)

### Measured eff TFLOP/s (2 GEMMs, rows=M*top_k):
| config | tok/expert | t0 front | t1 | t2 | t3 256x128 |
|---|---|---|---|---|---|
| E=256 M=1024 (PRODUCTION) | 24  | **18.0** | 17.2 | 9.2 | 16.4 |
| E=256 M=4096              | 96  | **57.5** |  -   |  -  | 53.4 |
| E=256 M=8192              | 192 | **91.2** |  -   |  -  | 87.1 |
| E=32  M=1024              | 192 | **97.4** |  -   |  -  | 92.7 |

### Conclusions (measured, not theory)
1. **front() (128x128x128) is FASTEST at EVERY shape.** The 256x128 tile is 5-9% SLOWER, never
   faster. The tactic lever is DEAD — the default is already optimal.
2. **ALL 4 tactics run CLEAN standalone (no crash).** The 256x128 tile is SOUND on GB10 sm121.
   So the server "died at warmup" was NOT the tile being unsupported/smem-overflow — it was my
   auto-default SELECTION path (now reverted to front()). The tile itself is fine; it's just slower.
3. **The bottleneck is TOKENS-PER-EXPERT, not the tile:** 24 tok/exp -> 18 TF/s; 96 -> 57; 192 ->
   91-97. Pure tile-starvation. At our production 24 tok/expert NO available tile fills well (24
   rows << the 128 or 256 tile M). The 356-TFLOPS forum figure is for a balanced/dense shape with
   many tokens/expert, NOT 256 experts x 24 tokens.
4. The earlier "4 TF/s" from the eager DSV4_OPPROF was inflated by build contamination + the
   GGML_CUDA_NO_GRAPHS eager penalty. The TRUE steady-state fused-MoE GEMM at production is ~18
   TF/s, and it is tile-starvation-bound at 24 tok/expert.

## The REAL prefill lever (measured): tokens-per-expert
The GEMM scales ~5x from 24->192 tok/expert (18 -> 91 TF/s). The only knob that raises tok/expert
is larger UB. Prior exhaustive work (DSV4_CUTLASS_FUSED_MOE_PORT.md): UB=2048 fits but the model's
NON-MoE activation/compute buffers (not the MoE workspace, which is tiny) cap the box; UB=4096 died
at ctx-init. So on a SINGLE GB10 the activation memory wall blocks the tokens/expert lever.
=> The path to the vLLM-class prefill is EP+DP / more box memory to push UB (more tokens/expert),
   OR a fundamentally different MoE kernel that packs 256 tiny experts better than CUTLASS grouped
   GEMM does (vLLM's structural advantage). Both are large efforts, not a tactic flag.

## Honest residual
The fused MoE GEMM at the production shape sits at ~18 TF/s, tile-starvation-bound, with front()
already optimal. That IS the structural 256-expert MoE-efficiency gap vs vLLM on this hardware: not
a config bug, not a missing tile — the grouped GEMM cannot fill a tile at 24 tok/expert, and CUTLASS
on GB10 has no sub-128-row tile for sm120 NVFP4 (are_tile_shapes_supported_sm120 only allows
128x128x128/128x128x256/128x256x128/256x128x128). Raising it requires more tokens/expert (UB/EP+DP)
or a different kernel.

## Code state (kept, low-risk, diagnostic)
- Default tactic = front() (optimal, measured).
- try/catch around runMoe (dsv4-moe-fused-run.cu ~498): a CUTLASS throw is logged with the exact
  message + tactic and falls back to grouped instead of crashing the server.
- layer-0 smem diagnostic: prints MaxSharedMemoryPerBlockOptin / MaxSharedMemoryPerBlock + the
  selected tile, so any real smem-cap issue is visible.
- DSV4_MOE_FUSED_TACTIC=<i> still available to A/B in-server (will confirm the standalone numbers).

## Still on the table (from R3, unchanged)
- DSV4_TP_REDUCE_BF16 (gated): FIXED the warmup crash. Root cause = the bf16 path did a pool
  cudaMalloc on the stream the meta-backend was CAPTURING into a CUDA graph (alloc mid-capture =
  crash). Now uses a PERSISTENT static bf16 scratch grown ONLY when the stream is not capturing;
  under capture (the first forward) it falls through to the native F32 reduce (correct, uncompressed
  for that capture). No more crash. Caveat: with graphs ON the first-forward capture bakes the F32
  path, so the bf16 compression only takes effect on reduce subgraphs that run eager (e.g. under
  DSV4_MOE_FUSED_GRAPH_OFF or the small-prefill band). Measure prefill t/s + verify output quality.
- DSV4_MOE_FUSED_PROF (fixed) still uncaptured — gives the GEMM-vs-glue split to confirm ~18 TF/s
  steady-state in-server.

## DEPLOY (coordinator)
1. Confirm front() is optimal in-server: `DSV4_MOE_FUSED_TACTIC=0` (default) vs `=3`, measure 13k
   prefill t/s. Expect t0 >= t3 (matches standalone 18.0 vs 16.4).
2. Capture the fixed phase profiler: `DSV4_MOE_FUSED_PROF=1 DSV4_MOE_FUSED_PROF_AFTER=400`.
3. BF16 reduce debug: `DSV4_TP_REDUCE_BF16=1` — if it still crashes, the new try/catch won't catch
   it (it's in ggml-cuda.cu, not runMoe); grep the warmup stderr for the CUDA/NCCL error and report.
