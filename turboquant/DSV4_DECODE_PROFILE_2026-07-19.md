# DSV4 decode profile @ 16.80 t/s — the empirical map to 45+

Config (binary 9832, feat/dsv4-mtp-fold e9df2712a): 2-node TP, FP4 nsparks model, 256K ctx,
UB2048, EP1 + sidecar + grouped + fused + W4A16_DECODE + LM_HEAD_F8 + ATTN_SPLIT + SHEXP_SPLIT
+ FOLD_PARTIAL_ADD, PARALLEL=1, GS4, SPEC="" (plain). Measured plain 16.80 t/s (deterministic).

Profiled with DSV4_STEP_GRAPH+STEP_OPPROF+STEP_TIME (instrumentation drops live t/s to 13.54,
but the per-op DISTRIBUTION is what matters). whole-token GPU = 71 ms; Σ per-op = 53.6 ms.

## Per-op breakdown (per token, 77 op-classes, top of 53.6 ms)
| op | ms | % | calls/tok | note |
|---|---|---|---|---|
| DSV4_MOE_GROUPED(M=1)          | 15.46 | 28.8% | 42.3x | routed MoE — memory says at scalar-FFMA compute floor |
| **collectives (GPU − Σop)**    | ~17.4 | ~24%  |  —    | cross-node AllReduce at subgraph boundaries, not op-timed |
| MUL_MAT f8 4096x1024           |  7.15 | 13.3% | 167x  | ANOMALY: ~4x/layer small F8 GEVM — fusion/redundancy suspect |
| MUL_MAT f8 4096x4096           |  4.02 |  7.5% | 42.3x | per-layer dense F8 projection |
| MUL_MAT f8 1024x16384          |  3.42 |  6.4% | 42.3x | per-layer (shexp up / ff) |
| MUL_MAT f8 4096x129280         |  2.20 |  4.1% | 1x    | lm_head (F8, already halved) |
| MUL_MAT bf16 4096x1024         |  1.69 |  3.2% | 41.3x | |
| MUL_MAT f32 16384x24           |  1.21 |  2.3% | 84.7x | |
| FLASH_ATTN_EXT D=512 nq=1      |  1.18 |  2.2% | 42.3x | |
| (norms/rope/argsort/getrows/…) | rest  |       |       | |

## Reading it — the road to 45 (22 ms/token)
Buckets: MoE 28.8% + collectives ~24% + dense-F8 (7.15+4.02+3.42+1.69+…≈17) ~30% + lm_head 4% + attn 2%.

- To hit 45 t/s we must cut ~37 ms of the 59.5 ms wall (16.80). Even zeroing ALL collectives
  (~14 ms real) AND halving all dense F8 (~7 ms) only reaches ~26 t/s. **45 is impossible without
  a faster MoE decode kernel** — MoE alone is ~13 ms real and at this kernel's floor. vLLM does
  46.4 on the SAME two boxes because its sm120 W4A16 MoE-decode kernel has a HIGHER floor. So the
  #1 lever for the goal is the core MoE-decode kernel (the W4A16 native port's remaining bulk),
  NOT MTP/fold (fold is ±7% around plain and a closed chapter).

## Ordered levers (best ROI first)
1. **MoE decode kernel → vLLM sm120 parity** (biggest, hardest, the repo's purpose). Current
   grouped M=1 = 15.5 ms/28.8% at "compute floor" for THIS kernel; vLLM's is faster. This is the
   only lever that can reach 45.
2. **Collectives ~17 ms / 24%** (task #10). In-graph AllReduce reduction / overlap with compute.
   Split ladder already collapsed some (FOLD_PARTIAL_ADD). Ceiling win ~+20-30% → ~20-22 t/s.
3. **The 167x MUL_MAT f8 4096x1024 = 7.15 ms / 13.3%** (~4x/layer small F8 GEVM). High call count
   smells like a per-head/per-group loop that could batch or fuse into one GEMM per layer. Identify
   the source (grouped_out wo_a? indexer head-shared K re-read per memory project_dsv4_real_context_map?)
   and fuse. Ceiling ~+6-8%.
4. Dense F8 projections mirrored across ranks — check any still-mirrored (unsplit) projection and
   TP-split it (ATTN_SPLIT/SHEXP_SPLIT already did attention + shared expert).

## Note
Profiling t/s (13.54) is NOT the real speed — DSV4_STEP_OPPROF puts CUDA events around every op.
Real plain = 16.80. Turn prof OFF for any speed claim.
