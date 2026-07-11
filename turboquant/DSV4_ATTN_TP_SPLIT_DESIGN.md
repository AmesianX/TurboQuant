# DSV4 attention TP head-split (group granularity) — implementation design
2026-07-12. Goal: kill the mirrored dense-attention duplication (~2.7GB/token/rank of the
5.7GB dense reads) → decode 13.0 → ~15.5-16 (MTP), on the road to 45~57.
Gate: `DSV4_ATTN_SPLIT=1` (default OFF, mirrors today's behavior byte-identically).

## Verified facts this design rests on (two deep code scans, 2026-07-12)
- Attention IS per-head independent between wq_b and wo_a; the ONLY cross-head mixing is
  `dsv4_grouped_out` (deepseek4.cpp:769-796): wo_a mixes 8 heads per group (8 groups),
  wo_b mixes the 8 group outputs. → **split unit = output group = 8 heads = 4096 rows of
  wq_b output = 1024 rows of wo_b input.** 2 ranks → 4 groups (32 heads) each.
- Indexer/DSA top-k (512) is per-query-token, SHARED across all 64 heads
  (sum over indexer heads → single score → one topk → one mask [n_kv,T], no head dim).
  → indexer + compressor + mask stay MIRRORED, broadcast to local heads. (deepseek4.cpp:1861-1927)
- Latent KV cache is single-head ([512, 1, T], MLA), broadcast to all q heads in FA
  (llama-graph.cpp:2181/2203). Stays MIRRORED (TBQ3 quant unaffected).
- attn_sinks {n_head}: per-head scalar into softmax → AXIS_0-split (group-aligned).
- q rms_norm (per-head over 512), dsv4_rope_tail fwd/inv (per-head): no cross-head mixing.
- Meta backend (ggml-backend-meta.cpp):
  - MUL_MAT: AXIS_1 w × MIRRORED act → AXIS_0 act (:648); AXIS_0 w × AXIS_0 act → PARTIAL
    (:659) → subgraph boundary → ONE AllReduce (:2273-2331, :2583-2630). Reshape/view/cpy
    handlers remap split axes through shape changes (:681-758 area). No allgather primitive —
    once split, the chain must stay split until a PARTIAL reduce.
  - FLASH_ATTN_EXT (:847-862): all-MIRRORED, or ALL-of-Q/K/V AXIS_2 (standard head-split).
    **Missing: mixed branch Q=AXIS_2, K/V=MIRRORED (kv-head=1 broadcast), mask=MIRRORED,
    sinks=AXIS_0 → out AXIS_1.** Semantically exact for MLA (every head reads whole latent).
  - MUL_MAT_ID EP branch (:672-676): src0 AXIS_2 × MIRRORED act; per-rank ids remap; down
    = PARTIAL, gate/up = benign-MIRRORED lie. **Missing: aligned-group branch** (see D2).
  - No PARTIAL folding: attention adds a 2nd AllReduce/layer (16KB payload) — acceptable
    (measured all-reduce ~30-60µs → +1.3-2.6ms/token vs −11.7ms reads saved).

## Edits

### 1. Config — src/llama-model.cpp llama_meta_device_get_split_state (~:499, BEFORE MLA fallback)
Env-gated `DSV4_ATTN_SPLIT`; arch==DEEPSEEK4 only; MTP/nextn layer already early-returns
MIRRORED (:489-497) — keep that ordering.
- `blk\.\d*\.attn_q_b\.weight`      → AXIS_1, reference attn_output_b.weight
  (regex does NOT match `blk.N.indexer.attn_q_b.weight` — verified name has `indexer.` infix)
- `blk\.\d*\.attn_output_a\.weight` → AXIS_2, reference attn_output_b.weight
- `blk\.\d*\.attn_output_b\.weight` → AXIS_0 (self)
- `blk\.\d*\.attn_sinks\.weight`    → AXIS_0, reference attn_output_b.weight (move BEFORE the
  existing sinks-mirror branch at :514)
- Everything else DSV4-attention (wq_a, attn_kv, q_a_norm, indexer.*, compressor, caches) stays
  MIRRORED as today.
- Granularity (meta get_split_granularity :690-758): must be GROUP-aligned —
  wo_b AXIS_0: 1024 rows (o_lora per group); wq_b AXIS_1: 4096 (8 heads × 512);
  wo_a AXIS_2: 1 group; sinks AXIS_0: 8 heads. Add DSV4 cases keyed off the reference tensor;
  rotation must be IDENTICAL for all four tensors of a layer (same reference ⇒ same rotation).

### 2. Meta — FA mixed branch (ggml-backend-meta.cpp handle_flash_attn_ext :847)
```
if (Q==AXIS_2 && K==MIRRORED && V==MIRRORED && (mask null|MIRRORED) && (sinks null|AXIS_0)):
    assert K/V kv-head dim == 1   // MLA broadcast case only
    return AXIS_1 with Q's segments   // FA out = [Dv, n_head, T, S], head dim = 1
```
Local kernel run: q [512,T,32,S] × k/v [512,n_kv,1] (GQA broadcast), sinks [32] — standard.

### 3. Meta — grouped_out chain
Decode chain after FA: out AXIS_1 [512,64,T] → inverse rope_tail (handle_generic non-scalar,
propagates) → cont/permute (:handle_cpy/view) → reshape [4096, 8, T] → AXIS_1 (group dim) →
`mul_mat_id(wo_a_g AXIS_2, o AXIS_1-aligned, ids MIRRORED-arange)` → **new branch needed**:
- D2 (chosen, full savings): treat as ALIGNED group split. Local: weight [4096,1024,4],
  act local groups [4096,4,T], ids must become local arange 0..3 (identity routing → the ids
  slice+rebase step the EP path already does; REUSE that machinery — check its location in the
  meta compute path, it slices ids and subtracts the expert base). Out = [1024,4,T] local,
  label AXIS_1. → reshape_2d [8192,T] → AXIS_0 (:handle_reshape) →
  `mul_mat(wo_b AXIS_0, low AXIS_0)` → existing PARTIAL branch (:659) → AllReduce. DONE.
- D1 (fallback if ids rebase is hairy): wo_a via EP-style zero-fill MIRRORED-lie out, wo_b stays
  MIRRORED weight, add name-match `attn_output_b` → PARTIAL (like "down"). Loses wo_b's 16.75MB/
  layer/rank saving but needs no new ids machinery. Implement D1 first if D2 stalls; D2 after.

### 4. DSV4 custom ops — audit op_params vs local ne
rope_tail fwd/inv, q rms_norm run on local [512,32,T]: verify the CUDA kernels take head count
from tensor->ne[1] (not from a baked op_param of 64) — deepseek4.cpp:664 asserts ne[1]==n_head
at BUILD time (full graph, 64: fine); check ggml/src/ggml-cuda/dsv4-*.cu kernel grid derivation.
Same audit for FP8_KV_QUANTIZE (KV path — stays mirrored, untouched) and INDEXER_* (mirrored).

### 5. Sanity/verify plan (per clean-baseline discipline)
1. Build .66 → rsync bins → md5 both boxes.
2. `DSV4_ATTN_SPLIT=0` (unset): byte-identical behavior, decode 13.0 regression check.
3. `=1` 2-node: expect meta-backend asserts → fix the next missing handler (each abort names
   the op); iterate. Known suspects: CONT/PERMUTE between FA and reshape; SET_ROWS none (KV
   path mirrored); the multislot path (DSV4_MULTISLOT splitter) — test PARALLEL=1 first.
4. Correctness gates: greedy 2-node output == mirrored baseline output (same seed/temp0);
   then perplexity spot-check (wiki chunk) vs baseline.
5. Perf: bench_decode.sh ladder — plain, +graphs, +F8 head, +MTP. Expect plain ~12.5→?
   (dense −2.7GB/token/rank → GPU ~62ms → wall ~75-80ms ≈ 12.5-13.3 plain, then MTP).
   Then re-profile (DSV4_KERNEL_PROF) → next lever (shexp split into MoE reduce / shuffle
   fusion / MoE GEVM bw).

## Byte math (why this is worth it)
Dense f8/layer today (each rank): wq_b 33.5 + wo_a 33.5 + wo_b 33.5 + shexp 25.2 + wq_a 4.2
+ kv 2.1 ≈ 132MB → split saves (wq_b+wo_a+wo_b)/2 = 50MB/layer → 2.15GB/token/rank.
Post-split/rank/token ≈ MoE 2.15 + dense 3.0 + head 0.5 ≈ 5.7GB → @≥200GB/s ≈ 28ms GPU.
Then shexp split (fold into MoE AllReduce: needs PARTIAL-folding or accept +1 reduce) →
another 0.54GB. Ceiling after all dense levers ≈ 4.6GB ≈ 20-23ms ≈ 43-50 plain-equivalent
with MTP → the 45~57 window. Prefill 1600 is a separate phase (W4A16 GEMM 32→50-60 TFLOP/s).
