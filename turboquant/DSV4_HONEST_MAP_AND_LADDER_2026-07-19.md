# DSV4 decode — honest map + split ladder (2026-07-19)

Goal (user, 2026-07-19): **prefill 1600+ / decode 45+ plain-or-MTP (50+ with MTP).**
Branch `feat/dsv4-w4a16-native-port`, fixes commit `c933fb0ec`.

## 1. Instruments are now honest (review fixes)

Multi-agent review (26 agents, all findings independently verified) found the profilers were
lying; fixed in c933fb0ec:
- STEP_OPPROF: events now OWNED by the slot that captured them (readback/replay-count only on
  that slot; reset destroys events -> next capture re-instruments). Was: stale timestamps
  re-accumulated on every replay of every slot.
- OPPROF: warmup skip discards in-flight pending pairs (cold 237ms-class outliers no longer
  drain into the "warm" table); no event recording inside an active step capture.
- decode dot cores (faithful_micro_dots.cuh): **fp32 accumulation** — the f16 chain at 2^-14
  prescale flushed O(1e-3)-activation products at the fp16 subnormal floor. Parity tests now
  cover small-activation + all-16-code cases. NOTE: the old "+2% = noise" W4A16_DECODE A/B was
  measured with the broken chain — re-measure.
- TP_REDUCE_BF16 scratch malloc failure = GGML_ABORT (was: silent per-rank F32 fallback ->
  mismatched collectives -> 2-node hang/corruption).
- build_moe_ffn: NULL guard for sidecar-skipped ffn_*_exps (clean abort, was segfault path).
- Perf-gate test init typo (i*7)%7==0: e2m1 codes 7/15 (±6) were never tested. Fixed in 8 files.

## 2. Honest decode map (2-node plain, 256K/ub2048, EP+grouped+fused+W4A16+EP_GRAPH+LM_HEAD_F8)

| config | wall ms/tok | t/s | step-time GPU ms | wall-GPU gap |
|---|---|---|---|---|
| baseline (no splits) | 89.3 | 11.2 | 86.9 | 2.4 ms (2.7%) |
| +ATTN_SPLIT | 62.9 | **15.90** | 61.1 | 1.8 ms |
| +SHEXP_SPLIT+FOLD_PARTIAL_ADD | 60.9 | **16.41** | 59.2 | 1.7 ms |

Ladder total so far: 11.2 -> 16.41 (+46.5%) from env levers alone (post review-fix binary).

**Post-split OPPROF map (probe4; 13.56 t/s WITH instrumentation — 2812 event nodes inflate the
step, 16.41 is the honest speed; RELATIVE shares are the point):** Σops 53.56ms, 77 classes.
- **DSV4_MOE_GROUPED 15.77ms (29.4%) — untouched by splits, now the #1 kernel target.**
- Splits verified in the map: q_b 1024x32768 6.91 -> 1024x16384 3.42 (halved); wo_a
  MUL_MAT_ID(n=8) 7.32 -> MUL_MAT 4096x1024 x167 calls 7.08 (regular per-rank GEMVs now).
- lm_head 2.21ms unchanged (mirrored; vocab-split candidate).
- Collectives bucket persists (~20ms class, unchanged by splits) — code target #1 alongside
  MoE GEVM tune.
- Quality sanity (streamed): coherent Korean continuation, no repetition/CJK garbage. Splits
  are quality-clean.

**Overturned conclusions:**
- The old "executor starves the GPU for 26ms" is DEAD — STEP_GRAPH already closed it (2.4ms).
  The vLLM gap lives INSIDE the graph, not in host scheduling.
- Dense F8 GEVMs are NOT inefficient kernels: ~205 GB/s per call (saturated). The cost is
  MIRRORED BYTES (both ranks read the same weights). Splitting bytes is the lever, and
  ATTN_SPLIT proved it: -25.8ms GPU (far above the +16.5% old estimate — that estimate was
  made with contaminated instruments).
- STEP_OPPROF total 65.75ms vs whole-step 86.9ms at baseline -> ~21ms is NCCL collectives
  (not per-op timed). First-class target.

Baseline per-op top (65.75ms total, 79 classes): MoE grouped 15.7 (127GB/s — tune headroom),
MUL_MAT_ID(f8,4096x1024,n=8) 7.3 (= wo_a grouped-LoRA, ids=arange identity, deepseek4.cpp:823
— COVERED by ATTN_SPLIT), q_b-class f8 GEVM 6.9, 8192x4096 f8 6.9, lm_head f8 2.4 (bf16
4.66 -> confirmed win), FA 1.5 (innocent).

## 3. Split ladder — what remains mirrored and what folds

ATTN_SPLIT covers q_b/wo_a/wo_b/sinks. SHEXP_SPLIT covers shared-expert up/gate/down with the
partial FOLDED into the MoE-down reduce (no extra collective). Deliberately mirrored: wq_a,
attn_kv, indexer, compressor (top-k + latent shared by all heads).

Key constraint (llama-model.cpp:620 comment): a non-folding dense split pays +1 AllReduce/layer
= 43 x 0.15ms = 6.5ms. So further splits must FOLD into an existing reduce or batch several
partials into one collective. NOTE: the wo_a MUL_MAT_ID (7.3ms) and q_b/wo_b GEVMs are already
covered by ATTN_SPLIT — the -25.8ms measured. Remaining candidates, in value order:
1. **Collectives ~21ms** — count them (DSV4_STEP_GRAPH_STATS / nccl debug), then: batch
   per-layer reduces, overlap via multistream inside the captured graph (vLLM recipe), and
   TP_REDUCE_BF16 for prefill-sized reduces (now crash-safe after c933fb0ec; decode reduces are
   under the 32768 floor -> decode needs count/overlap, not compression).
2. **MoE grouped GEVM 15.7ms @127GB/s** — bandwidth tune toward 200+ (doc lever ③): -6.5ms.
   Kernel-internal, no collective cost.
3. **lm_head 2.36ms** — vocab-split N/2 + gather 517KB logits to master only (sampling is
   master-side). Potential -1.1ms. Asymmetric (follower half idle) — or follower skips entirely
   and master keeps full (0 gain on master path; only worth it with the gather).
4. **wq_a / attn_kv compressor / indexer** — small (1.2 + 0.8 + 1.0ms class), top-k/latent
   shared by all heads (mirrored by design); only worth revisiting if a fold point exists.
NEXT MEASUREMENT: re-run the ladder winner with DSV4_STEP_OPPROF=1 to get the post-split
per-op table (the ATTN_SPLIT/SHEXP runs were STEP_TIME-only).

## 3b. Two diagnostics settled (2026-07-19 PM)

**MTP regression — CORRECTED DIAGNOSIS (the first "structural net loss" call was WRONG):**
plain 16.41 vs MTP 13.67 (SG on) / 13.91 (SG off) / 13.78 (GS=16) / 14.9 (GS=4, -n400).
Neither STEP_GRAPH nor GRAPH_SLOTS is the cause. The REAL numbers (first completed-response
measurement, GS=4): **τ = 2.12-2.20 (59% draft acceptance) — comfortably above the ~1.77
breakeven.** Implied round cost = 142-148ms vs ~105ms model (verify M=3 ~92 + draft ~13).
With a 105ms round and τ=2.17, MTP would do ~20.8 t/s and WIN. => there is a ~40ms/round
overhead leak, not an acceptance problem. Round anatomy: verify(target,M=3) + process()
mirror-decode(ctx_dft) + n_max sequential draft() decodes(ctx_dft) — up to 4 llama_decode
calls/round; suspects = per-call graph build/alloc on ctx_dft ("graphs reused = 0" class),
host-side fixed overhead, DFlash-era checkpoint-cache full-forward leak. DSV4_MTP_PROF
buckets (tgt-embd/mirror-decode/extract + draft decode/sample/embd + graph builds) localize
it. ALSO: measurement methodology note — cold first-request probes under-measure MTP (more
graph shapes to warm than plain); always measure request 2+.
Lesson recorded: GRAPH_SLOTS=16 at 256K/ub2048 OOMs the box (memwatch killed the server at
3GB free) — launcher's own ~2GB/slot warning stands; GS=4 is the measurement ceiling here.

**Decode MoE GEVM is at the scalar-FFMA compute floor, NOT bandwidth** (bench:
turboquant/dsv4_dec_gevm_bench.cu, 7 controlled variants):
| variant | GB/s-equiv | note |
|---|---|---|
| V0 current vec16 | 73.6 | shipping kernel |
| V2 no-ALU raw uint4 sum | 246 | pure weight-stream bandwidth CEILING |
| V3 2x uint4 ILP | 73.1 | no gain — not latency/ILP bound |
| V4 bit-trick half2 | 37.6 | SLOWER — __hfma2 half-rate on GB10 |
| V5 4-way accumulator | 73.9 | no gain — not accumulator-chain bound |
| V6 e2m1 LUT (shared) | 35.4 | SLOWER — LDS latency > the ALU it saves |
The 3.3x gap (73.6 vs 246) is pure instruction throughput: per-nibble dequant + FFMA is
~50 instr / 32 weight bytes vs V2's ~8. No scalar trick closes it (all tried, all fail or
regress). The ONLY structural fix is the tensor-core MMA path (the W4A16 GEMM the port built)
— but at M=1 decode the m16n8k16 MMA wastes 15/16 rows, so the win is bounded. This is the
same physics that caps vLLM/b12x decode at ~40. => MoE GEVM 15.77ms is largely IRREDUCIBLE at
M=1; do NOT spend more here. Reallocate to the collectives bucket.

## 4. Arithmetic to target

15.9 t/s = 62.9ms. Ladder projections (honest, each needs measurement):
shexp-split -3~5, MoE GEVM bw tune 127->200 -6.5, indexer split -3.5, lm_head -1.1,
collective count/overlap -5~8 => ~40-45ms => **22-25 t/s plain**. MTP τ~2 => **44-50**.
τ is the gatekeeper for 45+/50+ — after the ladder, MTP depth/acceptance work (vLLM uses
MAC=12-class deep speculation; our SPEC currently caps draft-n at 2).

Prefill 1600+ stays a separate track: SM12x NVFP4 GEMM authoring (prefill map 2026-07-16:
compute-bound, fused port exhausted at +1.3%).

## 5. Resume

Server launcher: `tp-serve/tp-w4a16.sh` (now forwards STEP_GRAPH/STEP_OPPROF/STEP_TIME).
Probe: scratchpad probe -> `grep print_timing /tmp/tp_MASTER.log`. Known-good measurement env:
PORT=8081 CTX=262144 UB=2048 PARALLEL=1 SPEC="" DSV4_EP=1 DSV4_MOE_GROUPED=1 DSV4_MOE_FUSED=1
DSV4_MOE_SIDECAR=.../nvfp4_sidecar_ep DSV4_MOE_W4A16_DECODE=1 DSV4_EP_DECODE_GRAPH=1
DSV4_LM_HEAD_F8=1 DSV4_ATTN_SPLIT=1 [+DSV4_SHEXP_SPLIT=1 DSV4_FOLD_PARTIAL_ADD=1]
DSV4_STEP_GRAPH=1 DSV4_STEP_TIME=1.
