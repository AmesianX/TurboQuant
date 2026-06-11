# DSV4 Performance — Status & Attack Plan (GB10)

Working doc to resume DeepSeek-V4-Flash perf work. Last updated 2026-06-11 (session 2: iq2_xs MMVQ).
Branch: `feat/deepseek4`.

## Current numbers (IQ2_XS-XL 3-split, GB10, greedy)

| Config | decode t/s | note |
|---|---|---|
| baseline f16 KV | **14.66** | 13.4 → 13.8 (fp8 QDQ fix) → 14.66 (iq2_xs mmvq + graph reuse, 2026-06-12) |
| + MTP (`--spec-type draft-mtp --spec-draft-p-min 0.75 --spec-draft-n-max 2`) | 15.5–16.2 free text / ~17 regular | accept 98–100% |
| tbq3 KV + MTP (production config) | **16.4–16.7** essay | accept 98%; 16.1 before graph-reuse fix; MTP gains are capped by single-slot thrash + verify-graph rebuilds (roadmap 1–2) |
| prefill | 160–164 t/s @14K prompt | ds4 reference: 343 |
⚠️ measure t/s only after `ss -tlnp | grep 8888` confirms YOUR pid owns the port — a wrapper-kill
(`kill $!` on the nohup pid) leaves the old server alive and silently serving your benchmark.

Launch (the production config):
```
build/bin/llama-server -m .../IQ2_XS-XL/DeepSeek-V4-Flash-IQ2_XS-XL-00001-of-00003.gguf \
  -c 16384 -ngl 999 -fa on --jinja -t 4 -ctk tbq3 -ctv tbq3 \
  --spec-type draft-mtp --spec-draft-p-min 0.75 --spec-draft-n-max 2 \
  --host 0.0.0.0 --api-key "occultsaint@X" --port 8888
```

## GRAPH REUSE FIXED (2026-06-12 ~01:00) — graphs reused 0 → 47/47

**Every DSV4 decode was rebuilding the full ~6800-node ggml graph per token** ("graphs reused = 0"):
`dsv4_graph_inputs` is registered via res->add_input but never overrode `can_reuse` → default false
vetoes reuse for every graph containing dsv4 masks. Fixed: builder records per-compress-layer
topology drivers (`add_reuse_key`: 256-padded n_comp_view, visible<=top_k branch, pad-mask branch —
phase-uniform already made everything else pos-invariant), `can_reuse` recomputes from the new
ubatch pos and compares. n_tokens>1 honestly returns false (chunk masks are pos-sized).
Result: plain f16 decode **13.8 → 14.66 t/s (+6%)**, greedy output identical, reused = 47/47.

### MTP round budget (measured, DSV4_MTP_PROF=1 instrumentation, production config, 61ms/round ≈ 2 tok)
| component | ms | note |
|---|---|---|
| verify GPU wait (tgt-embd sync in process()) | 27.8 | the actual model compute — kernel work hits this |
| unaccounted CPU | ~22 | verify graph build (now partly fixed) + server sampling/emit |
| draft() total | 9.8 | decode 5.0 (build+submit) + sample-sync 5.6; GPU inside ≈ 3.1 (vocab proj q5_K 1.9) |
| mirror decode submit | 1.6 | cheap |

### Remaining roadmap (ranked)
1. **Multi-slot graph cache**: ctx_dft alternates [draft n=1, draft n=1, process n=3] — single-slot
   gf_res_prev thrashes, so only the 2nd consecutive draft reuses. Caveat: on miss sched_reset
   invalidates the other slot's allocations → either per-slot sched (memory!) or cache built graphs
   and redo alloc only (build is the expensive part).
2. **Phase-uniform chunk graphs** (verify n=2-3 reuse + CUDA-graph eligibility): pad compressed views
   + data-driven masks for small n_tokens like the n=1 path. Unlocks reuse for the verify pass
   (biggest CPU item) AND prefill target ③ shares machinery.
3. **Draft head requant** (quality-safe — wrong drafts get rejected, never emitted): MTP head logits
   matmul is q5_K 4096x129280 = 1.9ms of every draft token; iq2/iq3 → ~0.8-1.2ms.
4. Deferred verify_h extraction (overlap mirror decode with server CPU) — small, ~1ms/round.

## Profiler (use this, not nsys)

`DSV4_KERNEL_PROF=1 build/bin/llama-completion -m <shard1> -ngl 999 -fa on -ctk f16 -ctv f16 -n 48 --temp 0 -p "Write a short essay about the history of the printing press."`
— forces CUDA graphs off, brackets every node with cudaEvents, dumps per-op table at exit.
MUL_MAT keys carry `(type,ne0xne1,n=batch)`; nodes >5ms get the graph node name appended.
nsys 2024.6 captures ZERO CUPTI events on GB10/CUDA13 — don't waste time on it.
⚠️ llama-completion defaults to sampling — ALWAYS pass `--temp 0` for comparable output.

## Decode GPU-time breakdown (48 tok, after QDQ fix, total ~4.7s)

| share | op | diagnosis |
|---|---|---|
| 21.0% | `MUL_MAT_ID(iq2_xs,4096x2048,n=1)` ×4300 | routed experts gate/up — **87 GB/s effective (peak 273)** |
| 12% | other iq2_xs (down 2048x4096 n=6, prefill n=256, …) | same kernel family — iq2_xs total ≈ 33% |
| 10.7% | `MUL_MAT(q8_0,1024x32768,n=1)` ×2064, 245µs/call | **wq_b** (MLA Q up-proj), 35.6MB/layer/token, 145 GB/s |
| 8.5% | `MUL_MAT_ID(q8_0,4096x1024,n=8)` | hash-layer experts |
| 7.3% | `MUL_MAT(q8_0,8192x4096,n=1)` | wo_b, 204 GB/s (okay-ish) |
| ~8% | CPY+CONT+GET_ROWS+SET_ROWS+CONCAT swarm | launch-bound data movement |
| 0.8% | DSV4_FP8_KV_QUANTIZE | FIXED (was 6.0%) |
| 2.2% | FLASH_ATTN_EXT(K=f16,D=512) | **attention is NOT the bottleneck — the D=512 MMA port idea is dead for DSV4** (still valid for Qwen/AMX3_1) |

`MUL_MAT(f32,128x3,n=14)` showing ~234ms = one-time cuBLAS init on first f32 GEMM. Ignore.

## NEXT TARGETS (by payoff)

### ① iq2_xs MMVQ — IN PROGRESS (2026-06-11 session 2)
- Kernel: `vec_dot_iq2_xs_q8_1` @ `ggml/src/ggml-cuda/vecdotq.cuh` (now split into `_grid` core); dispatch in `mmvq.cu`.
- **Diagnosis revised.** Microbench (test-backend-ops, DSV4 shapes added at tests/test-backend-ops.cpp "deepseek-v4-flash" block) shows the kernel itself streams at **~223 GB/s** DRAM-fed (n_used=32/77.6MB, L2-busted) — same as q8_0. The grid-lookup-bound theory was wrong; in-model slowness (~99-132 GB/s) is **cold-weight load latency under-hidden** (per-thread MLP too low; in-model q8_0 matvecs degrade the same way: wq_b 145 vs 222 isolated). ⚠️ Microbench traps: GB10 L2=24MB and test reruns reuse the same experts — footprint must be ≫24MB or numbers are L2-flattered.
- n_expert=256, n_expert_used=6 (not 8): gate/up = 2 unfused MUL_MAT_ID(n=1) ×~45 layers; down = MUL_MAT_ID(n=6) via mul_mat_vec_q_moe.
- **Gate/up fusion does NOT fire for DSV4**: per-layer swiglu clamp (`swiglu_clamp_exp`, llama-graph.cpp:1705) expands to CLAMP+SILU+CLAMP+MUL — no GGML_OP_GLU, so the MUL_MAT_ID+MUL_MAT_ID+GLU pattern never matches. Recovering it needs a new GGML_GLU_OP_SWIGLU_LIMITED (SWIGLU_OAI is close but has +1 on up and asymmetric clamp). Side quest — saves launches + y/ids reread, not the main prize.
- **SHIPPED** (in-model DSV4_KERNEL_PROF, 47 decode tok, greedy output unchanged each step):
  1. **small_k path opened for iq2_xs** (mmvq.cu `should_use_small_k`: K=4096 sat exactly on the `<` boundary, 16<16). rpb 1→4, per-thread vec_dots 1→4: gate/up 146.6 → **126.5 µs/call** (-14%).
  2. **smem staging of iq2xs_grid** in mul_mat_vec_q only (gated on rpb>1 so the 512-entry copy amortizes; `vec_dot_iq2_xs_q8_1_grid` core in vecdotq.cuh): gate/up → **106.9 µs** (cumulative **-27%**, 99→136 GB/s).
- **⚠️ MTP regime lesson (the big one).** Under MTP, gate/up verify batches run at n=2-3 → `mul_mat_vec_q_moe`, NOT the optimized ncols_dst=1 kernel; down runs at n=6×3=18 → outside mmvq entirely. So the n=1 wins only reach plain decode + draft steps; production essay went 15.79 → 16.06 (top of the old noise band). Staging the codebook in the moe kernel was tried and is a measured **+47-54% regression** at n=2-3/k=4096 (64-96-thread blocks; copy+barrier dominate) — reverted; comment in the kernel guards against re-adding.
- Tried and rejected: rpb=8 via 2*nwarps (gate/up 114µs, fewer blocks → tail imbalance); moe rows_per_block 2→4 (neutral); moe-kernel smem staging (above; in-model down n=6 liked it (-7.7%) but plain-decode-only — MTP regression wins).
- **`mul_mat_vec_q_moe` MTP regime — attempted, NO WIN (2026-06-11 late):**
  - In-model MTP prof (essay, 47×~5 verify tok) confirms the regime split: `MUL_MAT_ID(iq2_xs,4096x2048,n=1)` 12.8% (plain-decode + draft, gets the n=1 win) and `MUL_MAT_ID(iq2_xs,2048x4096,n=6)` 6.9% (down-proj, **the moe kernel** — n=6 = 2 verify tok × 3? actually ncols_dst batches the MTP candidate fan; this is the path to beat). Per-call baselines: n=1 144.4µs, n=6 149.4µs.
  - **Software pipeline (UF=4 staged x-block loads → registers → compute) REGRESSED +20%** in-model (n=6 149.4→179.5µs); the `q2[UF][rpb]`/`sc`/`dm` register arrays drop occupancy more than the extra in-flight loads recover. REG was already 80/thread. Reverted (working tree == 586c6f3d0). Code comment in mul_mat_vec_q_moe guards against retry.
  - **Lesson: this kernel is occupancy-bound, not latency-bound.** Any opt that adds registers loses. Real levers left: (a) cut REG below 64 to lift occupancy (split the fat fused n=1 kernel from the moe kernel so launch_bounds minBlocks can rise — the gate/up/bias/glu fusion template is what bloats it), (b) a SWIGLU_LIMITED GLU op to fuse gate+up and halve the n=1 launch count, (c) leave it — decode is within ~30% of the LPDDR ceiling and prefill (③) is the bigger lever now.
- Kernel facts: REG:80, 6 blocks/SM (~40% occ) for the small_k instance; iq2_xs row = 1184B (74B/256-elem block); per-call footprint = 6 experts × 2048 rows × 1184B = 14.55MB.
- Remaining headroom (136 vs ~220 isolated): software-pipelined kbx batch loads, occupancy push (force REG≤64 via launch_bounds minBlocks=8 — global attr, needs per-type kernel split), SWIGLU_LIMITED GLU op to recover gate+up fusion.
- Expert requant to q4_K is NOT an option (experts dominate 82GB; q4_K won't fit in 128GB).

### ② wq_b q8_0 GEMV (10.7%, 145 GB/s)
1024×32768 matvec: small-K wide-N shape — check mmvq q8_0 launch config for this aspect ratio. Possibly split-K or vectorized loads. Success = 145 → 200+ GB/s (+3% decode).

### ③ mid-prompt batched compressor (prefill 164 → target 300+)
First prompt ubatch takes the batched `dsv4_build_compressor_prefill` path, but every LATER ubatch goes through `dsv4_build_compressor_decode_chunk` (deepseek4.cpp ~line 933) = sequential per-token node chain. That's why -ub 1024 gave +2% only (graph arena for it was bumped to n_tokens*768 in `15c819e9e`, keep). Fix = batched formulation that consumes the carried recurrent state for non-first ubatches (phase-uniform-grade work, several hundred lines).

### ④ q8_0 small-matmul swarm (311 calls/token)
LoRA/HC fragmentation. Candidates: fuse q_a+kv (same input `cur`), wo_a 그룹 batch, or CUDA-graph-level batching. Lower priority than ①.

## Settled questions (don't redo)
- **MTP sweep**: n_max=2, p_min≈0.75 is optimal. Depth≥3 LOSES (MoE verify batch cost > marginal accept). Without p_min gate accept drops to 69% and it's slower than baseline. Per-request `speculative.n_max/p_min` do NOT reach the draft impl (launch args only).
- **tbq3 KV + MTP**: stable together, neutral at short ctx, decode holds at 14K ctx.
- **-ub 1024**: no prefill gain (bottleneck is ③, not GEMM batch).
- ~20 t/s baseline memory-bound ceiling estimate stands; after ① expect ~15.5 baseline / ~18 with MTP.

## Benchmark protocol
- essay: `Write a short essay about the history of the printing press.` max_tokens 400, temp 0 → free-text t/s + accept.
- counting: `Count from 1 to 30, comma-separated.` max_tokens 120 → regular-text ceiling.
- quality gates: matrix `[[3,4],[5,6]]×[[1,2],[7,8]] = [[31,38],[47,58]]`, Pauli 볼프강/베르너/에르빈 (`/tmp/test2.py <port>` if present).
- First request after server start = warmup (graph capture + cuBLAS init), measure from the second.
