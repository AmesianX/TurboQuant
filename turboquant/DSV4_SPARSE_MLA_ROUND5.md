# DSV4 Sparse MLA Attention — Round 5: the standalone tensor-core kernel is NUMERICALLY CORRECT

Branch: `feat/dsv4-sparse-mla-mma` (off `feat/dsv4-cutlass-fused-moe`, which has the working
fused MoE committed at `0b69b9c2b4`). Goal of round 5 = the de-risk's recommendation: build the
TMA-gather + FP8-dequant + WMMA tensor-core sparse-MLA kernel **STANDALONE with a synthetic
numeric gate** before any ggml wiring. This is the multi-week core that round-4 designed but did
not implement.

## ✅ PHASE 1 DONE: standalone kernel numerically correct on GB10 sm_121a

Staged build (scratchpad/sparse_mla/), each stage gated against a dense-over-selected reference
(the top-512 IS the model's designed sparsity, so sparse-over-selected == dense + (-inf mask)):

| stage | file | what | gate (cos vs ref) |
|-------|------|------|-------------------|
| ref   | ref.h | f32 dense-over-selected reference + FP8-E4M3-B128 quant model | — |
| k1    | k1_scalar.cu | scalar GPU baseline (idx addressing, online softmax) | 1.000000 |
| k2    | k2_wmma.cu | bf16 WMMA 16x16x16 QK^T + PV, flash online softmax | 0.999999 |
| k3    | k3_tma.cu | + TMA **Gather4** K-loop (2×256-wide tiles for D=512) | 0.999999 |
| k4    | k4_fp8.cu | + **FP8-E4M3-B128 dequant** in the tile loader (real DSV4 cache) | 0.999999 |

**k4 = the full kernel.** TMA Gather4 K-loop + FP8-E4M3-B128 dequant tile loader + bf16 WMMA
tensor-core QK^T/PV + flash online-softmax. MQA: 1 CTA per query token, all 64 heads share the
512 gathered keys (gathered ONCE), V==K latent (D=512).

Numeric gate, **robust across n_comp ∈ {512,1024,4096,16384} × seeds {1,7,99}**:
- vs **FP8-dequant** reference: **cos ≥ 0.999998** (isolates the kernel — gather+MMA+softmax are
  essentially exact; the 2×256 gather seam, the FP8 dequant, and the D=512/64-head MMA fragment
  layout are all CORRECT).
- vs **bf16 true** reference: **cos ≥ 0.99962** (this includes the FP8 quantization floor itself,
  which is the model's own designed storage precision). Exceeds the ≥0.999 gate either way.

=> The three things round-4 flagged as the hard correctness risks are SOLVED:
   (a) the D=512 MQA MMA fragment layout, (b) the FP8 dequant in the tile loader,
   (c) the 2×256 gather seam. All validated on real GB10 sm_121a hardware.

## HW constraints PINNED this round (drove the design; correct earlier assumptions)
- **GB10 sm_121 dynamic-smem cap = 99KB** (sharedMemPerBlockOptin), NOT 227KB (that's sm_100).
  A 128KB launch returns "invalid argument" and silently yields zeros — ALWAYS check
  `cudaGetLastError()` after the launch (this cost a debugging round).
- **GB10 = 48 SMs, 100KB smem/SM, 1536 threads/SM.** (Not 132 SMs.)
- TMA gather4 destination must be **128B-aligned**: use a `__shared__ alignas(128)` STATIC staging
  buffer (dynamic-smem offset alignment was not enough → "misaligned address"). Matches the proven
  g2/wide/perf2 microbenches (re-validated this round: gather correct, 256-wide OK, 0.12µs/512-row).

## ⏳ PHASE 1.5 BLOCKER (honest): the kernel is OCCUPANCY-BOUND, not yet faster than dense
Correctness is done; **absolute speed is not yet a win** and must not be wired to the server yet.

- k4 budget: sQ[64×512]bf16 persistent = **64KB** dominates → 86KB/CTA → **1 CTA/SM → only 48
  concurrent queries** on the 48-SM GB10. 13000 queries serialize into ~270 waves.
- Standalone attn-only throughput: **~26.8k queries/s, FLAT** across 8.5k/13k/32k query counts
  (n_comp=16384). The **flatness is the design goal confirmed** (vs dense's 343→224 droop). But
  the absolute rate (~0.037ms/query) is too slow to beat dense at prefill scale as-is.
- k5 (k5_occ.cu) attempt: store Q in smem as **FP8** (32KB not 64KB) + just-in-time dequant of
  Q sub-tiles for WMMA → 55KB dynamic. Correct (cos 0.99996 vs FP8-dequant) BUT **still 1 CTA/SM**
  (55KB dynamic + ~13KB static wscratch/qtile = ~68KB; need ≤50KB total for 2 CTA/SM) and actually
  slightly slower (22.4k q/s) — the FP8 Q dequant overhead outweighs the unchanged occupancy.

### Why it's slow (root causes for the optimization phase)
1. **Occupancy**: must get total (dynamic+static) smem ≤ 50KB for 2 CTA/SM (or multiple queries
   per CTA to amortize). The 64KB Q footprint is the wall — Q needs all 64 heads for every key
   block, so streaming-per-block doesn't shrink it; FP8-Q halves it but static scratch eats the gain.
2. **Latency-bound gather**: TMA gather4 issued single-threaded (`t==0`), barrier re-init + wait
   per gather4 (8 per block × 32 blocks), staging copy serialized after each. No gather/compute
   overlap. The proven primitive is fast (0.12µs/512 rows) but this driver loop doesn't pipeline it.
3. **Tiny KB=16** → 32 flash iterations, each a full sync barrier dance.

### k6 (k6_dtile.cu): D-tiling experiment to cut smem — correct but did NOT clear occupancy
Tiled D into DT=128 chunks so only [KB×128] K is staged (4KB not 16KB). Dynamic smem dropped to
**43KB** and numerics held (cos 0.99996 vs FP8-dequant). BUT:
- Occupancy STILL 1 CTA/SM: 43KB dynamic + ~13KB static (wscratch[8][256]f32=8KB + qtile[8][256]=4KB)
  = ~56KB total > 50KB. **Static smem counts against the 100KB/SM too** — must trim it (single
  shared wscratch via warp serialization, fold qtile dequant into wscratch) to ~6KB for 2 CTA/SM.
- D-tiling **re-gathers K twice** (QK^T pass + PV pass) → 18.4k q/s, SLOWER than k4's 26.8k. The
  smem win costs gather traffic. A 2-CTA/SM D-tiled kernel must also cache the gathered K across
  the QK^T→PV passes (or fuse them) to not lose the doubling. (WARPS=4 trim attempt: nan + still
  1 CTA/SM — the 256-thread copy loops assume 8 warps; needs a clean rewrite, not a sed.)
- Measured fact: removing the per-block gather entirely (k4, reuse block-0 K) only lifts 26.8k→31.2k
  (~15%). So **occupancy (48 concurrent CTAs), not gather latency, is the dominant cost.** The
  optimization MUST raise CTA/SM; pipelining the gather alone is secondary.

### The optimization path (multi-week, scoped — NOT a correctness risk, the kernel is proven)
- Double-buffer sK: issue gather4 for block n+1 while WMMA-computing block n (cp.async.bulk is async;
  use 2 barriers/2 sK buffers). This is the big one — hides the gather latency.
- Multi-warp TMA issue (one gather4 per warp) instead of t==0 single-issue.
- Minimize smem to ≤50KB: smaller static scratch (reuse one wscratch via warp serialization),
  keep Q in FP8 (32KB), sK double-buffer at 2×8KB if KB drops. Target 2+ CTA/SM.
- Consider 2 queries/CTA sharing the Q-dequant warps to amortize launch/softmax overhead.

## What is SAFE / shipped state
- This branch changes **ONLY scratchpad/** — zero edits to any source, default, or fused-MoE path
  (verified: `git diff 0b69b9c2b4 HEAD -- ':!scratchpad'` is empty). Default attention path is
  byte-identical. Fused MoE (the 1.29× from the parent branch) is intact and committed.
- NOT wired into the server: a perf-losing kernel would regress prefill exactly like the round-3
  vec path (0.61×). Per the task: do NOT ship broken to default. Wiring (phase 2) waits until the
  kernel beats dense in the standalone harness at prefill scale.

## NEXT (phase 1.5 then 2)
1. Optimize k4/k5 for occupancy + gather/compute pipelining until standalone attn-only throughput
   clears the dense path's per-query cost AND stays flat (the standalone harness is the gate —
   no server churn needed). Target: comfortably > dense at 13k+, flat to 43k.
2. THEN phase 2 wiring: ggml-cuda op behind `DSV4_SPARSE_ATTN=1`, replacing the dense+(-inf mask)
   path at deepseek4.cpp ~2654/2712 (raw window stays dense), capture-safe (persistent scratch,
   on-device topk indices). Pass Q + get_dsv4_attn_k (FP8 latent) + topk + Kscale.
3. Phase 3 verify 2-node: numeric gate in-server (sparse≈dense, Korean+Paris+coherent), then the
   KEY measurement — prefill tok/s at 8.5k/13k/32k/43k WITH vs WITHOUT, flatness signal, then
   combine DSV4_MOE_FUSED=1 + DSV4_SPARSE_ATTN=1 toward jasl's 1595-1722.

## Files (scratchpad/sparse_mla/)
- ref.h — reference + FP8-E4M3-B128 quant model + cos/rel-err gates
- k1_scalar.cu / k2_wmma.cu / k3_tma.cu / k4_fp8.cu — the staged, numerically-gated build
- k5_occ.cu — FP8-Q occupancy experiment (correct; occupancy not yet cleared)
- k4_perf.cu — prefill-scale throughput bench (flatness confirmed)
- g2.cu / wide.cu / perf2.cu — re-validated TMA Gather4 primitives (correct/256-wide/0.12µs)
