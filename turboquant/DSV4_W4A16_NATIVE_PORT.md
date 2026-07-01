# DSV4 W4A16 fused-MoE — native CuTe C++ port (resume spec)

Branch: `feat/dsv4-w4a16-native-port` (off `feat/dsv4-sparse-mla-mma`).

## Goal
Port b12x's SM121 W4A16 fused-MoE kernel (the one that gives vLLM **decode 38.5 / prefill
~1570 on GB10**) into native ggml-cuda, so llama.cpp reaches the SAME verified numbers.
NOT a from-scratch design — a faithful **source→source translation** of Aiden's kernel,
with b12x output as the bit-parity oracle.

## Why this is a translation, not invention (de-risk, all GREEN)
- b12x fast path = **cute-dsl (CUTLASS 4.x Python DSL)**, 329 py / 3 cu. Kernels are
  `@cute.jit` funcs → `cute.compile` → cubin. NOT a linkable .so.
- **Rejected**: C→Python bridge (arch damage), cubin-blob bridge (opaque, shape-locked,
  non-upstreamable). **Chosen**: native CuTe C++ port, **static AOT (nvcc)**. JIT (NVRTC)
  deferred — our serving config is fixed → shapes bounded → static suffices.
- Same abstraction framework confirmed on-box: CuTe C++ headers at
  `/home/user/work/flash-attention/csrc/cutlass/include/cute` — layout/tensor/mma_atom/
  copy_atom ✓, `arch/mma_sm80.hpp` (std bf16 m16n8k16) ✓, `mma_traits_sm120.hpp` +
  `numeric_types.hpp` (e2m1) ✓, `copy_sm90_tma.hpp` ✓. Every primitive Aiden uses has a
  C++ counterpart → no missing infra, only translation labor.
- **Key finding**: the hot MMA is **standard bf16 `mma.sync.m16n8k16`** (Ampere+), NOT an
  exotic sm121-only block-scaled MMA. FP4 (`e2m1`) appears only in weight *unpack* (cvt/bit
  ops), never in the multiply. W4A16 = 4-bit weight dequantized to bf16, standard bf16 MMA.
  → no MLIR-only atom blocker.

## Source of truth (b12x, extracted at ~/work/dsv4-serve/build/b12x)
- MoE kernel: `b12x/moe/fused/w4a16/kernel.py` (5290 L) + host.py/prepare.py/route_pack.py
- Primitives: `b12x/cute/fp4.py` — `bf16_mma_m16n8k16_f32` @2271,
  `packed_dequant_e2m1x4_to_bfloat2x2` @1963, `packed_dequant_e4m3x4_*` (scale) @2045+
- SM121 tuning: `_W4A16_REGS_SM121`, tile-config selection in kernel.py:108-347

## Strategy: correct-first, then fast
1. **Primitives** (DONE, this commit): FP4→bf16 unpack + bf16 MMA warp op, faithful PTX
   translation → `ggml/src/ggml-cuda/dsv4-w4a16/dsv4-w4a16-primitives.cuh`.
2. **Correctness**: naive native W4A16 MoE using std bf16 MMA + these primitives, wired to a
   custom ggml op. Gate = **bit-parity vs b12x output** (per-expert GEMM).
3. **Fast**: reconstruct Aiden's orchestration in CuTe C++ (smem swizzle, multi-stage
   cp.async/TMA pipeline, warp-spec, `_W4A16_REGS_SM121` occupancy). Gate = t/s toward 38.5.

## Parity plan
- Foundational primitive: e2m1 has 16 codes {±0,±.5,±1,±1.5,±2,±3,±4,±6}; the bit-trick is
  host-replicable → self-contained unit test (no b12x runtime needed). See test file.
- Kernel level: dump b12x per-expert GEMM inputs/outputs (fixed seed) → compare native.

## Status
- [x] De-risk (framework/atoms/instruction-class) — GREEN
- [x] Branch + spec
- [x] Foundational primitives translated (bf16 MMA, e2m1→bf16 unpack), PTX verbatim
- [x] **nvcc compile-check @ sm_121a — PASS** (nvcc 13.0; MMA executes no-error)
- [x] e2m1 unpack = **pre-scale intermediate** (FP4 × 2^-126), bias deferred to scale mul.
- [x] **Scale-dequant (e8m0) + mul.bf16x2 combine ported** — e8m0 byte b -> 2^(b-120)
      (= scale × 2^7, inf/nan handled); combine gives true_weight × 2^-119, epilogue
      restores × 2^119. **Full-value parity 80/80** (@ sm_121a) over realistic scales.
      LEARNING: 2^-119 intermediate underflows bf16 for true_weight < ~2^-7 (b<~106) —
      inherent to b12x (same bf16), trained scales sit near b=127. Documented in test.
- [x] **m16n8k16 bf16 MMA fragment layout verified 128/128** (@ sm_121a, vs fp32 CPU ref,
      first try). A/B/D thread-element mapping correct → primitives compose into a real MMA.

--- BRICKS (primitives) DONE. Below = 시공 / assembly ---
- [x] **3b: single-tile W4A16 GEMM assembled — 128/128** (@ sm_121a). FP4+e8m0 -> dequant
      -> bf16 smem tile -> MMA (3a layout) -> epilogue ×2^119, vs fp32 ref. First pour: the
      verified primitives compose into a working W4A16 GEMM. (dequant_w helper in test.)
- [x] **3c-i: K-loop accumulation — 128/128** (K=64, 4 tiles, in-place MMA accumulate,
      single epilogue ×2^119, vs fp32 ref @ sm_121a). Real per-expert GEMM compute skeleton done.
- [x] **3c-ii: b12x oracle established (the elegant way)** — instead of the kernel ABI, use
      b12x's OWN torch golden reference (`b12x/moe/fused/reference.py`, what its tests trust).
      Ran `_make_fp4_lut`/`_dequant_fp4` LIVE in the image: FP4 table **16/16 match** to our
      native kernel; e8m0 = float8_e8m0fnu = 2^(b-127) = ours. → our kernel's dequant+GEMM
      semantics == b12x's reference. Script: `oracle_b12x_dequant.py`.
      (NOTE: this is semantics-parity vs b12x's *reference*; bit-exact-vs-*kernel* accumulation
      order is a further optional check, but the reference is what b12x itself validates against.)

--- PERF PHASE (correct -> fast) ---
- [x] **Baseline: tiled multi-block GEMM 256x512x1024, 91/91 parity, ~2.87 TFLOP/s** naive
      (grid=(N/8,M/16) warps, 1 output tile each, no reuse/pipeline). @ sm_121a -O3.
      Target: b12x ~57–91 TFLOP/s → ~20–30x climb ahead. This is the number to beat.
- [x] **Lever 1 OCCUPANCY — MEASURED, NOT the bottleneck.** WPB sweep 1/4/8/16/32 →
      WPB=1 (baseline 2721) is FASTEST; more warps/block = slower (bigger block = less
      sched flexibility + more smem/block). My "1 warp/block is starved" guess was wrong;
      the data killed it. Real bottleneck = **arithmetic intensity**: naive re-dequants +
      re-loads B once per M-tile (16x redundant), A once per N-tile → tensor cores starve
      (~2-3% of peak). test_w4a16_gemm_occ.cu.
- [ ] **Lever 2 (real one): block-tiled GEMM w/ smem A+B reuse** — load a block-tile of A
      and dequant a block-tile of B into smem ONCE, all warps reuse for a big output tile.
      Classic AI boost. Bigger rewrite = the actual perf 몸통.
- [x] **ncu PROFILE (sudo) — real bottleneck FOUND.** Compute SOL 12.5%, Memory 27.5%,
      warp cycles/inst 69.6, **94% of stalls = short-scoreboard (smem-load → MMA dependency)**.
      Latency-bound on feeding the MMA: pack2 does ~12 scalar bf16 smem loads/MMA and the MMA
      stalls ~65 cyc waiting. Explains why levers 1-3 (occ/reuse/dequant) did nothing.
- [x] **Lever 4 `ldmatrix` — WIN, 4.4x.** A: ldmatrix.x4, B: ldmatrix.x2.trans (addressing
      correct first try, parity 2800/2800). **2.7 → 11.8 TFLOP/s @ 2048^3, BM128/BN64.**
      ALSO caught: the 256x512x1024 test was too small — only 32 blocks under-fill 48 SMs, so
      grid-size/occupancy confounded it (naive's 1024 tiny blocks looked faster). Must measure
      perf at GPU-saturating sizes. test_w4a16_ldmatrix.cu (default now 2048^3).
- [~] **ncu re-profile after ldmatrix:** Compute 26.8%, Memory **68.8%** (was 27.5%),
      cycles/inst 32.3 (was 69.6), scoreboard stall 30.5% (was 94%). ldmatrix worked; kernel
      is now MEMORY-bound with compute idle → cp.async is the profiler-indicated next lever.
- [x] **Lever 5 cp.async double-buffered prefetch — +15%, 11.8 → 13.6 TFLOP/s** (2048^3,
      BM128/BN64, parity 2800/2800, pipeline correct first try). Raw A(bf16)+B(fp4) cp.async'd
      16B-vectorized into 2 smem buffers, next tile prefetched behind current MMA. Modest gain
      (vs ldmatrix's 4.4x) — the dequant->smem step with 2 __syncthreads likely serializes and
      caps overlap. test_w4a16_cpasync.cu. Running total: 5.0x over naive.
- [x] **Lever 6 B-direct dequant-to-registers — +27%, 13.6 → 17.3 TFLOP/s** (BM128/BN32).
      Profiler said MIO-queue-full (35.7%): B's smem round-trip (Braw read→Bdeq write→ldmatrix
      read = 3 smem passes) overloaded the MIO pipe. Fix: drop Bdeq+ldmatrix-for-B, dequant raw
      FP4 straight into the B fragment registers (dequant is cheap/compute-idle). Removed a
      __syncthreads too. BN32 best (lower reg/MIO pressure). Parity 2800/2800. 6.4x over naive.
- [ ] Re-profile → next bottleneck. Candidates: register-blocked wider warp tiles, deeper
      (3-stage) pipeline, A-load reuse across N-blocks, occupancy/_W4A16_REGS_SM121.
- [ ] MoE routing/grouping/top-k + ggml op wiring → decode 38.5
- [ ] Register-blocked wider warp tiles + occupancy/_W4A16_REGS_SM121 re-tune
- [ ] MoE routing/grouping/top-k + ggml op wiring → decode 38.5
- [ ] MoE routing/grouping/top-k (prepare.py 1039 / route_pack.py 390)
- [ ] ggml custom op wiring + end-to-end serve parity → decode 38.5
- [ ] MoE routing/grouping + top-k epilogue (prepare.py / route_pack.py)
- [ ] Orchestration (multi-stage pipeline / swizzle / _W4A16_REGS_SM121 occupancy) → 38.5
- [ ] ggml custom op wiring + end-to-end serve parity

## Resume pointer
Read this doc + `dsv4-w4a16-primitives.cuh`. Next: nvcc compile-check, then host parity test,
then the naive correct MoE op. Oracle = b12x in image `sparkrun-vllm-ds4-gb10:gb10-local`.

## Perf calibration (2026-07-01)
- **GB10 raw bf16 MMA peak (register-only microbench) = 75.9 TFLOP/s** (bench_mma_peak.cu).
- Our W4A16 GEMM = 18.8 TFLOP/s = **25% of peak**. Real headroom (~3x to b12x-class, ~4x to peak).
- b12x's 91 TFLOP/s > bf16 peak → that's the FP4 path; realistic W4A16(bf16) target ~50-60 (65-80% of peak).
- Diminishing per-lever returns are because the SIMPLE levers are exhausted, NOT because we're near
  the ceiling. Remaining 3x needs the harder restructures below (b12x's actual techniques).

## Perf trajectory
| lever | TFLOP/s | vs prev | % of 76 peak |
|---|---|---|---|
| naive | 2.7 | — | 4% |
| ldmatrix | 11.8 | +340% | 16% |
| cp.async | 13.6 | +15% | 18% |
| B-direct (dequant→regs) | 17.3 | +27% | 23% |
| BK=32 tile reuse | 18.8 | +8% | 25% |

## Remaining levers (harder restructures, ~3x headroom to b12x-class)
- [x] **Lever 8 2D warp-tiling + register blocking — +70%, 18.8 → 32.0 TFLOP/s** (42% of peak).
      Warps WMxWN, each warp does an RM x RN register-blocked grid of MMA tiles reusing A-frags
      across RN and B-frags across RM. Best: BM128/BN128/BK32/WM2/WN2 → RM4 RN8, 4 warps, 32 MMA
      tiles/warp. Parity 2800/2800. test_w4a16_wt.cu. 11.8x over naive; ~1.6-1.9x to b12x-class.
- [ ] Deeper (3-4 stage) cp.async pipeline; RM/RN + swizzle fine-tune → toward 50-60
- [ ] Then MoE routing/top-k + ggml op wiring → decode 38.5

## Perf trajectory (updated)
naive 2.7 → ldmatrix 11.8 → cp.async 13.6 → B-direct 17.3 → BK32 18.8 → **warp-tile 32.0** (42% of 75.9 peak)
