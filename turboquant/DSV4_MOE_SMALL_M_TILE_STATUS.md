# DSV4 NVFP4 grouped-MoE — SMALLER-M tile lever: INFEASIBLE on sm120 (proven)

Goal: cut tile-starvation in `ggml/src/ggml-cuda/dsv4-moe-grouped.cu` (CUTLASS NVFP4
W4A4 grouped GEMM) by shrinking the MMA M-tile from 128 → 16/32/64 so the ~12–48
tokens/expert fill the tile (M-util at ub=512 is only 9.4%).

## VERDICT: a smaller-M block-scaled tile DOES NOT COMPILE on sm120/sm_121a.
Only **M = 128 (or larger: 256)** is feasible. Empirically proven by compile probe,
not theory. NO kernel was built/gated/served — there is nothing to select with a
`DSV4_MOE_SMALL_TILE` flag, so the production boxes were never touched.

## Empirical compile probe (`turboquant/patches/dsv4_moe_tile_m_probe.cu`)
Exact replica of the production type-tree (Sm120, OpClassBlockScaledTensorOp, NVFP4
nv_float4_t, SFVectorSize=16, KernelScheduleAuto→cooperative), parameterized on
`-DTBS_M`/`-DTBS_N`, forcing full instantiation via `Gemm::get_workspace_size`.
nvcc 13.0, `-arch=sm_121a`, same include paths as the ggml-cuda target:

| MmaTileShape (M,N,128) | compiles? |
|------------------------|-----------|
| 128 × 128             | **YES**   |
| 256 × 128             | YES       |
| 128 × 256             | YES       |
| 64 × 128              | **NO**    |
| 32 × 128              | **NO**    |
| 16 × 128              | **NO**    |
| 128 × 64              | **NO**    |
| 128 × 32              | **NO**    |

M cannot go below 128; N cannot go below 128 either. Both can only grow.

## Root cause (structural CUTLASS constraint, not a tuning knob)
NVFP4 block scales use an **indivisible 128-row block**: `Blk_MN = _128`
(`include/cutlass/detail/sm100_blockscaled_layout.hpp:51`, used by both
SFVectorSize=16 NVFP4 and =32 MXFP8).

In `sm120_blockscaled_mma_builder.inl` the SF SMEM/TMA layout is built as:
- `sSFA_shapeM = size<0>(TileShape)/Blk_MN`  (line 192)
- `sSFA_strideK = ... size<0>(TileShape)/Blk_MN * Blk_Elems` (line 197)

For M<128, `size<0>(TileShape)/128 == 0` → the SFA TMA SLayout gets a zero-size
mode → cute fails with:
`"TMA requires CTA_Tile and SLayout top-level size equivalence"`,
`"Shape Divisibility Condition"`, `warning: division by zero`
(`cute/atom/copy_traits_sm90_tma.hpp:739`, `cute/layout.hpp:1101/1109`).

N<128 fails the same way: SFB rounds N up to 128 (`MMA_N_SFB = ceil_div(MMA_N,128)*128`,
layout line 171) but the sm120 N-permute atom `Shape<_8,_2,_2>`
(`sm120_common.inl:146`, fixed for SFVectorSize 16/32) then can't tile a 64-wide CTA
→ same SLayout mismatch.

No escape hatches on sm120:
- The sm120 blockscaled builder uses ONE `TileShape_MNK` for MMA *and* SF SMEM; there
  is **no decoupled `PerSmTileShape`/output-tile** (unlike sm100), so you can't keep a
  128 SF block while emitting a smaller scheduler/output tile.
- `Blk_MN=128` is identical for the MXFP8 (SfVectorSize=32) config — switching the
  scale granularity doesn't lower the M block either.
- `PermTileM = min(M,128)` only *clamps* the MMA permute; the SF divisor still uses the
  raw TileShape M.

This is exactly the wall behind the old memory note "a 64x128x128 PingPong schedule
TODO": it cannot be satisfied for a block-scaled NVFP4 mainloop on sm120.

## DSV4 geometry (why M-util is low but tile COUNT is fine)
E=256, U=6, F_half=1024, D=4096, 48 SMs.
| ub  | rows  | avg tok/expert | M-tiles/expert (128) | M-util @128 |
|-----|-------|----------------|----------------------|-------------|
| 512 | 3072  | 12.0           | 1                    | 9.4%        |
| 1024| 6144  | 24.0           | 1                    | 18.8%       |
| 2048| 12288 | 48.0           | 1                    | 37.5%       |

gate/up active tiles = E·1·(1024/128)=2048; down = E·1·(4096/128)=8192 — both ≫ 48 SMs,
so the GEMM is NOT grid/tile-count starved; the waste is purely intra-tile (only
12–48 of 128 M-rows carry data). A smaller M-tile is the only way to recover that —
and it's the one thing sm120 block-scaling forbids.

## What IS feasible (memory-friendly, no small-M kernel)
1. **Bigger ubatch raises M-util for free** — ub=2048 → 37.5% vs ub=512 → 9.4% (4×).
   The blocker is the staging arena OOM, not the GEMM. The arena (`pf_*` in
   LayerWeights) is sized to `ub*U` rows; ub=2048 over 43 layers is the memory cost.
   Memory-friendlier sub-options to recover M-util without a global ub bump:
   - **Token-block MoE-only re-batching**: accumulate several attention ubatches and
     run the MoE GEMM on a *larger* combined M (only the MoE arena grows, not attn KV).
   - **Drop the N-tile from 128→ keep 128** (can't shrink) but **grow N to 256**
     (compiles) to amortize the wasted-M cost over fewer kernel launches — marginal.
2. The current 128-M path already over-saturates tiles; the realistic prefill lever is
   ub/M aggregation, bounded by the ~20% MoE share of prefill GPU time (dense GEMMs are
   38% and already compute-bound) → end-to-end ceiling ~+10–20% even at full M-util.

## Files
- Probe: `turboquant/patches/dsv4_moe_tile_m_probe.cu` (rebuild:
  `nvcc -std=c++17 --expt-relaxed-constexpr --expt-extended-lambda -arch=sm_121a \
   -I.../cutlass/include -I.../cutlass/tools/util/include -DTBS_M=64 -DTBS_N=128 \
   -c dsv4_moe_tile_m_probe.cu`).
- No change to `dsv4-moe-grouped.cu`, no branch, no build, no rsync, no serve — the
  lever as specified cannot be instantiated.
