# b12x W4A16 kernel — complete structural analysis (100% understanding)

Source: `~/work/dsv4-serve/build/b12x/b12x/moe/fused/w4a16/{kernel.py 5290L, prepare.py, host.py,
route_pack.py, micro.py}` + `b12x/cute/fp4.py`. Analyzed by 5 parallel deep-dives, synthesized here.
This is the reference for the faithful port. **No code until this is understood — this doc IS that understanding.**

## 0. THE BIG PICTURE — the two headline numbers come from TWO DIFFERENT KERNELS

| number | kernel | technique | regime |
|---|---|---|---|
| **prefill 1570** | Marlin-style tensor-core packed W4A16 GEMM (`_run_tile_large_m`) | m16n8k16 MMA, compute-bound | large M (16-64 tok/expert) |
| **decode 38.5** | **`micro.py` `MoEMicroKernelBackend` — FP4 GEVM/GEMV**, NO tensor cores | warp-reduced FP4×bf16 dot products, bandwidth-bound | tiny M (1-8 tok) |

My 32 TFLOP/s work is ONLY on the prefill GEMM path. **Decode 38.5 needs a separate GEMV micro-kernel I have not started.**

## 1. PREFILL — Marlin-style tensor-core W4A16 GEMM

### Data layout (OFFLINE, prepare.py — this is what I was missing)
- Weights pre-reordered offline into the **MMA-B-fragment layout**: `_repack_4bit_no_perm`. One Uint32 `q`
  = exactly the **8 FP4 codes** a thread needs (K∈{r,r+1,r+8,r+9} × N∈{c,c+8} = m16n8k16 footprint).
  `pack_idx=[0,2,4,6,1,3,5,7]` → even-K in slots 0-3, odd-K in 4-7. `_PACK_FACTOR=8`.
- Scales pre-packed: e4m3_k16 (16 K/scale) or **e8m0_k32 (32 K/scale) — DSV4 uses e8m0**. `[0,2,1,3]` N-perm
  (matches B-frag n,n+8), ×2^7 pre-bias. One scale word = 4 N-channel scales. e8m0→bf16 = 2^(byte-120),
  global ×2^119.

### Runtime mainloop (`_run_mma_pipeline`, kernel.py:1653) — TWO pipelines:
1. **smem pipeline** = 4-stage `cp.async` ring (A bf16, B packed, scales), `wait_group(_STAGES-2)` → 2 outstanding.
2. **register pipeline** = smem→reg **DOUBLE BUFFER** (`a_regs_cur/next`, `b_scale_cur/next`). Compute MMA on
   `*_cur` while `_load_next_fragment_bundle` loads `*_next` from smem → hides ldmatrix/ld_shared latency
   (THE anti-scoreboard-stall technique I omitted).
- **Batched dequant** (`_scaled_dequant_b_fragment`): `bq0=q<<8`(even-K), `bq1=q`(odd-K) → 2 dequant calls →
  8 bf16, ×broadcast scale lanes. Amortized over `cta_m_blocks` M-blocks (large-m reuses 1 b_frag across all M).
- **Weights-as-MMA-A trick**: dequant weights = MMA A operand, activations = B (stationary, reused).
- **A-fragment reuse**: 1 ldmatrix → 8 MMAs (large-m). swizzled smem for conflict-free ldmatrix.

### Scheduling (`_run_persistent_gemm`, kernel.py:891)
- **Persistent kernel**: grid_x = SMs × blocks_per_sm (resident wave). Occupancy: m=1 uses **118-158 regs →
  up to 4 blocks/SM** (I used 128 accum regs → only 2). Tile configs (tile_k,tile_n,threads): large-m =
  (64,256,256)/(64,128,128)/(128,64,128).
- **Bulk region** = grid-stride over MN tiles, no split-K. **Tail region** = split-K across CTAs (keeps SMs busy
  when few MN tiles). Split-K reduce = global scratch + ordered-turn spinlock; owner slice stores.
- **Fused MoE**: FC1 → grid_barrier → SwiGLU → grid_barrier → FC2, all one persistent grid.

### What my kernel (32 TFLOP/s, 43% of peak) is MISSING vs this:
1. Offline weight reorder into MMA-fragment layout (I pack at runtime = instruction-heavy).
2. Register-level double buffering (cur/next) — hides smem latency.
3. Weights-as-A operand swap.
4. Lower register footprint → higher occupancy (4 vs 2 blocks/SM).
5. Persistent kernel + tail split-K (I under-fill the GPU on small problems).

## 2. DECODE — FP4 GEVM micro-kernel (`micro.py MoEMicroKernelBackend`, 2363L) — FULLY ANALYZED
THE production decode path for m∈{1,2,4,8}. Single-launch persistent-CTA fused FFN, **no tensor cores**,
bandwidth-bound weight streaming. Simpler to port than the Marlin GEMM (no MMA tiling/split-K).

**Launch geometry:** block = **512 threads (16 warps)** always. grid_x = min(#SM·occ, max(fc1_tasks,fc2_tasks)),
task-strided. `_BLOCK_SIZE=16` (scale block), `k_segments=ceil(k/512)` = per-lane 16-blocks. m==1 = single-token
specialization.

**Per-thread partition:** FC1 task=(token, topk_slot, n-chunk); each **lane** owns `k_segments` contiguous
16-wide K-blocks; each **warp** owns `rows_per_warp=i_chunk/16` output rows; 16 warps tile the CTA chunk.

**Dot core (fp4.py):** `fp4_dot8_sum` (3732) = one 16-elem block dot: 8×`fma.rn.f16x2` into ONE f16x2 acc,
reduce lo+hi→f32. `fp4_dot8_dual_sum` (3622) = fused up+gate (2 interleaved acc chains, 2-way ILP) = the hot
path. `fp4_dot4_sum` (3567) = FC2 8-elem dot. `_f32acc` variants for precise mode. Weights decode inline via
`cvt.rn.f16x2.e2m1x2` — no LUT.

**Weight streaming (the bandwidth core):** modelopt NATIVE layout, NO offline repack, NO cp.async, NO smem
staging. Each lane reads its `k_segments·8` weight bytes as **128-bit `ld.global.nc.v4.u32`** (read-only cache)
straight into registers, decodes inline. E4M3 scales via swizzled 128×4 tiling (rb=row>>7, mode_a=(row>>5)&3,
mode_32=row&31); E8M0 via `[K/32,N]` + Marlin col-permute. Runtime ≈ (routed W1+W2 bytes)/DRAM BW; FP4 halves bytes.

**Activations:** f16x2 in smem (`+1` pad every 8 blocks, conflict-free). On hot DSV4 shape (k_segments==8, m==1,
gated) the 64 activation words are **hoisted into registers once per warp-task** and reused across 4 rows×{up,gate}
→ kills ~7× smem activation traffic. w4a16 mode does NOT quantize activations (just packs bf16→f16x2).

**Full fusion (one launch):** FC1 (stream W1, dot×scale, warp-reduce, ×alpha) → lane0 SwiGLU (sigmoid(g)·g·u or
relu²) → requant intermediate to swizzled global scratch → **single grid barrier** (atomic count + release/acquire
epoch; light per-token barrier for m==1) → FC2 (warp owns 2 (m==1)/4 K-rows, lane owns 8 n-values, **loop over
top-k accumulating router-weighted** `bsf·dot·(alpha·router_w)`, warp-reduce, lane0 writes bf16). FC2 epilogue
already does the top-k combine — no separate reduce launch.

**Why fast:** (1) bandwidth-bound, reads each expert weight once via 128-bit nc loads, FP4 halves bytes;
(2) zero tensor-core tile waste (GEMV via warp-shuffle, every fma useful); (3) activations register-resident;
(4) single launch, one in-kernel barrier, no global roundtrip; (5) native layout, no repack pass.

**Reimpl checklist:** 512-thread persistent block; lane owns k_segments 16-blocks; `ldg.nc.v4` weights + inline
`cvt e2m1x2`; 8×`fma.f16x2`→f16x2 acc→f32; fuse up+gate; smem acts (+1 pad/8) hoisted on hot shape; `__shfl_xor`
butterfly reduce; lane0 SwiGLU; requant to swizzled scratch; grid barrier; FC2 top-k accumulate + bf16 write.
Key locs: class 355, configure 588, barrier 643, m1-FC2 665, m>1-FC2 843, kernel body 1212-2256; dots fp4.py 3567/3622/3732.

## 3. FAITHFUL PORT PLAN (two separate deliverables)
- **Prefill (→1570-class)**: mirror the Marlin GEMM. Order: offline weight-reorder → batched dequant (done:
  faithful_dequant.cuh) → register double-buffer mainloop → weights-as-A → persistent+tail-split-K → fused MoE.
- **Decode (→38.5)**: mirror `micro.py` GEMV — a DIFFERENT kernel. FP4 dot-product + warp-reduce, activations
  in registers, fused FC1→swiglu→FC2 single launch. (Needs its own deep-dive of micro.py.)

## 4. Status
Analysis 100% for kernel.py/prepare.py/host.py/scheduling. micro.py (decode GEVM) needs a dedicated deep-dive
before its faithful port. faithful_dequant.cuh = first verified faithful chunk (mirrors _scaled_dequant_b_fragment).
