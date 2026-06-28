# DSV4 efficient MoE — the EXACT DeepGEMM/vLLM mechanism, why ours starves, and the verdict

Branch: feat/dsv4-sparse-mla-mma. Research-then-port deliverable for "implement the MoE
the way vLLM/DeepGEMM does it on sm120 to break the 18 TF/s tile-starvation."

Sources read in full (file:line below): local DeepGEMM (sglang venv + TRT-LLM bundle),
flashinfer trtllm-gen batched-MoE export headers, our grouped GEMM, prior rounds
(AIDEN_1600_STACK, DSV4_MOE_SMALL_M_TILE_STATUS, PREFILL_GAP R4–R6). Internet: DeepGEMM
issues #236/#317, CUTLASS #2800, vLLM PR #41834, SonicMoE (arxiv 2512.14080).

---

## TL;DR (the honest verdict the task asked for)

**The DeepGEMM/trtllm "efficient sm120 MoE" the prompt hypothesizes does not exist, and the
mechanism that makes DeepGEMM fast is NOT "avoid tile-starvation at few tokens/expert" — it is
(a) a Hopper/Blackwell-datacenter MMA absent on sm120, and (b) a LARGE batch. DeepGEMM's own
grouped layout pads each expert to BLOCK_M=128 and tile-starves identically to ours at 24–48
tok/expert.** Three independent facts, each with evidence:

1. **No sm120 kernel exists.** DeepGEMM impls are sm90 (`wgmma`) + sm100 (`tcgen05`/`tmem`)
   ONLY. trtllm-gen batched-MoE FP4 cubins are SM100 ONLY. Both MMA families are physically
   absent on sm120/sm121 (consumer Blackwell / GB10). Upstream confirms: DeepGEMM #236, #317;
   CUTLASS #2800; vLLM PR #41834 force-disables DeepGEMM on family-120.
2. **The "efficient" grouped layout still pads to BLOCK_M=128** → at 24–48 tok/expert it
   wastes the same 62–81% of the tile we do. Its speed comes from *large batch* (192+
   tok/expert) + the HW MMA, not from a small-M trick.
3. **A smaller block-scaled tile is impossible on sm120** (proven compile probe,
   DSV4_MOE_SMALL_M_TILE_STATUS): the NVFP4 SF layout's `Blk_MN=128` divisor forbids M<128
   AND N<128. Even SonicMoE (SOTA static-batching, arxiv 2512.14080) needs
   `avg_tok/expert ≥ 2×tile = 256` for its token-rounding to win — we're at 24–48.

⇒ The orthodox, hardware-independent lever is **tokens/expert ↑ via large -ub** (already
implemented R5/R6: `DSV4_INDEXER_QTILE` tiles the indexer transient that was the OOM wall, so
-ub can scale to 4096/8192 where the SAME 128-tile fills to 57/91 TF/s = 3.2×/5×). There is
no kernel to port that beats this on sm120; the GEMM is correct and the only knob is M.

---

## PART 1 — The EXACT DeepGEMM mechanism (kernel + file:line + M-grouping)

DeepGEMM groups ONLY the M-axis (N,K fixed across experts — experts share shape). Two layouts:

### (a) MGroupedContiguous (prefill) — tokens concatenated, each expert M-aligned to BLOCK_M
Source (readable): `deep_gemm/legacy/m_grouped_gemm.py:23-36,72`
```
num_pid_m = cdiv(M, BLOCK_SIZE_M)          # M = TOTAL tokens across ALL experts
batch_id  = tl.load(m_indices_ptr + pid_m*BLOCK_SIZE_M)   # each M-block reads ONE expert id
assert a.size(0) % get_mk_alignment_for_contiguous_layout() == 0   # per-expert M aligned to BLOCK_M
```
CUDA scheduler: `deep_gemm/include/deep_gemm/common/scheduler.cuh:75-77,131-133`
```
} else if (kGemmType == GemmType::MGroupedContiguous) {
    num_blocks = num_m_blocks * num_n_blocks;              // num_m_blocks = ceil_div(TOTAL_M, BLOCK_M)
    ...
    const auto offset = __ldg(grouped_layout + m_block_idx*BLOCK_M);   // expert offset per M-block
```
The mechanism: **one persistent GEMM over the concatenated token array**; each 128-row M-block
reads its expert id from `grouped_layout` to select that expert's weights. Tokens are
pre-sorted by expert and **each expert's segment is padded up to the M block size**
(`get_mk_alignment_for_contiguous_layout`). So a block never straddles two experts.

### (b) MGroupedMasked (decode, CUDA-graph) — per-expert count array, scheduler skips empties
`scheduler.cuh:78-79,158-174,250`
```
} else if (kGemmType == GemmType::MGroupedMasked) {
    num_m_blocks = ceil_div(__ldg(grouped_layout + current_group_idx), BLOCK_M);  // actual count, not padded
    ... return m_offset + m_block_idx*BLOCK_M < __ldg(grouped_layout + current_group_idx);  // mask tail rows
```
Same as TRT-LLM's `GroupedWithOffsetScheduler`
(`TensorRT-LLM/cpp/include/tensorrt_llm/deep_gemm/scheduler.cuh:439-467`):
```
m = m_boundary - m_offset;                         // ACTUAL token count for this expert
num_m_blocks = ceil_div(m, BLOCK_M);               // 24 tok -> ceil(24/128) = 1 block (STARVED)
```

### THE BLOCK_M (the number that decides everything)
`TensorRT-LLM/.../deep_gemm/jit_utils.cuh:148-149`
```
block_ms.push_back((!is_grouped_contiguous && shape_m <= 64) ? 64 : 128);
```
**BLOCK_M = 128 for the grouped path** (64 only for non-grouped small dense GEMM). So DeepGEMM's
grouped GEMM tiles each expert at 128 rows. **At 24 tok/expert it computes a 128-row tile of
which 24 rows are real = 19% fill — identical starvation to ours.** The contiguous layout does
NOT pack multiple experts into one tile (a block reads ONE expert id); the masked layout
ceil_divs the ACTUAL count up to 128. Neither escapes the 128-row floor.

### Why DeepGEMM is nonetheless fast (the real mechanism)
1. **Hopper `wgmma` (sm90) / Blackwell-DC `tcgen05` + tensor-memory `tmem` (sm100)** persistent
   warp-specialized mainloop + FFMA SASS interleaving → ~1350 TFLOPS FP8 on Hopper at LARGE M.
   `sm100_fp8_gemm_1d1d.cuh:35` gates on `__CUDA_ARCH__ >= 1000`; uses UMMA/tmem/2-CTA.
   `sm90_fp8_gemm_1d1d.cuh` uses `wgmma`. **Both instruction families are ABSENT on sm120.**
2. **LARGE batch.** Prefill at d8192 → 8192·top6/256 ≈ 192 tok/expert → even a 128-tile fills
   60%+. At 192 tok/expert the standalone bench (R4) shows 91 TF/s vs 18 at 24. The win is M,
   not the tile.
3. **EP halves experts/GPU** → 2× tok/expert at the same batch.

---

## PART 2 — Why OURS starves (and why it's the SAME starvation, not a worse one)

`ggml/src/ggml-cuda/dsv4-moe-grouped.cu`:
- `:51-66` NVFP4 W4A4, `ArchTag=Sm120`, `OpClassBlockScaledTensorOp`,
  `ThreadBlockShape<_128,_128,_128>`, 1×1×1 cluster.
- `:101-103` per-expert problem shape `PS(m,n,k)` with **m = tokens-this-expert** (M=tokens).
- `:208` `mpad = ((m+127)/128)*128` → each expert's M padded to 128.

This is CUTLASS `GroupProblemShape` (an array of per-expert GEMMs) — algorithmically the SAME
as DeepGEMM's MGroupedMasked (ceil_div each expert's M to 128). Both pad to 128. The fused path
(`dsv4-moe-fused.cu` + flashinfer `CutlassMoeFCRunner`, default-off `DSV4_MOE_FUSED`) is the
same block-scaled 128-tile family. **Our starvation = DeepGEMM's starvation at the same shape.**

Standalone bench (DSV4_PREFILL_GAP_ROUND4): 24 tok/exp → 18 TF/s; 96 → 57; 192 → 91. Pure
tile-starvation, scales ~5× with M. **The GEMM is correct and tactic-insensitive
(R3: tactic sweep ~0); the ONLY lever is tok/expert = M = -ub.**

---

## PART 3 — Could the contiguous/masked LAYOUT (the algorithmic part) help us anyway?

This is the genuinely orthodox question (layout is HW-independent). Answer: **No, not at our
shape**, for a structural reason:

- The block-scaled MMA computes ONE (weight,scale) matrix per tile. A 128-row M-tile in the
  contiguous layout therefore must contain rows from a SINGLE expert (a block reads one expert
  id, `scheduler.cuh:132`). To never straddle experts, each expert's M is padded to 128 → at
  24–48 tok/expert the padding IS the waste. Contiguous vs per-expert-grouped move the same
  bytes; both pad to 128.
- The only way the layout would help is **packing multiple small experts into one 128-tile with
  per-row weight selection** — which the tensor-core MMA forbids (no per-row K-operand swap).
- A smaller tile (so 24–48 fills it) is the real fix, and **it does not compile on sm120**
  (DSV4_MOE_SMALL_M_TILE_STATUS: `Blk_MN=128` NVFP4 SF divisor → M<128 and N<128 both fail
  cute `TMA SLayout size equivalence`; empirically NO for 16/32/64). CUTLASS #2800 confirms
  block-scaled FP4 is sm_100a-only at the DSL level.
- SonicMoE (arxiv 2512.14080) token-rounding needs `avg_tok/expert ≥ 2·tile = 256`; its kernel
  wins use wgmma/tcgen05 → not sm120. At our 24–48 it would round-pad to 128/256 = worse.

⇒ The contiguous-layout angle is exhausted and proven not to help below ~128 tok/expert. The
layout is not the lever; **M is.**

---

## PART 4 — trtllm-gen batched-MoE (tile_tokens_dim 8/16/32) — is THAT portable to sm120?

This is the one design that pads to a SMALL tile (8/16/32) instead of 128 — the apparent
escape. Verdict: **NO sm120 cubin exists; the small-tile GEMM is SM100-only.**
- Routing pads each expert to `tile_tokens_dim` (`flashinfer/utils.py:116-137`
  `calculate_tile_tokens_dim`: next_pow2(tok·k/E·1.3), clamp [8,128]) — this is HOST-side and
  portable.
- BUT the GEMM that consumes the small-tiled layout is a **prebuilt trtllm-gen cubin**
  (`csrc/trtllm_fused_moe_runner.cu:218` `TrtllmGenBatchedGemmRunner`). The export headers
  (`include/flashinfer/trtllm/batched_gemm/trtllmGen_bmm_export/BatchedGemmInterface.h`) are
  **SM100 only** ("work id scheduler based on UGETNEXTWORKID for SM100"). The flashinfer module
  factory is `gen_trtllm_gen_fused_moe_sm100_module` — **no sm120 variant**
  (`flashinfer/fused_moe/core.py:825`). The small-M FP4 MMA assembly lives in the SM100 cubin
  (tcgen05/tmem). We cannot ship it for sm121.
- The 8/16/32 tile fundamentally needs an MMA that issues sub-128 block-scaled tiles — the same
  thing sm120 forbids in CUTLASS. trtllm's cubin does it with tcgen05; that's the SM100 lock.

⇒ The small-tile batched-MoE is the SM100 escape; it has no sm120 path. Porting the routing
(host) without the cubin (device) buys nothing — our CUTLASS GEMM still can't accept a sub-128
tile.

---

## PART 5 — THE PLAN (what to actually do, since "port DeepGEMM" is a dead end on sm120)

The lever is tokens/expert = M = large -ub, made VRAM-safe by tiling the non-MoE quadratic
transients. This is ALREADY BUILT on this branch (R5/R6) and is the correct, orthodox answer:

| piece | status | file:line |
|---|---|---|
| Indexer qtile (breaks the O(n_head·n_comp·ub) OOM wall that capped -ub) | DONE, gated | `src/models/deepseek4.cpp:1898,2221-2227` `DSV4_INDEXER_QTILE` |
| Fused MoE single-workspace (removes per-layer arena → -ub no longer arena-capped) | DONE, gated | `dsv4-moe-fused-run.cu`, `DSV4_MOE_FUSED` |
| VRAM wall-finder probe | DONE, gated | `src/llama-context.cpp:1476-1517` `DSV4_PREFILL_VRAM_PROBE` |
| Standalone MoE TF/s bench (18/57/91 @ 24/96/192) | DONE | `turboquant/dsv4_fused_moe_bench.cu` |

Expected (Amdahl, MoE 57.5% @ 18 TF/s baseline, R6):
- -ub=4096 → MoE ~57 TF/s (3.2×): 0.425·T + 0.575·T/3.2 ≈ 0.60·T → **~1.66× prefill**
- -ub=8192 → MoE ~91 TF/s (5×):   0.425·T + 0.575·T/5   ≈ 0.54·T → **~1.85× prefill**
From ~330 t/s baseline → ~550 (ub4096) / ~610 (ub8192) floor. Residual to vLLM's 1600 is the
per-expert GEMM efficiency (91 vs ~150+ on a HW with sub-128 block-scaled tiles) — a hardware
ceiling on sm120, not a missing kernel.

**Deploy/measure (coordinator):**
```
DSV4_MOE_FUSED=1 DSV4_INDEXER_QTILE=2048 DSV4_PREFILL_VRAM_PROBE=1 \
  -b 8192 -ub 4096     # then -ub 8192 (-b is VRAM-free; keep -b >= -ub)
```
Report: compute-buffer MiB @ ub 1024/4096/8192 (KiB/token slope), max -ub that fits under the
4 GB watchdog, standalone MoE TF/s at achieved tok/expert, end-to-end 13k prefill t/s vs ub=1024.

**If ub=8192 still OOMs after qtile:** the probe names the wall. Masks → tile them per-query
the same way (FA is query-independent). Graph reservation (n_nodes·ub) → linear, slightly
smaller ub. EP (128 experts/node → 2× tok/expert) is the only structural alternative but needs
ncclAllGather and is strictly weaker than the ub lever — pursue only if tiling can't reach 4096.

---

## EVIDENCE INDEX (for audit)
- DeepGEMM impls sm90+sm100 only: `deep_gemm/include/deep_gemm/impls/` (no sm12x file).
  sm100 gate `sm100_fp8_gemm_1d1d.cuh:35` `__CUDA_ARCH__ >= 1000`, UMMA/tmem.
- DeepGEMM grouped scheduler/contiguous/masked: `scheduler.cuh:75-79,131-174,250`;
  legacy triton `m_grouped_gemm.py:23-36,72`.
- BLOCK_M=128: `TensorRT-LLM/.../deep_gemm/jit_utils.cuh:148-149`.
- trtllm-gen batched-MoE SM100-only: `BatchedGemmInterface.h` (UGETNEXTWORKID SM100),
  `flashinfer/fused_moe/core.py:825` `gen_trtllm_gen_fused_moe_sm100_module`,
  `csrc/trtllm_fused_moe_runner.cu:218`.
- Our grouped GEMM 128-tile + per-expert pad-to-128: `dsv4-moe-grouped.cu:51-66,101-103,208`.
- sm120 sub-128 block-scaled tile INFEASIBLE: `DSV4_MOE_SMALL_M_TILE_STATUS.md` (compile probe).
- Upstream sm120 gaps: DeepGEMM #236 (missing sm120 impl files), #317 (DSV4 sm120 kernels
  missing), CUTLASS #2800 (FP4 block-scaled sm_100a-only), vLLM #41834 (force-disable DeepGEMM
  family-120).
- SOTA needs ≥256 tok/expert: SonicMoE arxiv 2512.14080 (`T̄ₑ/Mtile ≥ 2`).
- The ub lever + walls: `DSV4_PREFILL_GAP_ROUND4/5/6.md`.
