# KV Outlier Isolation — Status & TODO (정석화)

Working doc to resume the outlier-isolation feature. Last updated 2026-06-09.
Branch: `feat/kv-outlier-isolation` (off `main` a93da4a4b).

---

## What it is
Pre-rotation dense-and-sparse outlier isolation for low-bit KV cache: per block,
pull the top-2 `|x|` channels out BEFORE the WHT, store them as `ol_idx[2]`+`ol_val[2]`
(fp16) appended to the block, quantize the residual, and add `scale·Σ Q_raw[idx]·ol_val`
back into the Q·K dot. Reclaims the coding gain that random rotation forfeits at 3-bit.
Toggle: env `AMX3_OUTLIERS` (0=off default, 2=on). Disabled = bit-identical no-op.

## DONE (committed, validated)
- **amx3_1** K (commit `f0b1fbf7b`): 108→114B, env-toggled. Recovers ~98% of 3bit→f16 gap.
  jan-nano math 7.1%→45.7% (f16 46.4%); Pauli 월프강→볼프강.
- **tbq3_1** K + amxv3_1 V alias (commit `42fee0ae2`): **head_dim 128, 4.83×**.
  Config: `AMX3_OUTLIERS=2 ... -fa -ctk tbq3 -ctv amx3`. K=tbq3_1(56B)+outlier, V=amxv3_1(50B).
  Validated Qwen3-30B: baseline 월프강 → outlier 볼프강. 2× better compression than amx3 (2.25×).
- Files touched: ggml-common.h (struct), cpy-utils.cuh (quantize), fattn-vec.cuh (K dot),
  set-rows.cu (g_amx3_outliers global + auto-init), fattn.cu (3 FATTN_VEC_CASE), CMakeLists.txt
  (2 instance lists), template-instances/fattn-vec-instance-tbq3_1-amxv3_1.cu.

## KEY FINDING — outlier benefit is HEAD_DIM-dependent
3-bit collapse only happens at LOW head_dim (random-rotation+scalar Gaussianizes better at high dim):
| head_dim | baseline 3-bit (Pauli) | outlier |
|---|---|---|
| 64  | full collapse (README 파이브라스, old single-WHT) | high value but UNVALIDATED |
| 128 | corrupts (월프강) | **restores (볼프강) — DONE** |
| 256 | already correct (볼프강) | no effect → **skip** |
| 512 | (expect like 256) | skip |
=> outlier value is concentrated at head_dim ≤128. 256/512 = plain tbq3 is enough.
256 outlier was built then REVERTED (commit-clean). Do NOT re-attempt 256/512.

## DEFERRED / ABANDONED
- **256/512**: skip outlier (baseline fine). Use plain tbq3 + TriAttention option.
- **64**: deferred. Can't validate (only gpt-oss is hd64 and it gives empty output via the
  test harness — harmony format; no other hd64 model). Hardest impl (tbq3_3 = xhead double-WHT,
  8-head batched quantize, struct shared with tbq3_2 V; tbq3_3 not even CLI-selectable —
  "Unsupported cache type"). Niche (only gpt-oss family). Needs a testable hd64 vehicle first.
- **Part B → attention promotion**: ABANDONED. Drives garbage (4-bit angle = 22.5° too coarse
  for faithful Q·K; fine for eviction ranking only). Branch was deleted. λ=50 MSE didn't fix it.

---

## 정석화 (proper-ization) TODO — for next session

### P1 — make the 128 win production-clean
1. **Rebuild** to match committed source — current `build/bin/llama-server` (06-09 09:36) was
   built from the reverted-256 source, so binary≠source. `cmake --build build --target llama-server -j 8`.
2. **Clean `-ctv tbq3` CLI alias** (remove the `-ctv amx3` workaround). In `common/arg.cpp`,
   `tbq_shortcuts_v["tbq3"]` currently → TBQ3_0 (resolves to tbq3_1). Want V-side "tbq3" → AMXV3_1
   (plain). Either change the V shortcut, or add a V branch in the head_dim resolver
   `tbq_map` (src/llama-context.cpp:3425-3467) so V resolves to the plain type (amxv3_1 @128).
   Goal: user types `-ctk tbq3 -ctv tbq3` and gets K=tbq3_1(outlier), V=amxv3_1(plain).
3. **Promote AMX3_OUTLIERS to a real CLI flag** (e.g. `--kv-outliers N`) instead of env-only.
   Add to common/arg.cpp + common_params + the cuda setter `ggml_cuda_set_*`. Env can stay as fallback.
4. **Rename the toggle** for clarity: `AMX3_OUTLIERS`/`g_amx3_outliers` now also drives tbq3 —
   rename to `KV_OUTLIERS`/`g_kv_outliers` (it's no longer amx-specific). Touch set-rows.cu +
   the externs in cpy-utils.cuh and the fattn-vec.cuh correction comments.

### P2 — correctness hardening
5. **V-corruption guard**: tbq3_1 is now K-only (its quantize extracts outliers; dequantize_V_tbq3_1
   ignores them → V corrupt if tbq3_1 used as V). The `tbq3_1-tbq3_1` instance/config is now unsafe.
   Either (a) GGML_ASSERT/reject tbq3_1 as V, or (b) document that V must be amxv3_1, or (c) add the
   outlier add-back to dequantize_V_tbq3_1 (WHT-domain Hadamard scatter — see notes below). Pick (a)/(b).
6. **Read-side note**: the K dot correction lives in `fattn-vec.cuh` (the caller, after warp_reduce),
   NOT in `vec_dot_fattn_vec_KQ_tbq3_1` (fattn-common.cuh). It IS active (test proved 월→볼). If a
   non-vec attention path is ever used for these types, the correction must be added there too.
7. **g_amx3_outliers cross-TU**: it's read in the quantizer (set-rows.cu TU — same TU as def, OK)
   and the K-dot correction is unconditional (reads ol_val, 0 when disabled — no global read in the
   hot path). So NO cross-TU issue (unlike the abandoned g_amx3_partb). Keep it that way.

### P3 — validation (run before declaring done)
8. Proper benchmark sweep with the FINAL clean binary, head_dim 128 model (Qwen3-30B or 14B):
   - math_bench (thinking-ON — no-think breaks the metric, baseline>f16; see status memo),
   - Pauli (no-think) baseline vs outlier,
   - long-context (RULER/needle or filler 200/500/1000) — the real argmax-retrieval stress.
   - Confirm: outlier ≈ f16, baseline collapses.
9. Compression accounting in README: tbq3+outlier = 3.5 bpw = 4.57× (K alone) / 4.83× (K56+V50).

### P4 — optional / future
10. **64 (gpt-oss)**: only if a testable hd64 vehicle appears. Plan: add ol to block_tbq3_2 (shared
    with _3), outlier extraction in `quantize_f32_tbq3_3_xhead` (cpy-utils.cuh:1663) in the correct
    pre-double-WHT domain, zero ol in `quantize_f32_tbq3_2_block` (V stays plain), K-dot correction
    for type_K==TBQ3_3 in fattn-vec.cuh accounting for `is_double_wht_K` scaling, and wire tbq3_3
    into arg.cpp (currently "Unsupported"). HIGH complexity — validate baseline collapse first.
11. **TriAttention as an option for 256+** (user request): plain tbq3 K + TriAttention token eviction.
    Needs an amx3-256 per-arch variant (current amx3/TriAttention is hd128-only — the "pending
    per-arch variants" item). Separate large feature.
12. **main merge** of feat/kv-outlier-isolation once P1-P3 done.

### Quick reference — commands
- Build: `cmake --build build --target llama-server -j 8`
- Run (128 outlier): `AMX3_OUTLIERS=2 build/bin/llama-server -m <hd128 model> -c 16384 -ngl 999 -fa --reasoning off -ctk tbq3 -ctv amx3 --port 8890`
- Pauli/math test harness: `/tmp/test2.py <port>` (matrix + German→Korean Pauli); math_bench: `turboquant/math_bench.py collect`
