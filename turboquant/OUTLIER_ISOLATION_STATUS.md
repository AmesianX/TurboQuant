# KV Outlier Isolation — Status & TODO (정석화)

Working doc to resume the outlier-isolation feature. Last updated 2026-06-09.
Branch: `feat/kv-outlier-isolation` (off `main` a93da4a4b).

> **2026-06-09 update — tbq3/tbqv3 set is now first-class (commit c31be8bbc).**
> The `-ctv amx3` alias was test-only and is replaced by a dedicated V type `tbqv3_1`.
> Production config: **`-ctk tbq3 -ctv tbqv3`** (K=tbq3_1 outlier, V=tbqv3_1 MSE-plain).
> tbq3_1 outlier isolation is now **intrinsic (always on, no env)** — the type defines the
> behavior. `AMX3_OUTLIERS` only gates amx3 (separate TriAttention family) now.
> Validated: Qwen3-30B-A3B (18G) + GLM-4.5-Air (69G, head_dim=128); tbqv3 V byte-identical to
> the amxv3 baseline. Scope unchanged: head_dim=128 only (256 = plain tbq3, 64 dropped).

---

## What it is
Pre-rotation dense-and-sparse outlier isolation for low-bit KV cache: per block,
pull the top-2 `|x|` channels out BEFORE the WHT, store them as `ol_idx[2]`+`ol_val[2]`
(fp16) appended to the block, quantize the residual, and add `scale·Σ Q_raw[idx]·ol_val`
back into the Q·K dot. Reclaims the coding gain that random rotation forfeits at 3-bit.
Toggle: tbq3_1 = intrinsic (always on). amx3_1 = env `AMX3_OUTLIERS` (0=off default, 2=on).

## DONE (committed, validated)
- **amx3_1** K (commit `f0b1fbf7b`): 108→114B, env-toggled. Recovers ~98% of 3bit→f16 gap.
  jan-nano math 7.1%→45.7% (f16 46.4%); Pauli 월프강→볼프강.
- **tbq3_1** K + amxv3_1 V alias (commit `42fee0ae2`): **head_dim 128, 4.83×**.
  (Original alias config `-ctk tbq3 -ctv amx3`; superseded by the tbqv3 set below.)
- **tbqv3_1** dedicated V type — the tbq3 set (commit `c31be8bbc`): **`-ctk tbq3 -ctv tbqv3`**.
  K=tbq3_1(56B, outlier, intrinsic), V=tbqv3_1(50B, MSE-plain, amxv3_1 동치 클론). No env needed.
  Full plumbing (enum/struct/traits/quantize/dequant/IWHT/fattn cases/instances/CLI/guards).
  Validated Qwen3-30B-A3B + GLM-4.5-Air(69G): 볼프강 outlier marker correct, V coherent;
  tbqv3 output byte-identical to amxv3 baseline on GLM. amx3/amxv3 kept separate (TriAttention).
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

## 정석화 (proper-ization) TODO

### P1 — make the 128 win production-clean  ✅ DONE (c31be8bbc)
1. ✅ **Rebuild** — binary rebuilt from committed source (was reverted-256 stale binary).
2. ✅ **Clean V type** — instead of an alias, added the dedicated `tbqv3_1` V type. The test-only
   `-ctv amx3` workaround is superseded; production config is `-ctk tbq3 -ctv tbqv3`
   (K=tbq3_1 outlier, V=tbqv3_1 MSE-plain). tbqv3 maps directly to the _1 (128-only) type.
3. ✅ **No env flag needed** — tbq3_1 outlier is now intrinsic to the type (always on), so there is
   no toggle to promote. amx3_1 keeps `AMX3_OUTLIERS` (separate TriAttention family).
4. ✅ **No rename needed** — since tbq3_1 no longer reads the env, `AMX3_OUTLIERS`/`g_amx3_outliers`
   is once again amx-specific; left as-is.

### P2 — correctness hardening  ✅ DONE (c31be8bbc)
5. ✅ **V-corruption guard**: V now uses the proper plain type tbqv3_1 (its own quantize/dequant).
   tbq3_1 and amx3_1 are rejected as a V cache type (llama-context.cpp + llama-kv-cache.cpp:
   "K-only / V-only" errors), and tbqv3_1 is rejected as K. No silent corruption path remains.
6. **Read-side note** (unchanged): the K dot correction lives in `fattn-vec.cuh` (the caller, after
   warp_reduce), NOT in `vec_dot_fattn_vec_KQ_tbq3_1` (fattn-common.cuh). It IS active (월→볼). If a
   non-vec attention path is ever used for these types, the correction must be added there too.
   The TBQ V IWHT (inverse-WHT of the V output) is in fattn-vec.cuh and must list every TBQ V type
   (tbqv3_1 was added there — omitting it yields a wrong-basis / garbage V output).
7. **g_amx3_outliers** is now read only by the amx3 quantizer (same TU as its def in set-rows.cu).
   tbq3_1 no longer touches it. No cross-TU issue.

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
- Run (tbq3 set, 128): `build/bin/llama-server -m <hd128 model> -c 16384 -ngl 999 -fa on --reasoning-budget 0 -ctk tbq3 -ctv tbqv3 --port 8890`
  (outlier is intrinsic to tbq3 now — no `AMX3_OUTLIERS` needed. hd128 models: Qwen3-30B-A3B,
  Qwen3-14B, GLM-4.5-Air, MiniMax-M2.x. Note: kill servers with `pkill -f "llama-serve[r]"` —
  the char-class avoids the pkill self-match footgun.)
- Pauli/math test harness: `/tmp/test2.py <port>` (matrix + German→Korean Pauli); math_bench: `turboquant/math_bench.py collect`
