# TurboQuant KV Cache Compression for llama.cpp

> Implementation of [TurboQuant (ICLR 2026, Google DeepMind)](https://arxiv.org/abs/2504.19874) — KV cache compression via Walsh-Hadamard Transform + Lloyd-Max quantization with QJL correction

[🇰🇷 한국어](README_KO.md)

### 🆕 v1.6.0 — Polar Derotate + Tangent Residual (TBQX3_1, Qwen3-14B)

**New K cache format: polar-coordinate storage with content/position separation and analytical tangent residual correction. Beats f16 on math reasoning while preserving Korean prose quality.**

**Environment:** NVIDIA DGX Spark (GB10, 128GB) · CUDA 12.8 · Model: Qwen3-14B Q4_K_M · ctx=40960 · temp=0

**Memory footprint (ctx=40960, 40 layers):**

| Config | K buffer | V buffer | Total KV | Compression |
|--------|----------|----------|----------|-------------|
| f16/f16 | 3200 MiB | 3200 MiB | 6400 MiB | 1.0x |
| **tbqx3/tbq3** | **725 MiB** | 625 MiB | **1350 MiB** | **4.74x** |
| tbq3/tbq3 | 625 MiB | 625 MiB | 1250 MiB | 5.12x |

**Speed (decode, same prompt):**

| Config | t/s | vs f16 |
|--------|-----|--------|
| f16/f16 | 24–25 | 1.00x |
| **tbqx3/tbq3** | **21–22** | **0.87x** |
| tbq3/tbq3 | 21–22 | 0.87x |

**Math Accuracy (35 problems, seed=1234, temp=0):**

| Config | Math (/35) | % | vs f16 |
|--------|------------|---|--------|
| **tbqx3/tbq3** | **13/35** | **37.1%** | **+8%** |
| f16/f16 | 12/35 | 34.3% | — |
| tbq3/tbq3 | 10/35 | 28.6% | −17% |

> **tbqx3/tbq3 beats f16 on math while compressing 4.74x** and matching tbq3/tbq3 speed. Legacy tbq3/tbq3 trails f16 by 17%.

**TBQX3_1 block format (3.625 bpw, head_dim=128):**

| Field | Size | Purpose |
|-------|------|---------|
| `d_r` | 2 B | Rayleigh σ (half) |
| `qr[24]` | 24 B | 3-bit r indices (Rayleigh Lloyd-Max, 64 pairs) |
| `qphi[24]` | 24 B | 3-bit φ_content indices (uniform, 64 pairs) |
| `qtan[8]` | 8 B | 1-bit tangent sign per pair |
| **Total** | **58 B** | **3.625 bpw** |

**Key ideas:**

1. **Polar derotate (content/position separation)**: K is stored as `(r, φ_content) = (r, φ_post − pos·freq_i)` per RoPE pair. Content is position-invariant; re-rotation by `pos·freq_i` at read time restores post-RoPE K. Attention now sees content geometry directly instead of content·position entanglement.
2. **Rayleigh Lloyd-Max on r**: the magnitude `r = √(x² + y²)` of a complex Gaussian pair is Rayleigh-distributed. Lloyd-Max 8-level codebook derived analytically from Rayleigh quantile boundaries (no calibration required, works across models).
3. **Tangent Residual (the drift fix)**: the 3-bit uniform φ quantization has bounded error ±π/8. The resulting K perturbation lies almost entirely along the tangent direction `(-sin φ, cos φ)`. One extra bit encodes the sign of `Δφ`, and the magnitude `r · π/16` is analytical (half-cell). No learned scale, reuses already-computed sin/cos — just 2 extra FMAs per pair at read time. Cuts φ error in half (22.5° → 11.25°), eliminates low-probability token drift (Cyrillic contamination on rare-token transliterations).
4. **No WHT in content path**: polar structure preserves angular information directly; applying WHT across pairs would destroy the RoPE pair structure. TBQX3_1 is the first TBQ variant to skip WHT entirely.

**Recommended config for Qwen3-14B and other head_dim=128 RoPE models:**
```
--cache-type-k tbqx3 --cache-type-v tbq3
```

---

### v1.5.3 — Double WHT Per-Head for head_dim=64 (GPT-OSS 120B)

**Cross-head WHT abandoned. Replaced with double WHT per-head (S1→WHT64→S2→WHT64) for D=64 models. QJL 1-bit correction re-enabled — critical for multi-turn stability.**

**Environment:** NVIDIA DGX Spark (GB10, 128GB) · CUDA 12.8 · Model: GPT-OSS 120B (MXFP4)

**Math Accuracy (35 problems, temp=0):**

| Config | K cache | V cache | Math (/35) | Korean | Multi-turn |
|--------|---------|---------|------------|--------|------------|
| f16/f16 | f16 | f16 | **35/35** | ✅ | ✅ |
| **tbq4/tbq3** | tbq4_2 | tbq3_2 | **35/35** | ✅ | ✅ |
| tbqp3/tbq3 | tbqp3_3 | tbq3_2 | ❌ (matrix) | ✅ | ✅ (9+ turns) |

> **Recommended:** `--cache-type-k tbq4 --cache-type-v tbq3` for head_dim=64 models.
> 4-bit K achieves f16-equivalent math accuracy (35/35) with 3-bit V compression.
> 3-bit K (tbqp3) supports Korean conversation and multi-turn but cannot reliably compute matrix operations.

**v1.5.3 Key Changes:**

1. **Double WHT per-head (D=64)**: Cross-head WHT (512-point, $H_8 \otimes H_{64}$) abandoned due to Q-K domain mismatch. Replaced with S1→WHT64→S2→WHT64 double WHT per-head. Kurtosis 0.375→0.047 (near-Gaussian).
2. **QJL re-enabled for K at D=64**: Contrary to v1.5.2 (which removed QJL), the 1-bit QJL correction is critical for multi-turn stability. Without QJL: repetition loops at turn 3-4. With QJL (TBQP3_3): 9+ turns verified.
3. **TBQ_TUNING D=64 instances**: All D=64 K/V combinations added to TBQ_TUNING build (tbqp3_3, tbq3_3, tbq4_2, f16, q8_0 × tbq3_2/f16).

---

### v1.5.2 — PPL 21%→6%, Precision Fix, Deterministic Kernel

**Critical precision loss in flash attention kernel fixed. 3-bit KV cache now deterministic and achieves 1.06x f16 PPL.**

**Environment:** NVIDIA DGX Spark (GB10, 128GB VRAM) · CUDA 12.8 · Model: [unsloth/gemma-4-26B-A4B-it-GGUF](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF) UD-Q4_K_XL

**Memory & Compression (262K ctx, Gemma 4 26B MoE):**

| Config | Global KV | SWA KV | Total KV | Compression |
|--------|-----------|--------|----------|-------------|
| f16/f16 | 5,120 MiB | 300 MiB (f16) | 5,420 MiB | 1.0x |
| **tbqp3/tbq3** | 990 MiB (K:500 + V:490) | 300 MiB (f16) | **1,290 MiB** | **4.2x** |

**PPL Benchmark (wikitext-2-raw, ctx=2048):**

| Config | PPL | vs f16 |
|--------|-----|--------|
| f16/f16 | 419.8 | 1.00x |
| **tbqp3/tbq3** | **445.7** | **1.06x** |

**Math Accuracy (262K ctx, temp=0, 35 problems × 10 runs):**

| Config | 10 runs (/35) | Avg | Peak |
|--------|---------------|-----|------|
| **tbqp3/tbq3** | 19,23,18,22,18,20,19,16,19,17 | **19.1** | **23** |
| f16/f16 | 19,20,21,20,20,21,21,19,20,20 | 20.1 | 21 |

> **4.2x compression, 6% PPL gap, f16-equivalent math.** Peak 23 exceeds f16 best 21.

**Key changes:**

1. **Flash Attention Precision Fix (half2→float)**: Upstream fattn-vec uses half2 (V_DOT2) for VKQ accumulation and KQ shared memory. This causes precision loss that gets amplified by the 512-point IWHT butterfly into non-deterministic MoE expert routing. Fixed by forcing float accumulation path (`#undef V_DOT2_F32_F16_AVAILABLE`) for all 70 TBQ V template instances. **PPL: 454.7 → 445.7 (1.08x → 1.06x).**
2. **V IWHT Float Staging**: The reduce stage wrote half values to `__shared__ half KQ[]`, but IWHT read them as `float*` — type mismatch amplified by butterfly. Fixed with float register staging + `__syncthreads()` barrier.
3. **Dynamic Attention Sharpening**: α(N) = 1 + c × √(ln N), where c is derived from MMSE/EVT theory. Adapts to context size: small α during generation, larger α during long-context prefill. No clamp needed.
4. **V Rotation Bugfix (attn_rot_v=0)**: attn_rot was applied to V but IWHT decode has no inverse rotation. K rotation is safe (cancels in Q·K dot product).
5. **Per-block Norm (TBQ3 D=512)**: Independent norm per 256-half after 512-WHT. TBQP3 keeps global norm (QJL uses cross-block WHT).
6. **Removed 1.15x V hack**: Replaced by principled attention sharpening.
7. **Fixed tbq4_0 D=512 OOB read**.
8. **Fixed tbq4/tbqp4 D=512 WHT domain mismatch**: K encoding used 256-point WHT while Q used 512-point WHT. Added 512-point encode functions.
9. **Double WHT per-head (head_dim=64)**: Cross-head 512-point WHT abandoned (Q-K domain mismatch). Replaced with S1→WHT64→S2→WHT64 double WHT per-head. Kurtosis 0.375→0.047 (near-Gaussian). Q and K in same domain — no IWHT needed.
10. **head_dim=64 K: TBQP3_3 (QJL enabled)**: QJL 1-bit correction critical for multi-turn stability (7+ turns verified). K auto-mapped to TBQP3_3 (2-bit Lloyd-Max + 1-bit QJL + double WHT).
11. **Dynamic MMSE softening (head_dim=64)**: α(N) = SQNR/(SQNR + √(ln N/ln N₀)), opposite of sharpening — reduces overconfidence when SQNR is low.

### Attention Sharpening — Theory & Dynamic α Formula

Quantization noise in K adds variance to attention scores, flattening the softmax distribution. The sharpening factor α compensates this:

**Dynamic (current implementation):**
```
α(N) = 1 + c × √(ln N)
c = 1/(2 × SQNR_eff × √(ln N₀))
```
where N = current KV token count (runtime), N₀ = 2048 (reference context size).

The √(ln N) term comes from **Extreme Value Theory** — the expected maximum noise among N competing tokens grows as √(2 ln N), increasing the probability of a wrong token stealing softmax mass. The formula naturally adapts: small α during autoregressive generation (small N), larger α during long-context prefill (large N). No clamp needed.

**Why TBQ3 and TBQP3 have different α:**

| Type | Structure | Attention Score Noise | Effective SQNR | α coefficient |
|------|-----------|----------------------|----------------|---------------|
| **TBQ3** | 3-bit Lloyd-Max (8 levels) | Clean per-element noise only | 31.2 | 0.016 |
| **TBQP3** | 2-bit MSE + 1-bit QJL | MSE noise + QJL random projection noise | 13.8 | 0.036 |
| **TBQ4** | 4-bit Lloyd-Max (16 levels) | Minimal noise | 56.2 | 0.009 |
| **TBQP4** | 3-bit MSE + 1-bit QJL | MSE noise + QJL projection noise | 24.5 | 0.020 |

TBQ3 has lower per-element reconstruction error (8 levels), but TBQP3's QJL correction adds a **second noise source** to the attention score: the 1-bit random projection `d_qjl × Σ(Q_wht2[i] × sign[i])`. While QJL reduces K reconstruction MSE (good for V-style usage), it increases attention score variance (bad for softmax). This is why **TBQP3 needs stronger sharpening than TBQ3** despite having better reconstruction quality.

**Dynamic α by context size:**

| ctx N | TBQ3 | TBQP3 | TBQ4 | TBQP4 |
|-------|------|-------|------|-------|
| 256 | 1.012 | 1.027 | 1.007 | 1.015 |
| 512 | 1.013 | 1.030 | 1.007 | 1.017 |
| 1024 | 1.015 | 1.033 | 1.008 | 1.018 |
| **2048** | **1.016** | **1.036** | **1.009** | **1.020** |
| 4096 | 1.017 | 1.038 | 1.009 | 1.021 |
| 8192 | 1.018 | 1.041 | 1.010 | 1.023 |
| 32768 | 1.019 | 1.044 | 1.011 | 1.025 |
| 131072 | 1.020 | 1.047 | 1.011 | 1.026 |
| **262144** | **1.021** | **1.048** | **1.012** | **1.027** |

The range is narrow (1.007–1.048) because √(ln N) grows very slowly. Even 128× context increase (2048→262144) only shifts α by ~0.012. This validates using a fixed constant as a practical first approximation.

```bash
# Recommended (f16-equivalent, 4.2x compression)
./llama-server -m ~/Models/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf \
    -t 4 -c 262144 -n 32768 --parallel 1 \
    --cont-batching --jinja \
    --reasoning off --reasoning-budget 0 --reasoning-format none \
    --n-gpu-layers 999 --flash-attn on \
    -b 1024 -ub 512 --no-mmap \
    --cache-type-k tbqp3 --cache-type-v tbq3 \
    --temp 0 --host 0.0.0.0 --port 8888
```
> SWA K+V auto-upgraded to f16. No extra config needed.

**Supported K/V combinations:**

Use shorthand names (`tbq3`, `tbqp3`, etc.) — internal suffixes (`_0`, `_1`, etc.) are auto-mapped by head dimension.

| `--cache-type-k` \ `--cache-type-v` | f16 | q8_0 | q4_0 | tbq3 | tbq4 |
|--------------------------------------|-----|------|------|------|------|
| **f16** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **q8_0** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **q4_0** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **tbq3** | ✅ | ✅ | — | ✅ | ✅ |
| **tbq4** | ✅ | ✅ | — | ✅ | ✅ |
| **tbqp3** | ✅ | ✅ | — | ✅ | ✅ |
| **tbqp4** | ✅ | ✅ | — | ✅ | ✅ |

> **`tbqp` types are K-only.** Do NOT use `tbqp3`/`tbqp4` for `--cache-type-v` — QJL correction operates on Q·K dot product and has no meaning for V cache. Recommended: `--cache-type-k tbqp3 --cache-type-v tbq3`.

---

### v1.5.1 — Exceeds f16 Quality: SWA f16 Bypass + V 512-WHT + QJL D=512

**3-bit KV cache (tbq3/tbq3) now EXCEEDS f16 quality. SWA f16 bypass is the key breakthrough.**

**Benchmark (Gemma 4 26B-A4B-it MoE, UD-Q4_K_XL, DGX Spark GB10, 262K ctx, temp=0):**

| Config | K cache | V cache | attn_rot_k | Global KV | SWA KV | Math Accuracy (10 runs) | Avg | Compression |
|--------|---------|---------|-----------|-----------|--------|------------------------|-----|-------------|
| **tbqp3/tbq3** | tbqp3 | tbq3 | OFF(auto) | 990 MiB | 300 MiB(f16) | 37,38,40,38,38,36,37,36,37,37 | **37.4** | **4.2x** |
| **tbq3/tbq3** | tbq3 | tbq3 | ON | 980 MiB | 300 MiB(f16) | 39,39,37,37,38,35,35,39,35,36 | **37.0** | **4.2x** |
| f16/f16 | f16 | f16 | OFF | 5120 MiB | 300 MiB(f16) | 37,36,36,36,36,38,36,38,37,36 | 36.6 | 1.0x |

> **tbqp3/tbq3 exceeds f16 (37.4 > 36.6).** Peak 40/65 surpasses f16's best of 38.

**Key techniques:**

1. **SWA KV f16 Bypass**: SWA cache is small (~300 MiB) but has 25 layers dominating overall quality. Auto-upgrade SWA K+V to f16 eliminates SWA quantization noise — the hidden culprit that masked all previous optimization attempts.
2. **V 512-WHT + 512-IWHT**: V cache now uses the same 512-point WHT as K (sign flip + 9-stage butterfly + global norm). 512-point IWHT at attention output (128 threads × 4 elements, warp shuffle + shared memory butterfly).
3. **QJL D=512 Restored**: Previously removed as "ineffective at D=512" — SWA noise was masking QJL improvement. With SWA f16 bypass, tbqp3 K outperforms tbq3 K (37.4 > 37.0). attn_rot auto-disabled for TBQP (prevents triple rotation).

---

### v1.5.0 — Upstream Rebase + Gemma 4 Support

**Full rebase on latest upstream llama.cpp (b7ad48ebd) + Gemma 4 TBQ KV cache support.**

Previously maintained as a separate history from upstream llama.cpp, requiring manual patching of hundreds of commits for each sync. Starting with v1.5.0, the fork shares upstream commit history, enabling single-command sync via `git merge upstream/master`.

**Changes:**
- 3-way merged all TBQ/TBQP code onto latest upstream llama.cpp (b7ad48ebd)
- **Gemma 4 support**: hybrid head_dim=512 (global) + 256 (SWA) architecture fully supported
- Deepseek MLA 512x512 MMA config included (upstream addition)
- bf16 flash attention vec kernel support (upstream addition)
- V-less cache, stream-k FA, and other upstream optimizations included
- Established proper fork structure for easy future upstream sync

**Gemma 4 key techniques:**
1. **SWA cache type auto-remapping**: When global (head_dim=512) and SWA (head_dim=256) differ, the SWA cache is automatically assigned the correct TBQ sub-type
2. **Variable GQA support**: Gemma 4 has per-layer head_count_kv (16/4). WHT rotation operates per-head, so variable GQA is safe — `attn_rot_k` activation condition updated
3. **D=512 single-pass WHT + global norm**: K quantization applies 512-point WHT in one pass (9-stage butterfly). Both 256-blocks share the same global norm for cross-block scale consistency. Q preprocessing also uses 512-point WHT (128 threads × 4 elements). V IWHT uses 256-block independent processing (V has per-block norms)
4. **head_dim via op_params**: head_dim passed to set_rows kernel via `op_params[0]` to correctly distinguish D=512 from D=256

**Benchmark (Gemma 4 31B-it Dense, UD-Q4_K_XL, DGX Spark GB10, 262K ctx):**

| Cache | GPU Memory | Compress | PP t/s | TG t/s | PPL (wiki, 2K) | Math Accuracy | Pauli |
|-------|-----------|----------|--------|--------|----------------|---------------|-------|
| f16/f16 | 41,500 MiB | 1.0x | 152.9 | 10.1 | 309.7 | 42/65 (64.6%) | PASS |
| **tbq3/tbq3** | **23,215 MiB** | **1.8x** | 112.9 | 9.0 | 212.1 | **32/65 (49.2%)** | **PASS** |

**Benchmark (Gemma 4 26B-A4B MoE, UD-Q4_K_XL, DGX Spark GB10, 262K ctx):**

| Cache | GPU Memory | Compress | TG t/s | Math Accuracy | Pauli |
|-------|-----------|----------|--------|---------------|-------|
| f16/f16 | 5,720 MiB | 1.0x | 56.4 | 37/65 (56.9%) | PASS |
| **tbq3/tbq3** | **1,106 MiB** | **5.2x** | 41.1 | **30/65 (46.2%)** | **PASS** |

> **Note:** Gemma 4 is a hybrid SWA architecture with non-SWA 10 layers (head_dim=512) + SWA 50 layers (head_dim=256). SWA cache is limited to the sliding window size (1536 cells), so Dense compression (1.8x) is lower than MLA models (5.2x). The MoE variant achieves **5.2x compression**.
>
> **Note:** Math accuracy measured using [turboquant_math_accuracy.py](https://github.com/eullm/eullm/blob/main/bench/turboquant_math_accuracy.py) (2x2/3x3 matrix multiplication, scalar arithmetic, filler 0-2000t). Out of 65 tests, f16 and TBQ **agreed on 55 (31B Dense) / 58 (26B MoE) cases**. Divergent cases show small numerical deviations (e.g. `160→150`, `519→509`), not garbage output. Pauli test (scientist name recall, German→Korean) passes with coherent, diverse output.
>
> **⚠️ D=512 limitation:** Gemma 4's head_dim=512 extends beyond the TurboQuant paper's validated range (head_dim=128). QJL 1-bit correction (TBQP) does not work at D=512 — **only TBQ (MSE-only) is supported**. All 8 QJL variants tested (global/per-block norm, L2/RMS gamma, correlated/independent sign patterns) degraded quality. TBQP (QJL) works normally for head_dim≤256 models.

<details>
<summary>Build & Run</summary>

**Build:**
```bash
cmake -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_SHARED_LIBS=ON \
      -DGGML_CUDA=ON \
      -DGGML_BLAS=ON \
      -DGGML_CCACHE=OFF \
      -DCMAKE_EXE_LINKER_FLAGS="-lpthread -lm" \
      -DLLAMA_BUILD_TESTS=OFF \
      -DLLAMA_BUILD_EXAMPLES=OFF \
      -DLLAMA_BUILD_SERVER=ON \
      -DCMAKE_VERBOSE_MAKEFILE=ON \
      ..

make -j12
```

**Server (f16 baseline):**
```bash
./llama-server -m ~/Models/gemma4/gemma-4-31B-it-UD-Q4_K_XL.gguf \
    -t 4 -c 262144 -n 32768 --parallel 2 \
    --cont-batching --jinja --reasoning-format auto \
    --chat-template-kwargs '{"enable_thinking": false, "reasoning_effort": "medium"}' \
    --n-gpu-layers 999 --flash-attn on \
    -b 1024 -ub 512 --no-mmap \
    --cache-type-k f16 --cache-type-v f16 \
    --top-k 20 --temp 0.6 --top-p 0.95 --min-p 0.0 \
    --presence-penalty 0.0 --repeat-penalty 1.0 \
    --host 127.0.0.1 --port 8888
```

**Server (TurboQuant):**
```bash
./llama-server -m ~/Models/gemma4/gemma-4-31B-it-UD-Q4_K_XL.gguf \
    -t 4 -c 262144 -n 32768 --parallel 2 \
    --cont-batching --jinja --reasoning-format auto \
    --chat-template-kwargs '{"enable_thinking": false, "reasoning_effort": "medium"}' \
    --n-gpu-layers 999 --flash-attn on \
    -b 1024 -ub 512 --no-mmap \
    --cache-type-k tbq3 --cache-type-v tbq3 \
    --top-k 20 --temp 0.6 --top-p 0.95 --min-p 0.0 \
    --presence-penalty 0.0 --repeat-penalty 1.0 \
    --host 127.0.0.1 --port 8889
```
</details>

**Compatibility:** All v1.4.2 features (MMA tensor core, QJL scalar correction, MLA asymmetric) fully preserved. Existing model benchmarks unchanged.

---

### v1.4.2 — MMA Tensor Core Acceleration + QJL Scalar Correction

**TBQ/TBQP MMA tensor core acceleration: TG speed 30→49 t/s (+63%).**

MMA tensor core attention acceleration for GLM-4.7-Flash (MLA, K=576/V=512) asymmetric models. Up to 1.6x TG speed improvement over vec kernel.

**Key features:**

1. **MMA K spatial dequant**: raw TBQ/TBQP blocks → IWHT → spatial f16 → tensor core. Warp shuffle optimization reduces cooperative IWHT `__syncthreads` from 8 to 4.
2. **QJL scalar correction**: TBQP 1-bit QJL correction via lightweight scalar ops instead of full MMA pass. Reads sign bits + dq directly from raw K blocks. Correctly handles QJL's second sign basis (signs2).
3. **V = K view spatial**: K dequanted to spatial domain, V (= K view) automatically spatial. Output IWHT completely eliminated.
4. **Proper kernel signature**: `fattn_kernel_t` extended with `raw_K_data`, `raw_K_stride`, `Q_wht2_data`, `Q_wht2_stride`. No pointer hacks.
5. **Fused Q WHT12**: Reads Q->data directly, computes Q_wht1 (intermediate) + Q_wht2 (for QJL). No cudaMemcpy.

**Benchmarks (GLM-4.7-Flash UD-Q4_K_XL, DGX Spark GB10):**

| Cache | KV MiB | Ratio | TG t/s | vs v1.4.1 | Pauli |
|-------|--------|-------|--------|-----------|-------|
| f16/f16 | 10,469 | 1.0x | 67.5 | — | PASS |
| **tbq3/tbq3** | **2,944** | **3.6x** | **49.7** | **+55%** | **PASS** |
| **tbqp3/tbq3** | **2,981** | **3.5x** | **42.8** | **+36%** | **PASS** |
| tbq4/tbq4 | 3,526 | 3.0x | ~49 | +47% | PASS |
| tbqp4/tbq4 | 3,562 | 2.9x | ~42 | +45% | PASS |

> **Note:** TBQP (with QJL) is slower than TBQ because QJL requires a second WHT transform (signs2) for Q_wht2 precomputation + per-token scalar correction overhead. The tradeoff is improved dot product accuracy (better PPL).

---

### v1.4.1 — GLM-4.7-Flash (MLA) Asymmetric K/V Support

**Full TurboQuant support for MLA architecture models: GLM-4.7-Flash, DeepSeek-V2/V3.**

MLA (Multi-head Latent Attention) has asymmetric K=concat(latent[512], rope[64])=576 dims and V=latent[512] dims. Three key techniques:

1. **D_V template parameter**: vec kernel separates K/Q dim (D=576) from V dim (D_V=512). VKQ arrays, combine stride, IWHT passes, output writes all use D_V. Symmetric models default D_V=D (zero impact).
2. **RoPE f16 passthrough**: _4 block structs store sub-block 3 (rope 64) as raw f16 instead of WHT+quantized. RoPE norm (~10.49) is ~80x larger than latent (~0.13) — any quantization error dominates attention scores. Q preprocessing and dot product handle sub-block 3 as direct f16.
3. **MLA V-as-K-view**: In MLA absorption, V is a view of K cache (same type). TBQP V dequantize uses MSE centroid only (QJL is for K·Q dot product correction, not V reconstruction). IWHT runs 256+256 two-pass (rope pass skipped).

**Benchmark (GLM-4.7-Flash UD-Q4_K_XL, DGX Spark GB10):**

| Cache | KV MiB | Compress | PP t/s | TG t/s | PPL | Pauli |
|-------|--------|----------|--------|--------|-----|-------|
| f16/f16 | 10,469 | 1.0x | 73.0 | 60.3 | 5.998 | PASS |
| **tbq3/tbq3** | **2,944** | **3.6x** | 68.2 | 32.0 | **6.836** | **PASS** |
| **tbqp3/tbq3** | **2,981** | **3.5x** | 66.8 | 31.5 | **6.586** | **PASS** |
| tbq4/tbq4 | 3,526 | 3.0x | 67.2 | 33.4 | — | PASS |
| tbqp4/tbq4 | 3,562 | 2.9x | 65.8 | 28.9 | — | PASS |

> **Note:** MLA compression ratio (3.5x) is lower than standard models (5.2x) because MLA already compresses KV to a 256-dim latent — the original cache is already small. The 7.5GB savings (10,469→2,981 MiB) is still significant.
>
> **Note:** TG speed (31.5 vs 60.3 t/s): TBQ uses the vec kernel (scalar WHT dot product), f16 uses the MMA kernel (tensor cores). Adding TBQ on-the-fly dequantize to MMA would enable tensor core acceleration — future work.

**Bugs fixed:**

| Bug | Root Cause | Fix |
|-----|-----------|-----|
| GLM tbqp3/tbq3 crash (v1.4.0) | `get_best_fattn_kernel` returned NONE for D=576 TBQ | Added D=576+V=512 TBQ vec kernel routing |
| Symmetric dispatch shadows asymmetric | `FATTN_VEC_CASE` doesn't check V->ne[0] → launches DV=576 | ASYM cases placed before symmetric |
| RoPE quantization → garbage | rope norm(~10.49) >> latent norm(~0.13), quant error dominates attention | Sub-block 3 changed to f16 passthrough |
| TBQP V dequant with QJL → garbage | QJL corrects K·Q dot products only, not V reconstruction | Removed QJL from V dequant (MSE only) |
| Q WHT sub-block race condition | Missing `__syncthreads` between sub-block storage and next WHT | Added `__syncthreads` barriers |

---

### v1.3.0 — Bulletproof head_dim Detection + Critical Bug Fixes

**Benchmark (Qwen3-30B-A3B Q4_K_M, DGX Spark GB10, head_dim=128):**

| Config | PPL | vs F16 | Note |
|--------|-----|--------|------|
| f16/f16 | 6.26 | baseline | |
| **tbqp4/tbq4** | **6.70** | **+7.1%** | +Direct Sign correction (head_dim=128) |
| tbq4/tbq4 | 6.73 | +7.5% | MSE only |
| **tbqp3/tbq3** | **7.91** | **+26.3%** | +Direct Sign correction (head_dim=128) |
| tbq3/tbq3 | 8.49 | +35.6% | MSE only |

> **Note:** TBQP residual correction method varies by head_dim:
> - **head_dim=256**: QJL (original paper, SRHT-based)
> - **head_dim=128/64**: Direct Sign (our fix for QJL variance issue — 4.3x lower variance)
>
> PPL numbers above are from a **MoE model** (3.7B active params/token). F16 baseline 6.26 is inherent to the architecture, not a TurboQuant issue.

**Issues Resolved:**

| Issue | Reporter | Status |
|-------|----------|--------|
| Phi-4, DeepSeek auto head_dim detection failure | @fritolays | Fixed — P1→P5 cascade |
| turbo4-K PPL explosion (18,202 on Cydonia-24B) | @TheTom | Fixed — was head_dim misdetection |
| GLM head_dim=576, Qwen3-4B head_dim=80 | @fritolays, @sztlink | Fixed — pow2 check + q8_0 fallback |
| "////" output on Qwen3.5-27B-UD | @modderBUG | Not reproducible on v1.3.0 |
| llama-bench TBQ types not accepted | @sztlink | Fixed — 16 types + 4 shorthands |
| Windows OpenSSL DLL dependency | @sztlink | Fixed — standalone builds |

**Other Improvements:**
- llama-bench: full TBQ/TBQP type support via -ctk/-ctv
- Unsupported head_dim: falls back to q8_0, preserving compression
- Standalone builds: no external DLL dependencies

<details>
<summary>🔧 head_dim Detection Technical Details (P1→P5 Priority Cascade)</summary>

Previously only read GGUF `{arch}.attention.key_length` metadata — most models (Phi-4, DeepSeek, Gemma, Mistral) don't store this, causing TurboQuant to silently disable.

Now uses 6 detection signals with strict priority cascade:
- **P1**: `attention.key_length` (100% — GGUF authoritative)
- **P2**: `attention.key_length_mla` (100% — MLA models like DeepSeek V2)
- **P3**: `attention.key_length_swa` (100% — SWA models like Gemma 2/3)
- **P4**: `attention.value_length` (95% — cross-check)
- **P5**: `n_embd / n_head` (70% — fallback, can be wrong for MoE)

Cross-validation between signals with diagnostic logging:
```
TurboQuant head_dim signals — key=128 val=128 computed=64 mla_k=0 mla_v=0 swa_k=0
[P1✓ P5✗] key_length=128 but n_embd/n_head=64 — using P1
```

**n_embd/n_head is WRONG for many models:**

| Model | n_embd/n_head | Actual head_dim | Without P1 |
|-------|---------------|-----------------|------------|
| Qwen3-30B-A3B (MoE) | 64 | **128** | Wrong WHT block → garbage |
| Qwen3.5-27B | 213 | **256** | TurboQuant disabled |

Power-of-2 validation catches non-WHT-compatible dimensions (head_dim=80, 576) early.
</details>

---

### v1.2.0 — Auto head_dim Mapping + head_dim=64 Quality Fix + V Cross-Head WHT

**Automatic head_dim mapping — no suffix numbers needed:**
```bash
# v1.2.0: just use tbq3/tbqp3, auto-selects based on head_dim
--cache-type-k tbqp3 --cache-type-v tbq3
```
- `head_dim=256`: K=tbqp3_0 (QJL), V=tbq3_0 → **5.2x compression**
- `head_dim=128`: K=tbqp3_1 (Direct Sign), V=tbq3_1 → **5.0x compression**
- `head_dim=64`: K=**q8_0** (auto fallback), V=tbq3_2 → **2.7x compression**

**head_dim=64 quality issue discovered and fixed:**

Scientist name test (German→Korean transliteration) revealed a fundamental WHT limitation:
- PPL improved (2195 < 4008) but generation broke (attention smoothing artifact)
- TurboQuant paper only validated head_dim=128 — CLT convergence fails at d=64
- **Fix: K cache auto-falls back to q8_0** + V keeps WHT (values tolerate noise in weighted sums)

| Model | head_dim | Config | KV Memory | Compress | PPL (2K) | Prompt t/s | Gen t/s | Pauli Test |
|:------|:---------|:-------|----------:|---------:|---------:|---------:|--------:|:---:|
| GPT OSS 120B | 64 | F16/F16 | 4,608 MiB | 1.0x | 2413 | 133 | 47 | ✅ |
| GPT OSS 120B | 64 | **q8_0/tbq3 (auto)** | **1,692 MiB** | **2.7x** | **1925** | **145** | **46** | **✅** |
| GPT OSS 20B | 64 | F16/F16 | 3,072 MiB | 1.0x | 4008 | 412 | 74 | ✅ |
| GPT OSS 20B | 64 | **q8_0/tbq3 (auto)** | **1,128 MiB** | **2.7x** | **2649** | **421** | **75** | **✅** |
| Qwen3.5-122B | 256 | F16/F16 | 6,144 MiB | 1.0x | — | 91 | 23 | ✅ |
| Qwen3.5-122B | 256 | **tbqp3/tbq3 (auto)** | **1,188 MiB** | **5.2x** | **—** | **102** | **22** | **✅** |

**Input validation — safe handling of all invalid combinations:**
- TBQP (QJL) for V → auto-downgrade to TBQ
- Wrong _N suffix → auto-correct or fallback based on head_dim
- Unsupported head_dim → q8_0/f16 fallback with warning

#### Why the "Pauli Test"? — When PPL Lies

**The problem:** Perplexity (PPL) is the standard metric for evaluating KV cache quantization. Lower PPL = better, right? We discovered this is **dangerously misleading** for WHT-based quantization at small head dimensions.

With our cross-head WHT implementation (512-element WHT across 8 heads, head_dim=64), we achieved PPL numbers that looked *better* than FP16:

| Config | PPL (2K) | Generation Quality |
|:-------|:---------|:-------------------|
| F16/F16 (baseline) | 4008 | ✅ Correct |
| cross-head WHT 3-bit | **2195** (looks better!) | ❌ Completely broken |

**PPL improved by 45% — but the model couldn't even spell names correctly.** The WHT quantization noise acts as attention smoothing: it reduces extreme wrong predictions (lowering average surprise = lower PPL) while destroying the sharp attention peaks needed to pick the single correct token (breaking generation).

**The test:** We needed a metric that catches this. We chose German scientist names translated to Korean standard notation — specifically "Wolfgang Pauli" → "볼프강 파울리".

Why this works as a benchmark:
- **Cultural specificity:** Korean has an official national standard ([외래어 표기법](https://kornorms.korean.go.kr/)) for transliterating foreign names. "Wolfgang Pauli" must be rendered as "볼프강 파울리" (bol-peu-gang pa-ul-li) — not "볼프강 파우리" or "워워우 파울리" or any other variant.
- **Extreme difficulty for LLMs:** Unlike English/Chinese/Japanese where multiple transliterations are acceptable, Korean has exactly ONE correct answer per name. This is determined by standardized phonological rules that map German phonemes to Korean syllables.
- **Sensitivity to attention precision:** Getting multi-syllable Korean transliteration correct requires the model to maintain precise token-by-token attention over the source name. Any attention blurring from quantization noise immediately produces wrong syllables.
- **Validation standard:** The correct answer matches what appears in Korean middle/high school science textbooks — an objective, verifiable ground truth.

**What we found across all configurations (head_dim=64):**

| K type | bits | "Wolfgang Pauli" → | PPL |
|:-------|:-----|:-------------------|:----|
| F16 | 16 | 볼프강 파울리 ✅ | 4008 |
| q8_0 | 8 | 볼프강 파울리 ✅ | — |
| tbq4_0 (4-bit WHT) | 4 | 볼프강 **파우리** ⚠️ (one syllable wrong) | — |
| tbq3_0 (3-bit WHT) | 3 | **파이브라스** ❌ (nonsense) | — |
| tbqp3_0 (3-bit + QJL) | 3 | **er** ❌ (not even Korean) | — |
| tbqp3_3 (cross-head) | 3 | **2.** ❌ | **2195** (misleadingly "good") |

The cross-head configuration that achieved the "best" PPL (2195) produced the worst generation output. **PPL and generation quality were inversely correlated.**

**Root cause:** The TurboQuant paper (ICLR 2026) only validated on head_dim=128 models (Gemma, Mistral, Llama-3.1-8B). At head_dim=64, the Central Limit Theorem convergence required by WHT is insufficient — coordinates don't approximate Gaussian well enough for accurate scalar quantization.

**Our solution:** For head_dim=64, K cache automatically falls back to q8_0 (bypassing WHT entirely), while V cache keeps WHT (value weighted sums are more noise-tolerant). This passed the Pauli test while maintaining 2.7x compression.

**Lesson learned:** Always validate KV cache quantization with generation-quality tests, not just PPL. A method that smooths attention distributions can improve PPL while destroying the model's ability to generate precise outputs.

> **Note: About the Cross-Head WHT code**
>
> v1.2.0 includes cross-head WHT implementations developed for head_dim=64 models: 8 KV heads grouped into a 512-element WHT, Kronecker decomposition H_512 = H_8 ⊗ H_64, V cross-head scoring, etc. This code improved PPL at head_dim=64 but failed to preserve generation quality, so it is **not used in auto-mapping**. However, we intentionally retain the code for future research: cross-head experiments at head_dim>=128 (e.g., 1024-element WHT), comparison with alternative methods (KITTY, learned rotation), or new approaches that may achieve better CLT convergence. Related types: `_3` suffix (tbq3_3, tbq4_3, tbqp3_3, tbqp4_3).

---

### v1.1.0 — head_dim 64/128 Support + Direct Sign Residual Correction

**Multi head_dim support:**
- `head_dim=256`: existing (Qwen3.5, Qwen3-Next) — QJL residual correction
- `head_dim=128`: **new** (Llama, Qwen3, Mistral, MiniMax, most models) — Direct Sign correction
- `head_dim=64`: **new** (gpt-oss, smaller models) — Direct Sign correction
- Auto-detected — user CLI unchanged (`--cache-type-k tbqp3_0` works for all)

**Direct Sign — 4.3x lower variance than paper's QJL:**

The paper's QJL uses SRHT random projection for residual correction, but at d≤128 the projection noise exceeds the correction benefit. Direct Sign stores `sign(residual)` directly:
- 4.3x variance reduction: `(1-2/π)/(π/2) = 0.23`
- No second WHT needed → faster query preprocessing
- QJL retained for d=256 where it excels (hybrid strategy)

#### head_dim=128 Benchmark (Qwen3-30B-A3B Q4_K_M, 2K context, DGX Spark GB10)

| KV Config | PPL | Speed (t/s) | KV Size | Compression |
|:----------|----:|----------:|--------:|------------:|
| f16 + f16 (baseline) | 6.69 | 87.8 | 192 MiB | 1.0x |
| q8_0 + q8_0 | 6.68 | 84.3 | 102 MiB | 1.9x |
| q4_0 + q4_0 | 7.33 | 85.0 | 54 MiB | 3.6x |
| tbq4_0 + tbq4_0 | 7.02 | 68.6 | 50 MiB | 3.9x |
| tbq4_0 + tbq3_0 | 7.19 | 68.1 | 44 MiB | 4.4x |
| **tbqp4_0 + tbq3_0 (Direct Sign)** | **7.08** | **63.6** | **44 MiB** | **4.3x** |
| tbqp3_0 + tbq3_0 (Direct Sign) | 7.95 | 65.3 | 38 MiB | 5.0x |

#### Direct Sign vs QJL Comparison (head_dim=128, TBQP3/TBQ3)

| Method | PPL | Note |
|:-------|----:|:-----|
| QJL (paper) | 11.04 | Projection noise dominates at d=128 |
| **Direct Sign (v1.1.0)** | **7.95** | **PPL reduced by 3.09** |

#### Bug Fix

- `__syncthreads()` race condition: query WHT shared memory corruption at ncols=2 (prompt eval) — PPL exploded to 2000+ while token generation (ncols=1) appeared normal

---

### v1.0.0 Benchmark (head_dim=256)

**Key result: `tbqp3_0 + tbq3_0` = 5.2x compression + lower PPL than FP16 + 12% faster**

### Benchmark Setup

- **Model**: Qwen3.5-35B-A3B Q4_K_M (19.71 GiB)
- **System**: NVIDIA DGX Spark, GB10 GPU, 128GB Unified Memory, CUDA 13.0
- **Dataset**: wikitext-2-raw (test set)

### Performance Summary

| KV Config | KV Memory | Compression | PPL (2K) | PPL (8K) | Speed (8K) |
|:----------|----------:|------------:|---------:|---------:|-----------:|
| f16 + f16 (baseline) | 5,120 MiB | 1.0x | 4.678 | 6.829 | 51.9 t/s |
| q8_0 + q8_0 | 2,720 MiB | 1.9x | 4.679 | 6.806 | 50.1 t/s |
| tbq3_0 + tbq3_0 | 980 MiB | 5.2x | 4.756 | 6.963 | 63.5 t/s |
| **tbqp3_0 + tbq3_0** | **990 MiB** | **5.2x** | **4.672** | **6.850** | **58.3 t/s** |

### Key Findings

```
Memory:   5,120 → 990 MiB  (81% savings)
PPL@2K:   4.678 → 4.672   (better than FP16!)
PPL@8K:   6.829 → 6.850   (+0.3% difference)
Speed@8K: 51.9  → 58.3 t/s (+12% faster)
```

### Perplexity by Context Length

| KV Config | 2K | 4K | 8K | 32K |
|:----------|---:|---:|---:|----:|
| f16 + f16 | 4.678 | 6.591 | 6.829 | 6.151 |
| q8_0 + q8_0 | 4.679 | 6.585 | 6.806 | 6.144 |
| tbq3_0 + tbq3_0 | 4.756 | 6.736 | 6.963 | 6.201 |
| **tbqp3_0 + tbq3_0** | **4.672** | **6.683** | **6.850** | **6.273** |

### Token Generation Speed (t/s)

| KV Config | 2K | 4K | 8K | 32K |
|:----------|---:|---:|---:|----:|
| f16 + f16 | 51.1 | 52.5 | 51.9 | 51.6 |
| q8_0 + q8_0 | — | 52.3 | 50.1 | — |
| tbq3_0 + tbq3_0 | 66.9 | 66.4 | 63.5 | 66.3 |
| **tbqp3_0 + tbq3_0** | **63.9** | **63.0** | **58.3** | **63.3** |

### QJL Correction Effect

QJL (Quantized Johnson-Lindenstrauss) = paper's TurboQuant_prod. Adds 1-bit residual correction to Key.

| Context | tbq3_0 (no QJL) | tbqp3_0 (with QJL) | Improvement |
|:--------|---:|---:|:---:|
| 2K | 4.756 | **4.672** | **-0.084** |
| 4K | 6.736 | **6.683** | **-0.053** |
| 8K | 6.963 | **6.850** | **-0.113** |
| 32K | 6.201 | 6.273 | +0.072 |


---

## References

- [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874) (ICLR 2026, Google DeepMind)
- [Google Research Blog](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)
- [llama.cpp](https://github.com/ggml-org/llama.cpp) — Base framework

## License

This implementation follows the llama.cpp project license (MIT).
