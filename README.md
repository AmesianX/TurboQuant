# TurboQuant KV Cache Compression for llama.cpp

> Implementation of [TurboQuant (ICLR 2026, Google DeepMind)](https://arxiv.org/abs/2504.19874) — KV cache compression via Walsh-Hadamard Transform + Lloyd-Max quantization with QJL correction

[🇰🇷 한국어](README_KO.md)

### 🆕 v1.5.3 — Double WHT Per-Head for head_dim=64 (GPT-OSS 120B)

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
