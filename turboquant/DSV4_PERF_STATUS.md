# DSV4 Performance — Status & Attack Plan (GB10)

Working doc to resume DeepSeek-V4-Flash perf work. Last updated 2026-06-11.
Branch: `feat/deepseek4` @ `0389bac8b` (pushed; main is at v1.8.0 docs — merge these at next release).

## Current numbers (IQ2_XS-XL 3-split, GB10, greedy)

| Config | decode t/s | note |
|---|---|---|
| baseline f16 KV | **13.8** | was 13.4 before the fp8 QDQ fix (`1d75505b0`) |
| + MTP (`--spec-type draft-mtp --spec-draft-p-min 0.75 --spec-draft-n-max 2`) | 15.5–16.2 free text / ~17 regular | accept 98–100% |
| tbq3 KV + MTP + QDQ fix (current server config) | **15.79** essay | accept 96% |
| prefill | 160–164 t/s @14K prompt | ds4 reference: 343 |

Launch (the production config):
```
build/bin/llama-server -m .../IQ2_XS-XL/DeepSeek-V4-Flash-IQ2_XS-XL-00001-of-00003.gguf \
  -c 16384 -ngl 999 -fa on --jinja -t 4 -ctk tbq3 -ctv tbq3 \
  --spec-type draft-mtp --spec-draft-p-min 0.75 --spec-draft-n-max 2 \
  --host 0.0.0.0 --api-key "occultsaint@X" --port 8888
```

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

### ① iq2_xs MMVQ — 33% of decode at ⅓ bandwidth ← START HERE
- Kernel: `vec_dot_iq2_xs_q8_1` @ `ggml/src/ggml-cuda/vecdotq.cuh:1020`; dispatch in `mmvq.cu` (MUL_MAT_ID n=1 path).
- Structure: per 32-elem subblock → 4× 512-entry `iq2xs_grid` lookups (`uint2`, 4KB table) + ksigns unpack + dp4a. Random-index global/constant loads are the suspected stall.
- Ideas, in test order:
  1. **Stage `iq2xs_grid` (4KB) into shared memory** at kernel start (one-time per block); same for ksigns if tabled.
  2. Check occupancy/launch shape of mmvq for ne0=4096, 8 experts, n=1 (rows-per-block / blocks count on GB10's SM count); try wider rows_per_cuda_block.
  3. Compare against `dequantize_mul_mat_vec` path if it still exists for iq2_xs.
- Measure with DSV4_KERNEL_PROF before/after; success = iq2_xs effective GB/s 87 → 150+ (decode +10–15%).
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
