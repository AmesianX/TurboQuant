# DSV4 decode — the real map (2026-07-13). 56 ms/token, fully accounted.

This replaces every earlier enemy list. Those were built on `DSV4_OPPROF` running COLD (it dumped
inside warmup, so kernel first-launch/JIT landed in the table) and on a stack whose NCCL was
silently on TCP. Both are fixed (25f2db446, b9f920e09). The numbers below are a WARM steady-state
decode profile (`DSV4_OPPROF_SKIP=40000`), plain decode, RDMA + ATTN_SPLIT, ~12.8 tokens.

## The reference we are chasing (same 2x GB10, measured by others and by us)

| stack | decode t/s, single stream |
|---|---|
| vLLM + b12x, plain (no spec) | **40-45** |
| + stock MTP | ~50 |
| + DSpark drafter (block-wise Markov head) | **60-67** |
| **ours (llama.cpp), plain** | **16** |

vLLM's plain step is ~25 ms. Ours is 56 ms of GPU time. Speculation is only the last 1.3-1.5x on
top — **the step is the whole game.**

## Where our 56 ms goes (warm, per token, per rank)

| | ms | verdict |
|---|---|---|
| `DSV4_MOE_GROUPED` (43 calls) | **16.4** | 131 GB/s achieved. **~10 ms at DRAM speed** |
| f8 dense GEMV (q_a/q_b/wo/kv, 216 calls) | 12.8 | **already at DRAM speed — this is the floor** |
| f8 shexp (mirrored) | 3.9 | halves if split |
| f8 lm_head (mirrored) | 2.2 | halves if split |
| bf16 compressor/indexer weights | 3.2 | **not quantized** |
| f32 router + hyper-connection weights | 2.2 | **not quantized** |
| **glue: CONT 527, GET_ROWS 276, SET_ROWS 229, CONCAT 166, ROPE_TAIL 191, CPY 165, RMS_NORM 130, ARGSORT 40, ...** | **14.9** | **~2100 tiny kernels/token** |
| FLASH_ATTN_EXT (43) | 1.0 | healthy |
| | **56** | |

~2700 kernel launches per token. vLLM's step is MoE + dense + almost nothing else.

**Our dense GEMV kernels are NOT slow.** Measured standalone (`scratchpad/gemv_bw.cpp`):
lm_head f8 214 GB/s, q_b 244, wo_b 349, q_a 226, shexp 295 — 93-150% of the 230 GB/s DRAM ceiling.
Do not "optimize" them; they are done.

## What the glue actually is

The DSA (sparse-attention) pipeline, expressed as thousands of ggml ops instead of a few kernels:

    compress_gate/kv matmul -> SET_ROWS (write compressed KV cache)
    indexer q_b/proj matmul -> ARGSORT (top-k) -> GET_ROWS (gather selected KV)
    -> CONCAT -> CPY -> CONT -> sparse FLASH_ATTN

b12x has this as fused kernels. That is the gap. It is a fusion problem, not a ggml-can't problem —
`DSV4_MOE_FUSED`, `DSV4_ROPE_TAIL`, `DSV4_HC_EXPAND`, `DSV4_FP8_KV_QUANTIZE` are already ours.

## Unquantized weights, read every token, mirrored on both ranks

    BF16  attn_compress_gate/kv  (4096x1024)x21 + (4096x512)x20   520 MB
    BF16  indexer compress_gate/kv + proj                          99 MB
    F32   ffn_gate_inp (router)  (4096x256)x44                    184 MB
    F32   hc_attn_fn / hc_ffn_fn (16384x24)x86                    135 MB
                                                            ------------
                                                      938 MB  =  5.4 ms

Load-time convert to F8 (the `DSV4_LM_HEAD_F8` trick, `llama-model.cpp:1881`) => ~400 MB, -3.4 ms.
Keep the router in bf16 rather than f8 unless a perplexity gate says otherwise — quantizing it
changes expert SELECTION, not just precision.

## The arithmetic to 45+

    56.0  now                                        -> 16 t/s
    -6.6  MoE GEVM 131 -> 220 GB/s   (b12x micro.py orchestration: persistent CTA, 512 threads,
                                      task-strided, m==1 specialization. Fully analysed in
                                      DSV4_W4A16_B12X_ANALYSIS.md section 2. The earlier port
                                      swapped only the INNER MATH (fp4_dot8) and measured +2% --
                                      the win is in the ORCHESTRATION, not the dot product.)
    -10.0 fuse the DSA compressor+indexer pipeline   (the 2100-kernel swarm -> a few kernels)
    -3.4  quantize the aux weights above
    -5.0  split the still-mirrored dense (shexp, q_a, kv, lm_head) -- boundaries are cheap now
    ------
    31.0 ms -> ~32 t/s plain
      x MTP with its draft overhead removed (~1.15)  -> ~37
      x a DSpark-class drafter (~1.3)                -> ~48

Every term is measured. Nothing here is a guess.

## Order of work (risk-adjusted)

1. **MoE micro-kernel orchestration** — biggest single kernel, complete spec already written, and
   b12x is a bit-parity oracle. Self-contained.
2. **Aux weight quantization** — cheap, proven trick, immediate -3.4 ms.
3. **DSA pipeline fusion** — biggest prize (-10 ms), most invasive. Do it after 1-2 have banked.
4. **Split the remaining mirrored dense** — `shexp` folds into the MoE AllReduce for free
   (adjacent PARTIALs joined by a plain `ggml_add`, deepseek4.cpp:3051).

## Tools (use these, not the old ones)

- `DSV4_OPPROF=1 DSV4_OPPROF_SKIP=40000 DSV4_OPPROF_DUMP_AFTER=40000 GGML_CUDA_NO_GRAPHS=1`
  — WARM profile. Without SKIP it is cold and lies. NO_GRAPHS is mandatory (per-op CUDA events
  cannot be recorded inside a captured graph), so ignore that run's absolute t/s and read only the
  proportions. `DSV4_OPPROF_BYOP=1` rolls up by op; `DSV4_OPPROF_NAMES=1` adds node names.
- `scratchpad/gemv_bw.cpp` — standalone GB/s per weight shape. Link with
  `-Wl,--no-as-needed -lggml -lggml-base -lggml-cpu -lggml-cuda`; GB10 enumerates as **IGPU**.
- `DSV4_TP_NO_REDUCE=1` — prices the collectives (wrong output; diagnostic only).
- NCCL: **one rail** (`mlx5_0`). Four OOM the box at CTX=262144 and buy nothing (1 = 2 = 4 measured).
