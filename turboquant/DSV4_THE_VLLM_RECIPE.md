# The vLLM recipe for DSV4 decode — from the reference implementation's own writeup

Source: **"DeepSeek V4 in vLLM: Efficient Long-context Attention"**, vLLM blog, 2026-04-24
(https://vllm.ai/blog/2026-04-24-deepseek-v4). Plus the DSA background post
(https://blog.vllm.ai/2025/09/29/deepseek-v3-2.html) and the DSpark thread on the NVIDIA forums.

They diagnose exactly the problem our graph-mode profile found, and they say what they did about it.

> **"This model requires many small, mostly memory-bound kernels. We need to avoid extra launches
> and HBM round-trips that would otherwise slow the full decode path."**

The value of fusion here is NOT the launch count (we proved that: deleting 313 kernels/token moved
the step 0.3 ms, so a kernel in a replayed graph costs ~1 us). It is the **HBM round-trip** -- the
intermediate tensor that gets written and read back.

## The three fusions, with their own speedup numbers

| # | fusion | their speedup | our ops (graph-mode measured) |
|---|---|---|---|
| 1 | **Compressor + RMSNorm + RoPE + cache insertion.** "After compression, the compressed K immediately goes through RMSNorm, RoPE, and insertion into the following attention's KV cache." | **1.4-3x** | the whole DSA compressor chain: SOFT_MAX 0.36, SUM_ROWS, MUL, CONCAT(pool), GET_ROWS(state_perm), SET_ROWS(state) 0.65, CONT+CPY(state stores) 0.84, ROPE_TAIL 0.25, FP8_KV_QUANTIZE 0.4 -- ~4-5 ms |
| 2 | **Inverse RoPE + fp8 quant -> o_lora fp8 batched matmul.** | **2-3x** | out-inverse ROPE_TAIL + `dsv4_grouped_out` -- ~0.5-1 ms |
| 3 | **Fused Q norm + KV RoPE + K insert** ("KV cache insertion for both the compressed path and the sliding-window path"). | **10-20x** | RMS_NORM(Q) 0.22 + ROPE_TAIL(Q,KV) ~0.5 + FP8_KV_QUANTIZE(raw) 0.4 + SET_ROWS(raw) 0.23 -- ~1.35 ms |

## Multi-stream

> "The operations before main attention are highly parallelizable. They break into three pieces:
> **indexer computation, main-attention KV compression, and sliding-window token insertion**."
> "For `c128a` layers, which have no indexer, we run main KV compression in parallel with SWA token
> insertion. For `c4a` layers, we run the full indexer pipeline on its own stream in parallel with
> main KV compression and SWA token insertion."
> "With these overlaps, we observe a **5-6% end-to-end latency reduction at low batch sizes**."

This is the answer to the other half of our finding: every tiny op runs on ONE block = ONE SM of 48,
so the GPU is ~98% idle during each. ggml-cuda already has the fork/join machinery
(`stream_ctx.concurrent_events`, `ggml_cuda_graph_evaluate_and_capture`) -- it just is not used here.
Note DSV4's compress-ratio array (deepseek4.cpp:3343) already tells us which layers are c4a
(ratio 4, 21 layers, indexer) vs c128a (ratio 128, 20 layers, no indexer).

## Unified page layout

> "Largest bucket: `c4a` main KV, SWA KV, `c4a` compressor state, `c128a` compressor state.
> Middle bucket: C4 indexer KV, C4 indexer compressor state. Smallest bucket: `c128a` main KV."

Five cache types into three page-size buckets. We keep the raw SWA window and the compressed KV in
SEPARATE tensors, which is why `k_all = ggml_concat(k_raw, kv_comp_cache)` (deepseek4.cpp:2686) has
to PHYSICALLY COPY 512 KB per layer per token -- 0.385 ms, plus the mask CPY+CONCAT (another ~0.5 ms).
Allocate them adjacent and `k_all` becomes a **view**: the concat disappears entirely.

## Sparse attention: never materialize the gather

From the DSA post: FlashMLA "fuses the two-path scores, causal mask, and attention into a single
kernel"; SGLang's DSA kernel "fuses the gather into the attention loop, **loading selected KV entries
directly from HBM to SRAM without materializing an intermediate tensor**"; top-k "can be expressed
with a fused kernel". vLLM uses DeepGEMM's lightning-indexer kernels + FlashMLA's sparse attention.

We do the opposite: ARGSORT -> GET_ROWS (materialize the selected KV) -> CONCAT -> FLASH_ATTN.
Our indexer score path is dormant at short context but turns on past `indexer_top_k * 4`.

## Where this leaves the arithmetic

    our step now   50.4 ms   (plain, 17.6 t/s)      MoE 14.8 + dense ~21 + glue 12.8 + FA 1
    vLLM           25   ms   (plain, 40-45 t/s)

Recipe value: fusion 1 (~-3 ms) + fusion 2 (~-0.5) + fusion 3 (~-1.2) + kill the k_all/mask concat
(~-0.9) + multi-stream (~-2.5) = **~-8 ms of the 12.8 ms of glue**. Plus MoE 14.8 -> ~10 and the rest
of the mirrored dense. That lands the step near 30 ms = ~33 t/s plain, and a working drafter on top.

## Order (biggest multiplier per unit of risk)

1. **Unified/adjacent KV allocation** -> `k_all` and the mask concat become views. No new kernel.
2. **Fusion 3** (Q norm + KV RoPE + K insert) -- their 10-20x, and it is a small, well-bounded kernel.
3. **Multi-stream** the three pre-attention branches -- ggml-cuda's fork/join already exists.
4. **Fusion 1** (the compressor step) -- biggest ms, biggest kernel.
5. **Fusion 2** (inverse RoPE + fp8 quant into the o_lora bmm).
