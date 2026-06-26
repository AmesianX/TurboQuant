# DSV4 prefill / DSA port — resolved diagnosis + port map (2026-06-26)

## The mystery (a week of going in circles), RESOLVED with cited research + profiling

**Symptom:** prefill ~100 t/s on 2-node FP4 TP; CodeMason agent (2.9k-token system prompt) = 147s/turn, every turn re-prefills (RS/recurrent cache can't trim backward).

### What it is NOT (dead theories, proven wrong):
- NOT network/RDMA bandwidth. Switched NCCL socket→**4-rail RoCE RDMA** (NCCL_IB_HCA=rocep1s0f0,rocep1s0f1,roceP2p1s0f0,roceP2p1s0f1, GID 3, all 200G ACTIVE) — prefill 102→102, **zero change**. AllReduce bandwidth is not the bottleneck.
- NOT the DSA "gather-vs-mask". Read ds4's `attention_prefill_mixed_softmax_kernel` — **ds4 ALSO uses a mask** (comp_mask), dense softmax over all n_comp, exactly like our `dsv4_build_compressed_mask_from_topk`. There is no gather to port. DSA sparsity in ds4 = same mask we have.
- NOT (mainly) attention. Profiled single-box IQ2 prefill vs context length: 232@700 → 214@12.6k tok = **FLAT** = NOT O(L²) attention-bound = **MoE-GEMM/compute-bound**.

### What it IS (cited, 3-0 verified — research task wt34jtcif):
- ds4/DwarfStar gets 343–458 t/s prefill on single GB10 with **fused hand-written CUDA (106 kernels, cublasGemmEx FP16 prefill)** — NOT FP4, NOT a sparse-attention trick. **Pure kernel/engine efficiency.**
- llama.cpp's **NVFP4 path is functionally broken** (block_nvfp4 can't dequant without F32 scale; ggml-org/llama.cpp#22042) → our FP4 MoE likely falls back slow.
- ggerganov's own bench: llama.cpp does **2009 t/s prefill on GPT-OSS-120B (dense MoE) single GB10** → the GEMM engine is NOT intrinsically slow; the V4-Flash gap is kernel-specific (compressor object-explosion O(n²) blocks -ub>256; FP4 broken; generic ggml MoE vs ds4 fused).
- Realistic single-GB10 V4-Flash prefill ceiling ≈ **460 t/s** (Entrpi/ds4-on-spark). 2-Spark vLLM (jasl fork, hazyumps recipe) = **330–613 prefill / 31–44 decode**.

## Numbers measured this session
| config | prefill | decode |
|---|---|---|
| 2-node FP4 TP -ub256 (project) | ~100 | ~15–19 |
| single-box IQ2 -ub256 (NOT the project) | 239 | 16.5 |
| ds4 single-box q2 (ref) | 343–458 | 13–16 |
| 2-Spark vLLM jasl (ref) | 330–613 | 31–44 |

## Port map — ds4 kernels → our ggml-cuda (TP-aware)
Pattern already established: `ggml/src/ggml-cuda/dsv4.cu` + `GGML_OP_DSV4_*` ops, called from `src/models/deepseek4.cpp`.

**Already ported (helpers):** hc_split_sinkhorn, hc_weighted_sum, hc_expand, fp8_kv_quantize, rope_tail.

**NOT ported (the heavy compute = the gap; deepseek4.cpp builds these with generic ggml mul_mat/soft_max):**
| ds4 kernel | role | TP handling | priority |
|---|---|---|---|
| `routed_moe` + `matmul_q8_0_preq_warp8`/`grouped_q8_0_a_preq_warp8` | MoE expert GEMM | **local expert shard only** (meta-backend splits experts; kernel does the local half) | **#1 = the bottleneck** |
| `attention_prefill_mixed_softmax` / `attention_indexed_mixed_*_online` | fused online-softmax attn | MIRRORED (runs identical per box, TP-safe, drop-in) | #2 (tractable, but profile says low ROI alone) |
| `indexer_scores_wmma*` | WMMA indexer scoring | mirrored | #3 |
| `indexer_topk_*` | fast top-k | mirrored | #3 |

**TP rule:** attention/indexer = mirrored → port as-is. MoE GEMM = operates on whatever local expert shard the meta-backend hands it (TP split is ABOVE the kernel). Each new op also needs a split-state entry in `ggml-backend-meta.cpp` (MIRRORED for attn/indexer; experts follow the existing MoE axis).

**New-op checklist per kernel:** ggml.h enum+API → ggml.c op-create+shape → dsv4.cu kernel+launcher → dsv4.cuh decl → ggml-cuda.cu dispatch+supports_op → ggml-backend-meta.cpp split-state → deepseek4.cpp swap generic build → build on .66 → rsync .67 md5 → 2-node verify (output coherent) → measure prefill.

## Honest expectation
- Porting **attention** alone ≈ no prefill win (profile: MoE-bound; ds4 masks too).
- The win is **MoE GEMM** (#1) + fixing the **FP4 broken path** (or moving to IQ2/Q2_K like ds4).
- This is multi-WEEK kernel engineering, build-verify per kernel. Not one-shottable; blind CUDA = breakage.

## Alternatives (if not the multi-week llama.cpp port)
- 2-Spark **jasl/vllm** fork (hazyumps recipe) = the documented fast path for THIS exact hardware (330–613 prefill, 31–44 decode, working prefix-cache). repos: hazyumps/deepseek-v4-flash-gb10, jasl/vllm.
- ds4 single-box (458) — but not 2-node, not our project.

## Current server state
Single-box IQ2 launched for the profile test (port 8080). Project config = `tp-serve/tp.sh` (FP4 2-node, now with V4.jinja tool template + RDMA 4-rail + GPU-sampler-on). The V4 chat template fix (tool_calls now parse) IS real and landed in tp.sh.
