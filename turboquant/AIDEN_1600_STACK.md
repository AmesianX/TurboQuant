# AIDEN 1600/45 STACK — complete understanding + port map

**Task:** read + understand EVERYTHING in "Aiden's" (jasl + community) DeepSeek-V4-Flash work on
2× DGX Spark (GB10, sm121) and map the exact port into our llama.cpp fork. READ/UNDERSTAND/MAP
only — no code changes here.

**Sources read in full:**
- LOCAL `/home/user/work/vllm-spark/flashinfer-src/` — trtllm batched-MoE, CUTLASS grouped-MoE, sm120 GEMMs, MLA, the Python tile heuristic.
- vLLM PR **#41834** (jasl) `github.com/vllm-project/vllm/pull/41834` + fork `jasl/vllm:codex/ds4-sm120-min-enable` (tag `sm120-pr-41834-stable-preview-20260626` @ `c766cbc6ff`).
- `github.com/hazyumps/deepseek-v4-flash-gb10` (branch **`master`**) — the proven 2×GB10 runbook.
- NVIDIA forum "SM121 CUTLASS NVFP4 356 TFLOPS"; `github.com/BTankut/dgx-spark-sglang-moe-configs`; TRT-LLM issue #11368.

---

## 0. CRITICAL REALITY CHECK — what "1600/45" actually is

The prompt's "1600 prefill / 45 decode" is the **best-case number from the PR description**, NOT a sustained single-stream chat number. Reconcile the three numbers carefully — they are different regimes:

| Source | Prefill tok/s | Decode tok/s | Regime |
|---|---|---|---|
| PR #41834 description (RTX SM120, 2× PRO 6000, TP=2, decode-gate ON) | **1595–1722** | 38–42 (40 @ C=1) | datacenter GPU, batched prefill @ d8192, decode gate ON |
| PR #41834 (GB10 SM121, 2-node TP=2) | **1722.5 @ d8192** | **38.5 @ d8192** | the "1600/45" the user cites — d8192 batched prefill |
| hazyumps GB10 runbook (proven, single-stream) | ~330–430 @ 2–17k | **31–42** (MTP n=2) | real single-stream chat, 384K ctx |

**So the 1600 is a PREFILL-THROUGHPUT number measured by feeding a large batch of prefill tokens
(`d8192` = 8192 batched tokens), not a TTFT speedup on a single prompt.** The hazyumps runbook —
which is the *actually-running, validated* recipe — reports ~405 prefill / ~31–42 decode single-stream.
The "45" is decode peak with MTP n=2 at long context (~42 observed, ~38.5 in PR).

**Implication for us:** the 1600 is reachable ONLY in the same regime — a large batched-prefill
(many tokens in one forward), where the MoE GEMM has enough tokens/expert to fill tiles AND the
sparse-MLA prefill kernel is the fast indexed-split one. Our single-prompt prefill will not hit 1600
just by porting kernels; it hits it the same way Aiden does — batch the prefill tokens (large `-ub`)
so the MoE is not tile-starved. This is the through-line of the whole stack.

---

## PART A — PREFILL ≈ 1600: the exact mechanism

### A.0 The surprising headline finding from the PR

**The 1595–1722 prefill on SM120/SM121 is NOT driven by a new MoE GEMM.** Per the PR #41834 diff
(verified file-by-file), MoE on SM120 is just **stock**:
- **FP8 DeepSeek-V4-Flash** → DeepGEMM "mega-MoE" grouped path (`fp8_fp4_mega_moe` / `transform_weights_for_mega_moe`, `vllm/utils/deep_gemm.py` L511+).
- **NVFP4 variant** → **Marlin W4A16** (because `support_deep_gemm()` in `vllm/platforms/cuda.py` deliberately *drops* family-120 → forces Marlin/CUTLASS).
- SM12x-tuned MoE config JSONs + the Marlin-MoE cudagraph attr fix (`csrc/.../marlin_moe_wna16/ops.cu` L440/L531/L534).

The two things that actually move the prefill needle to 1595–1722 are **attention-side**, not MoE:

1. **Indexed-D512 split/chunked sparse-MLA Triton prefill kernel** — new 3521-line file
   `vllm/v1/attention/backends/mla/sparse_mla_kernels.py`, gated `VLLM_DEEPSEEK_V4_INDEXED_D512_SPLIT_PREFILL=1`,
   `..._MIN_TOKENS=4096`. This is the prefill fast path.
2. **Removal of the per-step device→host sync** in the indexer metadata builder
   `vllm/models/deepseek_v4/sparse_mla.py` (`build_c128a_topk_metadata`): replaces
   `int(positions.max().item())` (a CUDA sync every step) with a CPU-side `max_seq_len-1` upper bound,
   128-aligned (`_C128A_TOPK_ALIGNMENT`). Eliminates a per-step stall.
3. **The fused tf32 Triton MQA-logits top-k indexer** (hazyumps `patches/sm12x_deep_gemm_fallbacks.py`,
   bind-mounted over `vllm/v1/attention/ops/deepseek_v4_ops/sm12x_deep_gemm_fallbacks.py`):
   replaces a bf16 cuBLAS head-loop (CUDA cores, tensor cores idle, ~1 GiB/iter score materialization)
   with a fused tf32 Triton MQA-logits kernel. **+29% prefill at 9k (313 → 405 tok/s)** on GB10. The
   bf16-matmul-inputs fix (was FP32 SGEMM) is what unfreezes concurrency (4-concurrent 0.1 → ~60 tok/s).

**This directly corroborates our own finding** (`project_dsv4_prefill_speed`, `project_dsv4_prefill_dsa_port`):
prefill is kernel-efficiency / indexer-projection bound, NOT FLOP-bound. Aiden's win is the same lever
we identified — the indexer/attention projection — plus eliminating a sync.

### A.1 BUT — the MoE *can* be the bottleneck, and Aiden's MoE is NOT tile-starved at our shape

Even though the PR's MoE is "stock," the reason it isn't a bottleneck at d8192 is structural, and it's
exactly where OUR CUTLASS port dies at 18 TF/s. Two distinct MoE designs exist in flashinfer-src:

#### (i) trtllm-gen **batched**-MoE — pads tokens to `tile_tokens_dim` (THE efficient design)
Files: `csrc/trtllm_fused_moe_runner.cu`, `.../routing_deepseek.cu`, `include/flashinfer/trtllm/fused_moe/RoutingKernel.cuh`, `flashinfer/utils.py`.

Mechanism (file:line):
- Each expert's token count is rounded UP to a full tile: `numCta = divUpLog2(count, mPaddingLog2)`
  = `ceil(count / tile_tokens_dim)` — `RoutingKernel.cuh:296,500`, `routing_deepseek.cu:350`.
  An expert with **6 tokens** and `tile_tokens_dim=8` → `numCta=1`, runs **one full MMA tile of 8 rows**,
  never a degenerate M=6 micro-GEMM. Empty experts → 0 CTAs, cost nothing.
- Ragged (expert→variable rows) is flattened to a dense CTA list via `cub::BlockScan` exclusive-sum
  (`RoutingKernel.cuh:299,503`), producing `CtaIdxXyToBatchIdx[i]`=expert and `CtaIdxXyToMnLimit[i]`=valid
  row bound (`min`-clamped so pad rows are masked, not garbage-MAC'd) — `:303-310`.
- Padded total = `permutedIdxSize = numNonExitingCtas << mPaddingLog2` (`:320-324`). This is the M the
  batched GEMM iterates.
- **`tile_tokens_dim` is auto-chosen from expected load** — `flashinfer/utils.py:116-137`
  `calculate_tile_tokens_dim`: `per = num_tokens*top_k/num_experts`, `*1.3` imbalance, `next_pow2`,
  clamp `[8, 64|128]`. For DSV4 (256 experts, top-6): ~6 tok/expert → tile **8**; ~24 → tile **32**.
- One launch, persistent grid sized to worst case, `numNonExitingCtas` trims surplus CTAs at runtime
  (`runner.h:84-86`). PDL chains routing→GEMM1→act→GEMM2→finalize with no full sync.

**Result:** every active expert ≈ one full, well-formed MMA tile. No tile starvation by construction.

#### (ii) FlashInfer **CUTLASS grouped**-MoE — does NOT pad tokens (tile-STARVED, = our path)
Files: `csrc/fused_moe/cutlass_backend/cutlass_fused_moe_kernels.cuh` (4881 lines).

- Per-expert row count = `expert_first_token_offset[e+1] - expert_first_token_offset[e]` (`kernels.cuh:1195-1198`),
  fed RAW as the CUTLASS **N** dimension (tokens are N, not M — `kernels.cuh:1198-1203`, comment
  "M and N transposed since we are using the #tokens as the N dimension").
- **No token padding to a tile multiple anywhere** (grep: zero `tile_tokens_dim` in this file). The only
  padding is on the scale-factor buffer (`alignToSfDim`, `kernels.cuh:896-956,1585-1624`) — does NOT
  change the GEMM matrix dims.
- sm120 tile = `CtaShape128x128x128B` default (TileN=128) — `cutlass_heuristic.cpp:488-503`. So for
  DSV4 with 6–24 tokens/expert: `ceil(N/128)=1` tile per expert, but the tile computes a **full 128-wide
  N**, of which only 6–24 rows are real. **Effective MMA utilization ≈ tokens/128 = 5–19%.** This is
  precisely the tile-starvation that pins our CUTLASS grouped port at ~18 TF/s.

#### THE CRUX ANSWER (question A)
> HOW does Aiden's MoE reach high TF/s at 256 experts × few tokens/expert when ours is tile-starved at 18 TF/s?

**Mechanism = (i) a different MoE kernel design (token-padding to a small `tile_tokens_dim`), PLUS (iii)
a much larger effective batch (d8192 batched prefill), PLUS (ii) EP halving experts/GPU.** Concretely:

- **(i) Different kernel — PRIMARY.** Aiden's efficient path is the trtllm-gen **batched**-MoE that pads
  each expert to a full `tile_tokens_dim` tile (8/16/32). Our `dsv4-moe-grouped.cu` is the CUTLASS
  **grouped** design that pads each expert's M up to **128**
  (`dsv4-moe-grouped.cu:208` `mpad=((m+127)/128)*128`) → at 6–24 tokens we pay a 128-row tile = the same
  5–19% utilization. **Our padding quantum is 128; Aiden's is 8.** That single number is the starvation.
- **(iii) Bigger effective batch.** At d8192 with 256 experts × top-6, tokens/expert ≈ `8192*6/256 ≈ 192`
  — now even a 128-tile is well-filled. vLLM fits d8192 because (a) weights are EP-sharded (less weight/GPU),
  (b) paged KV + fp8 KV, (c) chunked prefill. Our path OOMs above `-ub~1024` because experts are MIRRORED
  (full 78GB weights on each GPU) so there's no activation headroom.
- **(ii) EP.** hazyumps runs `--enable-expert-parallel` with TP=2 → **128 experts/GPU** (256/2). Each GPU's
  GEMM sees the same tokens but half the experts → ~2× tokens/expert AND half the weight footprint
  (the OOM fix, TUNING.md: "halves expert weight per node"). PR #41834 confirms `num_local_physical_experts==128`.
- **(iv) Marlin vs NVFP4:** NOT the mechanism. PR uses Marlin only for the NVFP4 *variant* on SM120 because
  DeepGEMM is force-disabled; the FP8 model uses DeepGEMM mega-MoE. Marlin W4A16 is a fallback, not the
  1600 driver.

**So to melt Aiden's MoE into our code we need BOTH: (A) shrink our padding quantum from 128→`tile_tokens_dim`
(port the trtllm batched-MoE token-padding routing), AND (B) get the batch up (large `-ub` made affordable
by EP — split experts across the 2 nodes instead of mirroring).**

### A.2 Question B — what `-ub` and how does memory fit

- **`--max-num-batched-tokens 4096`** (both PR and hazyumps). PR's d8192 prefill benchmark feeds 8192-token
  batches (`_DEEPSEEK_V4_SPARSE_MLA_PREFILL_WARMUP_TOKENS=8192`); `..._MIN_TOKENS=4096` gates the indexed-split
  prefill kernel on. `--max-num-seqs 2` (hazyumps) / 24 (throughput).
- **Memory fits because of EP + paged/fp8 KV + low gpu-mem-util headroom:**
  - `--enable-expert-parallel` → 128 experts/GPU (~half the MoE weight). **This is the OOM fix.**
  - `--kv-cache-dtype fp8`, `--block-size 256`, paged KV.
  - `--gpu-memory-utilization 0.80` (GB10) — 384K ctx at 0.80 → ~5.5× concurrency headroom.
  - `--max-model-len 393216` (384K).
  - **No NVFP4/Marlin pre-quant in the runbook** — hazyumps loads native safetensors (`--load-format safetensors`,
    ~148GB), only KV is fp8. (The NVFP4 path is a separate PR variant.)

**Our wall:** experts MIRRORED in the meta-backend → full weights on both GPUs → no headroom → OOM above
`-ub~1024`. **EP (split experts across the 2 nodes) is the missing piece for both the memory fit AND the
tokens/expert fill.**

---

## PART B — DECODE ≈ 45 (38.5–42): the exact mechanism

### B.1 MTP draft
- **`--speculative-config '{"method":"deepseek_mtp","num_speculative_tokens":2}'`** — the model's built-in
  Multi-Token-Prediction head, **n=2**, no separate draft model. "MTP ~doubles decode throughput" (hazyumps).
- The "essential" MTP correctness fix (PR #41834 `vllm/v1/spec_decode/llm_base_proposer.py`):
  - `logits = logits.to(torch.float32)` before `apply_top_k_top_p` (L1804+) — the MTP head emits bf16,
    the triton top-k/top-p sampler asserts fp32; without the cast any non-greedy chat kills the engine.
  - `_enable_probabilistic_draft_probs` True for `method=="mtp"` → sample from `softmax(draft_logits)` instead
    of argmax → acceptance **58.9% → 67.8%** (~9pp).
  - `_get_effective_spec_step_idx()` cycles `spec_step_idx % num_nextn_predict_layers` to route through the
    correct draft layer.
- CUDAGraph: DSV4 default **`FULL_AND_PIECEWISE` + torch.compile** (NOT breakable — breakable is 1.5–3.8×
  SLOWER for MTP decode on SM120). Under spec-decode, the per-token sparse-MLA attention is eager-broken out
  of the FULL graph (`breakable_cudagraph.py` `_BREAK_DSV4_ATTN_UNDER_FULL_FOR_SPEC`) because it
  cross-contaminates requests via per-request block_table/topk gather. Capture sizes force exact small
  interactive sizes `{1..32}*multiple` so FULL decode graphs don't replay padded virtual requests.

### B.2 Sparse-MLA decode kernel
- `vllm/model_executor/layers/sparse_attn_indexer.py`: on family-120, `cooperative_topk` (Hopper TMA kernel)
  is DISABLED; `_should_use_sm120_short_row_topk_decode()` picks `top_k_per_row_decode` when `topk_tokens==512`
  and width small, else `persistent_topk`.
- The SM120 MQA-logits/topk impl is the torch/triton fallback in `sm12x_deep_gemm_fallbacks.py` + `sm12x_mqa.py`
  (no native sm121 lightning-indexer kernel exists). `fp8_fp4_mqa_topk_indices` writes top-k indices WITHOUT
  materializing full fp32 logits (`deep_gemm.py:520+`).
- `indexer.py`: `sparse_indexer_max_logits_bytes()` = **256MB on SM12x** vs 512MB elsewhere (the GB10 smem/
  memory budget).

### B.3 The 2-node plumbing that makes it not-crash (hazyumps NETWORK.md — the reliability core)
- **NCCL 2.30.4 mandatory** via `LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libnccl.so.2.30.4` — PyTorch's bundled
  ~2.28 deadlocks ("No available shared memory broadcast block found in 60 seconds"). RDMA must engage:
  `--device=/dev/infiniband --cap-add=IPC_LOCK --ulimit memlock=-1:-1`, `NCCL_IB_DISABLE=0`,
  `NCCL_IB_HCA=rocep1s0f0`, `NCCL_SOCKET_IFNAME=enp1s0f0np0`. TCP fallback drops you to ~12 tok/s.
  RoCE single cable MTU 9000, `--master-port 29519`, `--distributed-executor-backend mp` (NOT Ray).
- Build: `TORCH_CUDA_ARCH_LIST=12.1a` / `FLASHINFER_CUDA_ARCH_LIST=12.1a` (default 12.0+PTX → non-native cubins).
- `--no-enable-flashinfer-autotune` (skip 10-min startup autotune).

---

## PART C — THE GB10 TILE/SMEM CONSTRAINT (why kernels must shrink)

From NVIDIA forum + TRT-LLM #11368 + BTankut configs:
- **GB10/sm121 shared memory = 101,376 bytes (~99 KiB)** — same as RTX 4090, vs ~228 KiB on B200.
  This is THE constraint, "not compute."
- Dense NVFP4 peak: **356 TFLOPS** (71% of 500 TF). MoE grouped-GEMM tiles that fit:
  - **256×128 (N=256) → 154 TFLOPS** (prefill-optimal, the "256x128 smem note").
  - **128×128 → ~147 TFLOPS** (decode).
- Stock B200 FP4 tiles **256×128×128B** and **128×128×256B** each need **>99 KiB → OutOfResources** on GB10
  (`fp4_gemm_template.h` routes sm120/121 down the B200 path with too-large tiles — the #11368 bug).
- SGLang default MoE config requests **147,456 bytes** → dies; BTankut's GB10 configs are shrunk to the
  101,376-byte budget (small BLOCK_M 16–32, num_stages 2). The fix is `tile_bytes × num_stages ≤ 101,376`.
- **Our `dsv4-moe-grouped.cu` already uses `ThreadBlockShape Shape<_128,_128,_128>` + `ClusterShape<_1,_1,_1>`
  + `StageCountAutoCarveout` (line 64-78)** — auto-carveout should keep it within smem, which is why it BUILDS
  and RUNS; it's the token-padding-to-128 (not smem) that starves it.

---

## PART D — PORT MAP into our fork (concrete, file:line)

Three independent workstreams. **#1 (token-padding) is the single highest-leverage change** and is purely
local to our MoE kernel — no TP/EP rework required to test it.

### D.1 [PRIMARY] Shrink MoE padding quantum 128 → `tile_tokens_dim` (port trtllm batched-MoE token-padding)

**Our problem:** `ggml/src/ggml-cuda/dsv4-moe-grouped.cu:208`
```
int mpad=((m+127)/128)*128;   // <-- pads each expert to 128 rows; 6–24 real → 5–19% fill
```
and the CUTLASS grouped GEMM (`prep` kernel `:93-116`, `GroupedGemm::run` `:151-170`) feeds per-expert
problem-shape `PS(m,n,k)` where the M-tile is 128. At decode/low-token shapes each expert pays a 128-row tile.

**Two options, in order of effort:**

**Option 1A (low risk, port the heuristic + smaller M-tile):** Add a `tile_tokens_dim`-style chooser and
drop the M-tile so each expert's padded M matches the real load.
- Port `calculate_tile_tokens_dim` (`flashinfer/utils.py:116-137`) verbatim into our host code: pick
  `tile = clamp(next_pow2(num_tokens*top_k/num_experts * 1.3), 8, 64)`.
- The CUTLASS grouped GEMM's M-tile is fixed at compile time (`ThreadBlockShape<_128,_128,_128>`,
  `dsv4-moe-grouped.cu:66`). To get an 8/16/32 M-tile we'd need to **instantiate the grouped GEMM with a
  smaller `ThreadBlockShape` M** (e.g. `<_16,_128,_128>` or `<_32,...>`) — but CUTLASS sm120 blockscaled
  MMA min M-tile is constrained; the CUTLASS CHANGELOG notes sm120/121 gained **128×32 and 128×64** tile
  support (M-tile stays 128, N shrinks). **So the grouped design cannot easily get an M-tile below ~128 on
  sm120.** This is why Option 1B (switch designs) is the real answer.

**Option 1B (the actual fix — port the trtllm-gen BATCHED MoE):** Replace/augment our grouped path with the
batched-MoE routing+padding so the padding quantum is `tile_tokens_dim` (8/16/32), not 128.
- Port the routing metadata kernels: `RoutingKernel.cuh` `routingIndicesClusterKernel` (numTokens≤1024 →
  decode/small-prefill, our case) → produces `numCta = ceil(count/tile)`, exclusive-sum, `CtaIdxXyToBatchIdx`,
  `CtaIdxXyToMnLimit`, `permutedIdxSize` (`:296-324`). For DeepSeek routing use `routing_deepseek.cu`
  (sigmoid+bias, top-2-in-group, top-4-groups, top-k experts, `mRouteScale` normalize — `:40-210`).
- The batched GEMM inner loop is a **prebuilt trtllm-gen cubin** (`TrtllmGenBatchedGemmRunner`,
  `trtllm_fused_moe_runner.cu:233-238,318-326`) — the FP4 MMA assembly is NOT in source, it's in
  `batched_gemm/trtllmGen_bmm_export` cubins. **Two sub-options:**
  - **1B-i:** ship the trtllm-gen FP4 batched cubin and call it through the metadata we build. Heavy
    integration (cubin + launcher), but it's literally Aiden's kernel.
  - **1B-ii (recommended, orthodox):** keep our CUTLASS grouped GEMM but feed it the **batched-padded**
    layout — i.e. pad each expert to `tile_tokens_dim` (8/16/32) instead of 128, then sub-tile the N (token)
    dimension. Since tokens are the N dim in our grouped GEMM (CUTLASS `(M=out, N=tokens, K)`), the relevant
    knob is **TileN**, and sm120 supports **TileN 32/64** (CHANGELOG 128×32/128×64). So: instantiate a second
    grouped-GEMM variant with `ThreadBlockShape<_128,_64,_128>` (or `_32`) for the decode/low-token shape, and
    change `mpad` from 128 to `tile_tokens_dim`. This keeps our proven CUTLASS path, gets the N-tile down to
    32–64, and lifts fill from ~10% to ~50–100%. **This is the minimal, orthodox melt of Aiden's idea.**
- Files to touch: `dsv4-moe-grouped.cu` (add `tile_tokens_dim` chooser + a 2nd small-N-tile GEMM
  instantiation + change `:208` and `:110-112` atom shapes to the chosen tile), `dsv4-moe-gemm.cu`
  (the `kGranN=128` tactic, `:27`), `dsv4-moe-grouped.cuh`.
- We already have a probe for exactly this: `turboquant/patches/dsv4_moe_tile_m_probe.cu` and
  `turboquant/DSV4_MOE_SMALL_M_TILE_STATUS.md` — resume there.

### D.2 [ENABLER] EP — split the 256 experts across the 2 nodes (meta-backend)

**Why:** EP is what gives Aiden (a) the memory headroom for large `-ub` (half the MoE weight/GPU) and
(b) ~2× tokens/expert. Our meta-backend currently **MIRRORS** experts (memory: "DSV4만 MLA mirror →
가속0, 용량전용") — both GPUs hold all 256 experts' weights.

**Map:**
- `ggml/src/ggml-cuda/dsv4-moe-grouped.cu` registry (`dsv4_moe_grouped_get_layer_nvfp4`, the per-rank NVFP4
  expert pointers): make each rank register only its **local 128 experts** (rank0: experts 0–127, rank1:
  128–255) instead of all 256. The grouped GEMM already iterates `ng` groups over a registry — feed it the
  local subset.
- Routing: tokens whose selected expert is non-local get masked/skipped on this rank (mirror trtllm's `-1`
  permuted index for non-local experts, `RoutingKernel.cuh:313-316`; `routing_deepseek.cu:364-411`). Each
  rank computes only its local experts' contribution.
- Combine: after both ranks' local MoE, **all-reduce (sum) the moe_out** across the 2 nodes — the per-token
  output is the sum over all selected experts, and each expert lives on exactly one rank, so a sum-allreduce
  reconstructs the full result. This is a NEW collective on the MoE output path.
- Meta-backend (`ggml/src/ggml-backend-meta.cpp`): today MoE expert tensors resolve via
  `GGML_BACKEND_SPLIT_AXIS_MIRRORED` (`:66-67`, and the NextN-MoE mirrored-tensor resolution at `:452-460`).
  To do EP we need a new split mode that shards the expert dimension (axis over `n_expert`) and inserts the
  sum-allreduce on the MoE output. This is the genuinely new infra. **Caveat from memory
  (`project_turboquant_cross_node_tp`): cross-node EP/MTP has been a dead-end for *decode* (latency-bound,
  RoCE amplifies). EP here is for PREFILL throughput + memory, where it's a win.** Scope EP to prefill/large-batch
  and keep decode on the mirrored path.

### D.3 [PREFILL ATTENTION] Indexed-split sparse-MLA + kill the per-step sync

**Why:** This is the OTHER half of Aiden's 1600 (the +29% indexer win + the sync removal). Maps onto our
existing sparse-MLA work.

**Map:**
- Our kernel: `ggml/src/ggml-cuda/fattn-sparse-mla.cu` (`sparse_mla_kernel` `:43`, TMA variant
  `sparse_mla_kernel_tma` `:195`). It already does the MQA-amortized gather + bf16 WMMA QK^T/PV. Aiden's win
  is the **fused tf32 Triton MQA-logits top-k** that replaces a bf16 cuBLAS head-loop. Check whether our
  indexer (the argsort/top-k that produces `kv_idx`, fed as `src[6]`) materializes full logits — if so,
  port the chunked/fused tf32 logits→topk (the `_fp8_mqa_logits_topk_triton` idea from
  `sm12x_deep_gemm_fallbacks.py`): per-head-chunk logits (cap score tensor ≤1 GiB,
  `_SM120_MQA_LOGITS_MAX_SCORE_BYTES`), fused top-512 without materializing the full fp32 score matrix.
- **Kill the per-step device→host sync** (Aiden's `positions.max().item()` → `max_seq_len-1`): audit our
  indexer metadata build (whatever computes the comp-segment width / `n_comp_view`) for any
  `cudaMemcpy`/`.item()`-style sync per decode step; replace with a CPU-side 128-aligned upper bound. This is
  the cheapest, highest-ROI prefill fix and matches `project_dsv4_1m_mtp_clear_rows_bug` lessons (sync/clear
  costs dominate).
- Gate the indexed-split prefill kernel on a min-token threshold like Aiden's `..._MIN_TOKENS=4096` so it
  only fires for batched prefill.

### D.4 [DECODE] MTP n=2 — we already have this; align the config
- We already run MTP (memory: real decode 19.5–24 t/s, 2-node). Aiden's deltas to adopt:
  - Ensure draft logits are **fp32 before top-k/top-p** (the bf16→fp32 cast bug) — check our sampler path.
  - Consider **probabilistic draft sampling** (sample from softmax, not argmax) for +9pp acceptance — this is
    a draft-sampler change, model-agnostic, low risk.
  - Keep `num_speculative_tokens=2` (Aiden's value; matches our setup).

### D.5 [PLUMBING] 2-node serving config to match Aiden (already mostly in our memory)
- `LD_PRELOAD` NCCL 2.30.4 if we hit the SHM-broadcast deadlock; RDMA caps; `mp` backend; MTU 9000;
  `TORCH_CUDA_ARCH_LIST=12.1a` for any flashinfer cubin we ship; `--no-enable-flashinfer-autotune`.
- fp8 KV (we already support quant-KV under `-sm tensor` mirrored, `project_dsv4_tp_quant_kv_and_graph_reuse`).

---

## SUMMARY — what makes 1600/45 and what to port

| Aiden mechanism | File/evidence | Why faster than ours | Our port target |
|---|---|---|---|
| **MoE token-padding to `tile_tokens_dim` (8/16/32)** | trtllm `RoutingKernel.cuh:296`, `utils.py:116-137` | our pad quantum is **128** (`dsv4-moe-grouped.cu:208`) → 5–19% tile fill at 6–24 tok/expert | **D.1** — shrink pad to `tile_tokens_dim`, small-N-tile (32/64) CUTLASS variant |
| **Large batched prefill (d8192)** | PR #41834 d8192 = 1722 | tokens/expert ≈192 fills tiles; ours OOMs >`-ub 1024` | **D.2** EP for headroom + raise `-ub` |
| **EP — 128 experts/GPU** | hazyumps `--enable-expert-parallel`, PR `num_local_physical_experts==128` | half weight/GPU (fits) + 2× tokens/expert | **D.2** — EP split + sum-allreduce in meta-backend (currently MIRRORED) |
| **Indexed-split sparse-MLA prefill** | `sparse_mla_kernels.py` (+3521), gate MIN_TOKENS=4096 | fast batched-prefill attention | **D.3** — fused tf32 MQA-logits topk into `fattn-sparse-mla.cu` |
| **Kill per-step `positions.max().item()` sync** | `sparse_mla.py build_c128a_topk_metadata` | removes per-step CUDA stall | **D.3** — CPU 128-aligned upper bound |
| **Fused tf32 Triton MQA-logits topk** | hazyumps `sm12x_deep_gemm_fallbacks.py` | +29% prefill (313→405); was bf16 cuBLAS SGEMM | **D.3** |
| **MTP n=2 + fp32 draft logits + probabilistic drafts** | `llm_base_proposer.py` L1804+ | ~2× decode; +9pp acceptance | **D.4** — align our MTP sampler |
| **GB10 smem ≤101,376 B; TileN 256/128 = 154/147 TF** | NVIDIA forum, TRT-LLM #11368 | our `StageCountAutoCarveout` already fits | constraint guardrail for D.1 |

**Bottom line:** Aiden's 1600 prefill = (token-pad-to-small-tile MoE) × (big batched prefill enabled by EP) ×
(indexed-split sparse-MLA + sync removal). Aiden's 45 decode = MTP n=2 + sm120 sparse-MLA topk fallback +
the reliability plumbing. The single change with the most leverage for us is **D.1 (drop the 128 pad to
`tile_tokens_dim`)**, which is purely local to `dsv4-moe-grouped.cu` and directly attacks the 18 TF/s
starvation; **D.2 (EP)** is the enabler that lets us batch big enough for it to matter and removes the OOM
wall; **D.3** is the prefill-attention half. Decode is already near Aiden's regime (we're at 19.5–24, the
runbook ~31–42 single-stream); D.4 closes that with the MTP sampler fixes.
