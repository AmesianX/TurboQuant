# DSV4 Sparse MLA Attention — Round 4 (the prefill ceiling)

Branch: feat/dsv4-cutlass-fused-moe (base acba31c79 = "sparse-attn round-3: correct/lossless
but 0.61x at 13k vec inefficiency"). Goal: make the top-512 sparse MLA run on TENSOR CORES
so prefill stays FLAT with context (jasl-like) instead of 340@8.5k -> 224@43k.

## GATING QUESTION ANSWERED

### 1. NO portable sm120 sparse-MLA kernel exists to vendor
Our /home/user/work/vllm-spark/flashinfer-src is FlashInfer **0.4.1**. jasl's sm120 sparse
prefill needs 0.6.13+. The 0.4.1 tree has: block-sparse attention (fa2/fa3 = sm80/sm90 only,
NO sm120), MLA kernels sm80/sm90/sm100(cutlass_mla) only. No DEEPSEEK_V4 / sparse_mla /
SM120_PREFILL kernel. Confirmed by exhaustive search.

### 2. The REAL upstream technique = TMA Gather4 (hardware), then dense MMA
Per NVIDIA TensorRT-LLM Blackwell blog: DeepSeek-V3.2 sparse MLA prefill uses the
**TMALDG.Gather4** instruction ("loads four rows from a 2D tensor, coalesces into a
contiguous destination") to handle the per-query gather, feeding tensor-core GEMM. Runs in
MQA mode (single latent KV head -- matches DSV4: head_count_kv=1, 64 Q heads, 512 latent).
Kernels live in FlashMLA / DeepGEMM / TRT-LLM sparse framework (not in our flashinfer 0.4.1).

### 3. ✅ TMA Gather4 WORKS ON OUR GB10 sm_121 (microbench PROVEN)
PTX `cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4` (CUDA13 cccl
cp_async_bulk_tensor_gather_scatter.h, gated `__CUDA_ARCH__>=1000`; sm121=1200 qualifies).
Microbench (scratchpad/gather/g2.cu): gathered source rows {0,2,4,6} into contiguous smem
CORRECTLY on "NVIDIA GB10 sm_121" (row starts 0,64,128,192 as expected, sync: no error).
=> the enabling primitive for tensor-core sparse MLA EXISTS and RUNS on our hardware.
CUTLASS 4.3 has NO high-level gather copy-atom; must use the raw PTX intrinsic.

## THE PROBLEM (why round-3 vec lost, why MMA is hard)
- top-512 is PER-QUERY: ggml_argsort_top_k -> [512, n_tokens], each query token picks its
  OWN 512 comp-rows (deepseek4.cpp:2654). No shared key block across queries.
- The round-3 sparse path is wired ONLY into fattn-VEC with cols_per_block=1 (one query/block)
  because per-query keys break the shared-tile load. VEC has no tensor cores -> 0.61x.
- fattn.cu:823 FORCE-ROUTES to VEC when src[6] (kv_idx) bound. MMA kernel
  (fattn-mma-f16.cuh:1920) takes the kv_idx params but GGML_UNUSED_VARS them (the gap).

## DESIGN: TMA-Gather4 sparse-MLA tensor-core kernel
Per query-tile of T queries (but each picks own keys), the upstream trick: process ONE query
(or a small group) per CTA, but use Gather4 to pull its 512 keys (128 Gather4 ops) into
contiguous smem, then run a DENSE MMA over the 512 contiguous keys x 512 head_dim. MQA: the
single latent KV head means all 64 Q heads of that token share the SAME gathered 512 keys ->
the gather cost is amortized over 64 heads, and the QK^T is [64 heads x 512 dim] @ [512 keys
x 512 dim]^T = a real tensor-core GEMM. THIS is the win vec misses: vec does the 64x512x512
as scalar FMAs; MMA does it as tensor-core tiles over the gathered-contiguous K.

Key insight (MQA amortization): per query TOKEN, gather 512 keys ONCE, then 64-head QK^T and
PV are dense tensor-core GEMMs over those 512 contiguous keys. That is tensor-core-friendly
even though selection is per-query -- the per-query gather feeds a per-query dense GEMM whose
M=64 (heads) is enough to fill MMA tiles.

## SCOPE / STATUS
This is a from-scratch flash-attention kernel (TMA-gather K-loop + online-softmax + MMA over
gathered K, FP8 latent KV dequant in the tile loader). Large. Round-4 deliverable so far:
the GATING ANSWER (no vendor kernel; TMA Gather4 proven on GB10) + the design. Implementation
is the next phase. Default-off: DSV4_SPARSE_ATTN unset = dense path, byte-identical.

## ✅ DE-RISK: TMA Gather4 throughput (GB10, measured)
scratchpad/gather/perf2.cu, 1000 concurrent CTAs x 200 iters x 128 gather4:
  - 2.56e7 gather4 ops in 24.88 ms -> ~0.97 ns/gather4
  - **0.12 us to gather a full 512-row (per-query) set** (amortized across 1000 CTAs)
For a 13k prefill: 13000 queries x 0.12us = ~1.6 ms of gather, fully overlappable with the
MMA compute. => the per-query gather is NEGLIGIBLE; the QK^T/PV MMA dominates (the tensor-core
win vec misses). The approach is SOUND. (Single-threaded issue in the bench; a real kernel
pipelines gather4 across warps -> even faster.)

## CONCRETE IMPLEMENTATION PLAN (next phase, multi-step w/ numeric gate each)
New ggml-cuda op (or MMA-kernel branch) `fattn_sparse_mla` gated by src[6]+DSV4_SPARSE_ATTN:
  Step A: per query token (CTA), build the 512 absolute comp-row indices from kv_idx
          (already have: kv_idx[ic*top_k + ord], offset n_raw). Plus the n_raw dense window.
  Step B: TMA Gather4 the 512 FP8-E4M3-B128 latent rows (512-wide = 8x 128B lines/row, or
          tile the 512 dim) into contiguous smem; dequant FP8->bf16 in the tile loader (reuse
          the dsv4 fp8 dequant + rope-tail handling). MQA: gather ONCE, shared by 64 heads.
  Step C: QK^T as MMA: Q[64 heads x 512] @ Kgathered[512 keys x 512]^T -> S[64 x 512].
          online softmax (running max/sum) over the 512 keys + the n_raw window.
  Step D: PV as MMA: P[64 x 512] @ Vgathered[512 keys x 512] (V=K latent). -> O[64 x 512].
  Step E: numeric gate vs the dense+mask path (cos > 0.999; top-512 IS the model's designed
          sparsity -> should be near-lossless). THEN measure prefill-vs-context flatness.
Wire: fattn.cu dispatch -> route src[6] to the NEW mma-gather kernel (not vec) when sm121+
  TMA available; keep vec as the fallback (DSV4_SPARSE_ATTN_VEC=1). Default off = dense.

## STATUS after Round 4 (this session)
GATING ANSWERED + core primitive DE-RISKED on real HW. The full TMA-gather MMA flash kernel
is the implementation phase (large). Did NOT ship a half-built kernel into the default path
(round-3 vec sparse remains the only wired sparse path, still env-gated + lossless + 0.61x).

## TMA constraint found: 512-wide latent needs 2x 256-wide gather4 tiles
cuTensorMapEncodeTiled rejects box width 512 bf16 (1024B) -> "invalid argument" (TMA box-dim
max ~256 elems). 256-wide (512B) works. So the 512 latent dim = 2 column-tiles per gathered
row (256 gather4 ops for 512 rows). Still ~0.24us/query, negligible. Validated:
scratchpad/gather/{g2,wide,perf2}.cu (gather correctness + 256-wide + throughput).

## HONEST STATUS / what's left
- ✅ Gating answer: no vendor sm120 sparse-MLA kernel (flashinfer 0.4.1); the upstream
  technique (TMA Gather4) is PROVEN to work + be fast on our GB10 sm121.
- ⏳ The full TMA-gather MMA flash-attention kernel (FP8 latent dequant tile loader, online
  softmax, 64-head QK^T+PV MMA over gathered K, ggml flash_attn_ext integration) is the
  large implementation phase. NOT started in code to avoid shipping a broken default.
- No binary change this round -> .66/.67 md5 unchanged from Round 3 (no rsync needed).
- Recommend next session: build the kernel standalone (synthetic numeric gate vs dense+mask)
  BEFORE ggml wiring, then integrate behind DSV4_SPARSE_ATTN (replacing the vec route).

## ROOT CAUSE of round-3 vec 0.61x (found): 64x redundant gather + scalar QK^T
fattn-vec.cuh:120-123: blockIdx.z = sequence*ne02 + head -> EACH (query, head) pair is a
separate block. With sparse cols_per_block=1, that's n_tokens x 64 heads blocks, and EACH
re-gathers the SAME 512 keys (kv_idx is per-query, identical across the 64 heads). So the MQA
amortization is LOST: the 512-key gather happens 64x per query, and the QK^T is scalar (no
tensor cores). The TMA-gather MMA kernel fixes BOTH: 1 block per query processes all 64 heads,
gathers the 512 keys ONCE (TMA), and does QK^T/PV as tensor-core MMA. => the design is doubly
justified (tensor cores + 64x gather dedup). This is the architecturally-correct fix, and the
primitive is proven on GB10.
