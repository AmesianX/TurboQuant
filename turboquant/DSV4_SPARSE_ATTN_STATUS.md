# DSV4 Sparse Flash-Attention (per-query top-512 gather) — resume doc

Branch: `feat/dsv4-sparse-attn-gather` (off baseline `68224235f` = live FP4 grouped-MoE + ep2/dp checkpoint).
Goal: honor the DSA indexer's top-512 selection by GATHERING only the selected compressed
keys instead of running DENSE QK^T over the whole compressed cache + (-inf) masking the rest.
Env gate: `DSV4_SPARSE_ATTN=1`. Default (unset) = byte-identical dense path. Perplexity-gated.
This is the committed multi-week prefill lever (n^2 compressed-cache attention is the dominant
growing prefill cost).

## STATUS: DESIGN COMPLETE + baseline secured. Kernel NOT yet written. Default path untouched.

## Verified integration points (line-exact, working tree of baseline 68224235f)
- Dense call: `src/models/deepseek4.cpp:2712` `build_attn_mha(q, k_all, v_all, nullptr, attn_mask_cnv, …)`.
- `k_all = ggml_concat(k_raw, kv_comp, dim=2)`, `v_all = k_all` (MLA mirrored, K==V latent, D=512).
  Layout `[head_dim=512, 1, n_kv]`; gather axis = dim 2 (the n_kv rows).
  - chunk/single-seq assembly: deepseek4.cpp:2506-2508 (`k_all = concat(k_raw, kv_comp_cache, 2)`).
  - raw segment = first `n_raw` rows (DENSE/local SWA, stays dense). comp segment = next
    `n_comp_view` rows, at OFFSET `n_raw`. topk ordinals index 0..n_comp_view-1 → real row = n_raw + ord.
- topk: deepseek4.cpp:2654 `topk = ggml_argsort_top_k(ctx0, index_scores, top_k)`,
  shape `[top_k<=512, n_tokens]`, DISTINCT 512 comp-key set PER query column. cb name "indexer_topk".
  Currently consumed ONLY to build an additive -inf/0 mask via
  `dsv4_build_compressed_mask_from_topk` (deepseek4.cpp:1846-1861) concatenated onto attn_mask
  at :2673, then thrown at DENSE flash-attn. Selection is used to MASK, never to GATHER. <-- the waste.
- hparams.indexer_top_k = 512; compress_ratio = 4; n_comp_view = ctx/4 (grows per chunk).
- padded rows already -inf in index_scores (deepseek4.cpp:2642-2650) → never selected. Good.
- multi-slot block-diagonal: top-k can't cross sequences (enforced in scores). 2-node TP: indexer
  latent MIRRORED → both ranks compute identical topk → identical gather, NO new cross-rank exchange.

## Kernel substrate DECISION: extend fattn-VEC, not fattn-mma.
Rationale (this is the key engineering call):
- Per-query DISTINCT gather is the whole point — every query column needs its OWN 512-row set.
- fattn-mma-f16.cuh loads K/V in CONTIGUOUS `nbatch_fa`-row tiles SHARED by all queries in a
  warp-tile (`flash_attn_ext_f16_load_tile`: `KV + i*stride_KV`, contiguous i). Per-query row sets
  fundamentally break that shared-tile model → would require a redesign of the tile loader and
  destroy the validated numerics. High risk, against the spirit of "reuse validated kernel".
- fattn-vec.cuh processes `ncols` query COLUMNS per block; with `ncols==1`, ONE query per block.
  Its KV loop (fattn-vec.cuh:859) is `for (k_VKQ_0 ...) sum = vec_dot_KQ(K + i_KQ*nb11, ...)` —
  per-row scalar-ish dot with online softmax + ·V accumulation (the validated machinery the task
  says to reuse). Injecting a gather = replace contiguous `i_KQ` with `i_KQ = n_raw + kv_idx[ord]`
  for the comp segment; keep raw window dense. Minimal, local, numerics-preserving change.
- NOTE vs project policy "fattn-mma is default / vec is fallback": this is a NEW env-gated op, NOT a
  change to the default attention. The default dense MMA path is untouched. Per-query gather is the
  one case where vec's one-query-per-block layout is the CORRECT substrate. (Flag this to user.)

## Op / plumbing plan (capture-safe, CUDA-graphs ON)
- New input rides as `dst->src[6]` = `kv_idx` (i32 [top_k, n_tokens]) — the existing fork already
  uses src[5]=k_rope as a TurboQuant side-channel in launch_fattn (fattn-common.cuh:2362). Clean.
- Reuse GGML_OP_FLASH_ATTN_EXT (no new ggml op) OR a thin custom op. Leaning: keep the op, add an
  optional src[6]; in launch_fattn / vec kernel, if kv_idx != null AND DSV4 sparse → gather comp seg.
  kv_idx is the on-graph `topk` tensor (already i32, already captured) → no malloc/thrust/sync.
- Persistent scratch only. No per-call allocation. Keep graphs captured.

## Wiring in deepseek4.cpp behind DSV4_SPARSE_ATTN=1
- In the chunk path (the `else` single-seq branch, n_tokens==512, n_comp_view > top_k):
  instead of building the -inf compressed mask + concat + dense FA, route to the gather-attn:
  pass `topk` as the kv_idx side-channel; K/V = concat(raw_window, comp_cache) UNCHANGED;
  mask = raw-window mask ONLY (comp segment handled by gather, not mask).
  Flag OFF → unchanged dense path, byte-identical.
- Guard: only when n_comp_view > top_k (else dense is already cheap and uniform path is taken at :2602).

## Correctness traps (carry forward)
- gather must hit the ATTENTION cache rows (get_dsv4_attn_k), NOT the indexer latent. comp ordinal
  -> real k_all row = n_raw + ord.
- padded rows never selected (already -inf in scores). Don't re-pad K/V for the gathered set; the
  256-pad logic at deepseek4.cpp:2681-2709 is for the DENSE n_kv — the gather length is fixed top_k,
  handle its own alignment.
- softmax ACCUMULATION ORDER is the only fp caveat vs dense (gather visits the 512 in topk/sorted
  order, dense visits in row order). Online softmax is order-robust to ~fp tolerance → perplexity
  gate will confirm lossless. This is the quality risk to watch.

## VERIFY protocol (when kernel lands)
- Build ONLY on .66; rsync llama-server+libllama.so* to .67; md5 match BOTH.
- 2-node launch (FP4): NCCL_IB_GID_INDEX=-1 PARALLEL=2 UB=512 CTX=262144 DSV4_MOE_GROUPED=1
  DSV4_MOE_SIDECAR=/home/user/Models/DeepSeek-V4-Flash-GGUF/FP4/nvfp4_sidecar SPEC=""
  DSV4_SPARSE_ATTN=1 bash tp-serve/tp.sh ALLRESTART
- (a) prefill tok/s on ~13k prompt WITH vs WITHOUT DSV4_SPARSE_ATTN (cache_prompt=false). Expect ~1.3-1.8x@13k.
- (b) quality lossless: Korean fluent + "프랑스의 수도는 파리" + coherent long answer + short
  perplexity/logit check sparse≈dense.
- (c) graphs still captured. (d) build+md5 both nodes.
- NEVER pkill — kill by pid: `kill -9 $(ps -C llama-server -o pid=)`. tp.sh WATCH_MIN_GB=4 watchdog.

## Live state at design time
- Server pid 4165769 running off baseline (libllama.so.0.0.9704, md5 d8d6790... matches .67). Leave it; default dense path is safe.
