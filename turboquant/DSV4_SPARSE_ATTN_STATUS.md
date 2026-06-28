# DSV4 Sparse Flash-Attention (per-query top-512 gather) — resume doc

Branch: `feat/dsv4-sparse-attn-gather` (off baseline `68224235f` = live FP4 grouped-MoE + ep2/dp checkpoint).
Goal: honor the DSA indexer's top-512 selection by GATHERING only the selected compressed
keys instead of running DENSE QK^T over the whole compressed cache + (-inf) masking the rest.
Env gate: `DSV4_SPARSE_ATTN=1`. Default (unset) = byte-identical dense path. Perplexity-gated.
This is the committed multi-week prefill lever (n^2 compressed-cache attention is the dominant
growing prefill cost).

## STATUS (round 3 — MEASURED): CORRECT & LOSSLESS, but SLOWER at 13k (vec inefficiency).
## DSV4_SPARSE_ATTN gather works end-to-end on 2-node FP4: lossless quality, graphs captured.
## REAL measured prefill @16,349 tok (CTX=262144, UB=512, cache_prompt=false):
##   DENSE  (flag off): 294.48 / 299.39 tok/s  (54.6 s)
##   SPARSE (flag on) : 182.02 / 180.94 tok/s  (89.8 s)   => ~0.61x  (a 1.6x SLOWDOWN, NOT a speedup)
## Quality SPARSE==DENSE: "프랑스의 수도는 파리입니다." / "dog." (identical greedy); 13k summary coherent;
## Korean AI-history long answer fluent+accurate. Numerics lossless (softmax order caveat = non-issue).
## CUDA graphs reused (9/10/49) under sparse — capture OK. Both ranks SPMD (DSV4_SPARSE_ATTN forwarded).
##
## ROOT CAUSE of the slowdown (as the coordinator predicted): the FLOP reduction is real (comp segment
## ~4087 rows -> gather 512 per query at 16k) BUT the vec kernel is one-query-per-block, NO tensor
## cores, far less efficient per-FLOP than the dense MMA (<512,512>) path. At 13-16k the per-FLOP
## penalty dominates the ~8x comp-FLOP saving. The big win is meant to be LONG context (100k+ ->
## ~49x comp reduction) where the saving outgrows the penalty — NOT YET measured (256k ctx loaded;
## a 100k-prompt A/B is the next data point). VERDICT so far: vec substrate is CORRECT but does NOT
## win at 13k. To win broadly we likely need an MMA-class gather (tensor cores) — harder.
##
## Build/sync TRAP HIT+FIXED: first sparse launch crashed slave = `undefined symbol
## ggml_flash_attn_ext_add_kv_idx` because rsync synced only llama-server+libllama.so* — the new
## symbol lives in libggml-base.so and the kernel in libggml-cuda.so. FIX: rsync ALL of
## {llama-server, libllama.so*, libggml-base.so*, libggml-cuda.so*}; md5 all 4 on BOTH nodes.
##
## NEXT (round 4): (a) 100k-prompt A/B to test the long-ctx crossover (the actual design target);
## (b) if vec still loses at long ctx, the lever needs an MMA tile-gather (tensor cores) — scope it.
## Default path (flag off) remains byte-identical/safe; server currently UP on sparse for inspection.

## STATUS (round 2): KERNEL + WIRING IMPLEMENTED, BUILDS CLEAN (libllama.so.0.0.9708),
## md5 SYNCED+VERIFIED on .66 AND .67 (server 13a69ff9.../lib 91e3dbaa...). NOT YET
## numerically validated / measured (round 3). Default path (flag off) untouched.
## Commits f35d47a3c, 48965947c on branch feat/dsv4-sparse-attn-gather.

### What landed (round 2)
- ggml: `ggml_flash_attn_ext_add_kv_idx(a, kv_idx, n_raw)` binds topk to src[6], n_raw at
  op_params[4] (op_params[3]=FA precision — do NOT reuse). Decl in ggml.h.
- fattn_kernel_t + all 4 kernels (vec/mma/tile/wmma) gain (kv_idx, top_k, n_raw) (mma/tile/wmma:
  unused; default path functionally identical). launch_fattn extracts src[6], FORCES
  parallel_blocks=1 under gather (KV split assumes contiguous rows).
- VEC kernel gather: pos [0,n_raw)=dense raw window (identity), [n_raw,n_raw+top_k)=comp rows at abs
  (n_raw + kv_idx[ord]); K_eff/V_eff/mask at translated abs row; OOB lanes -inf. cols_per_block
  FORCED to 1 under gather. Added D=512 F16/F16 vec instance + dispatch (was missing).
- fattn.cu get_best_fattn_kernel: src[6] bound -> FORCE VEC (else D=512 picks MMA, can't gather).
- deepseek4.cpp chunk path (DSV4_SPARSE_ATTN=1, single-slot, !uniform, n_comp_view>top_k): captures
  topk, builds ALL-ZERO comp mask (keeps K-layout mask contract; only K/V ADDRESSING differs),
  passes topk+n_raw to build_attn_mha. Flag off = unchanged.

### NEXT (round 3) — VALIDATE then MEASURE (do NOT claim results before)
1. Bring up 2-node (binary 9708). Confirm DENSE (flag off) sane first; then DSV4_SPARSE_ATTN=1:
   greedy temp0 same prompt sparse-vs-dense -> tokens+logits match to fp tol (softmax accumulation
   ORDER is the only caveat). Korean + "프랑스의 수도는 파리" + coherent long answer.
2. Prefill A/B ~13k cache_prompt=false, tok/s with vs without sparse. HONEST: 13k may be
   break-even-to-modest (vec < dense MMA per-FLOP; big win = long ctx). Report real number anyway.
3. Confirm graphs captured (DSV4_GRAPH_PROBE).
- If numeric FAILS: STOP, report. Suspects: kv_idx layout (topk [top_k,n_tokens], kernel reads
  kv_idx + ic0*top_k — verify contiguous row-major, top_k=ne[0]); all-zero mask vs slope; OOB lanes.
- Live server pid 4165769 STILL on OLD binary (dense). Restart = the validation step; it's the
  user's live FP4 service — confirm before restart.

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
