# DSV4 Expert-Parallel (EP) — RECOVERED design + status

**Branch:** `feat/dsv4-sparse-mla-mma` (HEAD after this work).
**Origin:** verified-working EP from `feat/ep2-dp-attention`, commits
`a0999ec22 → b8c796ec8 → 0d95186ab → 4a7fa3a2f`, cherry-picked here (all clean, no manual
conflict). EP was developed on the grouped-MoE/FP4 baseline `68224235f` (the merge base) and
**verified producing correct 2-node output** at `4a7fa3a2f`.

**Goal:** Workstream **D.2** of `AIDEN_1600_STACK.md` — split the 256 routed experts across the
2 nodes (128/rank) instead of MIRRORING, freeing ~half the MoE weight/node → headroom for a large
prefill ubatch (`-ub 4096`, then 8192).

**Gate:** `DSV4_EP=1`. Default OFF = today's tensor-split (FF axis-1/axis-0) behavior,
byte-identical. SPMD: both ranks must set it (tp.sh forwards it).

---

## 0. What EP actually does here (the recovered mechanism)

EP does NOT touch the DSV4 custom MoE ops (`dsv4-moe-grouped.cu` / `dsv4-moe-fused`). It works
entirely on the **generic `mul_mat_id` path** (mmq/mmf), via three pieces:

1. **Expert-dim weight split (axis 2 = n_expert).**
   `src/llama-model.cpp:564-572` (`llama_meta_device_get_split_state`): when `DSV4_EP` is set, any
   `*_exps.weight` tensor returns `GGML_BACKEND_SPLIT_AXIS_2`. The existing tensor-split allocator
   then materializes only this rank's expert slice — rank0: experts `[0,128)`, rank1: `[128,256)`
   — so **each node allocates only half the routed-expert weight** (the memory win, see §3).

2. **PARTIAL combine = 1 AllReduce per FFN layer.**
   `ggml/src/ggml-backend-meta.cpp:663-674`: a `mul_mat_id` whose `src[0]` is AXIS_2-split and whose
   `src[1]` (activations) is MIRRORED is resolved so that **only the DOWN projection** is PARTIAL
   (`is_down` = name contains "down"); gate/up stay MIRRORED. This is the `0d95186ab` deadlock fix:
   marking gate/up PARTIAL too inserted 3 AllReduces/layer → subgraph explosion → slot-init hang.
   Gate/up are a benign "MIRRORED lie" — their disjoint rank-local outputs are consumed only by the
   same rank's `glu→down`, never by a cross-rank-identity op. The single down-projection PARTIAL
   feeds the existing NCCL sum-AllReduce (`:2546-2560`) → the full per-token MoE output. **No new
   collective.** (`assume_sync` query → MIRRORED at single-rank/no-TP, so the AllReduce is a no-op
   and DSV4_EP is inert without TP.)

3. **Per-rank GLOBAL→LOCAL expert remap (expert_offset).**
   - Meta stamps each device-j tensor's global expert offset:
     `ggml-backend-meta.cpp:1299-1316` writes `op_params[0]=expert_offset` (= Σ experts on lower
     ranks) and `op_params[1]=1` (EP flag) onto the per-device `mul_mat_id` tensor.
   - The CUDA dispatcher routes EP nodes to mmq and pre-zeros dst:
     `ggml/src/ggml-cuda/ggml-cuda.cu:2926-2937` — `op_params[1]!=0` ⇒ `cudaMemsetAsync(dst,0)` (so
     the remote-expert output positions this rank doesn't write are 0 for the AllReduce) then forces
     `ggml_cuda_mul_mat_q` (the only path wired for the offset).
   - mmq passes the offset + zero-pads the compacted id buffers:
     `ggml/src/ggml-cuda/mmq.cu:181-199` — reads `expert_offset=op_params[0]`; under EP zeroes
     `ids_src1`/`ids_dst` to the safe index 0 before the helper (the `4a7fa3a2f` MXFP4 OOB fix:
     each rank compacts only its local experts, so trailing id slots stay uninitialized and
     `quantize_mmq_mxfp4` would gather garbage — the matmul ignores the padding via `expert_bounds`).
   - The id helper remaps:
     `ggml/src/ggml-cuda/mmid.cu:27-118` — `expert_global = expert + expert_offset`; a token's
     `ids[...]` (GLOBAL id) matches local block `expert` iff `expert_used == expert_global`, and
     the prefix count uses `expert_used ∈ [offset, expert_global)`. `offset==0` is byte-identical to
     upstream (non-EP path untouched). Threaded through `mmid.cuh:5`, `mmf.cu:87-89`.

**Correctness:** each token's top-6 experts are partitioned disjointly across ranks (expert e →
rank `e/128`); each rank computes only its owned experts' gate-weighted contributions and writes
them into its `moe_out` (dst pre-zeroed elsewhere). The down-projection PARTIAL → sum-AllReduce
reconstructs `Σ_{e∈top6} w_e·expert_e(x)` exactly. Verified correct on 2-node FP4 at `4a7fa3a2f`.

---

## 1. Sharding choice — axis-2 (whole experts), at LOAD, no sidecar

EP uses the **generic mul_mat_id path on the model's own MXFP4 `_exps.weight` tensors**, split by
the standard tensor-split allocator (axis 2). There is NO separate sidecar to regenerate and NO
re-slice at load — the meta-backend allocator already slices the loaded tensor per rank. This is
strictly simpler than the sidecar-by-expert option I first sketched, and it's the path that was
actually verified.

**Implication for the deploy config:** EP is mutually exclusive with the DSV4 custom-op sidecar
path. For EP, run with `DSV4_EP=1` and **leave `DSV4_MOE_FUSED` / `DSV4_MOE_GROUPED` /
`DSV4_MOE_SIDECAR` UNSET**. The model then loads its routed experts normally (axis-2 split) and
runs them on the generic mmq path. (Under the custom-op path the `_exps.weight` are TENSOR_SKIP'd
and all 256 experts live in the per-rank registry — that path does NOT honor the expert split, so
combining EP with the custom op would require a separate change and is out of scope; the verified
win is the generic-path EP.)

---

## 2. Routing / dispatch dataflow

Router (`build_moe_ffn`, `src/llama-graph.cpp`) unchanged and identical on both ranks (MIRRORED
inputs) → global top-6 ids + weights. No explicit token all-to-all: activations are MIRRORED
(MLA replicates attention/KV), so EP = "replicate input, mask experts locally, all-reduce the
down output" — one collective on the small `[n_embd, n_tokens]` output, less traffic than classic
EP dispatch. The local masking is the `expert_offset` remap in `mm_ids_helper` (§0.3).

**Capture-safety:** the EP additions are integer compares + two `cudaMemsetAsync` (dst-zero,
id-pad) on the stream — no host sync/malloc; `expert_offset`/EP-flag are op_params baked at
graph-build → constant across replays. The AllReduce is the existing capture-safe NCCL path.

---

## 3. Memory accounting — the win

DSV4-Flash routed MoE: `E=256`, top-6, `n_embd=7168`, `n_ff_exp=2048`, 3 projections, ~58 MoE
layers, FP4 (MXFP4 ≈ 0.5 B/elem packed + 1/16 B/elem block scale ≈ 0.5625 B/elem effective).

Per expert per layer (full FF): gate+up `2·7168·2048·0.5625 ≈ 16.5 MB` + down
`2048·7168·0.5625 ≈ 8.26 MB` ≈ **24.8 MB/expert/layer**.

| Layout | routed experts / node | MoE routed weight / node |
|---|---|---|
| **MIRRORED / tensor-split (today)** | 256 (every node holds all experts; tensor-split only halves the FF *within* each expert) | ~78 GB |
| **EP axis-2 (this path)** | **128 whole experts** | 128 · 24.8 MB · 58 ≈ **~40 GB** |

**Per-node routed-MoE weight ~78 GB → ~40 GB; ~38 GB/node freed.** The freeing happens at LOAD via
the axis-2 tensor-split allocator (each rank's simple tensor is the 128-expert slice) — confirmed
by the verified EP run loading successfully. That ~38 GB is the headroom for the large-`-ub`
prefill activation arena (~0.26 MB/token/layer; `-ub 4096`·58·0.26 ≈ 62 GB staging is the wall
that OOM-kills today above `-ub~1024` under mirrored experts).

---

## 4. Empirical reality (from the verified commit `4a7fa3a2f`)

The commit that verified EP also **measured** it on the grouped-MoE/FP4 2-node setup:

> A/B (same ~1680-tok prompt): EP=1 **101 t/s** (108–115 @ 2–4k) vs EP=0 tensor-split **109 t/s** →
> COMPARABLE. **EP does NOT speed up prefill.** … the 2-node prefill bottleneck is NOT the
> cross-node AllReduce EP restructures — it is per-box compute/engine efficiency (ggml kernels vs
> ds4 fused).

So EP, *by itself, at the ubatch sizes tested then*, did not raise prefill t/s. **The thesis of
this task is different:** EP is the MEMORY enabler that lets `-ub` go to 4096/8192, where the MoE
becomes compute-bound (≈192 tok/expert) — the regime that actually moves prefill toward 1600. The
recovered EP must be re-measured **at large `-ub`** (the coordinator's job), not at the small-ub
A/B that already showed parity. The open question EP answers is "does the ~38 GB headroom let
`-ub 4096` run at all, and is the big-batch MoE then fast enough."

---

## 5. Status

**DONE — cherry-picked clean onto `feat/dsv4-sparse-mla-mma` (no manual conflict), builds on .66:**
- `a0999ec22` axis-2 expert split + PARTIAL combine gate (`llama-model.cpp:564-572`,
  `ggml-backend-meta.cpp:663-674`).
- `b8c796ec8` expert-offset path: meta stamp (`ggml-backend-meta.cpp:1299-1316`), CUDA EP dispatch
  + dst-zero + force-mmq (`ggml-cuda.cu:2926-2937`), mmid kernel remap (`mmid.cu`/`mmid.cuh`),
  mmq offset (`mmq.cu:181-195`).
- `0d95186ab` 1-AllReduce/layer deadlock fix (down-only PARTIAL) + EP_DBG PARTIAL counter
  (`ggml-backend-meta.cpp:2352-2365`) + mmf offset (`mmf.cu:87-89`).
- `4a7fa3a2f` MXFP4 id-pad OOB fix (`mmq.cu:185-194`) + EP_SYNC diag (`ggml-cuda.cu`) + tp.sh
  MASTER_WRAP.
- tp.sh: forward `DSV4_EP` / `DSV4_EP_DBG` to the slave (SPMD parity).

---

## 6. ROUND 2 — EP on the FAST FUSED kernel (overhead removal)

### Coordinator's measurement (confirms §4's prediction)
Generic-path EP (`DSV4_EP=1`, no fused): **enables `-ub 2048`** (was OOM without EP → memory
halving CONFIRMED; `-ub 4096` still OOMs, watchdog-killed at MemAvailable 2 GB → real ceiling
`-ub 2048` on the generic path). BUT speed **~314–331 t/s = NO faster** than non-EP FUSED `-ub 1024`
(~330), even though EP gives **96 tok/expert** (vs 24). So EP's memory win is real, the **speed is
eaten by the generic mmq path overhead**: EP forces `ggml_cuda_mul_mat_q` (MXFP4 W4A16 mmq) +
`mm_ids_helper` remap instead of the fast fused NVFP4 W4A4 CUTLASS kernel the non-EP path uses.

### The fix: EP + FUSED coexist (remove the generic-mmq overhead)
flashinfer's `CutlassMoeFCRunner::runMoe` **natively supports Expert-Parallel** via
`MOEParallelismConfig(tp_size, tp_rank, ep_size, ep_rank)`. Verified in source:
- Binding passes `num_experts_total = num_experts_on_rank * ep_size` to runMoe
  (`flashinfer_cutlass_fused_moe_sm100_binding.cu:320,472`).
- Inside: `num_experts_per_node = full_num_experts / parallelism_config.ep_size`
  (`cutlass_fused_moe_kernels.cuh:3726`) → the weight arrays are indexed LOCALLY [0,128).
- `setLocalExperts` (`cutlass_fused_moe_kernels.cuh:92-105`): given `start_expert = ep_rank *
  num_experts_per_node` and `end_expert`, it takes the GLOBAL `token_selected_experts` (our `sel`,
  range [0,256)), keeps `expert ∈ [start,end)`, remaps to `local = expert - start`, and **naturally
  skips remote experts** (their contribution = 0 on this rank). Exactly EP semantics, done by the
  fast kernel internally — no `mm_ids_helper`, no generic mmq.

So EP+fused = **register only this rank's 128-expert shard** + tell the fused runner `ep_size=2,
ep_rank=rank, global num_experts=256`. The fused op output is the per-rank partial; the existing
`GGML_OP_DSV4_MOE_FUSED → PARTIAL` (meta-backend `:1112-1125`) already inserts exactly ONE
sum-AllReduce/layer to combine (down-only is moot — the whole fused op is one PARTIAL node). No new
collective, no meta-backend change.

### Implementation (this round)
1. **Sidecar shard (memory halving):** preconvert `--ep` writes `sidecar_rank{r}.bin` holding only
   experts `[r*128,(r+1)*128)` with FULL FF (2048). The blob/registry already accept arbitrary
   `n_expert` → `n_expert=128` flows through `build_fused_layer` (sizes fc1/fc2/SF/alphas to 128,
   halving the fused weight footprint). New header fields `ep`, `expert_base`, `n_expert_global`.
   (`dsv4-moe-grouped-blob.h`, `tools/dsv4-nvfp4-preconvert`, sidecar load `llama-model.cpp`.)
2. **Fused runner EP config:** `dsv4-moe-fused.cu` reads the per-rank `ep_size`/`ep_rank` from the
   registry (`expert_base`/`n_expert_global`) and passes `MOEParallelismConfig(1,0,ep_size,ep_rank)`
   + `num_experts = n_expert_global (256)` to `runMoe`, while `build_fused_layer` uses `E=128` local.
   (`dsv4-moe-fused-run.cu`.)
3. **Routing:** `sel` stays GLOBAL [0,256) (router unchanged); the runner's `setLocalExperts` does
   the global→local remap + remote-skip. No change to `build_moe_ffn`.
4. **Gate:** all of this only when `DSV4_EP=1` AND `DSV4_MOE_FUSED=1` AND sidecar is the EP shard.
   The generic-path EP (Round 1) stays for the no-fused A/B. Default (no DSV4_EP) byte-identical.
5. **NOT** apply the generic axis-2 `_exps.weight` split (llama-model.cpp:570) under EP+fused — the
   experts come from the sharded sidecar, and `_exps.weight` are TENSOR_SKIP'd anyway.

### Expected
EP+fused `-ub 2048` runs the FAST NVFP4 CUTLASS kernel at 96 tok/expert on a 128-expert shard
(half the fused weight/node, same memory headroom that unblocked `-ub 2048`). Should beat the
~314 generic-EP and the ~330 non-EP-fused-`-ub 1024`, because it has BOTH the fast kernel AND 4×
the tokens/expert (96 vs 24) → far better tile fill (the §A.1 starvation lever from AIDEN_1600).

### Profile capture for the coordinator (run on the 314 baseline to confirm the split)
On the EP `-ub 2048` generic run (2-node, graphs already off for fused but EP is generic):
```
# Per-op-class GPU time (2-node-safe, deferred — attributes mmq-MoE vs mm_ids_helper vs the rest):
DSV4_OPPROF=1 DSV4_OPPROF_DUMP_AFTER=8000 DSV4_OPPROF_TOP=50
# AllReduce count per layer (both ranks must match; ~58 = 1/layer):
DSV4_EP_DBG=1
```
Look for: `MUL_MAT_ID(...mxfp4...)` total ms vs `mm_ids_helper`/glue vs `FLASH_ATTN_EXT`; and the
`[EP_DBG] n_partial` count. Expectation from §4: the generic mmq MoE GEMM dominates (the slow
W4A16 path), AllReduce is ~1/layer and small — i.e. the overhead is the KERNEL, which EP+fused
removes.

### Deploy recipe (coordinator)
```
# 1) Generate the EP sidecar shard (128 experts/rank, full FF). On .66:
build/bin/llama-dsv4-nvfp4-preconvert --model <FP4.gguf> --out <EP_DIR> --n-ranks 2 --ep
#    -> EP_DIR/sidecar_rank0.bin (experts 0-127), sidecar_rank1.bin (128-255), each ~HALF the
#       FF-split sidecar size. rsync EP_DIR to .67 (same path). (NOTE: distinct dir from the
#       FF-split sidecar; do not overwrite it — keep both for A/B.)

# 2) Deploy EP on the FAST fused kernel (both ranks, watchdog ON). Env (tp.sh forwards DSV4_EP):
DSV4_EP=1 DSV4_MOE_GROUPED=1 DSV4_MOE_FUSED=1 DSV4_MOE_SIDECAR=<EP_DIR>  -ub 2048
#    (GROUPED required by the sidecar loader gate. Large prefill -> fused runMoe ep_size=2/ep_rank.
#     DECODE/small-M -> grouped HP FP32-activation path, now EP-aware. min_m stays 256, NOT forced.)
#    Expect at load: "[dsv4-moe-grouped] EP config: ep=1 ... ep_size=2 ep_rank=<r>" on each rank.

# 3) A/B vs the 314 generic-EP and the ~330 non-EP-fused-ub1024 baselines; verify NO hallucination
#    on "양자역학 발전에 핵심 역할 한 과학자 10명" (quality must match non-EP fused).
```
Binaries to rsync to .67 (md5, build 05:06): llama-server acf631f4, libllama.so.0.0.9745 70d3d559,
libggml-cuda.so.0.13.1 0c23f0a1, libggml.so.0.13.1 c02c2eb8; preconvert 3cbb6a76 (unchanged).

## 7. QUALITY FIX — EP decode must be FP32-activation (not W4A4)

**Bug:** EP+fused hallucinated on long/complex generations (correct on short). **Root cause:** my
round-2 change forced fused `min_m=1`, routing DECODE (M=1) through the fused **W4A4** path, which
quantizes ACTIVATIONS to FP4 (e2m1). The non-EP baseline keeps decode on the grouped **HIGH-PRECISION
path** (`dsv4-moe-grouped.cu:526` — "FP32 activations × dequantized NVFP4 weights, NO 4-bit activation
quantization"); `min_m=256` sends only large prefill to fused. FP4 activations on every generated
token drift the per-token logits → hallucinated names; short outputs survive the smaller error.

**Not the alpha/SF order, not a missing W4A8 kernel.** The flashinfer EP indexing is correct (weights/
SF/alpha all locally indexed [0,128), shard order = `start_expert+i`; getWorkspaceSize divides by
ep_size — all consistent). And the GROUPED *prefill* GEMM is ALSO W4A4 (`ElementA=nv_float4_t<e2m1>`,
`:51`), so W4A4 per se isn't the bug — DECODE specifically must be FP32-activation, which EP downgraded.

**Fix (commit 9ed7c2551):**
- `llama-graph.cpp` — DROP the EP `min_m=1` forcing; EP uses the SAME `min_m=256` as non-EP:
  decode/small-M → grouped HP (FP32 act), large prefill → fused (fast).
- `dsv4-moe-grouped.cu` — the 4 HP-decode kernels (`dec_gate_up_swiglu[_fused]`,
  `dec_down_scatter[_fused]`) are now EP-aware: `e = sel[row] - expert_base`; if `e∉[0,E_local)` skip
  (zero act / no scatter) → each rank emits its partial, the existing `GGML_OP_DSV4_MOE_GROUPED →
  PARTIAL` AllReduce sums both → exact full result at FP32-activation precision. `ep_base`/`E_local`
  from `g_ep` (0/E non-EP → no-op, byte-identical). Large-M grouped CUTLASS prefill is guarded (never
  reached under EP).

**Net under EP:** prefill = fast fused W4A4 (same as non-EP prefill — prompt-only, bounded error);
decode = FP32-activation HP shard (== non-EP decode quality, exact). Quality should now match the
non-EP fused baseline.

## 8. DEADLOCK FIX — EP grouped-decode must run eager (graphs off)

**Symptom:** after R3, 2-node EP+fused DECODE hung (GPU idle, no tokens, server up) even at
max_tokens=3. The R2 W4A4 fused decode did NOT hang; routing decode to the grouped HP path did.

**Analysis (why the coordinator's "asymmetric PARTIAL" framing isn't the mechanism):** the graph is
built by `build_moe_ffn`, which is SPMD — **both ranks emit the identical node graph**, hence the
identical PARTIAL-node count and the identical number of inter-subgraph AllReduces
(`n_subgraphs-1`, executor `ggml-backend-meta.cpp:2589`). So it is NOT an asymmetric collective
count. The EP HP kernels also can't shuffle-deadlock: the expert-skip `if(e∉[0,E_local))return` is
uniform per block (`e` depends only on `blockIdx.y`), so a warp either fully returns or fully runs;
`warp_reduce_sum`'s `__shfl_down_sync(0xffffffff)` is only reached by fully-active warps.

**Root cause (best-supported):** the grouped **M=1 decode being CUDA-graph-CAPTURED** alongside the
**eager** inter-subgraph cross-rank PARTIAL AllReduce. Non-EP decode warms its grouped buffers and
the warm→graphs-ON transition is benign there; the R2 fused decode was warm-from-prefill. Under
EP+fused, prefill uses the fused op so the grouped decode buffers are first touched at decode, and
the warm-up-gate transition (`decode_unwarmed`→graphs ON at step 2) + the per-rank EP skip interact
badly with the captured-replay-vs-eager-NCCL ordering → both ranks block.

**Fix (commit 926e5e6ae, `ggml-cuda.cu`):** under `DSV4_EP`, force grouped-**decode** graphs OFF
(`ep_decode`). M=1 decode barely benefits from graphs; eager execution is deterministically
SPMD-symmetric (both ranks run the same subgraph computes + the same AllReduces, no capture-timing
skew). Prefill (large M → fused op) is untouched. `g_ep==0` ⇒ unchanged. Also fixed the
`DSV4_EP_DBG` crash (`ggml-backend-meta.cpp`): the PARTIAL-count loop now guards to meta-buffer
nodes and prints `rank` + `n_subgraphs` so both ranks can be compared.

### Diagnostic plan if it STILL hangs after this fix
Run both ranks with `DSV4_EP_DBG=1 GGML_TP_DBG=1`. Compare per rank:
- `[EP_DBG] rank=R ... n_subgraphs=S n_partial=P` — **S and P MUST match across ranks** (they will,
  by SPMD). If they don't, the graph build diverged (investigate `build_moe_ffn` env per rank).
- `[tp] rank=R subgraph=i/S allreduce node=... ok=1` — both ranks must reach the SAME `i` count of
  AllReduces and the last one must print on both. If rank A stops at subgraph i and rank B at j≠i,
  the hang is the AllReduce at min(i,j)+1 — i.e. a genuine collective desync (would then point to a
  real topology asymmetry, not capture).
If both ranks show identical `n_subgraphs`/`n_partial` and the AllReduce logs advance in lockstep
then stall together, the remaining suspects are NCCL-on-stream + concurrent `cudaMalloc` (the
grouped decode buffer first-alloc) — in that case pre-warm the grouped decode buffers during a
no-op warmup step, or set `DSV4_MOE_DECODE_MAX` to route differently. (This fix should preempt that
by running eager from step 1.)

### Status
- DONE (built .66, committed): R2 EP+fused, R3 quality fix (FP32-act HP decode), R4 deadlock fix
  (EP grouped-decode eager) + EP_DBG crash guard.
- Binaries for .67 (md5, build 05:xx after R4): llama-server `f50de2cf`, libllama.so.0.0.9746
  `70d3d559`, libggml-cuda.so.0.13.1 `cfffb14e`, libggml.so.0.13.1 `c02c2eb8`; preconvert `3cbb6a76`.
- REMAINING (coordinator): redeploy R4; confirm decode produces tokens AND no hallucination on
  "양자역학 발전에 핵심 역할 한 과학자 10명"; if still hung, run the diagnostic plan above and report the
  per-rank `[EP_DBG]`/`[tp]` lines.
