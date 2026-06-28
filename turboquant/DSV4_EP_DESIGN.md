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

**REMAINING (coordinator):**
- Deploy `DSV4_EP=1` with `DSV4_MOE_FUSED`/`SIDECAR` UNSET, `-ub 4096` (then 8192), watchdog
  `WATCH_MIN_GB=4` ON; measure batched 13k/d8192 prefill t/s + per-node `nvidia-smi` (confirm
  ~40 GB routed-MoE + the `-ub 4096` arena fit, no OOM-kill).
- If the generic-path EP MoE is the bottleneck at large `-ub` (likely, per §4's "ggml kernels vs
  ds4 fused"), the next lever is making the DSV4 FUSED custom op honor the expert shard (load only
  128 experts/rank into the registry + reuse the down-PARTIAL AllReduce). That's a separate,
  larger change — only pursue if the memory headroom proves EP lets big-ub run but the generic
  kernel caps throughput.
