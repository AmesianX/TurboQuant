# DSV4 in-server MoE vs standalone bench gap — diagnosis (2026-06-29)

Branch feat/dsv4-sparse-mla-mma. Build .66 only.

## Context
512-cap lift (commits 8df8cb17e/bc412da70) WORKED: -ub=2048 now runs forward w=2048,
MoE M=2048, prefill 336->360 t/s. But only +7%, not the ~1.8x Amdahl predicted from the
bench's 9->57 TF/s. -ub=4096 cap-lifted runs w=4092 but OOM/crashes; -ub=2048 is the
memory-safe point.

## ROOT CAUSE of the bench gap (NOT a fixable-glue bug)

The bench (`dsv4_fused_moe_bench.cu:53`) uses **`MOEParallelismConfig pc(1,0,1,0)` = ep_size=1,
NO EP**, E=256. Its table (DSV4_PREFILL_GAP_ROUND4.md):
| M    | tok/expert | eff TF/s (front tactic) |
|------|-----------|-------------------------|
| 1024 | 24        | 18.0 |
| 4096 | 96        | 57.5 |
| 8192 | 192       | 91.2 |
(no M=2048 row; interpolating 48 tok/expert ~= **37 TF/s**.)

In-server is **EP2** (ep_size=2, 128 local experts, set per-layer from the model file,
llama-model.cpp:1787). runMoe is called (dsv4-moe-fused-run.cu:537-547) with:
  n_tokens = M = 2048 (full ubatch), n_expert_global = 256, pc(1,0,ep_size=2,ep_rank).
Internally num_experts_per_node = 256/2 = 128; routing sorts all 2048*6 = 12288 GLOBAL
assignments, keeps the ~6144 LOCAL ones over 128 experts = **48 tok/local-expert**.

### The math that explains +7%, not +80%
- M=512 (old cap)  -> 12 tok/expert -> ~12-15 TF/s (tile-starved, the old in-server point)
- M=2048 (cap lift)-> 48 tok/local-expert -> **~37 TF/s** (interpolated bench)
- M=4096 (OOM)     -> 96 tok/expert -> 57 TF/s (the number Amdahl was applied to)

So the MoE rose ~15 -> ~37 TF/s (a real ~2.5x on the MoE GEMM), but the Amdahl target
(57) needs 96 tok/expert = M=4096, which OOMs. The +7% end-to-end is consistent with
the MoE going 15->37 (not 15->57) under Amdahl (MoE ~55.8% of prefill).

**The in-server MoE invocation does NOT differ from the bench in a fixable way:** at the
SAME tok/expert (48) it hits the SAME ~37 TF/s as the bench would at M=2048. The gap to
57 is purely tok/expert (48 vs 96), gated by the M=4096 activation OOM — a SEPARATE wall
(the fused MoE workspace is linear+tiny; the OOM is the other O(ub) graph activations).

## What to MEASURE to confirm (coordinator)
1. `DSV4_MOE_FUSED_PROF=1` at cap-lifted -ub=2048: the 4 sub-phases
   (cvt_in/route/runMoe/cvt_out). Confirm runMoe(GEMM) ms implies ~37 TF/s at the
   in-server M, and quantify the glue (cvt_in+route+cvt_out) as a % — if glue >25% it's
   a secondary amortization target; if runMoe dominates, the gap is purely tok/expert.
2. `DSV4_OPPROF=1`: confirm DSV4_MOE_FUSED(M=...) shows M=2048 (one call/layer), not split.
3. Compute eff TF/s = (2 GEMMs FLOPs at 6144 local rows) / runMoe-ms; compare to bench@48.

## The real levers (both blocked / out of scope here)
- **96 tok/expert needs M=4096**, blocked by the activation OOM (NOT the MoE workspace).
  Fixing that OOM (the remaining O(ub) graph activations) is the path to 57 TF/s -> the
  ~1.8x. That is the next target, separate from the MoE kernel.
- A different MoE kernel that fills tiles at <48 tok/expert (sub-128 M-tile) — CUTLASS
  sm120 NVFP4 has no sub-128 tile (round4), so this is a kernel-rewrite, not a flag.

## Honest conclusion
The cap lift converted M=512->2048 = ~15->37 TF/s on the MoE = the +7% end-to-end. The
MoE IS converting the larger batch to higher TF/s exactly as the bench predicts for 48
tok/expert. The remaining gap to the bench's 57 (and the 1.8x) is tok/expert 48 vs 96,
which requires M=4096, blocked by a separate activation-memory OOM — that OOM is the next
real lever, not the MoE invocation.

## UPDATE (2026-06-29): INDEXER_FUSED=21 is COMPLETE, not a half-fused bug

The coordinator read INDEXER_FUSED=21 (not 43) as "only half the layers fuse". That is a
MISREAD of the model topology. The DSV4-Flash per-layer compress_ratio array
(deepseek4.cpp:3303) is:
  {0,0, 4,128, 4,128, ... ,4}  -> 2 dense(0) + 20 ratio==4 + 19 ratio==128 (41 main layers)
ONLY the ratio==4 layers have the lightning indexer (the `if (compress_ratio == 4)` guard at
deepseek4.cpp:2161/2225/2571). So there are ~20-21 indexer layers TOTAL, and INDEXER_FUSED=21
means **ALL of them ARE fused.** There is NO un-fused indexer layer.

### So what is the 261 GB at the wide chunk?
NOT the indexer (fully fused) and NOT the main attention KQ: tp.sh runs `-fa on`
(tp.sh:83), and build_attn_mha uses ggml_flash_attn_ext when flash_attn && kq_b==null
(llama-graph.cpp:2186) -> NO [n_kv,ub,n_head] materialization. With FA on + indexer fully
fused, the 261 GB is a DIFFERENT ub-scaled transient.

### NEED the probe's op-sum line to name it
The DSV4_PREFILL_VRAM probe already prints, on each new high-water width:
  `op-sum(ub-scaled,alloc) <OP> <MiB>`  (top-12)  and
  `ub-alloc[ 0] <MiB> <OP> [ne0,ne1,ne2,ne3] <name>`  (top-20 largest nodes)
THAT line (not relayed) names the 261 GB op + its [n_comp_view, ub, 64?] shape. Candidates
once indexer+attention are ruled out: the indexer's downstream argsort/mask at the resumed
RESERVE's n_comp_view = 8*ub/4 (the reserve over-provisions pos0 = max(n_batch, 8*ub),
llama-context.cpp:633), or a compressor transient on the 19 ratio==128 layers, or the
attn_mask concat + FA pad. The fix follows from WHICH op it is:
  - if it's the indexer mask/argsort at reserve -> shrink reserve_pos0 (8*ub is excessive)
  - if it's a [.,ub,64] KQ -> FA is silently falling back (check FLASH_ATTN_EXT count vs 20)
  - if it's the ratio==128 compressor -> tile it like the indexer
