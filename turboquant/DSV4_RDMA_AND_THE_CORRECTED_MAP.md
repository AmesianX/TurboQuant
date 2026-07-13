# DSV4 — the NCCL/TCP bug, and the map it corrects (2026-07-13)

Resumed from `FREEZE_2026-07-12.md`. Everything below is a controlled A/B (same build, same
prompt, same env except the one variable), on `feat/dsv4-w4a16-native-port`.

## The bug: every cross-rank AllReduce was going over TCP

`NCCL_IB_HCA` was hardcoded to `rocep1s0f0,...`. A driver update renamed the devices to
`mlx5_0..3`, the list matched nothing, and NCCL fell back to sockets. It said so, every run:

    NCCL INFO NCCL_IB_HCA set to rocep1s0f0,...
    NCCL INFO NET/IB : No device found.
    NCCL INFO Using network Socket

Four ACTIVE 200 Gb/s ConnectX-7 rails, unused. A 16 KB reduce cost **0.63 ms** (tens of us is
normal); 42 of them per token = **40% of decode**. Fixed by discovering the HCAs from
`/sys/class/infiniband` in `tp-serve/tp-w4a16.sh` (b9f920e09) and `tp-serve/tp.sh` (08edb19b7).

**After ANY driver/image change, verify: `NCCL_DEBUG=INFO` must print "Using network IB".**
NCCL topology env must be SYMMETRIC across ranks — set it on the master only and the bootstrap
mismatches (`Message truncated : received 32768 bytes instead of 1024`). 4 rails OOM the box at
CTX=262144 (per-rail buffers registered in the unified memory the model already fills); rail
count does not matter for a 16 KB reduce anyway (1 = 2 = 4 rails, measured).

## Results

| | before (TCP) | after (RDMA) | |
|---|---|---|---|
| decode, plain, CTX 65k | 10.78 | **13.69** | +27% |
| decode, plain, + ATTN_SPLIT | — | **15.96** | +48% vs start |
| decode, MTP, CTX 262k | 11.92 | **15.25** | +28% |
| 13k prefill | 358 | **425** | +19% |

Correctness re-verified (coherent output, correct arithmetic) at every step.

## What this DISPROVES — do not resume from these

1. **"The 0.2 ms subgraph boundary is host launch overhead; capture the whole step and every TP
   split flips positive."** (`FREEZE_2026-07-12.md` step 5, `DSV4_ATTN_TP_SPLIT_DESIGN.md`.)
   I built it — `DSV4_STEP_GRAPH` (0842e57fb) captures all 43 subgraphs AND the 42 NCCL
   AllReduces into ONE device graph, 97.6% replay, one launch per token. **It gains ~2%.**
   The executor loop was already fully async; the host ran ahead and queued every subgraph while
   the GPU was still on the first. The boundary cost was never the host — it was the collective.
   Keep the step graph (correct, gated off, right substrate) but stop expecting anything from it.

2. **"DSV4_ATTN_SPLIT is net-negative."** It was, *because the extra reduce per layer cost 0.63 ms*.
   At 0.15 ms it is **+16.5%** (13.69 -> 15.96), no code change. The gate from 41794b115 just
   became profitable. Same arithmetic should now be re-run for the shexp split.

3. **"MTP verify is not batched / does a full forward per candidate."** My hypothesis, REFUTED:
   there is exactly one target `llama_decode` per round (`server-context.cpp:3260`) carrying
   `n_tokens = 1 + n_draft`. The verify amortizes fine — K=3 costs **1.65x** a K=1 step for 3x
   the tokens.

## Where decode actually stands

Measured with `DSV4_MTP_PROF` (133 rounds, 255 tokens):

    round = 120 ms, 1.92 tokens/round
      target verify (K=3)   ~102 ms   (1.65x a K=1 step -- amortizes)
      draft() x2              10.2 ms  (2 one-layer decodes + 2 lm_heads -- near-irreducible)
      process() mirror         7.9 ms  (REDUNDANT: re-decodes the verify batch on ctx_dft;
                                        TAG_SPEC_AVOID_DRAFT_REEVAL, server-context.cpp:3313)

MTP nets **zero** today (15.9 vs 16.2 plain). Killing the removable draft overhead is worth
roughly +18%, not more.

**The arithmetic that matters:** a K=1 step is **61.7 ms**. The physics floor is ~17.5 ms
(4 GB/rank/token @ 230 GB/s) = 57 t/s. Speculation can multiply by at most ~1.3. So 45 t/s is
unreachable until the STEP itself drops to ~25-30 ms. **The step is the whole game; speculation
is the last 1.3x on top of it.**

## The tooling problem blocking the next step

`DSV4_OPPROF` cannot be trusted for per-op attribution:
- It requires `GGML_CUDA_NO_GRAPHS=1`, so it measures an **eager** pass while decode runs in
  graph mode — a different object.
- Its per-op times are stall-contaminated: with node names in the key, the graph's FIRST nodes
  (`blk.0.hc_attn_fn`, `blk.0.attn_q_a`) show **237 ms and 78 ms for a single call** — they are
  absorbing the pass-start queue drain. The profile totals 715 ms for two passes whose real cost
  is ~124 ms: **6x inflated**.
- The enemy list in `FREEZE_2026-07-12.md` (FA 23.2%, CONT 13.2%, glue ~15%, wall-gap 38%) came
  from this tool. Treat it as unverified.

**Next session starts here:** get a trustworthy graph-mode profile of the 61.7 ms step. Without
it, every roadmap number is a guess — which is exactly how the boundary theory survived a month.
(nsys 2024.6 captures no CUDA kernel data on GB10 — see the traps in FREEZE_2026-07-12.md.)
