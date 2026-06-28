# DSV4 Prefill/Decode Gap — Round 1 (profiler + TP all-reduce analysis)

Branch: feat/dsv4-sparse-mla-mma. Build .66 only. libggml-cuda.so md5 (.0.13.1): bdd9bbd32178c50f21b682d9c631424a

## What changed (this round)
- **NEW capture-safe per-op profiler `DSV4_OPPROF`** (ggml-cuda.cu).
  - struct `dsv4_opprof::profiler` (~line 96) + hook in the eager dispatch loop (~line 4864),
    key builder `ggml_dsv4_opprof_key` (~line 4730 area, just above ggml_cuda_graph_evaluate_and_capture).
  - Why the legacy `DSV4_KERNEL_PROF` died at load on 2 nodes: it forces graphs fully OFF for the
    whole graph AND does `cudaEventSynchronize()` PER NODE. Under the SPMD meta-backend that per-node
    device sync desyncs the cross-rank NCCL AllReduce (rank A blocks on a node sync while rank B's
    AllReduce on the same stream waits for A to enqueue) -> hang/death during warm-up.
  - `DSV4_OPPROF` fixes this: records a cudaEvent pair around each EAGER node on its own stream,
    NO per-node sync; elapsed times computed LAZILY (drain completed pairs each call, full drain +
    one cudaDeviceSynchronize at process exit). Never forces graphs off, never changes the execution
    path. Gated on `!use_cuda_graph` (recording timing events mid-capture is illegal).

## HOW THE COORDINATOR CAPTURES THE 13k FUSED PROFILE
Add to the existing crash-free launch (DSV4_MOE_FUSED=1 + the 2-node TP env), on BOTH ranks:
```
GGML_CUDA_NO_GRAPHS=1 DSV4_OPPROF=1 DSV4_OPPROF_TOP=50
```
- `GGML_CUDA_NO_GRAPHS=1` = the known-safe global graph kill-switch (already used for the crash-free
  baseline) -> every op runs eager -> EVERY op-class gets timed (dense FP8 proj, fused MoE, BF16
  GEMMs, FlashAttn, indexer, etc.). This run will be SLOWER than steady-state (graphs off) but the
  RELATIVE per-op % is what we need.
- Send ONE ~13k prefill. At process exit each rank prints `[DSV4_OPPROF] eager-pass GPU time total ...`
  with the sorted per-op-class table (ms, %, count, key).
- Alternative (lighter, keeps graphs ON for non-DSV4 ops): `DSV4_OPPROF=1` alone — only the ops that
  already run graphs-off (fused/grouped MoE + small-prefill band) get timed. Use the NO_GRAPHS run
  for the FULL breakdown.

**The dump auto-prints — no clean shutdown needed.** After `DSV4_OPPROF_DUMP_AFTER` timed ops
(default 8000 ≈ one 13k-prefill forward) each rank prints `[DSV4_OPPROF] (auto) ...` to stderr ONCE,
mid-run. So `kill -9 $(ps -C llama-server -o pid=)` (the mandated stop) is fine — grab the table from
the server log/stderr after the prefill completes. Tune the trigger with `DSV4_OPPROF_DUMP_AFTER=<N>`
(set 0 for exit-only). A second copy also prints at clean exit if the dtor runs.

## STATIC ANALYSIS — TP all-reduce (likely a big chunk of the 5x prefill gap)
Files: ggml-backend-meta.cpp (dispatch 2529-2577, delay logic 2138-2313),
       ggml-cuda.cu ggml_backend_cuda_comm_allreduce_nccl (1189-1228, SPMD branch).

Findings (code-read, to be confirmed by the profile):
1. **DSV4 MLA KV is MIRRORED, not split** (tp-serve.h) -> attention does NOT all-reduce. The per-layer
   AllReduce is ONLY the MoE down-proj combine. ~1 AllReduce per MoE layer => ~45-60 inter-node
   round-trips per forward.
2. **The reduce is fully SERIALIZED with compute, zero overlap.** Dispatch loop (2529): for each
   subgraph i -> async compute(i) -> `comm_allreduce(i)` enqueued on the SAME stream -> compute(i+1)
   reads the reduced result on that same stream. NCCL kernel and the GEMMs never run concurrently.
   The `get_i_delayed` optimization only REDUCES THE COUNT of reduces (folds the MoE-combine epilogue
   into the reducing subgraph); it does not overlap comm with compute.
3. vLLM (the 1595-1722 target) overlaps TP comm with compute (chunked reduce-scatter / pipelined AG).
   This serialized-reduce is the prime structural suspect for the prefill gap, and the per-layer
   tiny-payload reduce is the decode tax.

### Why naive overlap is non-trivial here
The AllReduce output feeds the immediately-following residual+norm of the SAME layer, so reduce-i
cannot overlap with compute that depends on it. Real overlap needs either (a) reduce-scatter +
deferred all-gather so the next layer's down-proj input GEMM overlaps the gather, or (b) splitting
the hidden dim and pipelining reduce-chunks against the tail of the producing GEMM. Round 2 target
once the profile confirms the % the reduce costs.

## NEXT (Round 2, after profile numbers come back)
- If TP reduce is a big %: implement reduce/compute overlap (chunked pipeline on a side stream) OR
  batch/fuse adjacent reduces.
- Larger UB (2048/4096) now that fused single-workspace replaced the 43 arenas — amortizes weight
  reads across more tokens/forward. Watch WATCH_MIN_GB.
- Fold in native FP8 dense GEMM (+4.5%, branch feat/dsv4-fp8-native-gemm).
- Decode->45: fix MTP graph-context sizing (ggml.c:1929 obj_new exhaustion) so MTP runs with fused
  (no sparse); the fused-aware grouped HP decode + MTP verify widths must coexist in the graph cache.
