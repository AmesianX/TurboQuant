# Cross-node Tensor Parallelism for DSV4 (branch: feat/tp-2node-dsv4)

Goal: split DeepSeek-V4-Flash across .66 + .67 (each 1x GB10) so each node reads
HALF the weights per token in parallel -> ~2x decode bandwidth -> ~2x t/s, served
by llama-server. Nobody upstream has cross-node TP (only single-node multi-GPU TP
+ pipeline-only RPC). So it's net-new, but ~80% of the machinery already exists.

## What already exists in this fork (verified)
- `LLAMA_SPLIT_MODE_TENSOR` graph-level TP; `llm_arch_supports_sm_tensor()` INCLUDES
  `LLM_ARCH_DEEPSEEK4` (src/llama-arch.cpp). So DSV4 is TP-eligible.
- `ggml/src/ggml-cuda/allreduce.cu` (PR #22299): custom 2-GPU AllReduce.
- NCCL linked (`GGML_CUDA_NCCL=ON`), `ncclCommInitAll` at ggml-cuda.cu:1365.
- Constraints: TP requires flash_attn ON; not compatible with KV-quant (tbq3) yet;
  hardcoded n_devices==2.

## The gap (what we must build)
The existing NCCL init is `ncclCommInitAll` = SINGLE PROCESS, multiple GPUs in ONE
machine. GB10 = 1 GPU/box, so it cannot span .66+.67 as-is. Cross-node needs:
- M1: `ncclCommInitRank` + multi-process SPMD launch (one rank/node, ncclUniqueId
  exchanged over the network), each rank loading its tensor shard.
- M2: route the TP graph's AllReduce through cross-node NCCL (the intra-node
  peer-copy path in allreduce.cu does NOT cross machines).
- M3: DSV4 specifics under TP (MoE expert split, MLA, recurrent compress-state,
  per-rank KV, sampling on rank 0).
- M4: speculative (MTP/n-gram) on top.

## M0 de-risk RESULT (PASSED, gate green for decode)
Bench: tools/tp-bench/tp_ar_bench.py via torchrun (bundled NCCL 2.28.9, matched on
both boxes). Launch: tools/tp-bench/launch_arbench{,_info}.sh, master 10.0.1.1:PORT.
- **Transport: genuine RDMA** -- NCCL `NET/IB ... RoCE provider=Mlx5 speed=200000`,
  "Using network IB" (not socket).
- **Dual-rail auto-bonded to 400 Gb/s**: both Sparks cabled on both ConnectX-7 QSFP
  ports; NCCL auto-builds virtual devices `rocep1s0f0+rocep1s0f1` (400000) and
  `roceP2p1s0f0+roceP2p1s0f1` (400000). M0 already used dual-rail.
- **AllReduce latency @ n_embd (4096 f32 = 16KB): ~21 us** (1024:~18us). DSV4 token
  ~ 86 AllReduces -> ~1.8ms/token sync. vs halved per-node weight read (~25ms): the
  sync is small -> **~1.85x decode projected** (20 -> ~37 t/s).
- **GPUDirect RDMA is DISABLED** (`GDR 0`, "GPU Direct RDMA Disabled for HCA ...").
  Data bounces GPU->host->NIC. Enabling GDR is an M2 latency lever.
- **KNOWN ISSUE**: AllReduce HANGS at >= 16384 elems (64KB) over this RoCE setup
  (16KB works). Decode TP (16KB hidden) is unaffected; PREFILL (large AllReduces)
  will need this fixed -- likely NCCL proto/algo or RoCE PFC/flow-control or GDR.

## NCCL VERSION MISMATCH (must fix before M1 build)
System libnccl differs: .66 = 2.30.7, .67 = 2.30.4. A llama.cpp binary links system
libnccl, so cross-node init hits "bootstrap Message truncated: received 144 vs 128"
(observed with the standalone C bench). torch worked because it bundles a MATCHED
2.28.9 on both. -> M1.0 prerequisite: align system NCCL on both boxes (or ship one
libnccl via LD_LIBRARY_PATH for both ranks).

## Network facts
- RoCE iface enp1s0f0np0 = 10.0.1.1 (.66) / 10.0.1.2 (.67). RDMA devs: rocep1s0f0,
  rocep1s0f1 (pci0000), roceP2p1s0f0/f1 (pci0002, likely the direct-connect pair).
- Reuse launch_ddp.sh env: NCCL_SOCKET_IFNAME=enp1s0f0np0, do NOT set NCCL_IB_DISABLE.

## Next: M1.0 = align NCCL versions, then ncclCommInitRank SPMD prototype.

## M1.0 COMPLETE (verified)
- Source rsynced .66->.67 (NOTE: do NOT `--exclude 'models/'` — it wrongly matches
  src/models/, dropping dflash.cpp -> llama_model_dflash link error. Use only
  *.gguf / build/ / .git / *.o / *.so excludes).
- After source change on .67, MUST `cmake .` (reconfigure) before build, else stale
  generated Makefiles omit new src/models/*.cpp -> "undefined reference to vtable".
- NCCL aligned to 2.30.7 on both via ~/nccl-align (LD_LIBRARY_PATH at launch; system
  untouched). VERIFIED: C-NCCL all-reduce cross-node, all sizes, ~26us@16KB. The
  earlier torch "16384 hang" was a torch/NCCL-2.28.9 artifact, NOT a transport limit.
- llama-server builds on BOTH boxes from the synced source.

## M1 DESIGN (the crux: SPMD-ify the meta-backend)
ggml-backend-meta.cpp implements TP as SINGLE-PROCESS over N local `simple_devs`:
splits each weight along an axis into N slices (get_split_state callback), allocates
all N locally, computes, and AllReduces across the local devices (ggml-cuda comm_init
-> ncclCommInitAll, single process). GB10 = 1 GPU/node, so this can't span nodes.

SPMD conversion = DECOUPLE "logical world size N / my rank R" from "local devices (1)":
1. comm_init SPMD path (ggml-cuda.cu ~1361): when env GGML_TP_NRANKS>1, use
   ncclCommInitRank(comm, NRANKS, uniqueId, RANK) instead of ncclCommInitAll. Bootstrap
   uniqueId: rank0 ncclGetUniqueId -> TCP send to peers (reuse nccl_ar_bench exchange),
   others recv. comms={1}. try_allreduce_nccl already loops comms -> 1 ncclAllReduce on
   the single local tensor = cross-rank sum. (verified transport.)
2. meta-device: build with simple_devs=[local GPU] but split_state n_devices=NRANKS;
   rank R materializes ONLY slice R locally (loader writes its slice), other slices are
   remote. The local partial is AllReduced cross-rank. Requires meta-backend to alloc/
   compute only the local slice (today it does all N). KEY CHANGE.
3. model loader (llama-model.cpp): tensor_split + set_tensor must write rank R's slice
   to the local device only. The split math (tensor_split_scan, ~636) already computes
   per-slice ne; thread global rank through.
4. server launch: SPMD = 2 processes (one/node), env GGML_TP_RANK/NRANKS/MASTER_ADDR/
   PORT (+ LD_LIBRARY_PATH=~/nccl-align, NCCL_SOCKET_IFNAME=enp1s0f0np0). rank0 owns
   tokenizer/sampling/HTTP; rank1 is a compute follower in the decode loop.
   -> needs a follower run-loop on rank!=0 (no HTTP) driven by the same graph.

RISK: this is a real reimplementation of the execution model (single-process ->
SPMD), not a small patch. Proceed incrementally: (M1a) comm_init SPMD + a standalone
2-proc allreduce test through ggml; (M1b) meta-device local-slice-only; (M1c) loader;
(M1d) follower loop. Search upstream/NCCL docs at each blocker (user directive).

## M1a DONE (both boxes, compiles+links)
ggml-cuda.cu: ggml_backend_cuda_comm_init_nccl_spmd() — GGML_TP_NRANKS>1 -> ncclCommInitRank
over global world (1 local GPU/rank), uniqueId via TCP bootstrap (GGML_TP_MASTER_ADDR/PORT).
try_allreduce_nccl reused (1 comm -> 1 ncclAllReduce on the single local tensor = cross-rank sum).

## M1b SCOPE confirmed at code level (ggml-backend-meta.cpp, 2262 lines)
- Reduction: n_reduce_steps=ceil(log2(n_devs)) butterfly, but when NCCL comm is up it calls
  comm_allreduce(comm_ctx, nodes) at ~line 2204 with the per-device tensor array.
- comm_init(simple_backends, n) at ~1636-1639 with ALL LOCAL backends. With 1 local backend
  n_devs=1 => meta-backend is a no-op passthrough (no split/reduce).
- THEREFORE M1b = decouple LOGICAL split (NRANKS, for weight sharding) from PHYSICAL local
  backends (=1). Everywhere the meta-backend iterates simple_backends/n_devs for alloc + graph
  build + compute, SPMD must: shard weights into NRANKS, allocate/compute ONLY my rank's slice
  on the local GPU, and reduce via the cross-rank comm_allreduce (M1a, ready) on the single
  local partial. This is a STRUCTURAL change to the meta-backend execution/alloc model, spread
  across buffer alloc (backend_config/bufs), per-device cgraphs (~1983-1993), and split-state
  application. Largest single piece of the project.
- Key edit anchors: ctx ctor ~1598-1646 (n_reduce_steps, comm_init), reduce path ~1983-2204.
- Then M1c (loader writes only slice R, llama-model.cpp tensor_split_scan ~636) and M1d
  (rank!=0 follower decode loop, no HTTP) and M3 (DSV4 hybrid correctness) and M4 (test+0.0.0.0).
