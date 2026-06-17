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
