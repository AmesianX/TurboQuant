# M0 de-risk for cross-node tensor parallelism (feat/tp-2node-dsv4).
# Uses torch.distributed (bundled NCCL 2.28.9, matched on both boxes) over RoCE
# to measure all-reduce latency for hidden-state-sized tensors across 2 NODES.
# A DSV4 decode token issues ~2 all-reduces/layer x 43 = ~86 collectives; the
# 4096-elem latency x 86 is the per-token sync overhead that must stay small vs
# the ~25ms halved per-node weight read for ~2x to survive.
#
# launch (mirror launch_ddp.sh) on EACH box with its node_rank:
#   NCCL_SOCKET_IFNAME=enp1s0f0np0 python3 -m torch.distributed.run \
#     --nnodes=2 --node_rank=<0|1> --nproc_per_node=1 \
#     --master_addr=10.0.1.1 --master_port=29503 tp_ar_bench.py
import os, time, torch, torch.distributed as dist

dist.init_process_group("nccl")
rank = dist.get_rank()
torch.cuda.set_device(0)
dev = torch.device("cuda", 0)

if rank == 0:
    print(f"{'elems':>10} {'bytes':>12} {'us/AR':>10} {'alg_GB/s':>10}")

for n in [1024, 4096, 16384, 65536, 262144, 1048576]:
    x = torch.ones(n, device=dev, dtype=torch.float32)
    for _ in range(30):
        dist.all_reduce(x)
    torch.cuda.synchronize()
    dist.barrier()
    iters = 1000
    t0 = time.perf_counter()
    for _ in range(iters):
        dist.all_reduce(x)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    if rank == 0:
        print(f"{n:>10} {n*4:>12} {dt*1e6:>10.2f} {n*4/dt/1e9:>10.2f}")

if rank == 0:
    print("# per-token sync ~= 86 x (latency at 4096 elems)")

dist.destroy_process_group()
