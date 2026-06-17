#!/bin/bash
RANK=$1
cd /home/user/work/TurboQuant/tools/tp-bench
export NCCL_SOCKET_IFNAME=enp1s0f0np0
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
exec python3 -m torch.distributed.run --nnodes=2 --node_rank="$RANK" --nproc_per_node=1 \
  --master_addr=10.0.1.1 --master_port=29511 tp_ar_bench.py
