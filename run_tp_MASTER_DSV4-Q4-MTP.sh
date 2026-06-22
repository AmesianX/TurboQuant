#!/usr/bin/env bash
# ============================================================================
#  TP  M A S T E R   (leader, rank 0)   ->  run on  .66  (10.0.1.1)
#
#  MODEL : DeepSeek-V4-Flash  Q4  (head-split MoE across .66 + .67)
#          + MTP self-speculative draft (NextN layer MIRRORED, leader-local)
#  2-shard set: DSV4-Q4-00001-of-00002.gguf (base) + 00002 (NextN/MTP head)
# ============================================================================
set -euo pipefail

MODEL="$HOME/Models/DeepSeek-V4-Flash-GGUF/Q4mtp/DSV4-Q4-00001-of-00002.gguf"
PORT=8080
API_KEY="tbq-dsv4"
MASTER_ADDR="10.0.1.1"
MASTER_PORT=29655
IFACE="enp1s0f0np0"
CTX=8192

cd "$(dirname "$0")"
[ -d "$HOME/nccl-align" ] && export LD_LIBRARY_PATH="$HOME/nccl-align:${LD_LIBRARY_PATH:-}"
export NCCL_SOCKET_IFNAME="$IFACE"
export GGML_TP_NRANKS=2
export GGML_TP_RANK=0
export GGML_TP_MASTER_ADDR="$MASTER_ADDR"
export GGML_TP_MASTER_PORT="$MASTER_PORT"
export DSV4_MULTISLOT=1        # Step C: batch concurrent slots' MTP-verify into one decode

echo "=============================================================="
echo " TP MASTER (rank 0)  DSV4-Q4 + MTP + MULTISLOT"
echo "   model : $MODEL"
echo "   serve : http://0.0.0.0:$PORT   (api-key: $API_KEY)"
echo "   peer  : $IFACE @ $MASTER_ADDR  (NCCL $MASTER_PORT / ctrl $((MASTER_PORT+1)))"
echo "=============================================================="

exec build/bin/llama-server \
  -m "$MODEL" \
  -c "$CTX" -ngl 999 -fa on -sm tensor -fit off --no-warmup --no-mmap \
  -ctk f16 -ctv f16 \
  --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-p-min 0.0 \
  --parallel 2 \
  --host 0.0.0.0 --port "$PORT" --api-key "$API_KEY"
