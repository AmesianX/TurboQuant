#!/usr/bin/env bash
# ============================================================================
#  TP  S L A V E   (follower, rank 1)   ->  run on  .67  (10.0.1.2)
#
#  MODEL : DeepSeek-V4-Flash  Q4  (head-split MoE)  -- byte-identical to master
#  2-shard set: DSV4-Q4-00001-of-00002.gguf (base) + 00002 (NextN/MTP head)
#  No HTTP. Replays leader's verify decodes / KV ops over MASTER_PORT+1.
# ============================================================================
set -euo pipefail

MODEL="$HOME/Models/DeepSeek-V4-Flash-GGUF/Q4mtp/DSV4-Q4-00001-of-00002.gguf"
PORT=8080
MASTER_ADDR="10.0.1.1"
MASTER_PORT=29655
IFACE="enp1s0f0np0"
CTX=8192

cd "$(dirname "$0")"
[ -d "$HOME/nccl-align" ] && export LD_LIBRARY_PATH="$HOME/nccl-align:${LD_LIBRARY_PATH:-}"
export NCCL_SOCKET_IFNAME="$IFACE"
export GGML_TP_NRANKS=2
export GGML_TP_RANK=1
export GGML_TP_MASTER_ADDR="$MASTER_ADDR"
export GGML_TP_MASTER_PORT="$MASTER_PORT"
export DSV4_MULTISLOT=1        # must match leader so the SPMD follower builds the identical graph
export DSV4_MS_DBG=1

echo "=============================================================="
echo " TP SLAVE (rank 1)  DSV4-Q4 + MTP + MULTISLOT -- follower (no HTTP)"
echo "   model  : $MODEL"
echo "   leader : $MASTER_ADDR  (NCCL $MASTER_PORT / ctrl $((MASTER_PORT+1)))"
echo "=============================================================="

exec build/bin/llama-server \
  -m "$MODEL" \
  -c "$CTX" -ngl 999 -fa on -sm tensor -fit off --no-warmup --no-mmap \
  -ctk f16 -ctv f16 \
  --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-p-min 0.0 \
  --parallel 2 \
  --host 0.0.0.0 --port "$PORT"
