#!/usr/bin/env bash
# ============================================================================
#  TP  S L A V E   (follower, rank 1)   ->  run on  .67  (10.0.1.2)
#
#  MODEL : Qwen3.6 27B Uncensored Heretic v2  (Native-MTP-Preserved, DENSE 27B)
#  MODE  : cross-node tensor-parallel (head-split across .66 + .67)
#
#  The follower has NO HTTP endpoint. It loads its weight shard, connects to the
#  leader's control channel (MASTER_ADDR:MASTER_PORT+1) and replays the leader's
#  verify decodes / KV ops so the per-layer NCCL AllReduce lines up.
#  REQUIRES the matched NCCL in ~/nccl-align (this box's system NCCL differs).
# ============================================================================
set -euo pipefail

# ---------------------------- config (edit here) ----------------------------
MODEL="$HOME/Models/Qwen36-MTP-test/model.gguf"   # MUST be byte-identical to the master's
PORT=8080
MASTER_ADDR="10.0.1.1"     # the MASTER box (.66) RoCE IP
MASTER_PORT=29620          # must match the master
IFACE="enp1s0f0np0"
CTX=8192
# ----------------------------------------------------------------------------

cd "$(dirname "$0")"
[ -d "$HOME/nccl-align" ] && export LD_LIBRARY_PATH="$HOME/nccl-align:${LD_LIBRARY_PATH:-}"
export NCCL_SOCKET_IFNAME="$IFACE"
export GGML_TP_NRANKS=2
export GGML_TP_RANK=1
export GGML_TP_MASTER_ADDR="$MASTER_ADDR"
export GGML_TP_MASTER_PORT="$MASTER_PORT"

echo "=============================================================="
echo " TP SLAVE (rank 1)  -- follower (no HTTP)"
echo "   model  : $MODEL"
echo "   leader : $MASTER_ADDR  (NCCL $MASTER_PORT / ctrl $((MASTER_PORT+1)))"
echo "   nccl   : ${LD_LIBRARY_PATH:-<system>}"
echo "=============================================================="

exec build/bin/llama-server \
  -m "$MODEL" \
  -c "$CTX" -ngl 999 -fa on -sm tensor -fit off --no-warmup \
  --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-p-min 0.75 \
  --host 0.0.0.0 --port "$PORT"
