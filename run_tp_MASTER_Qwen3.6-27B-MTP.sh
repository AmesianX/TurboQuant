#!/usr/bin/env bash
# ============================================================================
#  TP  M A S T E R   (leader, rank 0)   ->  run on  .66  (10.0.1.1)
#
#  MODEL : Qwen3.6 27B Uncensored Heretic v2  (Native-MTP-Preserved, DENSE 27B)
#  MODE  : cross-node tensor-parallel (head-split across .66 + .67)
#          + MTP self-speculative draft (NextN layer MIRRORED, runs leader-local)
#
#  Leader owns the HTTP endpoint / tokenizer / sampler and broadcasts every
#  context mutation to the follower over the control channel (MASTER_PORT+1).
#  Launch ORDER does not matter (leader blocks on accept, follower retries).
# ============================================================================
set -euo pipefail

# ---------------------------- config (edit here) ----------------------------
MODEL="$HOME/Models/Qwen36-MTP-test/model.gguf"   # Qwen3.6-27B Native-MTP (dense)
PORT=8080                  # web UI / OpenAI API port
API_KEY="tbq-mtp"          # required when binding 0.0.0.0
MASTER_ADDR="10.0.1.1"     # THIS box (.66) RoCE IP (ConnectX-7)
MASTER_PORT=29620          # NCCL bootstrap; control channel = MASTER_PORT+1
IFACE="enp1s0f0np0"        # RoCE interface
CTX=8192
# ----------------------------------------------------------------------------

cd "$(dirname "$0")"
[ -d "$HOME/nccl-align" ] && export LD_LIBRARY_PATH="$HOME/nccl-align:${LD_LIBRARY_PATH:-}"
export NCCL_SOCKET_IFNAME="$IFACE"
export GGML_TP_NRANKS=2
export GGML_TP_RANK=0
export GGML_TP_MASTER_ADDR="$MASTER_ADDR"
export GGML_TP_MASTER_PORT="$MASTER_PORT"

echo "=============================================================="
echo " TP MASTER (rank 0)"
echo "   model : $MODEL"
echo "   serve : http://0.0.0.0:$PORT   (api-key: $API_KEY)"
echo "   peer  : $IFACE @ $MASTER_ADDR  (NCCL $MASTER_PORT / ctrl $((MASTER_PORT+1)))"
echo "=============================================================="

exec build/bin/llama-server \
  -m "$MODEL" \
  -c "$CTX" -ngl 999 -fa on -sm tensor -fit off --no-warmup \
  --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-p-min 0.75 \
  --host 0.0.0.0 --port "$PORT" --api-key "$API_KEY"
