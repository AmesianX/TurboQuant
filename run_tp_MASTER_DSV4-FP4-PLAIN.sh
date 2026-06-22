#!/usr/bin/env bash
# ============================================================================
#  TP  M A S T E R  (leader, rank 0)  ->  run on  .66  (10.0.1.1)
#
#  MODEL : DeepSeek-V4-Flash  FP4-FP8-native  (SINGLE 156GB file)
#          MLA attention MIRRORED (DP), MoE/FFN tensor-split across .66 + .67.
#          PLAIN decode (no MTP) -- this is the 2-box baseline to beat.
# ============================================================================
set -euo pipefail

MODEL="$HOME/Models/DeepSeek-V4-Flash-GGUF/FP4/DeepSeek-V4-Flash-FP4-FP8-native.gguf"
PORT=8080
API_KEY="test1234@X"
MASTER_ADDR="10.0.1.1"
MASTER_PORT=29658
IFACE="enp1s0f0np0"
CTX=8192

cd "$(dirname "$0")"
[ -d "$HOME/nccl-align" ] && export LD_LIBRARY_PATH="$HOME/nccl-align:${LD_LIBRARY_PATH:-}"
export NCCL_SOCKET_IFNAME="$IFACE"
export GGML_TP_NRANKS=2
export GGML_TP_RANK=0
export GGML_TP_MASTER_ADDR="$MASTER_ADDR"
export GGML_TP_MASTER_PORT="$MASTER_PORT"
export DSV4_BATCHED_COMPRESSOR=1  # 배치 compressor (b0 cap 수정)
export DSV4_MULTISLOT=1        # 동시 슬롯 배치 → 집계 처리량

echo "=============================================================="
echo " TP MASTER (rank 0)  DSV4-FP4  PLAIN (no MTP)"
echo "   model : $MODEL"
echo "   serve : http://0.0.0.0:$PORT   (api-key: $API_KEY)"
echo "   peer  : $IFACE @ $MASTER_ADDR  (NCCL $MASTER_PORT / ctrl $((MASTER_PORT+1)))"
echo "=============================================================="

exec build/bin/llama-server \
  -m "$MODEL" \
  -c "$CTX" -ngl 999 -fa on -sm tensor -fit off --no-warmup --no-mmap \
  -b 512 -ub 256 -ctk f16 -ctv f16 \
  --host 0.0.0.0 --port "$PORT" --api-key "$API_KEY"
