#!/usr/bin/env bash
# ============================================================================
#  TP  S L A V E  (follower, rank 1)  ->  run on  .67  (10.0.1.2)
#
#  MODEL : DeepSeek-V4-Flash  FP4-FP8-native  (SINGLE 156GB file, byte-identical)
#  No HTTP. Replays leader's decodes / KV ops over MASTER_PORT+1.
#  PLAIN decode (no MTP).
# ============================================================================
set -euo pipefail

MODEL="$HOME/Models/DeepSeek-V4-Flash-GGUF/FP4/DeepSeek-V4-Flash-FP4-FP8-native-MTP.gguf"
PORT=8080
MASTER_ADDR="10.0.1.1"
MASTER_PORT=29658
IFACE="enp1s0f0np0"
CTX=8192

cd "$(dirname "$0")"
[ -d "$HOME/nccl-align" ] && export LD_LIBRARY_PATH="$HOME/nccl-align:${LD_LIBRARY_PATH:-}"
export NCCL_SOCKET_IFNAME="$IFACE"
export GGML_TP_NRANKS=2
export GGML_TP_RANK=1
export GGML_TP_MASTER_ADDR="$MASTER_ADDR"
export GGML_TP_MASTER_PORT="$MASTER_PORT"
export DSV4_BATCHED_COMPRESSOR=1  # 배치 compressor (b0 cap 수정)
export DSV4_MULTISLOT=1        # 동시 슬롯 배치 → 집계 처리량
export DSV4_VERIFY_REUSE=1     # MTP verify graph 재사용 (graphs reused=0 해결)

echo "=============================================================="
echo " TP SLAVE (rank 1)  DSV4-FP4  PLAIN -- follower (no HTTP)"
echo "   model  : $MODEL"
echo "   leader : $MASTER_ADDR  (NCCL $MASTER_PORT / ctrl $((MASTER_PORT+1)))"
echo "=============================================================="

exec build/bin/llama-server \
  -m "$MODEL" \
  -c "$CTX" -ngl 999 -fa on -sm tensor -fit off --no-warmup --no-mmap \
  -b 512 -ub 256 -ctk f16 -ctv f16 \
  --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-p-min 0.0 --parallel 2 \
  --host 0.0.0.0 --port "$PORT"
