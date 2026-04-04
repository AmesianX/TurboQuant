#!/bin/bash
# TurboQuant Server — OpenAI-compatible API on port 8081
# Usage: ./run-tbq-server.sh [tbq3|tbq4|tbqp3|tbqp4] [port]

KV_TYPE="${1:-tbqp3}"
PORT="${2:-8081}"
MODEL="/Users/ro/.ollama/models/blobs/sha256-4c27e0f5b5adf02ac956c7322bd2ee7636fe3f45a8512c9aba5385242cb6e09a"
TQ_DIR="$(dirname "$0")"

echo "🚀 Starting TurboQuant server"
echo "   Model: Gemma4 E4B"
echo "   K cache: ${KV_TYPE} (V cache: f16)"
echo "   API: http://localhost:${PORT}/v1/chat/completions"
echo ""

# Free Ollama's model memory
curl -s http://localhost:11434/api/generate -d '{"model":"gemma4:e4b","keep_alive":0}' > /dev/null 2>&1
sleep 2

exec "$TQ_DIR/build/bin/llama-server" \
  -m "$MODEL" \
  --host 0.0.0.0 --port "$PORT" \
  -ngl 0 \
  --cache-type-k "$KV_TYPE" --cache-type-v f16 \
  -c 4096
