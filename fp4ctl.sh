#!/usr/bin/env bash
# ============================================================================
#  DSV4-FP4 2-box TP server control  —  START / STOP / RESTART / STATUS
#  .66 = master (this box, HTTP 0.0.0.0:8080, api-key test1234@X)
#  .67 = slave/follower (10.0.1.2, no HTTP)
#
#  사용법:  bash fp4ctl.sh start | stop | restart | status
#
#  규칙(절대): pkill 금지. 오직 `kill -9 $(ps -C llama-server -o pid=)`.
#  stop은 양박스 GPU 메모리가 실제 회수될 때까지 기다림(재시작 경합 OOM 방지).
# ============================================================================
set -uo pipefail
cd "$(dirname "$0")"

SLAVE_HOST="10.0.1.2"
MASTER_SH="run_tp_MASTER_DSV4-FP4-MTP.sh"
SLAVE_SH="run_tp_SLAVE_DSV4-FP4-MTP.sh"
MLOG="/tmp/fp4_master.log"
SLOG="/tmp/fp4_slave.log"

_procs_66() { ps -C llama-server -o pid= 2>/dev/null | wc -l | tr -d ' '; }
_procs_67() { ssh "$SLAVE_HOST" 'ps -C llama-server -o pid= 2>/dev/null | wc -l' 2>/dev/null | tr -d ' '; }
_gpu_66()   { nvidia-smi 2>/dev/null | grep -c llama-server; }
_gpu_67()   { ssh "$SLAVE_HOST" 'nvidia-smi 2>/dev/null | grep -c llama-server' 2>/dev/null; }

cmd_stop() {
  echo "[stop] killing llama-server on both boxes (pid only, never pkill)..."
  for r in 1 2 3 4 5 6; do
    ps -C llama-server -o pid= 2>/dev/null | xargs -r kill -9 2>/dev/null
    ssh "$SLAVE_HOST" 'ps -C llama-server -o pid= 2>/dev/null | xargs -r kill -9 2>/dev/null'
    sleep 3
    local a b ga; a=$(_procs_66); b=$(_procs_67); ga=$(_gpu_66)
    echo "  round$r: .66 proc=$a gpu=$ga | .67 proc=$b"
    [ "$a" = "0" ] && [ "$b" = "0" ] && [ "$ga" = "0" ] && { echo "[stop] ✅ both boxes clean, GPU reclaimed"; return 0; }
  done
  echo "[stop] ⚠️ still not fully clean after 6 rounds — check manually"; return 1
}

cmd_start() {
  if [ "$(_procs_66)" != "0" ] || [ "$(_procs_67)" != "0" ]; then
    echo "[start] ⚠️ a server is already running — run 'restart' (or 'stop' first)"; return 1
  fi
  echo "[start] launching slave on $SLAVE_HOST ..."
  timeout 20 ssh "$SLAVE_HOST" "cd ~/work/TurboQuant && nohup bash $SLAVE_SH > $SLOG 2>&1 & disown; sleep 2; echo '  slave pid='\$(ps -C llama-server -o pid=|tr -d ' ')"
  echo "[start] launching master on .66 ..."
  nohup bash "$MASTER_SH" > "$MLOG" 2>&1 & disown
  sleep 6
  echo "  master pid=$(ps -C llama-server -o pid= | tr -d ' ')  (loading ~3min; HTTP 0.0.0.0:8080 key test1234@X)"
  echo "[start] watch readiness:  bash fp4ctl.sh status   (or: tail -f $MLOG)"
}

cmd_status() {
  echo "=== .66 master ==="; ps -C llama-server -o pid=,etime= 2>/dev/null || echo "  (down)"
  ss -tlnp 2>/dev/null | grep -q ':8080' && echo "  port 8080: LISTEN" || echo "  port 8080: (not up)"
  grep -qiE "all slots are idle|follower connected; serving" "$MLOG" 2>/dev/null && echo "  state: ✅ SERVING" || echo "  state: loading / not ready"
  echo "=== .67 slave ==="; ssh "$SLAVE_HOST" 'ps -C llama-server -o pid=,etime= 2>/dev/null || echo "  (down)"' 2>/dev/null
}

case "${1:-}" in
  start)   cmd_start ;;
  stop)    cmd_stop ;;
  restart) cmd_stop && cmd_start ;;
  status)  cmd_status ;;
  *) echo "usage: bash fp4ctl.sh start | stop | restart | status"; exit 1 ;;
esac
