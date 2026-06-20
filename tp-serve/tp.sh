#!/usr/bin/env bash
# ============================================================================
#  DSV4-Flash  2-box cross-node TP + MTP  serving control
#
#  Run the SAME script on both boxes; the role is auto-detected from the local
#  RoCE IP (10.0.1.1 = MASTER/rank0/HTTP, 10.0.1.2 = SLAVE/rank1/follower).
#
#  Commands (UPPERCASE):
#    START        start this box's role (master serves HTTP, slave follows)
#    STOP         stop llama-server on this box
#    RESTART      STOP then START on this box
#    ALLRESTART   (master only) STOP+START BOTH boxes in the right order
#    STATUS       show what's running on this box
#
#  Launch order does not matter (leader blocks on accept, follower retries),
#  but ALLRESTART brings the slave up first, then the master.
# ============================================================================
set -euo pipefail

# ---------------------------- config (edit here) ----------------------------
REPO="$HOME/work/TurboQuant"                 # repo root (has build/bin/llama-server)
MODEL="$HOME/Models/DeepSeek-V4-Flash-GGUF/Q4mtp/DSV4-Q4-00001-of-00002.gguf"
PORT=8080
API_KEY="tbq-dsv4"                           # required because we bind 0.0.0.0
CTX=8192
IFACE="enp1s0f0np0"                          # RoCE interface
MASTER_IP="10.0.1.1"
SLAVE_IP="10.0.1.2"
MASTER_PORT=29655                            # NCCL bootstrap; control channel = +1
SLAVE_SSH="10.0.1.2"                         # ssh target for the slave box
SPEC="--spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-p-min 0.75"
SELF="$REPO/tp-serve/tp.sh"                  # path of this script on each box
# ----------------------------------------------------------------------------

COMMON="-c $CTX -ngl 999 -fa on -sm tensor -fit off --no-warmup --no-mmap -ctk f16 -ctv f16 $SPEC"

# ---- role auto-detect by local RoCE IP -------------------------------------
detect_role() {
    local ips; ips="$(ip -4 -o addr show "$IFACE" 2>/dev/null | awk '{print $4}' | cut -d/ -f1)"
    if   grep -qx "$MASTER_IP" <<<"$ips"; then echo MASTER
    elif grep -qx "$SLAVE_IP"  <<<"$ips"; then echo SLAVE
    else echo UNKNOWN; fi
}
ROLE="$(detect_role)"
LOG="/tmp/tp_${ROLE}.log"

env_common() {
    [ -d "$HOME/nccl-align" ] && export LD_LIBRARY_PATH="$HOME/nccl-align:${LD_LIBRARY_PATH:-}"
    export NCCL_SOCKET_IFNAME="$IFACE"
    export GGML_TP_NRANKS=2
    export GGML_TP_MASTER_ADDR="$MASTER_IP"
    export GGML_TP_MASTER_PORT="$MASTER_PORT"
}

stop_local() {
    local pids; pids="$(ps -C llama-server -o pid= | tr -d ' ')"
    if [ -z "$pids" ]; then echo "[$ROLE] STOP: nothing running"; return 0; fi
    for p in $(ps -C llama-server -o pid=); do kill "$p" 2>/dev/null || true; done
    # give it a moment, then hard-kill any survivor
    sleep 2
    for p in $(ps -C llama-server -o pid=); do kill -9 "$p" 2>/dev/null || true; done
    echo "[$ROLE] STOP: killed [$pids]"
}

start_local() {
    cd "$REPO"
    if [ -n "$(ps -C llama-server -o pid=)" ]; then
        echo "[$ROLE] START: already running (use RESTART)"; return 0
    fi
    env_common
    case "$ROLE" in
        MASTER)
            export GGML_TP_RANK=0
            nohup build/bin/llama-server -m "$MODEL" $COMMON \
                --host 0.0.0.0 --port "$PORT" --api-key "$API_KEY" > "$LOG" 2>&1 &
            disown
            echo "[MASTER] START: serving http://0.0.0.0:$PORT (api-key $API_KEY), log $LOG" ;;
        SLAVE)
            export GGML_TP_RANK=1
            nohup build/bin/llama-server -m "$MODEL" $COMMON \
                --host 0.0.0.0 --port "$PORT" > "$LOG" 2>&1 &
            disown
            echo "[SLAVE] START: follower up (no HTTP), log $LOG" ;;
        *)
            echo "ERROR: role UNKNOWN — local $IFACE is neither $MASTER_IP nor $SLAVE_IP"; exit 1 ;;
    esac
}

status_local() {
    local pids; pids="$(ps -C llama-server -o pid=,etime= | tr '\n' ';')"
    echo "[$ROLE] STATUS: ${pids:-(not running)}"
}

case "${1:-}" in
    START)   start_local ;;
    STOP)    stop_local ;;
    RESTART) stop_local; start_local ;;
    STATUS)  status_local ;;
    ALLRESTART)
        if [ "$ROLE" != "MASTER" ]; then
            echo "ERROR: run ALLRESTART on the MASTER box ($MASTER_IP)"; exit 1
        fi
        echo "== ALLRESTART: stopping both =="
        ssh "$SLAVE_SSH" "$SELF STOP" || true
        stop_local
        echo "== ALLRESTART: starting slave then master =="
        ssh "$SLAVE_SSH" "$SELF START"
        sleep 3
        start_local ;;
    *)
        echo "DSV4 TP+MTP control (role: $ROLE)"
        echo "usage: $(basename "$0") START | STOP | RESTART | ALLRESTART | STATUS"
        exit 1 ;;
esac
