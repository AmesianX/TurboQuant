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
CTX=0           # 1M MTP ckpt prof
GRAPH_SLOTS=16  # TEST
SLOT_PROBE=     # diag: DSV4_SLOT_PROBE (pool slot actions)
RESULT_DEBUG=
VERIFY_REUSE=1
KERNEL_PROF=
GRAPH_PROBE=
STAGE_PROF=
                #  quant KV now allowed under -sm tensor for MLA (mirrored KV) — see llama-context.cpp guard.
IFACE="enp1s0f0np0"                          # RoCE interface
MASTER_IP="10.0.1.1"
SLAVE_IP="10.0.1.2"
MASTER_PORT=29655                            # NCCL bootstrap; control channel = +1
SLAVE_SSH="10.0.1.2"                         # ssh target for the slave box
# draft-mtp with a LOW p-min: the draft always proposes the full window so the verify batch is a fixed
# width every round. A high p-min (e.g. 0.75) makes the draft length vary with confidence, which makes
# the meta/CUDA graph re-capture every round (graphs reused = 0) and is a net slowdown — measured 6.5 t/s
# at p-min 0.75 vs 10.3 t/s at p-min 0.0 on DSV4 Q4 2-box. 0.0 also matches the model's standard sampling.
SPEC="--spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-p-min 0.0"
# MTP (≤64k): SPEC="--spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-p-min 0.0"; set GRAPH_SLOTS=16 VERIFY_REUSE=1 for graph reuse (disabled-default)
SELF="$REPO/tp-serve/tp.sh"                  # path of this script on each box
# ----------------------------------------------------------------------------

COMMON="-c $CTX --parallel 1 -ngl 999 -fa on -sm tensor -fit off --no-warmup --no-mmap -ctk tbq3 -ctv tbq3 $SPEC"

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
    if [ -n "${DSV4_MTP_PROF:-}" ]; then export DSV4_MTP_PROF; fi   # DIAG: build+alloc per step (set-e safe)
    export DSV4_GRAPH_SLOTS="${GRAPH_SLOTS:-1}"   # MTP graph-reuse slot pool (both ranks must match)
    if [ -n "${SLOT_PROBE:-}" ]; then export DSV4_SLOT_PROBE=1; fi
    if [ -n "${RESULT_DEBUG:-}" ]; then export LLAMA_GRAPH_RESULT_DEBUG="$RESULT_DEBUG"; fi
    if [ -n "${VERIFY_REUSE:-}" ]; then export DSV4_VERIFY_REUSE=1; fi
    if [ -n "${KERNEL_PROF:-}" ]; then export DSV4_KERNEL_PROF=1; fi
    if [ -n "${GRAPH_PROBE:-}" ]; then export DSV4_GRAPH_PROBE=1; fi
    if [ -n "${STAGE_PROF:-}" ]; then export DSV4_STAGE_PROF=1; fi
}

stop_local() {
    # `ps -C` exits 1 when nothing matches; under `set -e` + pipefail a bare assignment from it would
    # abort the script, so swallow the status (|| true) — empty output then means "nothing running".
    local pids; pids="$(ps -C llama-server -o pid= | tr -d ' ' || true)"
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
    local pids; pids="$(ps -C llama-server -o pid=,etime= | tr '\n' ';' || true)"  # || true: see stop_local
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
        # don't let a slave-side failure abort under set -e before the master is started — warn and go on
        ssh "$SLAVE_SSH" "$SELF START" || echo "[WARN] slave START failed ($SLAVE_SSH) — starting master anyway"
        sleep 3
        start_local ;;
    *)
        echo "DSV4 TP+MTP control (role: $ROLE)"
        echo "usage: $(basename "$0") START | STOP | RESTART | ALLRESTART | STATUS"
        exit 1 ;;
esac
