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
MODEL="${MODEL:-$HOME/Models/DeepSeek-V4-Flash-GGUF/Q4mtp/DSV4-Q4-00001-of-00002.gguf}"  # env-overridable (FP4 등)
PORT=8080
API_KEY="tbq-dsv4"                           # required because we bind 0.0.0.0
CTX="${CTX:-0}" # 0 = full native context (1M for DSV4, KV ~9.6GB). BUT 1M reserves the full 9.6GB KV up front,
                # starving working-memory headroom — a single ~27K prefill then crosses the watchdog and dies at
                # progress ~0.96. For real single-prompt use, set CTX=262144 (256K → KV ~2.4GB, +7GB headroom,
                # lets -ub go to 2048 for fast prefill). 1M is the demo/brag number, not a daily-driver value.
UB="${UB:-256}" # prefill micro-batch. 256 = memory-safe but slow prefill (many cross-node NCCL syncs at -sm tensor).
                # Bigger = faster prefill, more memory. CEILING 2048 (user: >2048 explodes memory). Raise only with
                # freed headroom (smaller CTX). Both ranks MUST match (graph shape) — forwarded via ALLRESTART FWD.
PARALLEL="${PARALLEL:-2}"  # server slots (--parallel). 1 = single stream. 2 = multi-slot concurrent serving
                # (PARALLEL>1 auto-sets DSV4_MULTISLOT=1 below — the batched-decode path; without it 2 slots
                # just contend and run SLOWER than single). MEASURED @1M Q4: single ~16 t/s, multi-slot 2-concurrent
                # ~22.8 t/s aggregate (~1.4x), min box-free ~7GB (safe). GS must stay 2 at 1M (see GRAPH_SLOTS).
GRAPH_SLOTS="${GRAPH_SLOTS:-2}"   # MTP graph-reuse pool. @1M MUST be 2: GS=4 multi-slot peaks at ~2.5GB box-free
                # and a real conversation pushed it to 1GB -> watchdog killed it. GS=2 holds ~7GB free, same speed
                # (GS=4 only +4% t/s). >64k contexts can raise this; at 1M, 2 is the ceiling. Old pool (GPU compute buffers, ~2GB/slot measured @ -ub 256). MEASURED box-free
                # left after a 6-question session: 2 slots=>~14GB, 4=>~10GB, 6=>~7GB, 16=>OOM. 4 is the balance
                # (more reuse cache, still ~10GB safety). These buffers live on the UNIFIED GPU pool — a cgroup
                # (host-only) can't see them, so the watchdog below (box-wide free mem) is the real OOM guard.
WATCH_MIN_GB="${WATCH_MIN_GB:-4}"  # memory watchdog: if box MemAvailable drops below this, kill ONLY llama-server (never Claude Code/ssh).
VERIFY_REUSE=1  # let DSV4 verify graphs qualify for reuse (pairs with the slot pool)
# --- memory protection (box = 124546 MiB; ~16GB box-free at idle after model+KV) ---------
#   The GB-scale OOM driver is the graph-slot pool on the GPU (~2GB/slot). It killed Claude Code + ssh via the
#   kernel's indiscriminate GLOBAL OOM-killer. The WATCH_MIN_GB watchdog (above) is the real guard — it watches
#   box-wide free mem (GPU + host) and kills only the server before the global OOM fires.
MEM_CAP="110G"  # cgroup v2 ceiling — HOST-side only, a secondary backstop. NOTE: on GB10 unified memory the
                # GPU allocations (model + graph compute buffers) are NOT charged to the cgroup, so this does
                # NOT bound the real OOM driver — it only catches a host-side runaway. The watchdog does the rest.
CACHE_RAM=2048  # prompt-cache byte limit (MiB) = 2GB share of the dynamic budget. (--cache-ram)
IFACE="enp1s0f0np0"                          # RoCE interface
MASTER_IP="10.0.1.1"
SLAVE_IP="10.0.1.2"
MASTER_PORT=29655                            # NCCL bootstrap; control channel = +1
SLAVE_SSH="10.0.1.2"                         # ssh target for the slave box
# draft-mtp with a LOW p-min: the draft always proposes the full window so the verify batch is a fixed
# width every round. A high p-min (e.g. 0.75) makes the draft length vary with confidence, which makes
# the meta/CUDA graph re-capture every round (graphs reused = 0) and is a net slowdown — measured 6.5 t/s
# at p-min 0.75 vs 10.3 t/s at p-min 0.0 on DSV4 Q4 2-box. 0.0 also matches the model's standard sampling.
SPEC="${SPEC---spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-p-min 0.0}"  # env-overridable: SPEC="" disables MTP
# MTP (≤64k): SPEC="--spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-p-min 0.0"; set GRAPH_SLOTS=16 VERIFY_REUSE=1 for graph reuse (disabled-default)
SELF="$REPO/tp-serve/tp.sh"                  # path of this script on each box
# ----------------------------------------------------------------------------

COMMON="-c $CTX -n 32768 --parallel $PARALLEL -b 512 -ub $UB -ngl 999 -fa on -sm tensor -fit off --no-warmup --no-mmap -ctk tbq3 -ctv tbq3 --cache-ram $CACHE_RAM --jinja --reasoning-format deepseek --reasoning off $SPEC ${EXTRA_ARGS:-}"

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
    export DSV4_GRAPH_SLOTS="${GRAPH_SLOTS:-1}"          # MTP graph-reuse slot pool (both ranks must match)
    if [ -n "${VERIFY_REUSE:-}" ]; then export DSV4_VERIFY_REUSE=1; fi
    # DSV4-specific multi-slot batched decode (splitter emits n_seqs>1 ubatches, batched over MoE/dense/norm/lm_head).
    # WITHOUT this, --parallel>1 just runs naive contending slots (slower than single). THIS env is what gives the
    # ~2x aggregate. Gated on PARALLEL>1 so single-stream serving is unaffected. Both ranks must match (SPMD).
    if [ "${PARALLEL:-1}" -gt 1 ]; then export DSV4_MULTISLOT=1; fi
    # O(1) batched chunk compressor (long multi-turn "괭" crash fix) is now DEFAULT-ON in the binary
    # — no env needed. Set DSV4_DISABLE_BATCHED_COMPRESSOR=1 only to force the old unrolled path for debug.
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
    # stop the memory watchdog sidecar too
    [ -f /tmp/tp_memwatch.pid ] && { kill "$(cat /tmp/tp_memwatch.pid)" 2>/dev/null || true; rm -f /tmp/tp_memwatch.pid; }
    echo "[$ROLE] STOP: killed [$pids]"
}

start_local() {
    cd "$REPO"
    if [ -n "$(ps -C llama-server -o pid=)" ]; then
        echo "[$ROLE] START: already running (use RESTART)"; return 0
    fi
    env_common
    # memory-capped cgroup v2 scope (no sudo). An over-budget OOM then kills ONLY this scope (llama-server),
    # never the rest of the box (Claude Code / ssh / OS). Skip gracefully if the user systemd manager isn't
    # reachable (e.g. a non-login ssh into the slave) — then launch plain (slave has no Claude Code to protect).
    local CGRUN=""
    if systemd-run --user --quiet --scope -p MemoryMax="$MEM_CAP" -p MemorySwapMax=0 -- true 2>/dev/null; then
        CGRUN="systemd-run --user --quiet --scope --collect -p MemoryMax=$MEM_CAP -p MemorySwapMax=0"
        echo "[$ROLE] cgroup cap ACTIVE: MemoryMax=$MEM_CAP MemorySwapMax=0"
    else
        echo "[$ROLE] WARN: systemd-run --user unavailable — launching WITHOUT cgroup cap"
    fi
    case "$ROLE" in
        MASTER)
            export GGML_TP_RANK=0
            nohup $CGRUN build/bin/llama-server -m "$MODEL" $COMMON \
                --host 0.0.0.0 --port "$PORT" --api-key "$API_KEY" > "$LOG" 2>&1 &
            disown
            echo "[MASTER] START: serving http://0.0.0.0:$PORT (api-key $API_KEY), log $LOG" ;;
        SLAVE)
            export GGML_TP_RANK=1
            nohup $CGRUN build/bin/llama-server -m "$MODEL" $COMMON \
                --host 0.0.0.0 --port "$PORT" > "$LOG" 2>&1 &
            disown
            echo "[SLAVE] START: follower up (no HTTP), log $LOG" ;;
        *)
            echo "ERROR: role UNKNOWN — local $IFACE is neither $MASTER_IP nor $SLAVE_IP"; exit 1 ;;
    esac

    # memory watchdog sidecar: the big dynamic memory (graph-slot GPU compute buffers) lives on the
    # unified GPU pool, which a cgroup can't see — so we guard the BOX-wide free memory instead. If it
    # drops below WATCH_MIN_GB we kill ONLY llama-server (controlled), pre-empting the kernel's global
    # OOM-killer which is indiscriminate (it took down Claude Code + ssh). Server is restartable; the box
    # (and Claude Code) survives.
    [ -f /tmp/tp_memwatch.pid ] && kill "$(cat /tmp/tp_memwatch.pid)" 2>/dev/null
    nohup bash -c '
        min='"$WATCH_MIN_GB"'
        while true; do
            avail=$(awk "/MemAvailable/{print int(\$2/1024/1024)}" /proc/meminfo)
            if [ "${avail:-99}" -lt "$min" ]; then
                echo "$(date +%T) MemAvailable ${avail}GB < ${min}GB -> killing llama-server (OOM pre-empt)"
                for p in $(ps -C llama-server -o pid=); do kill -9 "$p" 2>/dev/null; done
                sleep 3
            fi
            sleep 0.2
        done' > /tmp/tp_memwatch.log 2>&1 &
    echo $! > /tmp/tp_memwatch.pid
    disown
    echo "[$ROLE] memwatch: guarding box MemAvailable >= ${WATCH_MIN_GB}GB (kills server only, protects Claude Code)"
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
        # forward the slot/graph overrides so BOTH ranks build matching graphs (TP requires it).
        FWD="PARALLEL=$PARALLEL GRAPH_SLOTS=$GRAPH_SLOTS CTX=$CTX UB=$UB MODEL=$MODEL SPEC=\"$SPEC\""
        # don't let a slave-side failure abort under set -e before the master is started — warn and go on
        ssh "$SLAVE_SSH" "$FWD $SELF START" || echo "[WARN] slave START failed ($SLAVE_SSH) — starting master anyway"
        sleep 3
        start_local ;;
    *)
        echo "DSV4 TP+MTP control (role: $ROLE)"
        echo "usage: $(basename "$0") START | STOP | RESTART | ALLRESTART | STATUS"
        exit 1 ;;
esac
