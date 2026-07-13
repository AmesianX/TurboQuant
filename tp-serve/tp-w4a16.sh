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
MODEL="${MODEL:-$HOME/Models/DeepSeek-V4-Flash-GGUF/FP4/DeepSeek-V4-Flash-FP4-FP8-native-MTP.gguf}"  # FP4 (production default); env-overridable (Q4mtp 등)
PORT="${PORT:-8080}"
API_KEY="tbq-dsv4"                           # required because we bind 0.0.0.0
CTX="${CTX:-524288}" # 512K (agent daily-driver). 0=full 1M (brag, slow prefill). KV~4.8GB@tbq3.
                # starving working-memory headroom — a single ~27K prefill then crosses the watchdog and dies at
                # progress ~0.96. For real single-prompt use, set CTX=262144 (256K → KV ~2.4GB, +7GB headroom,
                # lets -ub go to 2048 for fast prefill). 1M is the demo/brag number, not a daily-driver value.
UB="${UB:-256}" # DSV4 ceiling: compressor builds ~O(n^2) graph objects, UB>256 OOMs the arena. Slow prefill is inherent.
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
# MTP only. (Tried draft-mtp,ngram-cache,ngram-map-k4v cascade 2026-06-26: 2.4 t/s DISASTER on DSV4 —
# n-gram drafts are variable-width (breaks MTP verify graph reuse -> graphs reused 5 vs 100s) AND a
# rejected n-gram draft rolls back via CHECKPOINT (~2s, not the cheap plane rollback MTP uses, since
# ngram isn't in need_n_rs_seq). Structural incompat with DSV4's fixed-width/graph-reuse MTP pipeline.)
SPEC="${SPEC---spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-p-min 0.0}"  # env-overridable: SPEC="" disables spec
# single MTP only: SPEC="--spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-p-min 0.0"; GRAPH_SLOTS=16 VERIFY_REUSE=1 for graph reuse
SELF="$REPO/tp-serve/tp-w4a16.sh"                  # path of this script on each box
# ----------------------------------------------------------------------------

# Anti-repetition default sampling. DSV4-Flash tends to loop/re-emit whole sections without hitting
# EOS (verbose "탁록전쟁"-style restatement). DRY penalizes repeated token *sequences* (best for that),
# repeat-penalty handles shorter loops. Env-overridable; a per-request body still overrides these.
SAMPLING="${SAMPLING:---repeat-penalty 1.1 --repeat-last-n 1024 --dry-multiplier 0.8 --dry-base 1.75 --dry-allowed-length 2 --dry-penalty-last-n 32768}"

# Deterministic decode for reproducible A/B measurement: temp 0 = greedy (no RNG), fixed seed.
# Both must be server-level so EVERY request is bit-identical across runs -> the only variable left
# is what we're testing (CPU vs GPU sampler etc.). Env-overridable (TEMP=0.7 SEED=... for real serving).
TEMP="${TEMP:-0}"
SEED="${SEED:-42}"

# -b (logical batch) must be >= -ub (micro-batch). Derive it so raising UB never violates that.
BB="$UB"; [ "$BB" -lt 512 ] && BB=512
COMMON="-c $CTX -n 32768 --parallel $PARALLEL -b $BB -ub $UB -ngl 999 -fa on -sm tensor -fit off --no-warmup --no-mmap -ctk ${CTK:-tbq3} -ctv ${CTV:-tbq3} --cache-ram $CACHE_RAM --jinja --chat-template-file $REPO/models/templates/deepseek-ai-DeepSeek-V4.jinja --reasoning-format deepseek --reasoning off --lock-server-params --temp $TEMP --seed $SEED $SPEC $SAMPLING ${EXTRA_ARGS:-}"

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
    export NCCL_SOCKET_IFNAME="$IFACE"          # TCP bootstrap rendezvous only
    # [RDMA] data plane over RoCE. 4x ConnectX-7 200G rails, RoCE v2 (GID 3).
    #
    # DISCOVER the HCAs, never hardcode them: this list used to read rocep1s0f0,... and a driver
    # update renamed the devices to mlx5_*. NCCL matched nothing, said "NET/IB : No device found",
    # and silently fell back to "Using network Socket" -- every AllReduce went over TCP. A 16 KB
    # reduce then cost 0.63 ms instead of tens of us, which is 40% of decode (measured: skipping
    # the reduces takes plain decode 10.7 -> 15.0 t/s). Verify with NCCL_DEBUG=INFO after any
    # driver change: the log MUST say "Using network IB", not Socket.
    # ONE rail by default: measured 1 = 2 = 4 rails, and four rails OOM the box at CTX=262144 (NCCL
    # registers per-rail buffers in the unified memory the model already fills). Override by hand
    # (NCCL_IB_HCA=mlx5_0,mlx5_1,...) only for a genuinely wire-bandwidth-bound workload.
    if [ -z "${NCCL_IB_HCA:-}" ]; then
        _hca="$(ls /sys/class/infiniband 2>/dev/null | head -1)"
        [ -n "$_hca" ] && export NCCL_IB_HCA="$_hca"
    fi
    export NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"
    export NCCL_IB_DISABLE=0
    export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-SYS}"
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
    # [GPU offload — kill CPU fallback under -sm tensor] penalties+DRY sampling + token_embd on GPU.
    # DSV4_GPU_SAMPLER=1 (default): sampler chain runs on the meta/CUDA backend (no CPU sampler split).
    # NOTE: GPU vs CPU sampler is ~parity on the real config (clean temp0/seed42 A/B still TODO); set
    # DSV4_GPU_SAMPLER=0 to revert to the upstream CPU sampler. DSV4_GPU_INPUT_EMBD=1: token_embd on GPU.
    export DSV4_GPU_SAMPLER="${DSV4_GPU_SAMPLER:-1}"
    export DSV4_GPU_INPUT_EMBD="${DSV4_GPU_INPUT_EMBD:-1}"
    # NVFP4 grouped-MoE op + sidecar weights + HP-activation threshold. Both ranks (SPMD)
    # MUST run identically; forwarded to the slave via the ALLRESTART FWD line below.
    [ -n "${DSV4_MOE_GROUPED:-}" ]    && export DSV4_MOE_GROUPED
    [ -n "${DSV4_MOE_FUSED:-}" ]      && export DSV4_MOE_FUSED
    [ -n "${DSV4_MOE_FUSED_TACTIC:-}" ]   && export DSV4_MOE_FUSED_TACTIC
    [ -n "${DSV4_MOE_FUSED_GRAPH_OFF:-}" ]&& export DSV4_MOE_FUSED_GRAPH_OFF
    [ -n "${DSV4_MOE_SIDECAR:-}" ]    && export DSV4_MOE_SIDECAR
    [ -n "${DSV4_MOE_DECODE_MAX:-}" ] && export DSV4_MOE_DECODE_MAX
    [ -n "${DSV4_MOE_W4A16_DECODE:-}" ] && export DSV4_MOE_W4A16_DECODE
    [ -n "${DSV4_EP_DECODE_GRAPH:-}" ] && export DSV4_EP_DECODE_GRAPH
    [ -n "${DSV4_LM_HEAD_F8:-}" ] && export DSV4_LM_HEAD_F8
    [ -n "${DSV4_ATTN_SPLIT:-}" ] && export DSV4_ATTN_SPLIT
    [ -n "${DSV4_MOE_FUSED_PROF:-}" ] && export DSV4_MOE_FUSED_PROF
    [ -n "${DSV4_MOE_FUSED_PROF_AFTER:-}" ] && export DSV4_MOE_FUSED_PROF_AFTER
    [ -n "${DSV4_PREFILL_VRAM_PROBE:-}" ] && export DSV4_PREFILL_VRAM_PROBE
    [ -n "${DSV4_MOE_FC1_FUSED:-}" ] && export DSV4_MOE_FC1_FUSED
    # [ep2-dp] Expert-parallel: split routed experts on the expert dim (axis2), 128/rank, so each box
    # loads HALF the MoE weight (memory headroom for large -ub). Runs the GENERIC mul_mat_id path with
    # the per-rank expert_offset -> NOT the sidecar custom op (leave DSV4_MOE_FUSED/SIDECAR unset for EP).
    # SPMD: both ranks MUST match -> forwarded to the slave on the ALLRESTART FWD line below.
    [ -n "${DSV4_EP:-}" ]             && export DSV4_EP
    [ -n "${DSV4_EP_DBG:-}" ]         && export DSV4_EP_DBG
    # DSV4 sparse-attention (per-query top-k comp-row gather). SPMD: both ranks build the SAME graph
    # shape (sparse skips the -inf comp mask concat + binds src[6]), so this MUST match on both ranks.
    [ -n "${DSV4_SPARSE_ATTN:-}" ]    && export DSV4_SPARSE_ATTN
    # capture-safe prefill arena cap + prefill-graph safety hatch (both ranks identical)
    [ -n "${DSV4_MOE_PREFILL_MAX:-}" ]       && export DSV4_MOE_PREFILL_MAX
    [ -n "${DSV4_MOE_PREFILL_GRAPH_OFF:-}" ] && export DSV4_MOE_PREFILL_GRAPH_OFF
    # DSA indexer query-tiling (O(ub^2) -> O(tile^2)) to unblock large UB. SPMD: both ranks MUST match
    # (same graph shape), so forward to the slave or it OOMs at large UB on the whole-ub indexer.
    [ -n "${DSV4_INDEXER_QTILE:-}" ]         && export DSV4_INDEXER_QTILE
    [ -n "${DSV4_ATTN_QTILE:-}" ]            && export DSV4_ATTN_QTILE
    [ -n "${DSV4_INDEXER_FUSED:-}" ]         && export DSV4_INDEXER_FUSED
    [ -n "${DSV4_HC_BF16:-}" ]               && export DSV4_HC_BF16
    [ -n "${DSV4_MLA_MMA:-}" ]               && export DSV4_MLA_MMA
    [ -n "${DSV4_COMPRESSOR_BF16:-}" ]       && export DSV4_COMPRESSOR_BF16
    [ -n "${DSV4_INDEXER_BF16:-}" ]          && export DSV4_INDEXER_BF16
    [ -n "${DSV4_RESUMED_RESERVE_MULT:-}" ]  && export DSV4_RESUMED_RESERVE_MULT
    [ -n "${NCCL_DEBUG:-}" ]          && export NCCL_DEBUG
    [ -n "${NCCL_IB_GID_INDEX:-}" ]   && export NCCL_IB_GID_INDEX
    # prefill graph-build profiling (off by default; both ranks identical). Used to characterize the
    # prefill bottleneck: per-chunk graph BUILD is ~2 ms, the dominant per-chunk cost is GPU compute +
    # ggml_backend_sched_alloc_graph (split/buffer scheduling, grows with the compressed-cache view).
    [ -n "${DSV4_MTP_PROF:-}" ]       && export DSV4_MTP_PROF
    [ -n "${DSV4_PREFILL_PROF:-}" ]   && export DSV4_PREFILL_PROF
    [ -n "${DSV4_KERNEL_PROF:-}" ]    && export DSV4_KERNEL_PROF
    # native sm120 FP8 (F8_E4M3_B128) tensor-core GEMM for the dense MLA/shexp projections.
    # Default OFF = byte-identical (dequant->F16 cuBLAS). SPMD: both ranks must match.
    [ -n "${DSV4_FP8_NATIVE:-}" ]     && export DSV4_FP8_NATIVE
    return 0   # never let a skipped optional-export ([ -n "" ] -> 1) fail env_common under set -e
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
            nohup $CGRUN ${MASTER_WRAP:-} build/bin/llama-server -m "$MODEL" $COMMON \
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
        # forward NVFP4 grouped-MoE env so the slave runs the identical SPMD config
        [ -n "${DSV4_MOE_GROUPED:-}" ]    && FWD="$FWD DSV4_MOE_GROUPED=$DSV4_MOE_GROUPED"
        [ -n "${DSV4_MOE_FUSED:-}" ]      && FWD="$FWD DSV4_MOE_FUSED=$DSV4_MOE_FUSED"
        [ -n "${DSV4_MOE_FUSED_TACTIC:-}" ]   && FWD="$FWD DSV4_MOE_FUSED_TACTIC=$DSV4_MOE_FUSED_TACTIC"
        [ -n "${DSV4_MOE_FUSED_GRAPH_OFF:-}" ]&& FWD="$FWD DSV4_MOE_FUSED_GRAPH_OFF=$DSV4_MOE_FUSED_GRAPH_OFF"
        [ -n "${DSV4_MOE_SIDECAR:-}" ]    && FWD="$FWD DSV4_MOE_SIDECAR=$DSV4_MOE_SIDECAR"
        [ -n "${DSV4_EP:-}" ]             && FWD="$FWD DSV4_EP=$DSV4_EP"
        [ -n "${DSV4_EP_DBG:-}" ]         && FWD="$FWD DSV4_EP_DBG=$DSV4_EP_DBG"
        [ -n "${DSV4_SPARSE_ATTN:-}" ]    && FWD="$FWD DSV4_SPARSE_ATTN=$DSV4_SPARSE_ATTN"
        [ -n "${DSV4_MOE_DECODE_MAX:-}" ] && FWD="$FWD DSV4_MOE_DECODE_MAX=$DSV4_MOE_DECODE_MAX"
        [ -n "${DSV4_MOE_W4A16_DECODE:-}" ] && FWD="$FWD DSV4_MOE_W4A16_DECODE=$DSV4_MOE_W4A16_DECODE"
        [ -n "${DSV4_EP_DECODE_GRAPH:-}" ] && FWD="$FWD DSV4_EP_DECODE_GRAPH=$DSV4_EP_DECODE_GRAPH"
        [ -n "${DSV4_LM_HEAD_F8:-}" ] && FWD="$FWD DSV4_LM_HEAD_F8=$DSV4_LM_HEAD_F8"
        [ -n "${DSV4_ATTN_SPLIT:-}" ] && FWD="$FWD DSV4_ATTN_SPLIT=$DSV4_ATTN_SPLIT"
        [ -n "${DSV4_MOE_FUSED_PROF:-}" ] && FWD="$FWD DSV4_MOE_FUSED_PROF=$DSV4_MOE_FUSED_PROF DSV4_MOE_FUSED_PROF_AFTER=${DSV4_MOE_FUSED_PROF_AFTER:-400}"
        [ -n "${DSV4_PREFILL_VRAM_PROBE:-}" ] && FWD="$FWD DSV4_PREFILL_VRAM_PROBE=$DSV4_PREFILL_VRAM_PROBE"
        [ -n "${DSV4_MOE_FC1_FUSED:-}" ] && FWD="$FWD DSV4_MOE_FC1_FUSED=$DSV4_MOE_FC1_FUSED"
        [ -n "${DSV4_MOE_PREFILL_MAX:-}" ]       && FWD="$FWD DSV4_MOE_PREFILL_MAX=$DSV4_MOE_PREFILL_MAX"
        [ -n "${DSV4_MOE_PREFILL_GRAPH_OFF:-}" ] && FWD="$FWD DSV4_MOE_PREFILL_GRAPH_OFF=$DSV4_MOE_PREFILL_GRAPH_OFF"
        [ -n "${DSV4_INDEXER_QTILE:-}" ]         && FWD="$FWD DSV4_INDEXER_QTILE=$DSV4_INDEXER_QTILE"
        [ -n "${DSV4_ATTN_QTILE:-}" ]            && FWD="$FWD DSV4_ATTN_QTILE=$DSV4_ATTN_QTILE"
        [ -n "${DSV4_INDEXER_FUSED:-}" ]         && FWD="$FWD DSV4_INDEXER_FUSED=$DSV4_INDEXER_FUSED"
        [ -n "${DSV4_HC_BF16:-}" ]               && FWD="$FWD DSV4_HC_BF16=$DSV4_HC_BF16 DSV4_MLA_MMA=${DSV4_MLA_MMA:-} CTK=${CTK:-tbq3} CTV=${CTV:-tbq3}"
        [ -n "${DSV4_COMPRESSOR_BF16:-}" ]       && FWD="$FWD DSV4_COMPRESSOR_BF16=$DSV4_COMPRESSOR_BF16"
        [ -n "${DSV4_INDEXER_BF16:-}" ]          && FWD="$FWD DSV4_INDEXER_BF16=$DSV4_INDEXER_BF16"
        [ -n "${DSV4_RESUMED_RESERVE_MULT:-}" ]  && FWD="$FWD DSV4_RESUMED_RESERVE_MULT=$DSV4_RESUMED_RESERVE_MULT"
        [ -n "${DSV4_MOE_GRAPH_OFF:-}" ]  && FWD="$FWD DSV4_MOE_GRAPH_OFF=$DSV4_MOE_GRAPH_OFF"
        [ -n "${DSV4_GRAPH_PROBE:-}" ]    && FWD="$FWD DSV4_GRAPH_PROBE=$DSV4_GRAPH_PROBE"
        [ -n "${DSV4_MTP_PROF:-}" ]       && FWD="$FWD DSV4_MTP_PROF=$DSV4_MTP_PROF"
        [ -n "${DSV4_PREFILL_PROF:-}" ]   && FWD="$FWD DSV4_PREFILL_PROF=$DSV4_PREFILL_PROF"
        [ -n "${DSV4_KERNEL_PROF:-}" ]    && FWD="$FWD DSV4_KERNEL_PROF=$DSV4_KERNEL_PROF"
        [ -n "${DSV4_FP8_NATIVE:-}" ]     && FWD="$FWD DSV4_FP8_NATIVE=$DSV4_FP8_NATIVE"
        [ -n "${NCCL_DEBUG:-}" ]          && FWD="$FWD NCCL_DEBUG=$NCCL_DEBUG"
        [ -n "${NCCL_IB_GID_INDEX:-}" ]   && FWD="$FWD NCCL_IB_GID_INDEX=$NCCL_IB_GID_INDEX"
        # NCCL topology/tuning must be SYMMETRIC across ranks. Overriding these on the master only
        # makes the ranks disagree on the rail count and the bootstrap exchange mismatches:
        # "Message truncated : received 32768 bytes instead of 1024" -> NCCL internal error.
        [ -n "${NCCL_IB_HCA:-}" ]         && FWD="$FWD NCCL_IB_HCA=$NCCL_IB_HCA"
        [ -n "${NCCL_MIN_NCHANNELS:-}" ]  && FWD="$FWD NCCL_MIN_NCHANNELS=$NCCL_MIN_NCHANNELS"
        [ -n "${NCCL_MAX_NCHANNELS:-}" ]  && FWD="$FWD NCCL_MAX_NCHANNELS=$NCCL_MAX_NCHANNELS"
        [ -n "${NCCL_NET_GDR_LEVEL:-}" ]  && FWD="$FWD NCCL_NET_GDR_LEVEL=$NCCL_NET_GDR_LEVEL"
        [ -n "${NCCL_BUFFSIZE:-}" ]       && FWD="$FWD NCCL_BUFFSIZE=$NCCL_BUFFSIZE"
        # Changes the K>1 verify graph SHAPE -- if only one rank takes the uniform path the two
        # build different graphs and the SPMD control stream mismatches.
        [ -n "${VERIFY_REUSE:-}" ]        && FWD="$FWD VERIFY_REUSE=$VERIFY_REUSE"
        # BOTH ranks MUST agree on step-graph capture or their NCCL calls desynchronize -> hang.
        [ -n "${DSV4_STEP_GRAPH:-}" ]     && FWD="$FWD DSV4_STEP_GRAPH=$DSV4_STEP_GRAPH"
        [ -n "${DSV4_STEP_GRAPH_SLOTS:-}" ] && FWD="$FWD DSV4_STEP_GRAPH_SLOTS=$DSV4_STEP_GRAPH_SLOTS"
        [ -n "${DSV4_STEP_GRAPH_STATS:-}" ] && FWD="$FWD DSV4_STEP_GRAPH_STATS=$DSV4_STEP_GRAPH_STATS"
        # Same rule, harder: if only one rank skips the reduce, the other blocks in NCCL forever.
        [ -n "${DSV4_TP_NO_REDUCE:-}" ]   && FWD="$FWD DSV4_TP_NO_REDUCE=$DSV4_TP_NO_REDUCE"
        # OPPROF needs no-graphs; BOTH ranks MUST match graph mode or the SPMD control stream mismatches (crash).
        [ -n "${DSV4_OPPROF:-}" ]         && FWD="$FWD DSV4_OPPROF=$DSV4_OPPROF DSV4_OPPROF_TOP=${DSV4_OPPROF_TOP:-60}"
        [ -n "${GGML_CUDA_NO_GRAPHS:-}" ] && FWD="$FWD GGML_CUDA_NO_GRAPHS=$GGML_CUDA_NO_GRAPHS"
        # don't let a slave-side failure abort under set -e before the master is started — warn and go on
        ssh "$SLAVE_SSH" "$FWD $SELF START" || echo "[WARN] slave START failed ($SLAVE_SSH) — starting master anyway"
        sleep 3
        start_local ;;
    *)
        echo "DSV4 TP+MTP control (role: $ROLE)"
        echo "usage: $(basename "$0") START | STOP | RESTART | ALLRESTART | STATUS"
        exit 1 ;;
esac
