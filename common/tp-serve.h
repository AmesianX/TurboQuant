// [tp-2node-dsv4] SPMD serving control plane for cross-node tensor parallelism.
//
// In SPMD TP the heavy compute is split across ranks (each computes its weight slice,
// joined by per-layer NCCL AllReduce). But a *server* is request-driven: only rank 0 has the
// HTTP endpoint / tokenizer / sampler and thus knows which batch to decode. The other ranks
// have no driver. This header adds a tiny control plane so rank>0 mirrors rank 0's context
// mutations (decode + KV-cache ops) exactly, keeping the replicated (MIRRORED) KV caches in
// lockstep so the NCCL collectives line up. Heavy tensors are NOT sent here -- only small op
// records (token ids, positions, seq ids). This is the vLLM/Megatron driver-broadcast pattern.
//
// Channel: a single TCP stream. rank 0 (leader, GGML_TP_MASTER_ADDR) listens on
// GGML_TP_MASTER_PORT+1; rank>0 (followers) connect. Leader sends op records; followers replay.

#pragma once

#include "llama.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <map>
#include <atomic>
#include <mutex>

#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>

namespace tpserve {

enum op_type : int32_t {
    TP_OP_DECODE        = 1,
    TP_OP_SEQ_RM        = 2,
    TP_OP_SEQ_ADD       = 3,
    TP_OP_SEQ_CP        = 4,
    TP_OP_SEQ_KEEP      = 5,
    TP_OP_CLEAR         = 6,
    TP_OP_STATE_SAVE    = 7, // mirror a KV snapshot (checkpoint / prompt-cache save)
    TP_OP_STATE_RESTORE = 8, // restore a previously mirrored KV snapshot
    TP_OP_STOP          = 99,
};

// Monotonic key naming each mirrored KV snapshot. The leader assigns it at save time and replays
// it at restore time; the follower keeps a parallel store keyed by it. Data is NEVER sent (the
// follower snapshots its own mirrored KV) -- only the small key/seq/flags. [tp-2node-dsv4]
inline uint64_t tp_next_state_key() { static std::atomic<uint64_t> k{1}; return k.fetch_add(1); }

inline int  tp_nranks() { static const int v = getenv("GGML_TP_NRANKS") ? atoi(getenv("GGML_TP_NRANKS")) : 1; return v; }
inline int  tp_rank()   { static const int v = getenv("GGML_TP_RANK")   ? atoi(getenv("GGML_TP_RANK"))   : 0; return v; }
inline bool tp_enabled(){ return tp_nranks() > 1; }
inline bool tp_is_leader()   { return tp_enabled() && tp_rank() == 0; }
inline bool tp_is_follower() { return tp_enabled() && tp_rank() != 0; }

// The single control socket (leader: one accepted fd; follower: the connected fd).
inline int & tp_fd() { static int fd = -1; return fd; }

// Serializes the multi-write op records on the leader: the server broadcasts from several threads
// (HTTP handlers + the main decode loop), and interleaved tp_send_all sequences would garble the
// stream and desync the follower. Each tp_bcast_* takes this lock for its whole record. [tp-2node-dsv4]
inline std::mutex & tp_send_mtx() { static std::mutex m; return m; }

// Registry of the contexts that participate in TP. In the current MTP design only ctx_tgt is
// registered (id 0); the NextN draft ctx_dft is intentionally NOT registered (direction B — it runs
// mirrored/solo on rank 0, see server-context.cpp). Both leader and follower register the SAME
// contexts in the SAME order during load. Every op record carries this id so the follower replays it
// on the matching context. [tp-2node-dsv4]
inline std::vector<llama_context *> & tp_ctxs() { static std::vector<llama_context *> v; return v; }
inline int tp_register_ctx(llama_context * ctx) {
    // Idempotent: load_model() re-runs on resume-from-sleep, so a plain push_back would re-register the
    // same context every wake — growing the registry unbounded and shifting the leader/follower id map.
    // Return the existing id if already registered; only a genuinely new context is appended.
    auto & v = tp_ctxs();
    for (size_t i = 0; i < v.size(); i++) { if (v[i] == ctx) { return (int) i; } }
    v.push_back(ctx);
    return (int) v.size() - 1;
}
inline int tp_ctx_id(const llama_context * ctx) {
    auto & v = tp_ctxs();
    for (size_t i = 0; i < v.size(); i++) { if (v[i] == ctx) { return (int) i; } }
    return -1; // not a TP context (e.g. warmup before registration) -> don't broadcast
}

inline void tp_send_all(int fd, const void * buf, size_t n) {
    const char * p = (const char *) buf;
    while (n > 0) {
        ssize_t k = ::send(fd, p, n, 0);
        if (k <= 0) { return; } // broken pipe: leave; caller will see follower die
        p += k; n -= (size_t) k;
    }
}
inline bool tp_recv_all(int fd, void * buf, size_t n) {
    char * p = (char *) buf;
    while (n > 0) {
        ssize_t k = ::recv(fd, p, n, 0);
        if (k <= 0) { return false; }
        p += k; n -= (size_t) k;
    }
    return true;
}

// ---- connection setup ---------------------------------------------------------------------

// Leader: bind+listen+accept ONE follower (2-node). Blocks until the follower connects.
inline void tp_leader_accept() {
    const char * pe = getenv("GGML_TP_MASTER_PORT");
    const int port = (pe ? atoi(pe) : 29600) + 1;
    int lfd = ::socket(AF_INET, SOCK_STREAM, 0);
    int one = 1; ::setsockopt(lfd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    sockaddr_in addr{}; addr.sin_family = AF_INET; addr.sin_addr.s_addr = INADDR_ANY; addr.sin_port = htons(port);
    ::bind(lfd, (sockaddr *) &addr, sizeof(addr));
    ::listen(lfd, 1);
    int fd = ::accept(lfd, nullptr, nullptr);
    ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    ::close(lfd);
    tp_fd() = fd;
}

// Follower: connect to the leader's control port (retry while the leader still loads its model).
inline void tp_follower_connect() {
    const char * host = getenv("GGML_TP_MASTER_ADDR"); if (!host) host = "127.0.0.1";
    const char * pe = getenv("GGML_TP_MASTER_PORT");
    const int port = (pe ? atoi(pe) : 29600) + 1;
    for (int attempt = 0; attempt < 18000; attempt++) {
        int fd = ::socket(AF_INET, SOCK_STREAM, 0);
        sockaddr_in addr{}; addr.sin_family = AF_INET; addr.sin_port = htons(port);
        ::inet_pton(AF_INET, host, &addr.sin_addr);
        if (::connect(fd, (sockaddr *) &addr, sizeof(addr)) == 0) {
            int one = 1; ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
            tp_fd() = fd;
            return;
        }
        ::close(fd);
        usleep(200000); // 0.2s
    }
}

// ---- leader: broadcast ops ----------------------------------------------------------------

// Every record is prefixed with the target context id (ctx_tgt=0; only registered contexts) so the follower
// replays it on the matching context. [tp-2node-dsv4]
// Serialize+send a batch (token path only; the server never uses batch.embd).
inline void tp_bcast_decode(int32_t ctx_id, const llama_batch & b) {
    if (!tp_is_leader() || tp_fd() < 0 || ctx_id < 0) { return; }
    std::lock_guard<std::mutex> lk(tp_send_mtx());
    const int32_t op = TP_OP_DECODE;
    const int32_t nt = b.n_tokens;
    if (getenv("GGML_TP_DBG")) { fprintf(stderr, "[tp-op] SEND ctx=%d DECODE nt=%d\n", ctx_id, nt); fflush(stderr); }
    tp_send_all(tp_fd(), &ctx_id, sizeof(ctx_id));
    tp_send_all(tp_fd(), &op, sizeof(op));
    tp_send_all(tp_fd(), &nt, sizeof(nt));
    tp_send_all(tp_fd(), b.token, sizeof(llama_token) * nt);
    tp_send_all(tp_fd(), b.pos,   sizeof(llama_pos)   * nt);
    tp_send_all(tp_fd(), b.n_seq_id, sizeof(int32_t)  * nt);
    for (int32_t i = 0; i < nt; i++) {
        tp_send_all(tp_fd(), b.seq_id[i], sizeof(llama_seq_id) * b.n_seq_id[i]);
    }
    tp_send_all(tp_fd(), b.logits, sizeof(int8_t) * nt);
}

// Drive a decode on a TP context: broadcast it to the follower (if leader + registered) then run it.
// Used everywhere the server/speculative path decodes ctx_tgt OR the MTP draft ctx_dft. [tp-2node-dsv4]
inline int32_t tp_decode(llama_context * ctx, const llama_batch & b) {
    if (tp_is_leader()) { tp_bcast_decode(tp_ctx_id(ctx), b); }
    return llama_decode(ctx, b);
}

inline void tp_bcast_seq_op(int32_t ctx_id, int32_t op, llama_seq_id s0, llama_seq_id s1, llama_pos p0, llama_pos p1, llama_pos d) {
    if (!tp_is_leader() || tp_fd() < 0 || ctx_id < 0) { return; }
    std::lock_guard<std::mutex> lk(tp_send_mtx());
    tp_send_all(tp_fd(), &ctx_id, sizeof(ctx_id));
    tp_send_all(tp_fd(), &op, sizeof(op));
    tp_send_all(tp_fd(), &s0, sizeof(s0));
    tp_send_all(tp_fd(), &s1, sizeof(s1));
    tp_send_all(tp_fd(), &p0, sizeof(p0));
    tp_send_all(tp_fd(), &p1, sizeof(p1));
    tp_send_all(tp_fd(), &d,  sizeof(d));
}

// Mirror a KV snapshot save/restore. `op` is TP_OP_STATE_SAVE or TP_OP_STATE_RESTORE; the follower
// snapshots/restores ITS OWN (mirrored) KV for `seq_id` under `key` -- no data crosses the wire.
inline void tp_bcast_state_op(int32_t ctx_id, int32_t op, uint64_t key, llama_seq_id seq_id, int32_t flags) {
    if (!tp_is_leader() || tp_fd() < 0 || ctx_id < 0) { return; }
    std::lock_guard<std::mutex> lk(tp_send_mtx());
    tp_send_all(tp_fd(), &ctx_id, sizeof(ctx_id));
    tp_send_all(tp_fd(), &op,     sizeof(op));
    tp_send_all(tp_fd(), &key,    sizeof(key));
    tp_send_all(tp_fd(), &seq_id, sizeof(seq_id));
    tp_send_all(tp_fd(), &flags,  sizeof(flags));
}

inline void tp_bcast_stop() {
    if (!tp_is_leader() || tp_fd() < 0) { return; }
    std::lock_guard<std::mutex> lk(tp_send_mtx());
    const int32_t ctx_id = 0;
    const int32_t op = TP_OP_STOP;
    tp_send_all(tp_fd(), &ctx_id, sizeof(ctx_id));
    tp_send_all(tp_fd(), &op, sizeof(op));
}

// ---- follower: receive+replay loop --------------------------------------------------------

// Drives ALL registered follower contexts (currently just ctx_tgt; the NextN draft ctx_dft is not
// registered) from the leader's op stream until TP_OP_STOP / EOF. Each record names its target
// context by id. [tp-2node-dsv4]
inline void tp_follower_loop() {
    const int fd = tp_fd();
    std::map<uint64_t, std::vector<uint8_t>> state_store; // mirrored KV snapshots, keyed by leader's key
    size_t state_store_bytes = 0;                         // running total, for the eviction cap below
    static constexpr size_t TP_STATE_STORE_CAP = 8ull * 1024 * 1024 * 1024; // 8 GiB cap (leader live set is far smaller)
    std::vector<llama_token>   toks;
    std::vector<llama_pos>     pos;
    std::vector<int32_t>       nsid;
    std::vector<int8_t>        logits;
    while (true) {
        int32_t ctx_id, op;
        if (!tp_recv_all(fd, &ctx_id, sizeof(ctx_id))) { break; }
        if (!tp_recv_all(fd, &op, sizeof(op))) { break; }
        if (op == TP_OP_STOP) { break; }
        llama_context * ctx = (ctx_id >= 0 && ctx_id < (int) tp_ctxs().size()) ? tp_ctxs()[ctx_id] : nullptr;
        llama_memory_t  mem = ctx ? llama_get_memory(ctx) : nullptr;
        if (op == TP_OP_DECODE) {
            int32_t nt;
            if (!tp_recv_all(fd, &nt, sizeof(nt))) { break; }
            toks.resize(nt); pos.resize(nt); nsid.resize(nt); logits.resize(nt);
            if (!tp_recv_all(fd, toks.data(), sizeof(llama_token) * nt)) break;
            if (!tp_recv_all(fd, pos.data(),  sizeof(llama_pos)   * nt)) break;
            if (!tp_recv_all(fd, nsid.data(), sizeof(int32_t)     * nt)) break;
            std::vector<std::vector<llama_seq_id>> seqs(nt);
            for (int32_t i = 0; i < nt; i++) {
                seqs[i].resize(nsid[i]);
                if (!tp_recv_all(fd, seqs[i].data(), sizeof(llama_seq_id) * nsid[i])) { goto done; }
            }
            if (!tp_recv_all(fd, logits.data(), sizeof(int8_t) * nt)) break;
            if (getenv("GGML_TP_DBG")) { fprintf(stderr, "[tp-op] RECV ctx=%d DECODE nt=%d\n", ctx_id, nt); fflush(stderr); }
            // Rebuild a batch and decode on the matching context (same shapes => NCCL lines up).
            llama_batch b = llama_batch_init(nt, 0, 1);
            b.n_tokens = nt;
            for (int32_t i = 0; i < nt; i++) {
                b.token[i]    = toks[i];
                b.pos[i]      = pos[i];
                b.n_seq_id[i] = nsid[i];
                for (int32_t s = 0; s < nsid[i]; s++) { b.seq_id[i][s] = seqs[i][s]; }
                b.logits[i]   = logits[i];
            }
            if (ctx) { llama_decode(ctx, b); }
            llama_batch_free(b);
        } else if (op == TP_OP_STATE_SAVE || op == TP_OP_STATE_RESTORE) {
            uint64_t key; llama_seq_id seq_id; int32_t flags;
            if (!tp_recv_all(fd, &key,    sizeof(key)))    break;
            if (!tp_recv_all(fd, &seq_id, sizeof(seq_id))) break;
            if (!tp_recv_all(fd, &flags,  sizeof(flags)))  break;
            if (ctx && op == TP_OP_STATE_SAVE) {
                const size_t sz = llama_state_seq_get_size_ext(ctx, seq_id, (llama_state_seq_flags) flags);
                std::vector<uint8_t> & dst = state_store[key];
                state_store_bytes -= dst.size(); // 0 for a fresh key (monotonic keys never collide)
                dst.resize(sz);
                llama_state_seq_get_data_ext(ctx, dst.data(), sz, seq_id, (llama_state_seq_flags) flags);
                state_store_bytes += sz;
                // [TurboQuant] Bound the follower snapshot store. The leader evicts old checkpoints
                // locally but sends no "free" signal, so without this cap state_store would grow
                // unbounded over a long serving session and OOM the follower (GB10 unified RAM = GPU
                // pool). Evict oldest-first (smallest monotonic key); the leader only ever restores
                // keys from its bounded live checkpoint set (the most recent), so the oldest evicted
                // here is never a key the leader will ask to restore.
                while (state_store_bytes > TP_STATE_STORE_CAP && state_store.size() > 1) {
                    auto oldest = state_store.begin();
                    state_store_bytes -= oldest->second.size();
                    state_store.erase(oldest);
                }
            } else if (ctx) {
                auto it = state_store.find(key);
                if (it != state_store.end() && !it->second.empty()) {
                    llama_state_seq_set_data_ext(ctx, it->second.data(), it->second.size(), seq_id, (llama_state_seq_flags) flags);
                } else {
                    // [tp-2node-dsv4] FIX#5: a STATE_RESTORE for a key the follower never
                    // stored means the leader<->follower snapshot sets have DESYNCED. A
                    // silent no-op leaves the follower's KV stale -> the next decode's NCCL
                    // collectives mismatch the leader -> hang/garbage. Fail loud so the
                    // desync is diagnosable instead of corrupting silently.
                    fprintf(stderr, "[tp-op] FATAL: STATE_RESTORE key=%llu MISS on follower "
                            "(ctx=%d seq=%d) — leader/follower snapshot desync\n",
                            (unsigned long long) key, ctx_id, (int) seq_id);
                    fflush(stderr);
                    abort();
                }
            }
        } else {
            llama_seq_id s0, s1; llama_pos p0, p1, d;
            if (!tp_recv_all(fd, &s0, sizeof(s0))) break;
            if (!tp_recv_all(fd, &s1, sizeof(s1))) break;
            if (!tp_recv_all(fd, &p0, sizeof(p0))) break;
            if (!tp_recv_all(fd, &p1, sizeof(p1))) break;
            if (!tp_recv_all(fd, &d,  sizeof(d)))  break;
            if (mem) switch (op) {
                case TP_OP_SEQ_RM:   llama_memory_seq_rm  (mem, s0, p0, p1);     break;
                case TP_OP_SEQ_ADD:  llama_memory_seq_add (mem, s0, p0, p1, d);  break;
                case TP_OP_SEQ_CP:   llama_memory_seq_cp  (mem, s0, s1, p0, p1); break;
                case TP_OP_SEQ_KEEP: llama_memory_seq_keep(mem, s0);            break;
                case TP_OP_CLEAR:    llama_memory_clear   (mem, true);          break;
                default: break;
            }
        }
    }
done:
    return;
}

} // namespace tpserve
