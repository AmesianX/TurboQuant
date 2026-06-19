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

#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>

namespace tpserve {

enum op_type : int32_t {
    TP_OP_DECODE  = 1,
    TP_OP_SEQ_RM  = 2,
    TP_OP_SEQ_ADD = 3,
    TP_OP_SEQ_CP  = 4,
    TP_OP_SEQ_KEEP= 5,
    TP_OP_CLEAR   = 6,
    TP_OP_STOP    = 99,
};

inline int  tp_nranks() { static const int v = getenv("GGML_TP_NRANKS") ? atoi(getenv("GGML_TP_NRANKS")) : 1; return v; }
inline int  tp_rank()   { static const int v = getenv("GGML_TP_RANK")   ? atoi(getenv("GGML_TP_RANK"))   : 0; return v; }
inline bool tp_enabled(){ return tp_nranks() > 1; }
inline bool tp_is_leader()   { return tp_enabled() && tp_rank() == 0; }
inline bool tp_is_follower() { return tp_enabled() && tp_rank() != 0; }

// The single control socket (leader: one accepted fd; follower: the connected fd).
inline int & tp_fd() { static int fd = -1; return fd; }

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

// Serialize+send a batch (token path only; the server never uses batch.embd).
inline void tp_bcast_decode(const llama_batch & b) {
    if (!tp_is_leader() || tp_fd() < 0) { return; }
    const int32_t op = TP_OP_DECODE;
    const int32_t nt = b.n_tokens;
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

inline void tp_bcast_seq_op(int32_t op, llama_seq_id s0, llama_seq_id s1, llama_pos p0, llama_pos p1, llama_pos d) {
    if (!tp_is_leader() || tp_fd() < 0) { return; }
    tp_send_all(tp_fd(), &op, sizeof(op));
    tp_send_all(tp_fd(), &s0, sizeof(s0));
    tp_send_all(tp_fd(), &s1, sizeof(s1));
    tp_send_all(tp_fd(), &p0, sizeof(p0));
    tp_send_all(tp_fd(), &p1, sizeof(p1));
    tp_send_all(tp_fd(), &d,  sizeof(d));
}

inline void tp_bcast_stop() {
    if (!tp_is_leader() || tp_fd() < 0) { return; }
    const int32_t op = TP_OP_STOP;
    tp_send_all(tp_fd(), &op, sizeof(op));
}

// ---- follower: receive+replay loop --------------------------------------------------------

// Drives the follower context purely from the leader's op stream until TP_OP_STOP / EOF.
inline void tp_follower_loop(llama_context * ctx) {
    llama_memory_t mem = llama_get_memory(ctx);
    const int fd = tp_fd();
    std::vector<llama_token>   toks;
    std::vector<llama_pos>     pos;
    std::vector<int32_t>       nsid;
    std::vector<llama_seq_id>  flat_seq;
    std::vector<int8_t>        logits;
    while (true) {
        int32_t op;
        if (!tp_recv_all(fd, &op, sizeof(op))) { break; }
        if (op == TP_OP_STOP) { break; }
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
            // Rebuild a batch and decode (same shapes => matching graph => NCCL lines up).
            llama_batch b = llama_batch_init(nt, 0, 1);
            b.n_tokens = nt;
            for (int32_t i = 0; i < nt; i++) {
                b.token[i]    = toks[i];
                b.pos[i]      = pos[i];
                b.n_seq_id[i] = nsid[i];
                for (int32_t s = 0; s < nsid[i]; s++) { b.seq_id[i][s] = seqs[i][s]; }
                b.logits[i]   = logits[i];
            }
            llama_decode(ctx, b);
            llama_batch_free(b);
        } else {
            llama_seq_id s0, s1; llama_pos p0, p1, d;
            if (!tp_recv_all(fd, &s0, sizeof(s0))) break;
            if (!tp_recv_all(fd, &s1, sizeof(s1))) break;
            if (!tp_recv_all(fd, &p0, sizeof(p0))) break;
            if (!tp_recv_all(fd, &p1, sizeof(p1))) break;
            if (!tp_recv_all(fd, &d,  sizeof(d)))  break;
            switch (op) {
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
