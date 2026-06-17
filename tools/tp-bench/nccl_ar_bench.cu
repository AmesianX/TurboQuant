// M0 de-risk for cross-node tensor parallelism (feat/tp-2node-dsv4).
// Measures NCCL all-reduce latency/bandwidth across 2 NODES over RoCE for
// hidden-state-sized tensors. A DSV4 decode token would issue ~2 all-reduces
// per layer x 43 layers ~= 86 collectives; this tells us the per-token sync
// overhead and whether ~2x (halved per-node weight read) survives it.
//
// Bootstrap: rank0 generates the ncclUniqueId and ships it to rank1 over a
// plain TCP socket (no MPI dependency). Launch rank0 on .66, rank1 on .67.
//
// build: nvcc -O3 -o nccl_ar_bench nccl_ar_bench.cu -lnccl
// run (.66): ./nccl_ar_bench 0 10.0.1.1 29555
// run (.67): ./nccl_ar_bench 1 10.0.1.1 29555

#include <nccl.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>

#define CK(x)   do { auto e=(x); if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} } while(0)
#define NK(x)   do { auto r=(x); if(r!=ncclSuccess){fprintf(stderr,"NCCL %s:%d %s\n",__FILE__,__LINE__,ncclGetErrorString(r));exit(1);} } while(0)

// rank0: bind/listen/accept, send id. rank1: connect (with retry), recv id.
static void exchange_id(int rank, const char* ip, int port, ncclUniqueId* id) {
    const size_t N = sizeof(ncclUniqueId); // 128
    if (rank == 0) {
        int s = socket(AF_INET, SOCK_STREAM, 0);
        int one = 1; setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
        sockaddr_in a{}; a.sin_family = AF_INET; a.sin_port = htons(port);
        a.sin_addr.s_addr = INADDR_ANY;
        if (bind(s, (sockaddr*)&a, sizeof(a)) < 0) { perror("bind"); exit(1); }
        listen(s, 1);
        int c = accept(s, nullptr, nullptr);
        if (c < 0) { perror("accept"); exit(1); }
        size_t off = 0; const char* p = (const char*)id;
        while (off < N) { ssize_t w = write(c, p+off, N-off); if (w<=0){perror("write");exit(1);} off += w; }
        close(c); close(s);
    } else {
        int s = -1;
        for (int try_=0; try_<100; ++try_) {
            s = socket(AF_INET, SOCK_STREAM, 0);
            sockaddr_in a{}; a.sin_family = AF_INET; a.sin_port = htons(port);
            inet_pton(AF_INET, ip, &a.sin_addr);
            if (connect(s, (sockaddr*)&a, sizeof(a)) == 0) break;
            close(s); s = -1; usleep(200000); // 0.2s, wait for rank0
        }
        if (s < 0) { fprintf(stderr,"rank1 connect failed\n"); exit(1); }
        size_t off = 0; char* p = (char*)id;
        while (off < N) { ssize_t r = read(s, p+off, N-off); if (r<=0){perror("read");exit(1);} off += r; }
        close(s);
    }
}

int main(int argc, char** argv) {
    if (argc < 4) { fprintf(stderr,"usage: %s <rank 0|1> <master_ip> <port>\n", argv[0]); return 1; }
    int rank = atoi(argv[1]);
    const char* ip = argv[2];
    int port = atoi(argv[3]);
    const int nranks = 2;

    CK(cudaSetDevice(0));
    ncclUniqueId id;
    if (rank == 0) NK(ncclGetUniqueId(&id));
    exchange_id(rank, ip, port, &id);

    ncclComm_t comm;
    NK(ncclCommInitRank(&comm, nranks, id, rank));
    cudaStream_t st; CK(cudaStreamCreate(&st));
    if (rank == 0) printf("%-12s %-14s %-14s %-12s\n", "elems", "bytes", "lat_us/AR", "alg_GB/s");

    // sizes spanning a DSV4 hidden vector (n_embd=4096) up to a fat activation
    const int sizes[] = {1024, 4096, 16384, 65536, 262144, 1048576};
    for (int si = 0; si < (int)(sizeof(sizes)/sizeof(int)); ++si) {
        size_t n = sizes[si];
        float* d; CK(cudaMalloc(&d, n*sizeof(float)));
        CK(cudaMemset(d, 1, n*sizeof(float)));

        const int warm = 30, iters = 1000;
        for (int i=0;i<warm;i++) NK(ncclAllReduce(d,d,n,ncclFloat,ncclSum,comm,st));
        CK(cudaStreamSynchronize(st));

        cudaEvent_t e0,e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
        CK(cudaEventRecord(e0, st));
        for (int i=0;i<iters;i++) NK(ncclAllReduce(d,d,n,ncclFloat,ncclSum,comm,st));
        CK(cudaEventRecord(e1, st));
        CK(cudaStreamSynchronize(st));
        float ms=0; CK(cudaEventElapsedTime(&ms, e0, e1));
        double lat_us = (double)ms*1000.0/iters;
        double bytes = n*sizeof(float);
        // bus bandwidth for ring all-reduce ~ 2*(n-1)/n * bytes; report alg bw = bytes/time
        double gbps = bytes / (lat_us*1e-6) / 1e9;
        if (rank == 0) printf("%-12zu %-14.0f %-14.2f %-12.2f\n", n, bytes, lat_us, gbps);
        CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1)); CK(cudaFree(d));
    }

    if (rank == 0) {
        // DSV4 per-token sync model: ~86 all-reduces of n_embd=4096
        printf("\n# interpret: per-token sync ~= 86 x (lat at 4096 elems)\n");
    }
    NK(ncclCommDestroy(comm));
    return 0;
}
