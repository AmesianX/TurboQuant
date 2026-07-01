// Perf lever 2: BLOCK-TILED GEMM with smem A/B reuse. A block computes a BMxBN
// output tile: it loads A[BMxBK] and dequants B[BKxBN] into smem ONCE per K-step,
// then all warps reuse them. Kills the naive kernel's redundant per-tile dequant/load.
// NWARPS = BM/16 (each warp owns one 16-row M-subtile, computes all BN/8 N-subtiles).
//   for cfg in "64 64 32" "128 64 32" "64 128 32"; do set -- $cfg; \
//     nvcc -arch=sm_121a -O3 -DBM=$1 -DBN=$2 -DBK=$3 test_w4a16_gemm_tiled.cu -o /tmp/t && /tmp/t; done
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <cuda_bf16.h>
#include "dsv4-w4a16-primitives.cuh"
using namespace dsv4::w4a16;

#ifndef BM
#define BM 64
#endif
#ifndef BN
#define BN 64
#endif
#ifndef BK
#define BK 32
#endif
#define M 256
#define N 512
#define K 1024
#define NWARPS (BM/16)
#define NNT (BN/8)

__device__ __host__ float fp4_val(int c){ static const float m[8]={0,.5f,1,1.5f,2,3,4,6}; float v=m[c&7]; return (c&8)?-v:v; }
__device__ __forceinline__ uint32_t pack2(__nv_bfloat16 lo,__nv_bfloat16 hi){
    unsigned short l=*reinterpret_cast<unsigned short*>(&lo), h=*reinterpret_cast<unsigned short*>(&hi);
    return ((uint32_t)h<<16)|(uint32_t)l; }
__device__ __forceinline__ __nv_bfloat16 dequant_w(uint8_t code,uint8_t sbyte){
    uint32_t e_lo,e_hi; dequant_e2m1x4_to_bf16x4((uint32_t)code<<12,e_lo,e_hi);
    uint32_t s_lo,s_hi; dequant_e8m0x4_to_bf16x4((uint32_t)sbyte,s_lo,s_hi);
    uint32_t prod=mul_bf16x2(e_hi&0xFFFFu, s_lo&0xFFFFu);
    unsigned short lo16=prod&0xFFFFu; return *reinterpret_cast<__nv_bfloat16*>(&lo16); }

__global__ void gemm_tiled(const __nv_bfloat16* __restrict__ A, const uint8_t* __restrict__ Bc,
                           uint8_t sbyte, float* __restrict__ D){
    int bm = blockIdx.y * BM, bn = blockIdx.x * BN;
    int wib = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    __shared__ __nv_bfloat16 As[BM*BK];
    __shared__ __nv_bfloat16 Bs[BK*BN];
    float acc[NNT][4];
    #pragma unroll
    for (int i=0;i<NNT;i++){ acc[i][0]=acc[i][1]=acc[i][2]=acc[i][3]=0; }
    for (int k0=0; k0<K; k0+=BK){
        for (int idx=threadIdx.x; idx<BM*BK; idx+=blockDim.x)
            As[idx] = A[(bm+idx/BK)*K + k0 + idx%BK];
        for (int idx=threadIdx.x; idx<BK*BN; idx+=blockDim.x)
            Bs[idx] = dequant_w(Bc[(k0+idx/BN)*N + bn + idx%BN], sbyte);
        __syncthreads();
        for (int kk=0; kk<BK; kk+=16){
            int r0 = wib*16;
            uint32_t a0=pack2(As[(r0+g  )*BK+kk+t*2+0], As[(r0+g  )*BK+kk+t*2+1]);
            uint32_t a1=pack2(As[(r0+g+8)*BK+kk+t*2+0], As[(r0+g+8)*BK+kk+t*2+1]);
            uint32_t a2=pack2(As[(r0+g  )*BK+kk+t*2+8], As[(r0+g  )*BK+kk+t*2+9]);
            uint32_t a3=pack2(As[(r0+g+8)*BK+kk+t*2+8], As[(r0+g+8)*BK+kk+t*2+9]);
            #pragma unroll
            for (int nt=0; nt<NNT; nt++){
                int c = nt*8 + g;
                uint32_t b0=pack2(Bs[(kk+t*2+0)*BN+c], Bs[(kk+t*2+1)*BN+c]);
                uint32_t b1=pack2(Bs[(kk+t*2+8)*BN+c], Bs[(kk+t*2+9)*BN+c]);
                mma_m16n8k16_bf16_f32(acc[nt][0],acc[nt][1],acc[nt][2],acc[nt][3], a0,a1,a2,a3, b0,b1);
            }
        }
        __syncthreads();
    }
    const float comp=0x1p119f; int r0=wib*16;
    #pragma unroll
    for (int nt=0; nt<NNT; nt++){
        int cc = bn + nt*8;
        D[(bm+r0+g  )*N+cc+t*2+0]=acc[nt][0]*comp; D[(bm+r0+g  )*N+cc+t*2+1]=acc[nt][1]*comp;
        D[(bm+r0+g+8)*N+cc+t*2+0]=acc[nt][2]*comp; D[(bm+r0+g+8)*N+cc+t*2+1]=acc[nt][3]*comp;
    }
}

int main(){
    const int SBYTE=129;
    static __nv_bfloat16 hA[M*K]; static uint8_t hBc[K*N];
    for(int i=0;i<M*K;i++) hA[i]=__float2bfloat16((float)((i%5)-2));
    for(int i=0;i<K*N;i++) hBc[i]=(uint8_t)(((i*7)%7)+(i%15));
    float scale=ldexpf(1.f,SBYTE-127);
    __nv_bfloat16* dA; uint8_t* dBc; float* dD;
    cudaMalloc(&dA,sizeof hA); cudaMalloc(&dBc,sizeof hBc); cudaMalloc(&dD,(size_t)M*N*sizeof(float));
    cudaMemcpy(dA,hA,sizeof hA,cudaMemcpyHostToDevice); cudaMemcpy(dBc,hBc,sizeof hBc,cudaMemcpyHostToDevice);
    dim3 grid(N/BN, M/BM); int block=NWARPS*32;
    gemm_tiled<<<grid,block>>>(dA,dBc,(uint8_t)SBYTE,dD);
    static float hD[M*N]; cudaMemcpy(hD,dD,sizeof hD,cudaMemcpyDeviceToHost);
    cudaError_t e=cudaDeviceSynchronize();
    int fails=0,checked=0;
    for(int m=0;m<M;m+=37) for(int n=0;n<N;n+=41){ float s=0;
        for(int k=0;k<K;k++) s+=__bfloat162float(hA[m*K+k])*fp4_val(hBc[k*N+n])*scale;
        checked++; if(fabsf(hD[m*N+n]-s)>1e-3f*(fabsf(s)+1)) fails++; }
    cudaEvent_t t0,t1; cudaEventCreate(&t0); cudaEventCreate(&t1); int iters=200;
    for(int w=0;w<5;w++) gemm_tiled<<<grid,block>>>(dA,dBc,(uint8_t)SBYTE,dD);
    cudaEventRecord(t0); for(int it=0;it<iters;it++) gemm_tiled<<<grid,block>>>(dA,dBc,(uint8_t)SBYTE,dD);
    cudaEventRecord(t1); cudaEventSynchronize(t1); float ms=0; cudaEventElapsedTime(&ms,t0,t1);
    double flop=2.0*M*N*K*iters;
    printf("BM%d BN%d BK%d (w%d) %s | parity %d/%d | %.3f ms | %.1f GFLOP/s\n",
           BM,BN,BK,NWARPS, e==cudaSuccess?"ok":"ERR", checked-fails, checked, ms/iters, flop/(ms/1e3)/1e9);
    return fails?1:0;
}
