// Perf lever 1: OCCUPANCY. Same tiled W4A16 GEMM, but pack WPB warps per block
// (each warp still computes one 16x8 output tile) so the SM holds more warps and
// hides memory/dequant latency. Sweep WPB via -DWPB=. WPB=1 == baseline.
//   for w in 1 4 8 16; do nvcc -arch=sm_121a -O3 -DWPB=$w test_w4a16_gemm_occ.cu -o /tmp/o && /tmp/o; done
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <cuda_bf16.h>
#include "dsv4-w4a16-primitives.cuh"
using namespace dsv4::w4a16;

#ifndef WPB
#define WPB 8
#endif
#define M 256
#define N 512
#define K 1024

__device__ __host__ float fp4_val(int c){ static const float m[8]={0,.5f,1,1.5f,2,3,4,6}; float v=m[c&7]; return (c&8)?-v:v; }
__device__ __forceinline__ uint32_t pack2(__nv_bfloat16 lo,__nv_bfloat16 hi){
    unsigned short l=*reinterpret_cast<unsigned short*>(&lo), h=*reinterpret_cast<unsigned short*>(&hi);
    return ((uint32_t)h<<16)|(uint32_t)l; }
__device__ __forceinline__ __nv_bfloat16 dequant_w(uint8_t code,uint8_t sbyte){
    uint32_t e_lo,e_hi; dequant_e2m1x4_to_bf16x4((uint32_t)code<<12,e_lo,e_hi);
    uint32_t s_lo,s_hi; dequant_e8m0x4_to_bf16x4((uint32_t)sbyte,s_lo,s_hi);
    uint32_t prod=mul_bf16x2(e_hi&0xFFFFu, s_lo&0xFFFFu);
    unsigned short lo16=prod&0xFFFFu; return *reinterpret_cast<__nv_bfloat16*>(&lo16); }

__global__ void gemm_occ(const __nv_bfloat16* __restrict__ A, const uint8_t* __restrict__ Bc,
                         uint8_t sbyte, float* __restrict__ D){
    int wib = threadIdx.x >> 5;                 // warp in block
    int wid = blockIdx.x * WPB + wib;           // global warp/tile id
    const int NT = N / 8;
    int tm = (wid / NT) * 16, tn = (wid % NT) * 8;
    if (tm >= M) return;
    int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    __shared__ __nv_bfloat16 Bts[WPB][16*8];
    __nv_bfloat16* Bt = Bts[wib];
    float d0=0,d1=0,d2=0,d3=0;
    for (int k0=0; k0<K; k0+=16){
        for (int idx=lane; idx<128; idx+=32)
            Bt[idx] = dequant_w(Bc[(k0+idx/8)*N + tn + idx%8], sbyte);
        __syncwarp();
        #define A_(r,c) A[(tm+(r))*K + k0 + (c)]
        #define B_(r,c) Bt[(r)*8 + (c)]
        uint32_t a0=pack2(A_(g,t*2+0),A_(g,t*2+1)),   a1=pack2(A_(g+8,t*2+0),A_(g+8,t*2+1));
        uint32_t a2=pack2(A_(g,t*2+8),A_(g,t*2+9)),   a3=pack2(A_(g+8,t*2+8),A_(g+8,t*2+9));
        uint32_t b0=pack2(B_(t*2+0,g),B_(t*2+1,g)),   b1=pack2(B_(t*2+8,g),B_(t*2+9,g));
        #undef A_
        #undef B_
        mma_m16n8k16_bf16_f32(d0,d1,d2,d3, a0,a1,a2,a3, b0,b1);
        __syncwarp();
    }
    const float comp=0x1p119f;
    D[(tm+g  )*N+tn+t*2+0]=d0*comp; D[(tm+g  )*N+tn+t*2+1]=d1*comp;
    D[(tm+g+8)*N+tn+t*2+0]=d2*comp; D[(tm+g+8)*N+tn+t*2+1]=d3*comp;
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
    int tiles=(M/16)*(N/8); dim3 grid((tiles+WPB-1)/WPB); int block=WPB*32;
    gemm_occ<<<grid,block>>>(dA,dBc,(uint8_t)SBYTE,dD);
    static float hD[M*N]; cudaMemcpy(hD,dD,sizeof hD,cudaMemcpyDeviceToHost);
    cudaError_t e=cudaDeviceSynchronize();
    int fails=0,checked=0;
    for(int m=0;m<M;m+=37) for(int n=0;n<N;n+=41){ float s=0;
        for(int k=0;k<K;k++) s+=__bfloat162float(hA[m*K+k])*fp4_val(hBc[k*N+n])*scale;
        checked++; if(fabsf(hD[m*N+n]-s)>1e-3f*(fabsf(s)+1)) fails++; }
    cudaEvent_t t0,t1; cudaEventCreate(&t0); cudaEventCreate(&t1); int iters=200;
    for(int w=0;w<5;w++) gemm_occ<<<grid,block>>>(dA,dBc,(uint8_t)SBYTE,dD);
    cudaEventRecord(t0); for(int it=0;it<iters;it++) gemm_occ<<<grid,block>>>(dA,dBc,(uint8_t)SBYTE,dD);
    cudaEventRecord(t1); cudaEventSynchronize(t1); float ms=0; cudaEventElapsedTime(&ms,t0,t1);
    double flop=2.0*M*N*K*iters;
    printf("WPB=%-2d block=%-4d %s | parity %d/%d | %.3f ms | %.1f GFLOP/s\n",
           WPB, block, e==cudaSuccess?"ok":"ERR", checked-fails, checked, ms/iters, flop/(ms/1e3)/1e9);
    return fails?1:0;
}
