// Perf lever 5 (profiler-directed): cp.async double-buffered prefetch. Kernel is now
// memory-bound (68.8%) with compute idle (26.8%) -> overlap the global load of the NEXT
// K-tile with the MMA on the CURRENT tile, and vectorize loads to 16B. Raw A(bf16)+B(fp4)
// are cp.async'd into double-buffered smem; B is dequanted per step then fed via ldmatrix.
//   nvcc -arch=sm_121a -O3 -DBM=128 -DBN=64 test_w4a16_cpasync.cu -o /tmp/ca && /tmp/ca
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <cuda_bf16.h>
#include "dsv4-w4a16-primitives.cuh"
using namespace dsv4::w4a16;

#ifndef BM
#define BM 128
#endif
#ifndef BN
#define BN 64
#endif
#define BK 16
#define M 2048
#define N 2048
#define K 2048
#define NWARPS (BM/16)
#define NNT (BN/8)

__device__ __host__ float fp4_val(int c){ static const float m[8]={0,.5f,1,1.5f,2,3,4,6}; float v=m[c&7]; return (c&8)?-v:v; }
__device__ __forceinline__ __nv_bfloat16 dequant_w(uint8_t code,uint8_t sbyte){
    uint32_t e_lo,e_hi; dequant_e2m1x4_to_bf16x4((uint32_t)code<<12,e_lo,e_hi);
    uint32_t s_lo,s_hi; dequant_e8m0x4_to_bf16x4((uint32_t)sbyte,s_lo,s_hi);
    uint32_t prod=mul_bf16x2(e_hi&0xFFFFu, s_lo&0xFFFFu);
    unsigned short lo16=prod&0xFFFFu; return *reinterpret_cast<__nv_bfloat16*>(&lo16); }
__device__ __forceinline__ uint32_t smem_u32(const void* p){ return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void ldm_x4(uint32_t&r0,uint32_t&r1,uint32_t&r2,uint32_t&r3,uint32_t a){
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n":"=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3):"r"(a)); }
__device__ __forceinline__ void ldm_x2t(uint32_t&r0,uint32_t&r1,uint32_t a){
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n":"=r"(r0),"=r"(r1):"r"(a)); }
__device__ __forceinline__ void cp_async16(void* s,const void* g){
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"::"r"((uint32_t)__cvta_generic_to_shared(s)),"l"(g)); }

__global__ void gemm_cpa(const __nv_bfloat16* __restrict__ A, const uint8_t* __restrict__ Bc,
                         uint8_t sbyte, float* __restrict__ D){
    int bm=blockIdx.y*BM, bn=blockIdx.x*BN, tid=threadIdx.x, wib=tid>>5, lane=tid&31, g=lane>>2, t=lane&3;
    __shared__ __align__(16) __nv_bfloat16 Araw[2][BM*BK];
    __shared__ __align__(16) uint8_t Braw[2][BK*BN];
    __shared__ __align__(16) __nv_bfloat16 Bdeq[BK*BN];
    float acc[NNT][4];
    #pragma unroll
    for(int i=0;i<NNT;i++){ acc[i][0]=acc[i][1]=acc[i][2]=acc[i][3]=0; }
    const int nsteps=K/BK, ACH=BM*(BK/8), BCH=BK*(BN/16);
    auto loadtile=[&](int step,int buf){
        int k0=step*BK;
        for(int c=tid;c<ACH;c+=blockDim.x){ int row=c/(BK/8), h=c%(BK/8);
            cp_async16(&Araw[buf][row*BK+h*8], &A[(bm+row)*K + k0 + h*8]); }
        for(int c=tid;c<BCH;c+=blockDim.x){ int kr=c/(BN/16), nc=c%(BN/16);
            cp_async16(&Braw[buf][kr*BN+nc*16], &Bc[(k0+kr)*N + bn + nc*16]); }
    };
    loadtile(0,0); asm volatile("cp.async.commit_group;\n");
    for(int s=0;s<nsteps;s++){
        int cur=s&1;
        if(s+1<nsteps){ loadtile(s+1,cur^1); asm volatile("cp.async.commit_group;\n"); asm volatile("cp.async.wait_group 1;\n"); }
        else asm volatile("cp.async.wait_group 0;\n");
        __syncthreads();
        for(int idx=tid;idx<BK*BN;idx+=blockDim.x) Bdeq[idx]=dequant_w(Braw[cur][idx], sbyte);
        __syncthreads();
        uint32_t a0,a1,a2,a3;
        ldm_x4(a0,a1,a2,a3, smem_u32(&Araw[cur][(wib*16+(lane%16))*BK + (lane/16)*8]));
        #pragma unroll
        for(int nt=0;nt<NNT;nt++){ uint32_t b0,b1;
            ldm_x2t(b0,b1, smem_u32(&Bdeq[(lane%16)*BN + nt*8]));
            mma_m16n8k16_bf16_f32(acc[nt][0],acc[nt][1],acc[nt][2],acc[nt][3], a0,a1,a2,a3, b0,b1); }
        __syncthreads();
    }
    const float comp=0x1p119f; int r0=wib*16;
    #pragma unroll
    for(int nt=0;nt<NNT;nt++){ int cc=bn+nt*8;
        D[(bm+r0+g  )*N+cc+t*2+0]=acc[nt][0]*comp; D[(bm+r0+g  )*N+cc+t*2+1]=acc[nt][1]*comp;
        D[(bm+r0+g+8)*N+cc+t*2+0]=acc[nt][2]*comp; D[(bm+r0+g+8)*N+cc+t*2+1]=acc[nt][3]*comp; }
}

int main(){
    const int SBYTE=129;
    static __nv_bfloat16 hA[M*K]; static uint8_t hBc[K*N];
    for(int i=0;i<M*K;i++) hA[i]=__float2bfloat16((float)((i%5)-2));
    for(int i=0;i<K*N;i++) hBc[i]=(uint8_t)((i*37+11)&0xFF); // all 256 nibble pairs -> every e2m1 code incl. 15 (-6)
    float scale=ldexpf(1.f,SBYTE-127);
    __nv_bfloat16* dA; uint8_t* dBc; float* dD;
    cudaMalloc(&dA,sizeof hA); cudaMalloc(&dBc,sizeof hBc); cudaMalloc(&dD,(size_t)M*N*sizeof(float));
    cudaMemcpy(dA,hA,sizeof hA,cudaMemcpyHostToDevice); cudaMemcpy(dBc,hBc,sizeof hBc,cudaMemcpyHostToDevice);
    dim3 grid(N/BN,M/BM); int block=NWARPS*32;
    gemm_cpa<<<grid,block>>>(dA,dBc,(uint8_t)SBYTE,dD);
    static float hD[M*N]; cudaMemcpy(hD,dD,sizeof hD,cudaMemcpyDeviceToHost);
    cudaError_t e=cudaDeviceSynchronize();
    int fails=0,checked=0;
    for(int m=0;m<M;m+=37) for(int n=0;n<N;n+=41){ float s=0;
        for(int k=0;k<K;k++) s+=__bfloat162float(hA[m*K+k])*fp4_val(hBc[k*N+n])*scale;
        checked++; if(fabsf(hD[m*N+n]-s)>1e-3f*(fabsf(s)+1)){ if(fails<6) printf("  MISS[m%d n%d] got=%g ref=%g\n",m,n,hD[m*N+n],s); fails++; } }
    cudaEvent_t t0,t1; cudaEventCreate(&t0); cudaEventCreate(&t1); int iters=200;
    for(int w=0;w<5;w++) gemm_cpa<<<grid,block>>>(dA,dBc,(uint8_t)SBYTE,dD);
    cudaEventRecord(t0); for(int it=0;it<iters;it++) gemm_cpa<<<grid,block>>>(dA,dBc,(uint8_t)SBYTE,dD);
    cudaEventRecord(t1); cudaEventSynchronize(t1); float ms=0; cudaEventElapsedTime(&ms,t0,t1);
    double flop=2.0*M*N*K*iters;
    printf("cp.async BM%d BN%d (w%d) %s | parity %d/%d | %.3f ms | %.1f GFLOP/s (ldmatrix 11.8k)\n",
           BM,BN,NWARPS, e==cudaSuccess?"ok":"ERR", checked-fails, checked, ms/iters, flop/(ms/1e3)/1e9);
    return fails?1:0;
}
