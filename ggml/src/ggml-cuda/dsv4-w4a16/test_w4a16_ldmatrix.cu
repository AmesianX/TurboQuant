// Perf lever 4 (profiler-directed): feed the MMA with ldmatrix instead of scalar
// pack2 smem loads (which caused 94% short-scoreboard stalls). Block stages A[BMxBK]
// + dequant B[BKxBN] in smem, then each warp loads fragments via:
//   A: ldmatrix.x4        (16x16, row-major)
//   B: ldmatrix.x2.trans  (16x8, K-consecutive fragment for the mma B operand)
// Correctness vs fp32 CPU ref + throughput vs the 2.7 TFLOP/s baseline.
//   nvcc -arch=sm_121a -O3 -DBM=64 -DBN=64 test_w4a16_ldmatrix.cu -o /tmp/lm && /tmp/lm
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
#define BK 16
// NOTE: sizes must SATURATE the GPU (48 SMs) or grid-size/occupancy confounds the
// perf comparison — a small GEMM makes the many-tiny-blocks naive kernel look faster
// than it is. 2048^3 gives every config plenty of blocks; measure efficiency here.
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
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        :"=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3):"r"(a)); }
__device__ __forceinline__ void ldm_x2t(uint32_t&r0,uint32_t&r1,uint32_t a){
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
        :"=r"(r0),"=r"(r1):"r"(a)); }

__global__ void gemm_ldm(const __nv_bfloat16* __restrict__ A, const uint8_t* __restrict__ Bc,
                         uint8_t sbyte, float* __restrict__ D){
    int bm=blockIdx.y*BM, bn=blockIdx.x*BN;
    int wib=threadIdx.x>>5, lane=threadIdx.x&31, g=lane>>2, t=lane&3;
    __shared__ __nv_bfloat16 As[BM*BK];
    __shared__ __nv_bfloat16 Bs[BK*BN];
    float acc[NNT][4];
    #pragma unroll
    for(int i=0;i<NNT;i++){ acc[i][0]=acc[i][1]=acc[i][2]=acc[i][3]=0; }
    for(int k0=0;k0<K;k0+=BK){
        for(int idx=threadIdx.x; idx<BM*BK; idx+=blockDim.x) As[idx]=A[(bm+idx/BK)*K + k0 + idx%BK];
        for(int idx=threadIdx.x; idx<BK*BN; idx+=blockDim.x) Bs[idx]=dequant_w(Bc[(k0+idx/BN)*N + bn + idx%BN], sbyte);
        __syncthreads();
        uint32_t a0,a1,a2,a3;
        ldm_x4(a0,a1,a2,a3, smem_u32(&As[(wib*16 + (lane%16))*BK + (lane/16)*8]));
        #pragma unroll
        for(int nt=0; nt<NNT; nt++){
            uint32_t b0,b1;
            ldm_x2t(b0,b1, smem_u32(&Bs[(lane%16)*BN + nt*8]));
            mma_m16n8k16_bf16_f32(acc[nt][0],acc[nt][1],acc[nt][2],acc[nt][3], a0,a1,a2,a3, b0,b1);
        }
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
    for(int i=0;i<K*N;i++) hBc[i]=(uint8_t)(((i*7)%7)+(i%15));
    float scale=ldexpf(1.f,SBYTE-127);
    __nv_bfloat16* dA; uint8_t* dBc; float* dD;
    cudaMalloc(&dA,sizeof hA); cudaMalloc(&dBc,sizeof hBc); cudaMalloc(&dD,(size_t)M*N*sizeof(float));
    cudaMemcpy(dA,hA,sizeof hA,cudaMemcpyHostToDevice); cudaMemcpy(dBc,hBc,sizeof hBc,cudaMemcpyHostToDevice);
    dim3 grid(N/BN,M/BM); int block=NWARPS*32;
    gemm_ldm<<<grid,block>>>(dA,dBc,(uint8_t)SBYTE,dD);
    static float hD[M*N]; cudaMemcpy(hD,dD,sizeof hD,cudaMemcpyDeviceToHost);
    cudaError_t e=cudaDeviceSynchronize();
    int fails=0,checked=0;
    for(int m=0;m<M;m+=37) for(int n=0;n<N;n+=41){ float s=0;
        for(int k=0;k<K;k++) s+=__bfloat162float(hA[m*K+k])*fp4_val(hBc[k*N+n])*scale;
        checked++; if(fabsf(hD[m*N+n]-s)>1e-3f*(fabsf(s)+1)){ if(fails<6) printf("  MISS[m%d n%d] got=%g ref=%g\n",m,n,hD[m*N+n],s); fails++; } }
    cudaEvent_t t0,t1; cudaEventCreate(&t0); cudaEventCreate(&t1); int iters=200;
    for(int w=0;w<5;w++) gemm_ldm<<<grid,block>>>(dA,dBc,(uint8_t)SBYTE,dD);
    cudaEventRecord(t0); for(int it=0;it<iters;it++) gemm_ldm<<<grid,block>>>(dA,dBc,(uint8_t)SBYTE,dD);
    cudaEventRecord(t1); cudaEventSynchronize(t1); float ms=0; cudaEventElapsedTime(&ms,t0,t1);
    double flop=2.0*M*N*K*iters;
    printf("ldmatrix BM%d BN%d (w%d) %s | parity %d/%d | %.3f ms | %.1f GFLOP/s (baseline 2721)\n",
           BM,BN,NWARPS, e==cudaSuccess?"ok":"ERR", checked-fails, checked, ms/iters, flop/(ms/1e3)/1e9);
    return fails?1:0;
}
