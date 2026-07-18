// Perf lever 8 (몸통): 2D warp-tiling + register blocking. Warps arranged WM x WN;
// each warp computes an RM x RN grid of 16x8 MMA tiles (RM=BM/16/WM, RN=BN/8/WN),
// loading RM A-fragments + RN B-fragments and doing RM*RN MMAs — reusing each A-frag
// across RN and each B-frag across RM (register-level arithmetic intensity). This lets
// BN grow (cutting A global traffic ~1/BN) without per-warp register blow-up.
// Structure: cp.async double-buffer + ldmatrix-A + B-direct-dequant (levers 4-7).
//   nvcc -arch=sm_121a -O3 -DBM=128 -DBN=128 -DWM=4 -DWN=2 -DBK=32 test_w4a16_wt.cu -o /tmp/wt && /tmp/wt
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
#define BN 128
#endif
#ifndef BK
#define BK 32
#endif
#ifndef WM
#define WM 4
#endif
#ifndef WN
#define WN 2
#endif
#ifndef ST
#define ST 3
#endif
#define RM (BM/16/WM)
#define RN (BN/8/WN)
#define NWARPS (WM*WN)
#define M 2048
#define N 2048
#define K 2048

__device__ __host__ float fp4_val(int c){ static const float m[8]={0,.5f,1,1.5f,2,3,4,6}; float v=m[c&7]; return (c&8)?-v:v; }
__device__ __forceinline__ __nv_bfloat16 dequant_w(uint8_t code,uint8_t sbyte){
    uint32_t e_lo,e_hi; dequant_e2m1x4_to_bf16x4((uint32_t)code<<12,e_lo,e_hi);
    uint32_t s_lo,s_hi; dequant_e8m0x4_to_bf16x4((uint32_t)sbyte,s_lo,s_hi);
    uint32_t prod=mul_bf16x2(e_hi&0xFFFFu, s_lo&0xFFFFu);
    unsigned short lo16=prod&0xFFFFu; return *reinterpret_cast<__nv_bfloat16*>(&lo16); }
__device__ __forceinline__ uint32_t pack2b(__nv_bfloat16 lo,__nv_bfloat16 hi){
    unsigned short l=*reinterpret_cast<unsigned short*>(&lo),h=*reinterpret_cast<unsigned short*>(&hi); return ((uint32_t)h<<16)|(uint32_t)l; }
__device__ __forceinline__ uint32_t smem_u32(const void* p){ return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void ldm_x4(uint32_t&r0,uint32_t&r1,uint32_t&r2,uint32_t&r3,uint32_t a){
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n":"=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3):"r"(a)); }
__device__ __forceinline__ void cp_async16(void* s,const void* g){
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"::"r"((uint32_t)__cvta_generic_to_shared(s)),"l"(g)); }

__global__ void gemm_pipe(const __nv_bfloat16* __restrict__ A, const uint8_t* __restrict__ Bc,
                        uint8_t sbyte, float* __restrict__ D){
    int bm=blockIdx.y*BM, bn=blockIdx.x*BN, tid=threadIdx.x, wib=tid>>5, lane=tid&31, g=lane>>2, t=lane&3;
    int wm=wib/WN, wn=wib%WN;
    __shared__ __align__(16) __nv_bfloat16 Araw[ST][BM*BK];
    __shared__ __align__(16) uint8_t Braw[ST][BK*BN];
    float acc[RM][RN][4];
    #pragma unroll
    for(int i=0;i<RM;i++) for(int j=0;j<RN;j++){ acc[i][j][0]=acc[i][j][1]=acc[i][j][2]=acc[i][j][3]=0; }
    const int nsteps=K/BK, ACH=BM*(BK/8), BCH=BK*(BN/16);
    auto loadtile=[&](int step,int buf){ int k0=step*BK;
        for(int c=tid;c<ACH;c+=blockDim.x){ int row=c/(BK/8), h=c%(BK/8);
            cp_async16(&Araw[buf][row*BK+h*8], &A[(bm+row)*K + k0 + h*8]); }
        for(int c=tid;c<BCH;c+=blockDim.x){ int kr=c/(BN/16), nc=c%(BN/16);
            cp_async16(&Braw[buf][kr*BN+nc*16], &Bc[(k0+kr)*N + bn + nc*16]); } };
    #pragma unroll
    for(int i=0;i<ST-1;i++){ if(i<nsteps){ loadtile(i,i%ST); asm volatile("cp.async.commit_group;\n"); } }
    for(int s=0;s<nsteps;s++){
        int cur=s%ST, nx=s+ST-1;
        if(nx<nsteps){ loadtile(nx,nx%ST); asm volatile("cp.async.commit_group;\n"); }
        if(s==nsteps-1) asm volatile("cp.async.wait_group 0;\n"); else asm volatile("cp.async.wait_group %0;\n"::"n"(ST-2));
        __syncthreads();
        const uint8_t* Bs=Braw[cur];
        #pragma unroll
        for(int kk=0;kk<BK;kk+=16){
            uint32_t af[RM][4];
            #pragma unroll
            for(int i=0;i<RM;i++){ int mr=(wm*RM+i)*16;
                ldm_x4(af[i][0],af[i][1],af[i][2],af[i][3], smem_u32(&Araw[cur][(mr+(lane%16))*BK + kk + (lane/16)*8])); }
            #pragma unroll
            for(int j=0;j<RN;j++){ int nn=(wn*RN+j)*8+g;
                uint32_t b0=pack2b(dequant_w(Bs[(kk+t*2+0)*BN+nn],sbyte), dequant_w(Bs[(kk+t*2+1)*BN+nn],sbyte));
                uint32_t b1=pack2b(dequant_w(Bs[(kk+t*2+8)*BN+nn],sbyte), dequant_w(Bs[(kk+t*2+9)*BN+nn],sbyte));
                #pragma unroll
                for(int i=0;i<RM;i++)
                    mma_m16n8k16_bf16_f32(acc[i][j][0],acc[i][j][1],acc[i][j][2],acc[i][j][3], af[i][0],af[i][1],af[i][2],af[i][3], b0,b1);
            }
        }
        __syncthreads();
    }
    const float comp=0x1p119f;
    #pragma unroll
    for(int i=0;i<RM;i++) for(int j=0;j<RN;j++){
        int rb=bm+(wm*RM+i)*16, cb=bn+(wn*RN+j)*8;
        D[(rb+g  )*N+cb+t*2+0]=acc[i][j][0]*comp; D[(rb+g  )*N+cb+t*2+1]=acc[i][j][1]*comp;
        D[(rb+g+8)*N+cb+t*2+0]=acc[i][j][2]*comp; D[(rb+g+8)*N+cb+t*2+1]=acc[i][j][3]*comp; }
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
    gemm_pipe<<<grid,block>>>(dA,dBc,(uint8_t)SBYTE,dD);
    static float hD[M*N]; cudaMemcpy(hD,dD,sizeof hD,cudaMemcpyDeviceToHost);
    cudaError_t e=cudaDeviceSynchronize();
    int fails=0,checked=0;
    for(int m=0;m<M;m+=37) for(int n=0;n<N;n+=41){ float s=0;
        for(int k=0;k<K;k++) s+=__bfloat162float(hA[m*K+k])*fp4_val(hBc[k*N+n])*scale;
        checked++; if(fabsf(hD[m*N+n]-s)>1e-3f*(fabsf(s)+1)){ if(fails<6) printf("  MISS[m%d n%d] got=%g ref=%g\n",m,n,hD[m*N+n],s); fails++; } }
    cudaEvent_t t0,t1; cudaEventCreate(&t0); cudaEventCreate(&t1); int iters=200;
    for(int w=0;w<5;w++) gemm_pipe<<<grid,block>>>(dA,dBc,(uint8_t)SBYTE,dD);
    cudaEventRecord(t0); for(int it=0;it<iters;it++) gemm_pipe<<<grid,block>>>(dA,dBc,(uint8_t)SBYTE,dD);
    cudaEventRecord(t1); cudaEventSynchronize(t1); float ms=0; cudaEventElapsedTime(&ms,t0,t1);
    double flop=2.0*M*N*K*iters;
    printf("pipe(ST=%d) BM%d BN%d BK%d WM%d WN%d (RM%d RN%d w%d) %s | parity %d/%d | %.3f ms | %.1f GFLOP/s (peak 75.9k)\n",
           ST,BM,BN,BK,WM,WN,RM,RN,NWARPS, e==cudaSuccess?"ok":"ERR", checked-fails, checked, ms/iters, flop/(ms/1e3)/1e9);
    return fails?1:0;
}
