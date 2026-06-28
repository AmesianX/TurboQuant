// DENSE baseline at the DSV4 shape: one CTA per query, dense QK^T over ALL n_visible comp keys
// (no top-k gather) -> softmax -> PV. This is what the current DSV4 path effectively does (dense
// flash-attn over the concat'd K with an (-inf) mask). Per-query cost grows with n_visible (the
// O(n^2) prefill droop). We compare the SPARSE kernel's per-query q/s against THIS at matched shape.
// For a fair "attention work" baseline we run dense over n_vis keys (= the visible comp window).
#include "ref.h"
#include <mma.h>
#include <cuda_fp8.h>
using namespace nvcuda;
#define WARPS 8
#define KBT 16                 // dense key-block (bigger ok, no gather)
extern __shared__ char smem[];
__device__ __forceinline__ float f8(uint8_t v){__nv_fp8_e4m3 f;memcpy(&f,&v,1);return (float)f;}

// dense over n_vis keys (rows 0..n_vis-1 of the FP8 cache). HQ heads, D dim. Q persistent bf16 (64KB)
// -> 1 CTA/SM (same as the all-heads sparse). This is the honest dense per-query comparison.
__global__ void __launch_bounds__(256) dense(
        const uint8_t* __restrict__ K8, const float* __restrict__ Kscale, int n_vis,
        const float* __restrict__ Q, float scale, float* __restrict__ Obase){
    const int t=threadIdx.x,warp=t>>5,lane=t&31; float* O=Obase+(size_t)blockIdx.x*HQ*D;
    __nv_bfloat16* sQ=(__nv_bfloat16*)smem;            // [HQ*D] 64KB
    __nv_bfloat16* sK=sQ+HQ*D;                          // [KBT*D] 64KB? -> 128KB over cap.
    // dense must also fit 99KB. Use KBT=16 like sparse for apples-to-apples smem. (dense pays many blocks.)
    __nv_bfloat16* sP=sK+KBT*D;
    float* sS=(float*)(sP+HQ*KBT);
    __shared__ float m_run[HQ],l_run[HQ]; __shared__ float wscratch[WARPS][256];
    for(int i=t;i<HQ*D;i+=256) sQ[i]=__float2bfloat16(Q[i]);
    for(int i=t;i<HQ*D;i+=256) O[i]=0.f; if(t<HQ){m_run[t]=-INFINITY;l_run[t]=0.f;}
    __syncthreads();
    for(int kb0=0; kb0<n_vis; kb0+=KBT){
        int kc=min(KBT,n_vis-kb0);
        for(int r=warp;r<kc;r+=WARPS){ int row=kb0+r; for(int d=lane;d<D;d+=32){int blk=d/B128; sK[r*D+d]=__float2bfloat16(f8(K8[(size_t)row*D+d])*Kscale[(size_t)row*NBLK+blk]);} }
        for(int r=kc;r<KBT;r++) for(int d=lane+warp*32; d<D; d+=256) sK[r*D+d]=__float2bfloat16(0.f);
        __syncthreads();
        for(int tile=warp;tile<(HQ/16)*(KBT/16);tile+=WARPS){int mt=tile/(KBT/16),nt=tile%(KBT/16);
            wmma::fragment<wmma::accumulator,16,16,16,float> acc;wmma::fill_fragment(acc,0.f);
            for(int dt=0;dt<D/16;dt++){wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major> a;wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::col_major> b;
                wmma::load_matrix_sync(a,sQ+(mt*16)*D+dt*16,D);wmma::load_matrix_sync(b,sK+(nt*16)*D+dt*16,D);wmma::mma_sync(acc,a,b,acc);}
            wmma::store_matrix_sync(sS+(mt*16)*KBT+nt*16,acc,KBT,wmma::mem_row_major);}
        __syncthreads();
        for(int h=warp;h<HQ;h+=WARPS){float* sr=sS+h*KBT;float bmx=-INFINITY;
            for(int j=lane;j<kc;j+=32){float v=sr[j]*scale;if(v>bmx)bmx=v;}
            for(int o=16;o>0;o>>=1)bmx=fmaxf(bmx,__shfl_xor_sync(0xffffffff,bmx,o));
            float mo=m_run[h],mn=fmaxf(mo,bmx),corr=(mo==-INFINITY)?0.f:expf(mo-mn);
            float bs=0;for(int j=lane;j<KBT;j+=32){float pp=(j<kc)?expf(sr[j]*scale-mn):0.f;sP[h*KBT+j]=__float2bfloat16(pp);bs+=pp;}
            for(int o=16;o>0;o>>=1)bs+=__shfl_xor_sync(0xffffffff,bs,o);
            if(lane==0){m_run[h]=mn;l_run[h]=l_run[h]*corr+bs;}
            for(int d=lane;d<D;d+=32)O[h*D+d]*=corr;}
        __syncthreads();
        for(int tile=warp;tile<(HQ/16)*(D/16);tile+=WARPS){int mt=tile/(D/16),dt=tile%(D/16);
            wmma::fragment<wmma::accumulator,16,16,16,float> acc;wmma::fill_fragment(acc,0.f);
            for(int kt=0;kt<KBT/16;kt++){wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major> a;wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::row_major> b;
                wmma::load_matrix_sync(a,sP+(mt*16)*KBT+kt*16,KBT);wmma::load_matrix_sync(b,sK+(kt*16)*D+dt*16,D);wmma::mma_sync(acc,a,b,acc);}
            wmma::store_matrix_sync(wscratch[warp],acc,16,wmma::mem_row_major);
            for(int e=lane;e<256;e+=32){int rr=e/16,cc=e%16;O[(mt*16+rr)*D+dt*16+cc]+=wscratch[warp][e];}}
        __syncthreads();
    }
    for(int h=warp;h<HQ;h+=WARPS){float inv=1.0f/l_run[h];for(int d=lane;d<D;d+=32)O[h*D+d]*=inv;}
}
int main(){
    cudaSetDevice(0);cudaDeviceProp pr;cudaGetDeviceProperties(&pr,0);
    int NC=getenv("NC")?atoi(getenv("NC")):16384; int NVIS=getenv("NVIS")?atoi(getenv("NVIS")):NC;
    int NQ=getenv("NQ")?atoi(getenv("NQ")):1;
    Problem p=make_problem(NC,1234);
    uint8_t* dK8;float* dKs,*dQ,*dO;
    cudaMalloc(&dK8,p.Kfp8.size());cudaMalloc(&dKs,p.Kscale.size()*4);cudaMalloc(&dQ,p.Q.size()*4);cudaMalloc(&dO,(size_t)NQ*HQ*D*4);
    cudaMemcpy(dK8,p.Kfp8.data(),p.Kfp8.size(),cudaMemcpyHostToDevice);cudaMemcpy(dKs,p.Kscale.data(),p.Kscale.size()*4,cudaMemcpyHostToDevice);cudaMemcpy(dQ,p.Q.data(),p.Q.size()*4,cudaMemcpyHostToDevice);
    size_t smem=(size_t)HQ*D*2+KBT*D*2+HQ*KBT*2+HQ*KBT*4;
    printf("DENSE smem=%.1fKB n_vis=%d NQ=%d\n",smem/1024.0,NVIS,NQ);
    cudaFuncSetAttribute(dense,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem);
    int mb;cudaOccupancyMaxActiveBlocksPerMultiprocessor(&mb,dense,256,smem);printf("occupancy=%d CTA/SM\n",mb);
    dense<<<NQ,256,smem>>>(dK8,dKs,NVIS,dQ,p.scale,dO);
    if(cudaDeviceSynchronize()){printf("sync FAIL %s\n",cudaGetErrorString(cudaGetLastError()));return 1;}
    cudaEvent_t a,b;cudaEventCreate(&a);cudaEventCreate(&b);cudaEventRecord(a);
    int IT=10;for(int i=0;i<IT;i++)dense<<<NQ,256,smem>>>(dK8,dKs,NVIS,dQ,p.scale,dO);
    cudaEventRecord(b);cudaEventSynchronize(b);float ms;cudaEventElapsedTime(&ms,a,b);ms/=IT;
    printf("DENSE n_vis=%d: %.3f ms/launch => %.0f queries/s\n",NVIS,ms,NQ/(ms/1000.0));
    return 0;
}
