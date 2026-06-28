#include "ref.h"
#include <mma.h>
#include <cudaTypedefs.h>
#include <cuda_fp8.h>
using namespace nvcuda;
#define KB 16
#define WARPS 8
#define NWMMA_M (HQ/16)
#define NWMMA_N (KB/16)
#define NWMMA_D (D/16)
#define TMA_W 256
#define NTILE (D/TMA_W)
extern __shared__ char smem[];
__device__ __forceinline__ void cp_gather4(uint32_t d,const CUtensorMap* tm,int c0,int r0,int r1,int r2,int r3,uint32_t b){
  asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes [%0],[%1,{%2,%3,%4,%5,%6}],[%7];"::"r"(d),"l"(tm),"r"(c0),"r"(r0),"r"(r1),"r"(r2),"r"(r3),"r"(b):"memory");}
__device__ __forceinline__ float f8(uint8_t v){__nv_fp8_e4m3 f;memcpy(&f,&v,1);return (float)f;}
// per-CTA query: idx is shared (same 512 for all CTAs in this bench), O per CTA
__global__ void __launch_bounds__(256) k4(const __grid_constant__ CUtensorMap tmK,const float* Q,const int* idx,const float* Kscale,float scale,float* Obase){
  const int t=threadIdx.x,warp=t>>5,lane=t&31; float* O=Obase+(size_t)blockIdx.x*HQ*D;
  __nv_bfloat16* sQ=(__nv_bfloat16*)smem;__nv_bfloat16* sK=sQ+HQ*D;__nv_bfloat16* sP=sK+KB*D;float* sS=(float*)(sP+HQ*KB);
  __shared__ float m_run[HQ],l_run[HQ];__shared__ float wscratch[WARPS][256];__shared__ alignas(128) uint8_t sStage[4*TMA_W];__shared__ alignas(8) uint64_t bar;
  for(int i=t;i<HQ*D;i+=256) sQ[i]=__float2bfloat16(Q[i]);
  for(int i=t;i<HQ*D;i+=256) O[i]=0.f; if(t<HQ){m_run[t]=-INFINITY;l_run[t]=0.f;}
  uint32_t bar_s=(uint32_t)__cvta_generic_to_shared(&bar);__syncthreads();
  for(int kb0=0;kb0<TOPK;kb0+=KB){
    for(int g=0;g<KB/4;g++)for(int ct=0;ct<NTILE;ct++){
      if(t==0){asm volatile("mbarrier.init.shared.b64 [%0],1;"::"r"(bar_s));asm volatile("fence.proxy.async.shared::cta;");
        uint32_t ds=(uint32_t)__cvta_generic_to_shared(sStage);int r0=idx[kb0+g*4],r1=idx[kb0+g*4+1],r2=idx[kb0+g*4+2],r3=idx[kb0+g*4+3];
        cp_gather4(ds,&tmK,ct*TMA_W,r0,r1,r2,r3,bar_s);asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;"::"r"(bar_s),"r"((uint32_t)(4*TMA_W)));
        asm volatile("{.reg .pred p;L:mbarrier.try_wait.parity.shared::cta.b64 p,[%0],0;@!p bra L;}"::"r"(bar_s));}
      __syncthreads();
      for(int e=t;e<4*TMA_W;e+=256){int i=e/TMA_W,c=e%TMA_W,dcol=ct*TMA_W+c,row=idx[kb0+g*4+i],blk=dcol/B128;sK[(g*4+i)*D+dcol]=__float2bfloat16(f8(sStage[e])*Kscale[(size_t)row*NBLK+blk]);}
      __syncthreads();}
    for(int tile=warp;tile<NWMMA_M*NWMMA_N;tile+=WARPS){int mt=tile/NWMMA_N,nt=tile%NWMMA_N;
      wmma::fragment<wmma::accumulator,16,16,16,float> acc;wmma::fill_fragment(acc,0.f);
      for(int dt=0;dt<NWMMA_D;dt++){wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major> a;wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::col_major> b;
        wmma::load_matrix_sync(a,sQ+(mt*16)*D+dt*16,D);wmma::load_matrix_sync(b,sK+(nt*16)*D+dt*16,D);wmma::mma_sync(acc,a,b,acc);}
      wmma::store_matrix_sync(sS+(mt*16)*KB+nt*16,acc,KB,wmma::mem_row_major);}
    __syncthreads();
    for(int h=warp;h<HQ;h+=WARPS){float* sr=sS+h*KB;float bmx=-INFINITY;for(int j=lane;j<KB;j+=32){float v=sr[j]*scale;if(v>bmx)bmx=v;}
      for(int o=16;o>0;o>>=1)bmx=fmaxf(bmx,__shfl_xor_sync(0xffffffff,bmx,o));float mo=m_run[h],mn=fmaxf(mo,bmx),corr=(mo==-INFINITY)?0.f:expf(mo-mn);
      float bs=0;for(int j=lane;j<KB;j+=32){float pp=expf(sr[j]*scale-mn);sP[h*KB+j]=__float2bfloat16(pp);bs+=pp;}for(int o=16;o>0;o>>=1)bs+=__shfl_xor_sync(0xffffffff,bs,o);
      if(lane==0){m_run[h]=mn;l_run[h]=l_run[h]*corr+bs;}for(int d=lane;d<D;d+=32)O[h*D+d]*=corr;}
    __syncthreads();
    for(int tile=warp;tile<NWMMA_M*NWMMA_D;tile+=WARPS){int mt=tile/NWMMA_D,dt=tile%NWMMA_D;
      wmma::fragment<wmma::accumulator,16,16,16,float> acc;wmma::fill_fragment(acc,0.f);
      for(int kt=0;kt<NWMMA_N;kt++){wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major> a;wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::row_major> b;
        wmma::load_matrix_sync(a,sP+(mt*16)*KB+kt*16,KB);wmma::load_matrix_sync(b,sK+(kt*16)*D+dt*16,D);wmma::mma_sync(acc,a,b,acc);}
      wmma::store_matrix_sync(wscratch[warp],acc,16,wmma::mem_row_major);for(int e=lane;e<256;e+=32){int rr=e/16,cc=e%16;O[(mt*16+rr)*D+dt*16+cc]+=wscratch[warp][e];}}
    __syncthreads();}
  for(int h=warp;h<HQ;h+=WARPS){float inv=1.0f/l_run[h];for(int d=lane;d<D;d+=32)O[h*D+d]*=inv;}
}
int main(){cudaSetDevice(0);Problem p=make_problem(16384,1234);
  float *dQ,*dO,*dKs;uint8_t* dK8;int* dI;int NQ=getenv("NQ")?atoi(getenv("NQ")):13000;
  cudaMalloc(&dQ,p.Q.size()*4);cudaMalloc(&dO,(size_t)NQ*HQ*D*4);cudaMalloc(&dK8,p.Kfp8.size());cudaMalloc(&dKs,p.Kscale.size()*4);cudaMalloc(&dI,TOPK*4);
  cudaMemcpy(dQ,p.Q.data(),p.Q.size()*4,cudaMemcpyHostToDevice);cudaMemcpy(dK8,p.Kfp8.data(),p.Kfp8.size(),cudaMemcpyHostToDevice);cudaMemcpy(dKs,p.Kscale.data(),p.Kscale.size()*4,cudaMemcpyHostToDevice);cudaMemcpy(dI,p.idx.data(),TOPK*4,cudaMemcpyHostToDevice);
  CUtensorMap tm{};uint64_t dims[2]={(uint64_t)D,(uint64_t)p.n_comp};uint64_t str[1]={(uint64_t)D};uint32_t box[2]={TMA_W,1};uint32_t es[2]={1,1};
  cuTensorMapEncodeTiled(&tm,CU_TENSOR_MAP_DATA_TYPE_UINT8,2,dK8,dims,str,box,es,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_NONE,CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  size_t smem=(size_t)HQ*D*2+KB*D*2+HQ*KB*2+HQ*KB*4;cudaFuncSetAttribute(k4,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem);
  k4<<<NQ,256,smem>>>(tm,dQ,dI,dKs,p.scale,dO);cudaDeviceSynchronize();
  cudaEvent_t a,b;cudaEventCreate(&a);cudaEventCreate(&b);cudaEventRecord(a);
  int IT=10;for(int i=0;i<IT;i++)k4<<<NQ,256,smem>>>(tm,dQ,dI,dKs,p.scale,dO);
  cudaEventRecord(b);cudaEventSynchronize(b);float ms;cudaEventElapsedTime(&ms,a,b);ms/=IT;
  printf("NQ=%d queries: %.3f ms/launch  => %.0f queries/s (sparse-MLA attn only, n_comp=16384)\n",NQ,ms,NQ/(ms/1000.0));
  printf("sync=%s\n",cudaGetErrorString(cudaDeviceSynchronize()));
  return 0;}
