#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#define NROW_TOTAL 16384
typedef unsigned short bf16;
__global__ void gatherBench(const __grid_constant__ CUtensorMap tmap, const int* idx, bf16* out, int niter) {
    __shared__ alignas(128) bf16 tile[4*64];
    __shared__ alignas(8) uint64_t bar;
    uint32_t dst_s=(uint32_t)__cvta_generic_to_shared(tile), bar_s=(uint32_t)__cvta_generic_to_shared(&bar);
    bf16 acc=0;
    for(int it=0; it<niter; it++){
      for(int g=0; g<128; g++){
        if(threadIdx.x==0){
          asm volatile("mbarrier.init.shared.b64 [%0],1;"::"r"(bar_s));
          asm volatile("fence.proxy.async.shared::cta;");
          int b=g*4; int r0=idx[b],r1=idx[b+1],r2=idx[b+2],r3=idx[b+3],c0=0;
          asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes "
            "[%0],[%1,{%2,%3,%4,%5,%6}],[%7];"::"r"(dst_s),"l"(&tmap),"r"(c0),"r"(r0),"r"(r1),"r"(r2),"r"(r3),"r"(bar_s):"memory");
          asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;"::"r"(bar_s),"r"((uint32_t)(4*64*2)));
          asm volatile("{.reg .pred p;L:mbarrier.try_wait.parity.shared::cta.b64 p,[%0],0;@!p bra L;}"::"r"(bar_s));
        }
        __syncthreads();
        acc ^= tile[threadIdx.x&255];
      }
    }
    if(threadIdx.x<256) out[threadIdx.x]=acc;
}
int main(){
  cudaSetDevice(0);
  bf16* d; cudaMalloc(&d,(size_t)NROW_TOTAL*64*2); cudaMemset(d,1,(size_t)NROW_TOTAL*64*2);
  CUtensorMap tm{}; uint64_t dims[2]={64,NROW_TOTAL}; uint64_t str[1]={64*2}; uint32_t box[2]={64,1}; uint32_t es[2]={1,1};
  CUresult r=cuTensorMapEncodeTiled(&tm,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,d,dims,str,box,es,
    CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_NONE,CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if(r){const char*s;cuGetErrorString(r,&s);printf("encode FAIL %s\n",s);return 2;}
  int hidx[512]; for(int i=0;i<512;i++) hidx[i]=(i*131)%NROW_TOTAL;
  int* didx; cudaMalloc(&didx,512*4); cudaMemcpy(didx,hidx,512*4,cudaMemcpyHostToDevice);
  bf16* dout; cudaMalloc(&dout,256*2);
  // launch enough CTAs to fill the GPU (proxy for n_tokens queries): 132 SMs
  cudaEvent_t a,b; cudaEventCreate(&a);cudaEventCreate(&b);
  int NITER=200, CTAS=1000;
  gatherBench<<<CTAS,128>>>(tm,didx,dout,2); cudaDeviceSynchronize();
  cudaEventRecord(a); gatherBench<<<CTAS,128>>>(tm,didx,dout,NITER); cudaEventRecord(b); cudaEventSynchronize(b);
  float ms; cudaEventElapsedTime(&ms,a,b);
  double gathers = (double)CTAS*NITER*128;
  printf("sync: %s\n", cudaGetErrorString(cudaDeviceSynchronize()));
  printf("%d CTAs x %d iters x 128 gather4 = %.2e gather4 ops in %.2f ms\n", CTAS,NITER,gathers,ms);
  printf("per gather4 (amortized over %d CTAs): %.2f ns; per 512-row gather: %.2f us\n", CTAS, ms*1e6/gathers, ms*1e3/(CTAS*NITER));
  return 0;
}
