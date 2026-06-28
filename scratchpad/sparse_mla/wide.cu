#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
typedef unsigned short bf16;
#define NROW 64
#define W 256   // try 256 bf16 = 512 bytes per gathered row
__global__ void k(const __grid_constant__ CUtensorMap tm, bf16* out){
  __shared__ alignas(128) bf16 tile[4*W];
  __shared__ alignas(8) uint64_t bar;
  uint32_t ds=(uint32_t)__cvta_generic_to_shared(tile), bs=(uint32_t)__cvta_generic_to_shared(&bar);
  if(threadIdx.x==0){
    asm volatile("mbarrier.init.shared.b64 [%0],1;"::"r"(bs));
    asm volatile("fence.proxy.async.shared::cta;");
    int c0=0,r0=0,r1=5,r2=10,r3=20;
    asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes "
      "[%0],[%1,{%2,%3,%4,%5,%6}],[%7];"::"r"(ds),"l"(&tm),"r"(c0),"r"(r0),"r"(r1),"r"(r2),"r"(r3),"r"(bs):"memory");
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;"::"r"(bs),"r"((uint32_t)(4*W*2)));
    asm volatile("{.reg .pred p;L:mbarrier.try_wait.parity.shared::cta.b64 p,[%0],0;@!p bra L;}"::"r"(bs));
  }
  __syncthreads();
  if(threadIdx.x<4) out[threadIdx.x]=tile[threadIdx.x*W]; // first elem of each gathered row
}
int main(){
  cudaSetDevice(0);
  bf16* d; cudaMalloc(&d,(size_t)NROW*W*2);
  bf16 h[NROW*W]; for(int r=0;r<NROW;r++) for(int c=0;c<W;c++) h[r*W+c]=(bf16)(r*1000+c);
  cudaMemcpy(d,h,sizeof(h),cudaMemcpyHostToDevice);
  CUtensorMap tm{}; uint64_t dims[2]={W,NROW}; uint64_t str[1]={W*2}; uint32_t box[2]={W,1}; uint32_t es[2]={1,1};
  CUresult r=cuTensorMapEncodeTiled(&tm,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,d,dims,str,box,es,
    CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_NONE,CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if(r){const char*s;cuGetErrorString(r,&s);printf("encode(W=%d) FAIL: %s\n",W,s);return 2;}
  bf16* o; cudaMalloc(&o,8);
  k<<<1,128>>>(tm,o); cudaError_t e=cudaDeviceSynchronize();
  printf("W=%d (%d bytes/row) gather4: %s\n",W,W*2,cudaGetErrorString(e));
  if(!e){bf16 ho[4];cudaMemcpy(ho,o,8,cudaMemcpyDeviceToHost);
    printf("gathered rows {0,5,10,20} first elem: %d %d %d %d (expect 0 5000 10000 20000)\n",ho[0],ho[1],ho[2],ho[3]);}
  return 0;
}
