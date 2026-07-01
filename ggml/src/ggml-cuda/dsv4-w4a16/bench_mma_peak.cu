#include <cstdio>
#include <cuda_bf16.h>
#include "dsv4-w4a16-primitives.cuh"
using namespace dsv4::w4a16;
__global__ void peak(float* out, int iters){
    float d0=0,d1=0,d2=0,d3=0;
    uint32_t a0=1,a1=2,a2=3,a3=4,b0=5,b1=6;
    for(int i=0;i<iters;i++){
        mma_m16n8k16_bf16_f32(d0,d1,d2,d3,a0,a1,a2,a3,b0,b1);
        mma_m16n8k16_bf16_f32(d0,d1,d2,d3,a0,a1,a2,a3,b0,b1);
        a0^=__float_as_uint(d0); b1^=__float_as_uint(d3);  // dep chain, prevent hoist
    }
    if(threadIdx.x==999) out[0]=d0+d1+d2+d3;
}
int main(){
    float* o; cudaMalloc(&o,4);
    int blocks=48*8, threads=256, iters=20000;   // fill GPU
    for(int w=0;w<3;w++) peak<<<blocks,threads>>>(o,iters);
    cudaEvent_t t0,t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0); peak<<<blocks,threads>>>(o,iters); cudaEventRecord(t1);
    cudaEventSynchronize(t1); float ms; cudaEventElapsedTime(&ms,t0,t1);
    double warps=(double)blocks*threads/32;
    double flop=warps*iters*2*(2.0*16*8*16);   // 2 mmas/iter, 4096 flop each
    printf("bf16 MMA peak (register-only): %.1f TFLOP/s (%.2f ms)\n", flop/(ms/1e3)/1e12, ms);
    printf("our GEMM 18.8 TFLOP/s = %.0f%% of this peak\n", 18.8/(flop/(ms/1e3)/1e12)*100);
    return 0;
}
