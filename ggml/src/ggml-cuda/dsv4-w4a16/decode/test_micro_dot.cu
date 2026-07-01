// Parity for the faithful (bit-trick) decode dot: 16 FP4 . 16 f16 == f32 ref, after 2^14 comp.
//   nvcc -arch=sm_121a -O3 test_micro_dot.cu -o /tmp/md && /tmp/md
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <cuda_fp16.h>
#include "faithful_micro_dots.cuh"
using namespace dsv4::w4a16::decode;

__host__ float fp4_val(int c){ static const float m[8]={0,.5f,1,1.5f,2,3,4,6}; float v=m[c&7]; return (c&8)?-v:v; }

__global__ void run(const uint32_t* wpk, const uint32_t* xh, float* out){
    out[0] = fp4_dot8_sum_prescale(wpk[0], wpk[1], xh[0],xh[1],xh[2],xh[3],xh[4],xh[5],xh[6],xh[7]);
}

int main(){
    int codes[16]; float acts[16];
    for(int i=0;i<16;i++){ codes[i]=(i%7)+1; acts[i]=(float)((i%3)+1); }  // non-cancelling, nonzero
    // pack codes: byte j = (codes[2j] low nibble | codes[2j+1] high nibble)
    uint32_t wpk[2]={0,0};
    for(int j=0;j<8;j++){ uint32_t b=(codes[2*j]&0xF)|((codes[2*j+1]&0xF)<<4); wpk[j/4]|=b<<((j%4)*8); }
    // pack acts: f16x2 word i = (acts[2i] low, acts[2i+1] high)
    uint32_t xh[8];
    for(int i=0;i<8;i++){ __half l=__float2half(acts[2*i]), h=__float2half(acts[2*i+1]);
        uint16_t lo=*(uint16_t*)&l, hi=*(uint16_t*)&h; xh[i]=((uint32_t)hi<<16)|lo; }
    float ref=0; for(int i=0;i<16;i++) ref += fp4_val(codes[i]) * acts[i];

    uint32_t *dw,*dx; float* dout;
    cudaMalloc(&dw,8); cudaMalloc(&dx,32); cudaMalloc(&dout,4);
    cudaMemcpy(dw,wpk,8,cudaMemcpyHostToDevice); cudaMemcpy(dx,xh,32,cudaMemcpyHostToDevice);
    run<<<1,1>>>(dw,dx,dout);
    float got_prescale; cudaMemcpy(&got_prescale,dout,4,cudaMemcpyDeviceToHost);
    cudaError_t e=cudaDeviceSynchronize();
    float got = ldexpf(got_prescale, 14);   // compensate the 2^-14 (caller folds into scale)
    printf("fp4_dot8 (bit-trick): %s | got=%.3f ref=%.3f | %s\n", cudaGetErrorString(e), got, ref,
           fabsf(got-ref)<1e-2f*(fabsf(ref)+1) ? "MATCH" : "MISMATCH");
    return fabsf(got-ref)<1e-2f*(fabsf(ref)+1) ? 0 : 1;
}
