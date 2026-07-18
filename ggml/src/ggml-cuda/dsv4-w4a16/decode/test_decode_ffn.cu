// Complete faithful decode MoE FFN GEVM (m=1): FC1(gate/up) -> SwiGLU -> FC2(down) + top-k
// combine, fused in ONE kernel (single block -> __syncthreads is the FC1/FC2 barrier, the
// m==1 fast case of b12x's per-token barrier). Uses b12x's dot cores (bit-trick decode).
// Block scale = 1 here (FP4 values direct) to verify the FFN math; real e8m0 scales fold in
// via the same 2^k compensation as b12x's GEMM. Verified vs fp32 CPU reference.
//   nvcc -arch=sm_121a -O3 test_decode_ffn.cu -o /tmp/ffn && /tmp/ffn
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <cuda_fp16.h>
#include "faithful_micro_dots.cuh"
using namespace dsv4::w4a16::decode;

#define H 128     // hidden
#define I 64      // intermediate
#define E 2       // active experts (top-k)
#define KB_H (H/16)   // K-blocks for FC1 contraction (over H)
#define KB_I (I/16)   // K-blocks for FC2 contraction (over I)

__host__ __device__ float fp4_val(int c){ float m[8]={0,.5f,1,1.5f,2,3,4,6}; float v=m[c&7]; return (c&8)?-v:v; }

// Weights as FP4 codes packed 2/byte along the contraction dim.
// Wup/Wgate: [E][I][H] ; Wdown: [E][H][I]. Packed: rows of (dim/2) bytes.
// device dot over a contiguous FP4 row (len L, multiple of 16) vs f16 activations (smem).
__device__ float row_dot(const uint8_t* wrow, const uint32_t* xh, int Lblocks){
    float acc = 0.f;
    for(int b=0;b<Lblocks;b++){
        const uint32_t* w = reinterpret_cast<const uint32_t*>(wrow + b*8); // 8 bytes = 16 codes
        const uint32_t* x = xh + b*8;                                       // 8 f16x2 = 16 acts
        acc += fp4_dot8_sum_prescale(w[0],w[1], x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7]);
    }
    return ldexpf(acc, 14);   // compensate 2^-14 prescale -> true (scale=1 case)
}

__global__ void decode_ffn(const uint8_t* Wup, const uint8_t* Wgate, const uint8_t* Wdown,
                           const uint32_t* xin, const float* rw, float* out){
    int t = threadIdx.x;                 // 0..H-1 (block = H threads)
    __shared__ uint32_t xh[H/2];         // input activation f16x2 (H f16)
    __shared__ float act[E][I];          // SwiGLU output per expert
    __shared__ uint32_t acth[E][I/2];    // act as f16x2 for FC2 dots
    for(int i=t;i<H/2;i+=blockDim.x) xh[i]=xin[i];
    __syncthreads();
    // FC1 + SwiGLU: threads cover E*I (e,i) pairs
    for(int idx=t; idx<E*I; idx+=blockDim.x){
        int e=idx/I, i=idx%I;
        const uint8_t* up_row  = Wup  + (e*I + i)*(H/2);
        const uint8_t* gt_row  = Wgate+ (e*I + i)*(H/2);
        float up  = row_dot(up_row, xh, KB_H);
        float gate= row_dot(gt_row, xh, KB_H);
        float silu = gate / (1.f + expf(-gate));
        act[e][i] = silu * up;
    }
    __syncthreads();
    // pack act to f16x2 for FC2 dots
    for(int idx=t; idx<E*(I/2); idx+=blockDim.x){
        int e=idx/(I/2), j=idx%(I/2);
        __half lo=__float2half(act[e][2*j]), hi=__float2half(act[e][2*j+1]);
        uint16_t l=*(uint16_t*)&lo, h=*(uint16_t*)&hi; acth[e][j]=((uint32_t)h<<16)|l;
    }
    __syncthreads();
    // FC2 + top-k combine: threads cover H outputs
    for(int hh=t; hh<H; hh+=blockDim.x){
        float o=0.f;
        for(int e=0;e<E;e++){
            const uint8_t* dn_row = Wdown + (e*H + hh)*(I/2);
            o += rw[e] * row_dot(dn_row, acth[e], KB_I);
        }
        out[hh]=o;
    }
}

static int run_ffn_case(float xscale, float denfloor, float reltol, const char * name){
    // host data
    static uint8_t hWup[E*I*(H/2)], hWgate[E*I*(H/2)], hWdown[E*H*(I/2)];
    for(int i=0;i<(int)sizeof hWup;i++)  hWup[i]  =(uint8_t)((i*5+1)&0xFF);
    for(int i=0;i<(int)sizeof hWgate;i++)hWgate[i]=(uint8_t)((i*7+3)&0xFF);
    for(int i=0;i<(int)sizeof hWdown;i++)hWdown[i]=(uint8_t)((i*3+2)&0xFF);
    float x[H]; for(int i=0;i<H;i++) x[i]=0.5f*(((i%5)-2))*xscale;  // f16-exact (xscale = power of 2)
    uint32_t xh[H/2]; for(int i=0;i<H/2;i++){ __half l=__float2half(x[2*i]),h=__float2half(x[2*i+1]);
        uint16_t a=*(uint16_t*)&l,b=*(uint16_t*)&h; xh[i]=((uint32_t)b<<16)|a; }
    float rw[E]={0.6f,0.4f};

    // fp32 reference
    float ref[H];
    float ract[E][I];
    for(int e=0;e<E;e++) for(int i=0;i<I;i++){
        float up=0,gate=0;
        for(int k=0;k<H;k++){ int cu=(hWup[(e*I+i)*(H/2)+k/2]>>((k&1)*4))&0xF;
                              int cg=(hWgate[(e*I+i)*(H/2)+k/2]>>((k&1)*4))&0xF;
            up += fp4_val(cu)*__half2float(__float2half(x[k])); gate += fp4_val(cg)*__half2float(__float2half(x[k])); }
        float silu=gate/(1.f+expf(-gate)); ract[e][i]=silu*up;
    }
    for(int hh=0;hh<H;hh++){ float o=0;
        for(int e=0;e<E;e++){ float p=0;
            for(int n=0;n<I;n++){ int cd=(hWdown[(e*H+hh)*(I/2)+n/2]>>((n&1)*4))&0xF;
                p += fp4_val(cd)*__half2float(__float2half(ract[e][n])); }
            o += rw[e]*p; }
        ref[hh]=o; }

    uint8_t *dU,*dG,*dD; uint32_t* dx; float *drw,*dout;
    cudaMalloc(&dU,sizeof hWup); cudaMalloc(&dG,sizeof hWgate); cudaMalloc(&dD,sizeof hWdown);
    cudaMalloc(&dx,sizeof xh); cudaMalloc(&drw,sizeof rw); cudaMalloc(&dout,H*4);
    cudaMemcpy(dU,hWup,sizeof hWup,cudaMemcpyHostToDevice); cudaMemcpy(dG,hWgate,sizeof hWgate,cudaMemcpyHostToDevice);
    cudaMemcpy(dD,hWdown,sizeof hWdown,cudaMemcpyHostToDevice); cudaMemcpy(dx,xh,sizeof xh,cudaMemcpyHostToDevice);
    cudaMemcpy(drw,rw,sizeof rw,cudaMemcpyHostToDevice);
    decode_ffn<<<1,H>>>(dU,dG,dD,dx,drw,dout);
    float got[H]; cudaMemcpy(got,dout,H*4,cudaMemcpyDeviceToHost);
    cudaError_t e=cudaDeviceSynchronize();
    cudaFree(dU); cudaFree(dG); cudaFree(dD); cudaFree(dx); cudaFree(drw); cudaFree(dout);
    int fails=0; float maxrel=0;
    for(int hh=0;hh<H;hh++){ float r=fabsf(got[hh]-ref[hh])/(fabsf(ref[hh])+denfloor); maxrel=fmaxf(maxrel,r);
        if(r>reltol){ if(fails<5) printf("  MISS[%d] got=%.6g ref=%.6g\n",hh,got[hh],ref[hh]); fails++; } }
    printf("decode FFN %-10s (FC1->SwiGLU->FC2): %s | %d/%d pass | maxrel=%.4f\n",
           name, cudaGetErrorString(e), H-fails, H, maxrel);
    return fails?1:0;
}

int main(){
    int fails = 0;
    // Case 1: O(1) activations (the original gate).
    fails += run_ffn_case(1.0f, 1e-3f, 2e-2f, "O(1)");
    // Case 2: x scaled by 2^-12 -> SwiGLU outputs ~1e-3 feed FC2, whose prescaled products sit
    // at the fp16 subnormal floor. Catches any regression to an f16 accumulation chain, which
    // flushed/quantized exactly this regime. Denominator floor scaled down so the check has teeth.
    fails += run_ffn_case(0.000244140625f, 1e-5f, 2e-2f, "2^-12");
    return fails?1:0;
}
