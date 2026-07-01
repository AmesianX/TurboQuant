// Brick 3a: single m16n8k16 bf16 MMA tile — isolate the fragment layout.
// D[16x8] = A[16x16] @ B[16x8], verified against an fp32 CPU reference.
// Values are small ints (bf16-exact) so comparison is exact.
//   nvcc -arch=sm_121a test_gemm_tile.cu -o /tmp/g && /tmp/g
#include <cstdint>
#include <cstdio>
#include <cuda_bf16.h>
#include "dsv4-w4a16-primitives.cuh"

using namespace dsv4::w4a16;

__device__ __forceinline__ uint32_t pack2(__nv_bfloat16 lo, __nv_bfloat16 hi) {
    unsigned short l = *reinterpret_cast<unsigned short *>(&lo);
    unsigned short h = *reinterpret_cast<unsigned short *>(&hi);
    return ((uint32_t)h << 16) | (uint32_t)l;
}

// Canonical PTX m16n8k16 (.f32.bf16.bf16.f32, row.col) fragment layout.
//   lane -> groupID g = lane>>2 (0..7), tid-in-group t = lane&3 (0..3)
__global__ void mma_tile(const __nv_bfloat16 * A, const __nv_bfloat16 * B, float * D) {
    int lane = threadIdx.x;
    int g = lane >> 2, t = lane & 3;
    #define A_(r,c) A[(r)*16 + (c)]     // 16x16 row-major
    #define B_(r,c) B[(r)*8  + (c)]     // 16x8, r=k, c=n
    uint32_t a0 = pack2(A_(g,   t*2+0), A_(g,   t*2+1));
    uint32_t a1 = pack2(A_(g+8, t*2+0), A_(g+8, t*2+1));
    uint32_t a2 = pack2(A_(g,   t*2+8), A_(g,   t*2+9));
    uint32_t a3 = pack2(A_(g+8, t*2+8), A_(g+8, t*2+9));
    uint32_t b0 = pack2(B_(t*2+0, g), B_(t*2+1, g));
    uint32_t b1 = pack2(B_(t*2+8, g), B_(t*2+9, g));
    #undef A_
    #undef B_
    float d0 = 0, d1 = 0, d2 = 0, d3 = 0;
    mma_m16n8k16_bf16_f32(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1);
    D[(g  )*8 + t*2+0] = d0;
    D[(g  )*8 + t*2+1] = d1;
    D[(g+8)*8 + t*2+0] = d2;
    D[(g+8)*8 + t*2+1] = d3;
}

int main() {
    __nv_bfloat16 hA[16*16], hB[16*8];
    float refD[16*8];
    for (int m = 0; m < 16; m++) for (int k = 0; k < 16; k++)
        hA[m*16 + k] = __float2bfloat16((float)(((m + 2*k) % 5) - 2));
    for (int k = 0; k < 16; k++) for (int n = 0; n < 8; n++)
        hB[k*8 + n] = __float2bfloat16((float)(((3*k + n) % 5) - 2));
    for (int m = 0; m < 16; m++) for (int n = 0; n < 8; n++) {
        float s = 0;
        for (int k = 0; k < 16; k++)
            s += __bfloat162float(hA[m*16+k]) * __bfloat162float(hB[k*8+n]);
        refD[m*8 + n] = s;
    }
    __nv_bfloat16 *dA, *dB; float *dD;
    cudaMalloc(&dA, sizeof hA); cudaMalloc(&dB, sizeof hB); cudaMalloc(&dD, sizeof refD);
    cudaMemcpy(dA, hA, sizeof hA, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, sizeof hB, cudaMemcpyHostToDevice);
    mma_tile<<<1, 32>>>(dA, dB, dD);
    float hD[16*8]; cudaMemcpy(hD, dD, sizeof hD, cudaMemcpyDeviceToHost);
    cudaError_t e = cudaDeviceSynchronize();
    int fails = 0;
    for (int i = 0; i < 16*8; i++) if (hD[i] != refD[i]) {
        if (fails < 6) printf("  MISS [m%d n%d] got=%g ref=%g\n", i/8, i%8, hD[i], refD[i]);
        fails++;
    }
    printf("mma tile: %s | fragment parity: %d/128 pass\n", cudaGetErrorString(e), 128 - fails);
    return fails ? 1 : 0;
}
