// Brick 3b (시공 first pour): single-tile W4A16 GEMM = dequant(B) + MMA + epilogue.
// D[16x8] = A[16x16] @ dequant(Bcode[16x8], scale_byte). B weights are FP4 (e2m1)
// with one e8m0 block scale; assembled from the VERIFIED primitives:
//   dequant -> bf16 smem tile (FP4 * 2^-119)  ->  MMA (brick 3a layout)  ->  *2^119.
// Verified against an fp32 CPU reference (A * FP4 * 2^(b-127)).
//   nvcc -arch=sm_121a test_w4a16_tile.cu -o /tmp/w && /tmp/w
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <cuda_bf16.h>
#include "dsv4-w4a16-primitives.cuh"

using namespace dsv4::w4a16;

__device__ __host__ float fp4_val(int c) {
    static const float m[8] = {0, .5f, 1, 1.5f, 2, 3, 4, 6};
    float v = m[c & 7]; return (c & 8) ? -v : v;
}

__device__ __forceinline__ uint32_t pack2(__nv_bfloat16 lo, __nv_bfloat16 hi) {
    unsigned short l = *reinterpret_cast<unsigned short *>(&lo);
    unsigned short h = *reinterpret_cast<unsigned short *>(&hi);
    return ((uint32_t)h << 16) | (uint32_t)l;
}

// One FP4 weight -> bf16 (value = FP4 * 2^(b-127) * 2^-119), via verified primitives.
__device__ __forceinline__ __nv_bfloat16 dequant_w(uint8_t code, uint8_t sbyte) {
    uint32_t e_lo, e_hi; dequant_e2m1x4_to_bf16x4((uint32_t)code << 12, e_lo, e_hi);
    uint32_t im = e_hi & 0xFFFFu;                       // FP4 * 2^-126
    uint32_t s_lo, s_hi; dequant_e8m0x4_to_bf16x4((uint32_t)sbyte, s_lo, s_hi);
    uint32_t sc = s_lo & 0xFFFFu;                       // 2^(b-120)
    uint32_t prod = mul_bf16x2(im, sc);                 // low lane = FP4 * 2^(b-127) * 2^-119
    unsigned short lo16 = prod & 0xFFFFu;
    return *reinterpret_cast<__nv_bfloat16 *>(&lo16);
}

__global__ void w4a16_tile(const __nv_bfloat16 * A, const uint8_t * Bcode, uint8_t sbyte, float * D) {
    int lane = threadIdx.x, g = lane >> 2, t = lane & 3;
    __shared__ __nv_bfloat16 Bt[16 * 8];
    for (int idx = lane; idx < 128; idx += 32) Bt[idx] = dequant_w(Bcode[idx], sbyte);
    __syncwarp();
    #define A_(r,c) A[(r)*16 + (c)]
    #define B_(r,c) Bt[(r)*8 + (c)]
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
    const float comp = 0x1p119f;   // epilogue: undo the 2^-119 weight offset
    D[(g  )*8 + t*2+0] = d0 * comp;
    D[(g  )*8 + t*2+1] = d1 * comp;
    D[(g+8)*8 + t*2+0] = d2 * comp;
    D[(g+8)*8 + t*2+1] = d3 * comp;
}

int main() {
    const int SBYTE = 130;                 // e8m0 scale => 2^(130-127) = 8
    __nv_bfloat16 hA[16*16]; uint8_t hBc[16*8]; float refD[16*8];
    for (int m = 0; m < 16; m++) for (int k = 0; k < 16; k++)
        hA[m*16+k] = __float2bfloat16((float)(((m + 2*k) % 5) - 2));
    for (int k = 0; k < 16; k++) for (int n = 0; n < 8; n++)
        hBc[k*8+n] = (uint8_t)((k*5 + n*3 + 1) & 0xF);   // all 16 e2m1 codes incl. 7 (+6) and 15 (-6)
    float scale = ldexpf(1.f, SBYTE - 127);
    for (int m = 0; m < 16; m++) for (int n = 0; n < 8; n++) {
        float s = 0;
        for (int k = 0; k < 16; k++)
            s += __bfloat162float(hA[m*16+k]) * fp4_val(hBc[k*8+n]) * scale;
        refD[m*8+n] = s;
    }
    __nv_bfloat16 *dA; uint8_t *dBc; float *dD;
    cudaMalloc(&dA, sizeof hA); cudaMalloc(&dBc, sizeof hBc); cudaMalloc(&dD, sizeof refD);
    cudaMemcpy(dA, hA, sizeof hA, cudaMemcpyHostToDevice);
    cudaMemcpy(dBc, hBc, sizeof hBc, cudaMemcpyHostToDevice);
    w4a16_tile<<<1, 32>>>(dA, dBc, (uint8_t)SBYTE, dD);
    float hD[16*8]; cudaMemcpy(hD, dD, sizeof hD, cudaMemcpyDeviceToHost);
    cudaError_t e = cudaDeviceSynchronize();
    int fails = 0;
    for (int i = 0; i < 128; i++) if (fabsf(hD[i] - refD[i]) > 1e-3f * (fabsf(refD[i]) + 1)) {
        if (fails < 6) printf("  MISS [m%d n%d] got=%g ref=%g\n", i/8, i%8, hD[i], refD[i]);
        fails++;
    }
    printf("w4a16 tile: %s | parity: %d/128 pass\n", cudaGetErrorString(e), 128 - fails);
    return fails ? 1 : 0;
}
