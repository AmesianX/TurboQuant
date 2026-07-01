// Standalone compile-check + FP4 dequant parity for the W4A16 foundational primitives.
//   nvcc -arch=sm_121a test_primitives.cu -o /tmp/t && /tmp/t
// (falls back to any sm_80+ arch for a pure syntax check — the MMA needs sm_80+).
#include <cstdint>
#include <cstdio>
#include <cuda_bf16.h>
#include "dsv4-w4a16-primitives.cuh"

using namespace dsv4::w4a16;

// Reference NVFP4 (e2m1) magnitude table, indexed by the 3-bit exp|mant field.
__device__ __host__ float fp4_e2m1_value(int code4) {
    static const float mag[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    float v = mag[code4 & 0x7];
    return (code4 & 0x8) ? -v : v;
}

// Dequant every 4-bit e2m1 code and report the bf16 the primitive produces.
__global__ void dequant_dump(float * out) {
    int c = threadIdx.x;            // 0..15
    if (c >= 16) return;
    uint32_t packed = (uint32_t)c << 12;   // code in nibble [12:15] of low lane
    uint32_t lo, hi;
    dequant_e2m1x4_to_bf16x4(packed, lo, hi);
    // pass-1 (hi = out1) handles the [12:15] nibble; low bf16 of the low lane holds our code
    __nv_bfloat16 b = *reinterpret_cast<__nv_bfloat16 *>(&hi);
    out[c] = __bfloat162float(b);
}

// Trivial MMA compile-check (identity-ish accumulate, correctness not asserted here).
__global__ void mma_smoke(float * acc) {
    float d0 = acc[0], d1 = acc[1], d2 = acc[2], d3 = acc[3];
    uint32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0, b0 = 0, b1 = 0;
    mma_m16n8k16_bf16_f32(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1);
    acc[0] = d0; acc[1] = d1; acc[2] = d2; acc[3] = d3;
}

// Full dequant chain: e2m1 weight (code c) x e8m0 scale (byte b) -> bf16, which
// equals true_weight * 2^-119 (epilogue restores). Grid over (c, b).
__global__ void dequant_full(const int * codes, const int * bytes, int n, float * out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    // e2m1 intermediate for code c -> lands in hi(out1) low bf16
    uint32_t e_lo, e_hi;
    dequant_e2m1x4_to_bf16x4((uint32_t)codes[i] << 12, e_lo, e_hi);
    uint32_t im = e_hi & 0xFFFFu;            // FP4(c) * 2^-126 in low lane
    // e8m0 scale for byte b -> lands in lo(h0) low bf16
    uint32_t s_lo, s_hi;
    dequant_e8m0x4_to_bf16x4((uint32_t)bytes[i], s_lo, s_hi);
    uint32_t sc = s_lo & 0xFFFFu;            // 2^(b-120) in low lane
    uint32_t prod = mul_bf16x2(im, sc);      // low lane = FP4(c) * 2^(b-127) * 2^-119
    __nv_bfloat16 pb = *reinterpret_cast<__nv_bfloat16 *>(&prod);
    out[i] = __bfloat162float(pb);
}

int main() {
    float *d_out; cudaMalloc(&d_out, 16 * sizeof(float));
    dequant_dump<<<1, 16>>>(d_out);
    float h[16]; cudaMemcpy(h, d_out, sizeof h, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // NOTE: the e2m1 bit-trick emits a PRE-SCALE intermediate (exponent bias is
    // deferred to the block-scale multiply, b12x kernel.py:710). So the raw output is
    // NOT the FP4 table value; only the SIGN is meaningful here. True value-parity comes
    // after the scale-dequant primitive is ported and multiplied in. Real fidelity oracle
    // = bit-exact vs b12x's packed_dequant_e2m1x4_to_bfloat2x2 output (TODO next step).
    int sign_fails = 0;
    printf("code | raw bf16 (pre-scale) | sign ok\n");
    for (int c = 0; c < 16; c++) {
        bool want_neg = (c & 0x8);
        bool got_neg  = (h[c] < 0.f) || (1.f / h[c] < 0.f);  // catch -0.0
        bool sign_ok  = (want_neg == got_neg);
        sign_fails += !sign_ok;
        printf(" %2d  |      %9.3g       | %s\n", c, h[c], sign_ok ? "yes" : "NO");
    }
    int fails = sign_fails;  // gate on sign + compile/run only, at this stage
    float *d_acc; cudaMalloc(&d_acc, 4 * sizeof(float));
    float acc0[4] = {1, 2, 3, 4}; cudaMemcpy(d_acc, acc0, sizeof acc0, cudaMemcpyHostToDevice);
    mma_smoke<<<1, 32>>>(d_acc);
    cudaError_t e = cudaDeviceSynchronize();
    printf("mma smoke: %s\n", cudaGetErrorString(e));

    // ---- full dequant chain parity: (e2m1 x e8m0) * 2^119 == FP4 * 2^(b-127) ----
    // Realistic block-scale bytes. NOTE: the 2^-119 intermediate means the product
    // underflows bf16 (->0) once true_weight < ~2^-7 (e.g. b=100 => 2^-146, below the
    // bf16 min denormal 2^-133). That is inherent to b12x's design (same bf16), not a
    // port bug; trained-model block scales sit near b=127. Test the representable range.
    const int bs[] = {120, 124, 127, 130, 134};   // normal e8m0 bytes (no inf/nan)
    int hc[16 * 5], hb[16 * 5], n = 0;
    for (int c = 0; c < 16; c++) for (int bi = 0; bi < 5; bi++) { hc[n] = c; hb[n] = bs[bi]; n++; }
    int *dc, *db; float *dfo;
    cudaMalloc(&dc, n * sizeof(int)); cudaMalloc(&db, n * sizeof(int)); cudaMalloc(&dfo, n * sizeof(float));
    cudaMemcpy(dc, hc, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(db, hb, n * sizeof(int), cudaMemcpyHostToDevice);
    dequant_full<<<(n + 63) / 64, 64>>>(dc, db, n, dfo);
    float hfo[16 * 5]; cudaMemcpy(hfo, dfo, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    int full_fails = 0;
    for (int i = 0; i < n; i++) {
        float restored = ldexpf(hfo[i], 119);                       // undo the 2^-119
        float ref      = fp4_e2m1_value(hc[i]) * ldexpf(1.f, hb[i] - 127);
        if (restored != ref) { full_fails++;
            if (full_fails <= 4) printf("  MISS c=%d b=%d got=%g ref=%g\n", hc[i], hb[i], restored, ref); }
    }
    printf("full-chain parity: %d/%d pass\n", n - full_fails, n);

    printf("dequant sign gate: %d/16 pass\n", 16 - fails);
    return (fails || full_fails) ? 1 : 0;
}
