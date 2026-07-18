// FAITHFUL decode GEVM dot core (b12x micro.py fp4_dot8_sum), ptxas-portable variant.
//
// b12x's original uses `cvt.rn.f16x2.e2m1x2` (VERIFIED: works on GB10 only via cute-dsl's
// MLIR/NVVM backend; raw ptxas 13.0/13.2 rejects it on sm_121/120/100 — proven by testing +
// b12x's own e2e test passing under cute-dsl). For an nvcc/ptxas C++ port we substitute
// b12x's OWN e2m1->f16 bit-trick (from its GEMM path packed_dequant_e2m1x4_to_half2x2) — same
// author, same numeric values, only the decode instruction differs. Algorithm (FP4 GEVM dot,
// f16x2 fma chain, lo+hi->f32 reduce) is b12x's verbatim.
//
// e2m1 bit-trick emits a PRE-SCALE fp16 (FP4 * 2^-14, exp bias deferred) exactly like the GEMM
// path (which folds the constant into the block/global scale). The dot returns true_dot * 2^-14;
// the caller folds 2^14 into the per-expert scale (as b12x folds its GEMM 2^119).
//
// ACCUMULATION IS FP32, NOT the b12x f16 fma chain. At 2^-14 prescale a product of a prescaled
// weight and an O(1e-3) activation sits at ~2^-24 -- the very bottom of fp16's subnormal range --
// so an f16 chain quantizes it to 1-3 significand bits or flushes it to zero. That is safe only
// at TRUE scale (b12x's cvt.rn.f16x2.e2m1x2 decode) or with bf16's 8-bit exponent (the GEMM
// path). Decoding at prescale forces the accumulator up to fp32; the GEVM is bandwidth-bound
// (~125 GB/s vs 17 TFLOP/s idle ALU), so the extra FFMA lanes are free.
#pragma once
#include <cstdint>
#include <cuda_fp16.h>

namespace dsv4 { namespace w4a16 { namespace decode {

// b12x e2m1->fp16 bit-trick (half2 variant, shr 3). One byte (2 FP4 nibbles n0=low,n1=high)
// -> f16x2 = (prescale_fp16(n0), prescale_fp16(n1)). Place n0 at bits[12:15] of low lane,
// n1 at bits[12:15] of high lane, then: out = (in&0x80008000) | ((in&0x70007000)>>3).
__device__ __forceinline__ uint32_t e2m1_byte_to_f16x2_prescale(uint32_t byte) {
    uint32_t in = ((byte & 0xFu) << 12) | ((byte >> 4) << 28);
    return (in & 0x80008000u) | ((in & 0x70007000u) >> 3);
}

// b12x fp4_dot8_sum (16-elem block dot), bit-trick decode. u_a,u_b = 8 FP4 bytes (16 codes);
// x0..x7 = 8 f16x2 (16 acts). Returns dot * 2^-14 (caller compensates via scale).
__device__ __forceinline__ float fp4_dot8_sum_prescale(
        uint32_t u_a, uint32_t u_b,
        uint32_t x0, uint32_t x1, uint32_t x2, uint32_t x3,
        uint32_t x4, uint32_t x5, uint32_t x6, uint32_t x7) {
    const uint32_t xs[8] = {x0,x1,x2,x3,x4,x5,x6,x7};
    const uint32_t us[8] = {u_a & 0xFFu, (u_a>>8)&0xFFu, (u_a>>16)&0xFFu, (u_a>>24)&0xFFu,
                            u_b & 0xFFu, (u_b>>8)&0xFFu, (u_b>>16)&0xFFu, (u_b>>24)&0xFFu};
    float acc_lo = 0.f, acc_hi = 0.f;   // fp32: prescaled products underflow an f16 chain
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        uint32_t hbits = e2m1_byte_to_f16x2_prescale(us[i]);
        __half2 h = *reinterpret_cast<__half2*>(&hbits);
        float2 hf = __half22float2(h);   // exact: prescaled fp16 values are representable
        float2 xf = __half22float2(*reinterpret_cast<const __half2*>(&xs[i]));
        acc_lo = fmaf(hf.x, xf.x, acc_lo);
        acc_hi = fmaf(hf.y, xf.y, acc_hi);
    }
    return acc_lo + acc_hi;   // lo + hi reduce (b12x shape, fp32 chain)
}

// b12x fp4_dot4_sum (FC2 8-elem dot): u = 4 FP4 bytes (8 codes); x0..x3 = 4 f16x2 (8 acts).
// Returns dot * 2^-14 (caller compensates). Bit-trick decode (b12x-faithful).
__device__ __forceinline__ float fp4_dot4_sum_prescale(
        uint32_t u, uint32_t x0, uint32_t x1, uint32_t x2, uint32_t x3) {
    const uint32_t xs[4] = {x0,x1,x2,x3};
    const uint32_t us[4] = {u & 0xFFu, (u>>8)&0xFFu, (u>>16)&0xFFu, (u>>24)&0xFFu};
    float acc_lo = 0.f, acc_hi = 0.f;   // fp32: see fp4_dot8_sum_prescale
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        uint32_t hb = e2m1_byte_to_f16x2_prescale(us[i]);
        __half2 h = *reinterpret_cast<__half2*>(&hb);
        float2 hf = __half22float2(h);
        float2 xf = __half22float2(*reinterpret_cast<const __half2*>(&xs[i]));
        acc_lo = fmaf(hf.x, xf.x, acc_lo);
        acc_hi = fmaf(hf.y, xf.y, acc_hi);
    }
    return acc_lo + acc_hi;
}

// b12x fp4_dot8_dual_sum: fused up+gate, shared activations, 2 accumulator chains.
// up = dot(up_w, x), gate = dot(gate_w, x). Both * 2^-14.
__device__ __forceinline__ void fp4_dot8_dual_sum_prescale(
        float & up, float & gate,
        uint32_t up_a, uint32_t up_b, uint32_t gate_a, uint32_t gate_b,
        uint32_t x0, uint32_t x1, uint32_t x2, uint32_t x3,
        uint32_t x4, uint32_t x5, uint32_t x6, uint32_t x7) {
    up   = fp4_dot8_sum_prescale(up_a,  up_b,  x0,x1,x2,x3,x4,x5,x6,x7);
    gate = fp4_dot8_sum_prescale(gate_a,gate_b,x0,x1,x2,x3,x4,x5,x6,x7);
}

// b12x warp_reduce: butterfly sum over 32 lanes (log2(32)=5 shuffle steps).
__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

}}} // namespace
