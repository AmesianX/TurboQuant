// FAITHFUL mirror of b12x's B-fragment staging methods (NOT my invention).
// Line-by-line port of b12x/moe/fused/w4a16/kernel.py:
//   _scaled_dequant_b_fragment (@ ~2219 region), bfloat2_broadcast_lane, _elem2_mul.
// This is the register-staged batched dequant that keeps b12x's MMA fed: one packed
// weight word q (8 FP4 codes) + one packed scale word s -> the full 8-value B fragment
// with only 2 dequant primitive calls (vs my earlier 4 wasteful per-code calls).
#pragma once
#include <cstdint>
#include <cuda_bf16.h>
#include "dsv4-w4a16-primitives.cuh"

namespace dsv4 { namespace w4a16 {

// b12x bfloat2_broadcast_lane(s, lane): replicate the bf16 at `lane` (0 or 1) of the
// bfloat2 register `s` into both halves.
__device__ __forceinline__ uint32_t bfloat2_broadcast_lane(uint32_t s, int lane){
    unsigned short h = (lane == 0) ? (unsigned short)(s & 0xFFFFu) : (unsigned short)(s >> 16);
    return ((uint32_t)h << 16) | (uint32_t)h;
}

// b12x _scaled_dequant_b_fragment(frag, q, s):
//   bq1 = q; bq0 = q << 8
//   (b0_0,b0_1) = dequant_e2m1x4(bq0);  (b1_0,b1_1) = dequant_e2m1x4(bq1)
//   scale by broadcast lanes 0/1, write frag[2][2].
// frag layout: frag[0][*] from bq0*scale0, frag[1][*] from bq1*scale1.
__device__ __forceinline__ void scaled_dequant_b_fragment(uint32_t frag[2][2], uint32_t q, uint32_t s){
    uint32_t bq1 = q;
    uint32_t bq0 = q << 8;
    uint32_t b0_0, b0_1, b1_0, b1_1;
    dequant_e2m1x4_to_bf16x4(bq0, b0_0, b0_1);   // 4 codes -> 2 bf16-pairs
    dequant_e2m1x4_to_bf16x4(bq1, b1_0, b1_1);   // 4 codes -> 2 bf16-pairs
    uint32_t s0 = bfloat2_broadcast_lane(s, 0);
    uint32_t s1 = bfloat2_broadcast_lane(s, 1);
    frag[0][0] = mul_bf16x2(b0_0, s0);
    frag[0][1] = mul_bf16x2(b0_1, s0);
    frag[1][0] = mul_bf16x2(b1_0, s1);
    frag[1][1] = mul_bf16x2(b1_1, s1);
}

}} // namespace
