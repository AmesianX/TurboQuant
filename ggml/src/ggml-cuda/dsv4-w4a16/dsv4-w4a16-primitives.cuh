// DSV4 W4A16 fused-MoE — foundational device primitives (native port).
//
// Faithful CUDA C++ translation of b12x's cute-dsl primitives. These are inline-PTX:
// the PTX is byte-identical to the reference; only the host wrapper changes
// (MLIR `llvm.inline_asm`  ->  CUDA `asm volatile`). This is the proof-of-concept
// brick showing the port is a source->source translation, not a reimplementation.
//
// Source of truth (extracted at ~/work/dsv4-serve/build/b12x):
//   - bf16 warp MMA:  b12x/cute/fp4.py :: bf16_mma_m16n8k16_f32          @ line 2271
//   - FP4 unpack:      b12x/cute/fp4.py :: packed_dequant_e2m1x4_to_bfloat2x2 @ 1963
//
// See turboquant/DSV4_W4A16_NATIVE_PORT.md for the full plan and parity gates.
#pragma once

#include <cstdint>

namespace dsv4 {
namespace w4a16 {

// ---------------------------------------------------------------------------
// FE2M1 (FP4) x4  ->  BF16 x4   branchless register dequant.
//
// Port of `packed_dequant_e2m1x4_to_bfloat2x2` (b12x/cute/fp4.py:1963).
// `packed` holds four e2m1 values in the top nibbles [12:15] of two 16-bit lanes;
// pass 1 (`out1`) converts those nibbles, pass 2 (`out2`) shifts the next nibbles
// [8:11] up and converts them. Each converts by keeping the sign bit (mask
// 0x80008000) and repositioning the 3-bit e2m1 field (mask 0x70007000) into the
// bf16 exponent slot (>>6). Result: two bfloat2 (packed as uint32) = 4 bf16.
//
// NOTE: return convention matches the reference exactly (lo = out2, hi = out1).
__device__ __forceinline__ void dequant_e2m1x4_to_bf16x4(
        uint32_t packed, uint32_t & lo, uint32_t & hi) {
    uint32_t o_lo, o_hi;  // %0 <- out2, %1 <- out1  (reference ordering)
    asm volatile(
        "{\n"
        "  .reg .b32 q, o1, o2, t;\n"
        "  and.b32 o1, %2, 0x80008000;\n"   // sign bits, nibble @ [12:15]
        "  and.b32 t,  %2, 0x70007000;\n"   // e2m1 3-bit field
        "  shr.u32 t,  t, 6;\n"             // -> bf16 exponent slot
        "  or.b32  o1, o1, t;\n"
        "  shl.b32 q,  %2, 4;\n"            // bring nibble @ [8:11] up to [12:15]
        "  and.b32 o2, q, 0x80008000;\n"
        "  and.b32 t,  q, 0x70007000;\n"
        "  shr.u32 t,  t, 6;\n"
        "  or.b32  o2, o2, t;\n"
        "  mov.b32 %0, o2;\n"
        "  mov.b32 %1, o1;\n"
        "}\n"
        : "=r"(o_lo), "=r"(o_hi)
        : "r"(packed));
    lo = o_lo;
    hi = o_hi;
}

// ---------------------------------------------------------------------------
// Warp MMA:  mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
//
// Port of `bf16_mma_m16n8k16_f32` (b12x/cute/fp4.py:2271). Standard Ampere+
// tensor-core MMA (CuTe atom SM80_16x8x16_F32BF16BF16F32_TN). A = 4 regs (8 bf16),
// B = 2 regs (4 bf16), accumulator D = 4 f32 (in-place: read as C, written as D).
__device__ __forceinline__ void mma_m16n8k16_bf16_f32(
        float & d0, float & d1, float & d2, float & d3,
        uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
        uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32\n"
        "  {%0, %1, %2, %3},\n"
        "  {%4, %5, %6, %7},\n"
        "  {%8, %9},\n"
        "  {%10, %11, %12, %13};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "f"(d0), "f"(d1), "f"(d2), "f"(d3));
}

// ---------------------------------------------------------------------------
// E8M0 (power-of-2 block scale) x4  ->  BF16 x4, materialized as (scale * 2^7).
//
// Port of `packed_dequant_e8m0x4_to_bfloat2x2` (b12x/cute/fp4.py:2117). `packed`
// holds 4 e8m0 bytes {b0,b1,b2,b3}. Each byte b -> bf16 exponent field (b+7)<<7,
// i.e. value 2^((b+7)-127) = 2^(b-120) = true_scale(2^(b-127)) * 2^7. Special:
// b>=248 -> +inf (0x7f80), b==255 -> nan (0x7fc0). Packing (interleaved to match
// the e2m1 fragment lane order): lo = bf16(h0,h2), hi = bf16(h1,h3).
//
// The 2^7 here + the e2m1 unpack's 2^-126 leave the MMA inputs scaled by 2^-119;
// the kernel epilogue multiplies the f32 accumulator by 2^119 to compensate.
__device__ __forceinline__ void dequant_e8m0x4_to_bf16x4(
        uint32_t packed, uint32_t & lo, uint32_t & hi) {
    uint32_t o_lo, o_hi;
    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  .reg .u32 b0,b1,b2,b3, h0,h1,h2,h3, t0,t1;\n"
        "  and.b32 b0, %2, 0x000000ff;\n"
        "  shr.u32 b1, %2, 8;  and.b32 b1, b1, 0x000000ff;\n"
        "  shr.u32 b2, %2, 16; and.b32 b2, b2, 0x000000ff;\n"
        "  shr.u32 b3, %2, 24;\n"
        "  add.u32 h0, b0, 7; add.u32 h1, b1, 7; add.u32 h2, b2, 7; add.u32 h3, b3, 7;\n"
        "  shl.b32 h0, h0, 7; shl.b32 h1, h1, 7; shl.b32 h2, h2, 7; shl.b32 h3, h3, 7;\n"
        "  setp.ge.u32 p, b0, 248; selp.b32 h0, 0x00007f80, h0, p;\n"
        "  setp.ge.u32 p, b1, 248; selp.b32 h1, 0x00007f80, h1, p;\n"
        "  setp.ge.u32 p, b2, 248; selp.b32 h2, 0x00007f80, h2, p;\n"
        "  setp.ge.u32 p, b3, 248; selp.b32 h3, 0x00007f80, h3, p;\n"
        "  setp.eq.u32 p, b0, 255; selp.b32 h0, 0x00007fc0, h0, p;\n"
        "  setp.eq.u32 p, b1, 255; selp.b32 h1, 0x00007fc0, h1, p;\n"
        "  setp.eq.u32 p, b2, 255; selp.b32 h2, 0x00007fc0, h2, p;\n"
        "  setp.eq.u32 p, b3, 255; selp.b32 h3, 0x00007fc0, h3, p;\n"
        "  shl.b32 t0, h2, 16; or.b32 %0, h0, t0;\n"
        "  shl.b32 t1, h3, 16; or.b32 %1, h1, t1;\n"
        "}\n"
        : "=r"(o_lo), "=r"(o_hi)
        : "r"(packed));
    lo = o_lo;
    hi = o_hi;
}

// Element-wise BF16x2 multiply (mul.bf16x2). Port of `bfloat2_mul`
// (b12x/cute/fp4.py). Used to combine the e2m1 weight fragment with its
// e8m0 block scale: weight_bf16x2 = mul_bf16x2(e2m1_frag, e8m0_scale_frag).
__device__ __forceinline__ uint32_t mul_bf16x2(uint32_t a, uint32_t b) {
    uint32_t d;
    asm volatile("mul.bf16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
    return d;
}

}  // namespace w4a16
}  // namespace dsv4
