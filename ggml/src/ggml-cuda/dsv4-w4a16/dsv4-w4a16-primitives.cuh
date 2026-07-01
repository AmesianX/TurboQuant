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

}  // namespace w4a16
}  // namespace dsv4
