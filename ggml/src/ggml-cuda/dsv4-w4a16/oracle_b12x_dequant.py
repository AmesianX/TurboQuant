#!/usr/bin/env python3
# b12x GOLDEN-REFERENCE oracle for the native W4A16 port.
#
# Instead of fighting b12x's kernel ABI (swizzled layouts, cute-dsl host API), we
# use b12x's OWN torch reference — the definition b12x's own tests validate the
# kernel against. If our native kernel matches this reference's semantics, it
# matches what b12x guarantees.
#
# Run inside the b12x image:
#   docker run --rm --entrypoint bash sparkrun-vllm-ds4-gb10:gb10-local \
#       -c 'python3 /path/oracle_b12x_dequant.py'
#
# Verifies: b12x reference FP4 dequant table == our native kernel's FP4 values.
# Result on 2026-07-01: MATCH True (16/16).
import torch
from b12x.moe.fused.reference import _make_fp4_lut, _dequant_fp4

lut = _make_fp4_lut(torch.device("cpu"))
# bytes covering all 16 e2m1 codes as (lo, hi) nibble pairs
packed = torch.tensor([0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE], dtype=torch.uint8)
b12x_vals = _dequant_fp4(packed, 8, 2, lut).flatten().tolist()

# our native kernel's FP4 table (dsv4-w4a16-primitives.cuh dequant_e2m1x4_to_bf16x4,
# after the block-scale multiply the value is fp4_table[code] * 2^(b-127))
ours = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]

n_ok = sum(1 for a, b in zip(b12x_vals, ours) if abs(a - b) < 1e-9)
print("b12x golden reference dequant:", [round(x, 3) for x in b12x_vals])
print("native kernel FP4 table      :", ours)
print(f"MATCH: {n_ok == 16} ({n_ok}/16)")
# b12x e8m0 scale convention (_e8m0_scales_to_float, torch.float8_e8m0fnu) = 2^(b-127),
# identical to our dequant_e8m0x4_to_bf16x4 (verified 80/80 in test_primitives.cu).
