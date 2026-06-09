// Force float accumulation for TBQ V types
#include "../common.cuh"
#undef V_DOT2_F32_F16_AVAILABLE

// Asymmetric baseline: F16 keys + TBQV3_1 values (MSE plain V)

#include "../fattn-vec.cuh"

DECL_FATTN_VEC_CASE(128, GGML_TYPE_F16, GGML_TYPE_TBQV3_1);
