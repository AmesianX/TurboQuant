// Force float accumulation for TBQ V types
#include "../common.cuh"
#undef V_DOT2_F32_F16_AVAILABLE

// Asymmetric: Q4_0 keys + TBQ4_1 values

#include "../fattn-vec.cuh"

DECL_FATTN_VEC_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_TBQ4_1);
