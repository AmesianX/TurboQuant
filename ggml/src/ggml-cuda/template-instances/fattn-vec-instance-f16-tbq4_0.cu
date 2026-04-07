// Force float accumulation for TBQ V types
#include "../common.cuh"
#undef V_DOT2_F32_F16_AVAILABLE

// Asymmetric: F16 keys + TBQ4_0 values

#include "../fattn-vec.cuh"

DECL_FATTN_VEC_CASE(256, GGML_TYPE_F16, GGML_TYPE_TBQ4_0);
