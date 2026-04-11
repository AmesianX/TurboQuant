// Force float accumulation for TBQ K types
#include "../common.cuh"
#undef V_DOT2_F32_F16_AVAILABLE

// TurboQuant 128-block: TBQP4_1 keys + Q8_0 values

#include "../fattn-vec.cuh"

DECL_FATTN_VEC_CASE(128, GGML_TYPE_TBQP4_1, GGML_TYPE_Q8_0);
