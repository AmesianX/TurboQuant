// Force float accumulation for TBQ K types
#include "../common.cuh"
#undef V_DOT2_F32_F16_AVAILABLE

// TurboQuant cross-head: TBQP3_3 keys + Q8_0 values

#include "../fattn-vec.cuh"

DECL_FATTN_VEC_CASE(64, GGML_TYPE_TBQP3_3, GGML_TYPE_Q8_0);
