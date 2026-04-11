// Force float accumulation for TBQ K types
#include "../common.cuh"
#undef V_DOT2_F32_F16_AVAILABLE

// TurboQuant 576-block: TBQ3_4 keys + Q8_0 values

#include "../fattn-vec.cuh"

DECL_FATTN_VEC_CASE(576, GGML_TYPE_TBQ3_4, GGML_TYPE_Q8_0);
