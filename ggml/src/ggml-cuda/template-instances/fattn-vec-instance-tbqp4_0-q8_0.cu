// Force float accumulation for TBQ K types
#include "../common.cuh"
#undef V_DOT2_F32_F16_AVAILABLE

#include "../fattn-vec.cuh"
DECL_FATTN_VEC_CASE(256, GGML_TYPE_TBQP4_0, GGML_TYPE_Q8_0);
DECL_FATTN_VEC_CASE(512, GGML_TYPE_TBQP4_0, GGML_TYPE_Q8_0);
