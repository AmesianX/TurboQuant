// Force float accumulation for TBQ V types
#include "../common.cuh"
#undef V_DOT2_F32_F16_AVAILABLE

// TurboQuant 576-block: TBQP4_4 keys + TBQ4_4 values

#include "../fattn-vec.cuh"

DECL_FATTN_VEC_CASE(576, GGML_TYPE_TBQP4_4, GGML_TYPE_TBQ4_4);
