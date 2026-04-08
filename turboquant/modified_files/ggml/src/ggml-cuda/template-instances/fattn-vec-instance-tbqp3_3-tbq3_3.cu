// Force float accumulation for TBQ V types
#include "../common.cuh"
#undef V_DOT2_F32_F16_AVAILABLE

// TurboQuant cross-head: GGML_TYPE_TBQP3_3 keys + GGML_TYPE_TBQ3_3 values

#include "../fattn-vec.cuh"

DECL_FATTN_VEC_CASE(64, GGML_TYPE_TBQP3_3, GGML_TYPE_TBQ3_3);
