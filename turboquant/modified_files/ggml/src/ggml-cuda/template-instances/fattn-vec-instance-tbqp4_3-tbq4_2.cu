// Force float accumulation for TBQ V types
#include "../common.cuh"
#undef V_DOT2_F32_F16_AVAILABLE

// TurboQuant double WHT per-head: TBQP4_3 keys + TBQ4_2 values

#include "../fattn-vec.cuh"

DECL_FATTN_VEC_CASE(64, GGML_TYPE_TBQP4_3, GGML_TYPE_TBQ4_2);
