// Force float accumulation for TBQ K types
#include "../common.cuh"
#undef V_DOT2_F32_F16_AVAILABLE

// TurboQuant cross-head: TBQ4_3 keys + F16 values

#include "../fattn-vec.cuh"

DECL_FATTN_VEC_CASE(64, GGML_TYPE_TBQ4_3, GGML_TYPE_F16);
