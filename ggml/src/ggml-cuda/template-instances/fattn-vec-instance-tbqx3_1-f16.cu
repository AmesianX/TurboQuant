// Force float accumulation so polar path stays in fp32
#include "../common.cuh"
#undef V_DOT2_F32_F16_AVAILABLE

// TurboQuant Polar Derotate 128-block: TBQX3_1 keys + F16 values

#include "../fattn-vec.cuh"

DECL_FATTN_VEC_CASE(128, GGML_TYPE_TBQX3_1, GGML_TYPE_F16);
