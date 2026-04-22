// Force float accumulation so AMX polar path stays in fp32
#include "../common.cuh"
#undef V_DOT2_F32_F16_AVAILABLE

// AMX: 128-WHT + polar. K = AMX3_1, V = AMXV3_1 (tbq3_1 동치 레이아웃).

#include "../fattn-vec.cuh"

DECL_FATTN_VEC_CASE(128, GGML_TYPE_AMX3_1, GGML_TYPE_AMXV3_1);
