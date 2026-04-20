// Force float accumulation so AMX polar path stays in fp32
#include "../common.cuh"
#undef V_DOT2_F32_F16_AVAILABLE

// AMX 128-block: AMX3_1 keys + F16 values.

#include "../fattn-vec.cuh"

DECL_FATTN_VEC_CASE(128, GGML_TYPE_AMX3_1, GGML_TYPE_F16);
