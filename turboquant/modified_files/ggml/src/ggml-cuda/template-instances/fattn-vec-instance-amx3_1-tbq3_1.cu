// Force float accumulation for TBQ V types
#include "../common.cuh"
#undef V_DOT2_F32_F16_AVAILABLE

// AMX 128-block: AMX3_1 keys + TBQ3_1 values.

#include "../fattn-vec.cuh"

DECL_FATTN_VEC_CASE(128, GGML_TYPE_AMX3_1, GGML_TYPE_TBQ3_1);
