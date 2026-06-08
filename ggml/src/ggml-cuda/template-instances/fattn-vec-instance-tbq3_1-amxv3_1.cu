// Force float accumulation so the WHT path stays in fp32
#include "../common.cuh"
#undef V_DOT2_F32_F16_AVAILABLE

// tbq3_1 (K, +outlier) x amxv3_1 (V, plain 50B). Alias config: -ctk tbq3 -ctv amx3.

#include "../fattn-vec.cuh"

DECL_FATTN_VEC_CASE(128, GGML_TYPE_TBQ3_1, GGML_TYPE_AMXV3_1);
