// Force float accumulation so the WHT path stays in fp32
#include "../common.cuh"
#undef V_DOT2_F32_F16_AVAILABLE

// tbq3 set: tbq3_1 (K, +outlier) x tbqv3_1 (V, MSE plain 50B). Config: -ctk tbq3 -ctv tbqv3.

#include "../fattn-vec.cuh"

DECL_FATTN_VEC_CASE(128, GGML_TYPE_TBQ3_1, GGML_TYPE_TBQV3_1);
