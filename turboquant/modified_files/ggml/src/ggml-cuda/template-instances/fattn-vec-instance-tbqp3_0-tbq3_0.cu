// Force float accumulation for TBQ V types to prevent half2 precision loss
// that causes non-determinism when amplified by IWHT butterfly + MoE FFN.
// Must include common.cuh first (which defines V_DOT2_F32_F16_AVAILABLE),
// then undef it before fattn-vec.cuh sees it.
#include "../common.cuh"
#undef V_DOT2_F32_F16_AVAILABLE

#include "../fattn-vec.cuh"
DECL_FATTN_VEC_CASE(256, GGML_TYPE_TBQP3_0, GGML_TYPE_TBQ3_0);
DECL_FATTN_VEC_CASE(512, GGML_TYPE_TBQP3_0, GGML_TYPE_TBQ3_0);
