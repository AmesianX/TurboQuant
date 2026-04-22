// v1.5.2 precision fix: force float accumulation to prevent half2 precision loss
// being amplified by the 512-point IWHT butterfly (V = latent 512 for MLA models).
// Required for GLM-4.7-Flash / DeepSeek-V2/V3 asymmetric K=576/V=512 dispatch.
#include "../common.cuh"
#undef V_DOT2_F32_F16_AVAILABLE

// GLM asymmetric: K=576 (TBQP3_4), V=512 (_0/f16/q8_0/same-type)

#include "../fattn-vec.cuh"

DECL_FATTN_VEC_CASE_ASYM(576, 512, GGML_TYPE_TBQP3_4, GGML_TYPE_TBQ3_0);
DECL_FATTN_VEC_CASE_ASYM(576, 512, GGML_TYPE_TBQP3_4, GGML_TYPE_TBQ4_0);
DECL_FATTN_VEC_CASE_ASYM(576, 512, GGML_TYPE_TBQP3_4, GGML_TYPE_F16);
DECL_FATTN_VEC_CASE_ASYM(576, 512, GGML_TYPE_TBQP3_4, GGML_TYPE_Q8_0);
DECL_FATTN_VEC_CASE_ASYM(576, 512, GGML_TYPE_TBQP3_4, GGML_TYPE_TBQP3_4);
