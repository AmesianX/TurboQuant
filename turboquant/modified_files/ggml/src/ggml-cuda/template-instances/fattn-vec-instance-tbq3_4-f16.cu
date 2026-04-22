// v1.5.2 precision fix: force float accumulation to prevent half2 precision loss
// being amplified by the 512-point IWHT butterfly (V = latent 512 for MLA models).
// Required for GLM-4.7-Flash / DeepSeek-V2/V3 asymmetric K=576/V=512 dispatch.
#include "../common.cuh"
#undef V_DOT2_F32_F16_AVAILABLE

// TurboQuant 576-block: TBQ3_4 keys + F16 values

#include "../fattn-vec.cuh"

DECL_FATTN_VEC_CASE(576, GGML_TYPE_TBQ3_4, GGML_TYPE_F16);
