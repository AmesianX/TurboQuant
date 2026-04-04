// TurboQuant 3-bit KV cache: TBQ3_0 keys + F16 values

#include "../fattn-vec.cuh"

DECL_FATTN_VEC_CASE(256, GGML_TYPE_TBQ3_0, GGML_TYPE_F16);
