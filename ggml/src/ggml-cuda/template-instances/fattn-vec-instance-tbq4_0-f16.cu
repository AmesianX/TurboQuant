// TurboQuant KV cache: TBQ4_0 keys + F16 values
// Only D=256 is valid (TBQ4_0 block size = QK_K = 256)

#include "../fattn-vec.cuh"

DECL_FATTN_VEC_CASE(256, GGML_TYPE_TBQ4_0, GGML_TYPE_F16);
DECL_FATTN_VEC_CASE(512, GGML_TYPE_TBQ4_0, GGML_TYPE_F16);
