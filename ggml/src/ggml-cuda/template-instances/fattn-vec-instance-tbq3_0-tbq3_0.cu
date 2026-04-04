// TurboQuant 3-bit K + 3-bit V (full TBQ compression)

#include "../fattn-vec.cuh"

DECL_FATTN_VEC_CASE(256, GGML_TYPE_TBQ3_0, GGML_TYPE_TBQ3_0);
DECL_FATTN_VEC_CASE(512, GGML_TYPE_TBQ3_0, GGML_TYPE_TBQ3_0);
