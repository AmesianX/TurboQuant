#pragma once

#include "common.cuh"
#include "convert.cuh"
#include "vecdotq.cuh"

#include <cstdint>

#define FATTN_KQ_STRIDE       256
#define HALF_MAX_HALF         __float2half(65504.0f/2) // Use neg. of this instead of -INFINITY to initialize KQ max vals to avoid NaN upon subtraction.
#define SOFTMAX_FTZ_THRESHOLD -20.0f                   // Softmax exp. of values smaller than this are flushed to zero to avoid NaNs.

// log(2) = 0.6931, by adding this to the KQ maximum used for the softmax the numerical range representable
//     by the VKQ accumulators is effectively being shifted up by a factor of 2.
// This reduces issues with numerical overflow but also causes larger values to be flushed to zero.
// However, as the output from FlashAttention will usually be used as an input for a matrix multiplication this should be negligible.
// Still, the value range should be shifted as much as necessary but as little as possible.
// The macro on the following line shifts it by a factor of 2**3=8, as was needed to fix https://github.com/ggml-org/llama.cpp/issues/18606 .
#define FATTN_KQ_MAX_OFFSET (3.0f*0.6931f)

typedef void (* fattn_kernel_t)(
        const char * __restrict__ Q,
        const char * __restrict__ K,
        const char * __restrict__ V,
        const char * __restrict__ mask,
        const char * __restrict__ sinks,
        const int  * __restrict__ KV_max,
        float      * __restrict__ dst,
        float2     * __restrict__ dst_meta,
        const float scale,
        const float max_bias,
        const float m0,
        const float m1,
        const uint32_t n_head_log2,
        const float logit_softcap,
        const int32_t ne00, const uint3   ne01, const int32_t ne02, const int32_t ne03,
                            const int32_t nb01, const int32_t nb02, const int32_t nb03,
        const int32_t ne10, const int32_t ne11, const int32_t ne12, const int32_t ne13,
                            const int32_t nb11, const int32_t nb12, const int64_t nb13,
                            const int32_t nb21, const int32_t nb22, const int64_t nb23,
                            const int32_t ne31, const int32_t ne32, const int32_t ne33,
                            const int32_t nb31, const int32_t nb32, const int64_t nb33,
        const char * __restrict__ raw_K_data,
        const int32_t raw_K_stride,
        const char * __restrict__ Q_wht2_data,
        const int32_t Q_wht2_stride,
        const char * __restrict__ k_rope_data,      // TurboQuant: MLA _4 rope slice (f16). nullptr unless src[5] set.
        const int32_t k_rope_stride);               // bytes per K cell row in k_rope tensor

typedef float (*vec_dot_KQ_t)(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8 , const void * __restrict__ Q_ds);

template <int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_f16(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8 , const void * __restrict__ Q_ds_v) {

    const half2 * K_h2 = (const half2 *) K_c;
    GGML_UNUSED(Q_q8);
    GGML_UNUSED(Q_ds_v);

    constexpr int cpy_nb = ggml_cuda_get_max_cpy_bytes();
    constexpr int cpy_ne = cpy_nb / 4;

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < D/2; k_KQ_0 += nthreads*cpy_ne) {
        __align__(16) half2 tmp[cpy_ne];
        ggml_cuda_memcpy_1<sizeof(tmp)>(tmp, K_h2 + k_KQ_0 + (threadIdx.x % nthreads)*cpy_ne);
#pragma unroll
        for (int k_KQ_1 = 0; k_KQ_1 < cpy_ne; ++k_KQ_1) {
#ifdef V_DOT2_F32_F16_AVAILABLE
            ggml_cuda_mad(sum,                tmp[k_KQ_1] , ((const half2  *) Q_v)[k_KQ_0/nthreads + k_KQ_1]);
#else
            ggml_cuda_mad(sum, __half22float2(tmp[k_KQ_1]), ((const float2 *) Q_v)[k_KQ_0/nthreads + k_KQ_1]);
#endif // V_DOT2_F32_F16_AVAILABLE
        }
    }

    return sum;
}

template <int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_bf16(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8 , const void * __restrict__ Q_ds_v) {

    const nv_bfloat162 * K_bf16 = (const nv_bfloat162 *) K_c;
    GGML_UNUSED(Q_q8);
    GGML_UNUSED(Q_ds_v);

    constexpr int cpy_nb = ggml_cuda_get_max_cpy_bytes();
    constexpr int cpy_ne = cpy_nb / 4;

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < D/2; k_KQ_0 += nthreads*cpy_ne) {
        __align__(16) nv_bfloat162 tmp[cpy_ne];
        ggml_cuda_memcpy_1<sizeof(tmp)>(tmp, K_bf16 + k_KQ_0 + (threadIdx.x % nthreads)*cpy_ne);
#pragma unroll
        for (int k_KQ_1 = 0; k_KQ_1 < cpy_ne; ++k_KQ_1) {
#ifdef V_DOT2_F32_F16_AVAILABLE
            // FIXME replace macros in vector FA kernel with templating and use FP32 for BF16
            ggml_cuda_mad(sum, ggml_cuda_cast<float2>(tmp[k_KQ_1]), __half22float2(((const half2 *) Q_v)[k_KQ_0/nthreads + k_KQ_1]));
#else
            ggml_cuda_mad(sum, ggml_cuda_cast<float2>(tmp[k_KQ_1]), ((const float2 *) Q_v)[k_KQ_0/nthreads + k_KQ_1]);
#endif // V_DOT2_F32_F16_AVAILABLE
        }
    }

    return sum;
}

template<int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_q4_0(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {

    const block_q4_0 * K_q4_0 = (const block_q4_0 *) K_c;
    GGML_UNUSED(Q_v);

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < int(D/sizeof(int)); k_KQ_0 += nthreads) {
        const int k_KQ = k_KQ_0 + (nthreads == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads);

        const int ib    = k_KQ /  QI8_1;
        const int iqs4  = k_KQ %  QI4_0;
        const int shift = k_KQ & (QI8_1/2);

        int v;
        ggml_cuda_memcpy_1<sizeof(int), 2>(&v, K_q4_0[ib].qs + sizeof(int)*iqs4);
        v = (v >> shift) & 0x0F0F0F0F;
        const int u = Q_q8[k_KQ_0/nthreads];

        const int sumi = ggml_cuda_dp4a(v, u, 0);

        const float2 Q_ds = ((const float2 *) Q_ds_v)[k_KQ_0/nthreads];
        sum += __half2float(K_q4_0[ib].d) * (sumi*Q_ds.x - (8/QI8_1)*Q_ds.y);
    }

    return sum;
}

template<int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_q4_1(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {

    const block_q4_1 * K_q4_1 = (const block_q4_1 *) K_c;
    GGML_UNUSED(Q_v);

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < int(D/sizeof(int)); k_KQ_0 += nthreads) {
        const int k_KQ = k_KQ_0 + (nthreads == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads);

        const int ib    = k_KQ /  QI8_1;
        const int iqs4  = k_KQ %  QI4_1;
        const int shift = k_KQ & (QI8_1/2);

        int v;
        ggml_cuda_memcpy_1<sizeof(int)>(&v, K_q4_1[ib].qs + sizeof(int)*iqs4);
        v = (v >> shift) & 0x0F0F0F0F;
        const int u = Q_q8[k_KQ_0/nthreads];

        const int sumi = ggml_cuda_dp4a(v, u, 0);

        const float2 K_dm = __half22float2(K_q4_1[ib].dm);
        const float2 Q_ds = ((const float2 *) Q_ds_v)[k_KQ_0/nthreads];

        sum += K_dm.x*Q_ds.x*sumi + K_dm.y*Q_ds.y/QI8_1;
    }

    return sum;
}

template<int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_q5_0(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {

    const block_q5_0 * K_q5_0 = (const block_q5_0 *) K_c;
    GGML_UNUSED(Q_v);

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < int(D/sizeof(int)); k_KQ_0 += nthreads) {
        const int k_KQ = k_KQ_0 + (nthreads == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads);

        const int ib    = k_KQ /  QI8_1;
        const int iqs4  = k_KQ %  QI5_0;
        const int iqs8  = k_KQ %  QI8_1;
        const int shift = k_KQ & (QI8_1/2);

        int v;
        ggml_cuda_memcpy_1<sizeof(int), 2>(&v, K_q5_0[ib].qs + sizeof(int)*iqs4);
        v = (v >> shift) & 0x0F0F0F0F;

        {
            int vh;
            ggml_cuda_memcpy_1<sizeof(int), 2>(&vh, K_q5_0[ib].qh);
            vh >>= iqs8 * QI5_0;

            v |= (vh <<  4) & 0x00000010; // 0 ->  4
            v |= (vh << 11) & 0x00001000; // 1 -> 12
            v |= (vh << 18) & 0x00100000; // 2 -> 20
            v |= (vh << 25) & 0x10000000; // 3 -> 28
        }

        const int u = Q_q8[k_KQ_0/nthreads];

        const int sumi = ggml_cuda_dp4a(v, u, 0);

        const float2 Q_ds = ((const float2 *) Q_ds_v)[k_KQ_0/nthreads];

        sum += __half2float(K_q5_0[ib].d) * (sumi*Q_ds.x - (16/QI8_1)*Q_ds.y);
    }

    return sum;
}

template<int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_q5_1(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {

    const block_q5_1 * K_q5_1 = (const block_q5_1 *) K_c;
    GGML_UNUSED(Q_v);

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < int(D/sizeof(int)); k_KQ_0 += nthreads) {
        const int k_KQ = k_KQ_0 + (nthreads == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads);

        const int ib    = k_KQ /  QI8_1;
        const int iqs4  = k_KQ %  QI5_1;
        const int iqs8  = k_KQ %  QI8_1;
        const int shift = k_KQ & (QI8_1/2);

        int v;
        ggml_cuda_memcpy_1<sizeof(int)>(&v, K_q5_1[ib].qs + sizeof(int)*iqs4);
        v = (v >> shift) & 0x0F0F0F0F;

        {
            int vh;
            ggml_cuda_memcpy_1<sizeof(int)>(&vh, K_q5_1[ib].qh);
            vh >>= iqs8 * QI5_0;

            v |= (vh <<  4) & 0x00000010; // 0 ->  4
            v |= (vh << 11) & 0x00001000; // 1 -> 12
            v |= (vh << 18) & 0x00100000; // 2 -> 20
            v |= (vh << 25) & 0x10000000; // 3 -> 28
        }

        const int u = Q_q8[k_KQ_0/nthreads];

        const int sumi = ggml_cuda_dp4a(v, u, 0);

        const float2 K_dm = __half22float2(K_q5_1[ib].dm);
        const float2 Q_ds = ((const float2 *) Q_ds_v)[k_KQ_0/nthreads];

        sum += K_dm.x*Q_ds.x*sumi + K_dm.y*Q_ds.y/QI8_1;
    }

    return sum;
}

template <int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_q8_0(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {

    const block_q8_0 * K_q8_0 = (const block_q8_0 *) K_c;
    GGML_UNUSED(Q_v);

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < int(D/sizeof(int)); k_KQ_0 += nthreads) {
        const int k_KQ = k_KQ_0 + (nthreads == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads);

        const int ib  = k_KQ / QI8_0;
        const int iqs = k_KQ % QI8_0;

        int v;
        ggml_cuda_memcpy_1<sizeof(v), 2>(&v, K_q8_0[ib].qs + 4*iqs);

        const float2 * Q_ds = (const float2 *) Q_ds_v;
        const float Q_d = Q_ds[k_KQ_0/nthreads].x;

        sum += vec_dot_q8_0_q8_1_impl<float, 1>(&v, &Q_q8[k_KQ_0/nthreads], K_q8_0[ib].d, Q_d);
    }

    return sum;
}

template <typename Tds, int ni>
static __device__ __forceinline__ void quantize_q8_1_to_shared(
    const float * __restrict__ x, const float scale, int * __restrict__ yq32, void * __restrict__ yds) {

    float vals[sizeof(int)] = {0.0f};
#pragma unroll
    for (int l = 0; l < int(sizeof(int)); ++l) {
        vals[l] = (ni == WARP_SIZE || threadIdx.x < ni) ? scale * x[4*threadIdx.x + l] : 0.0f;
    }

    float amax = fabsf(vals[0]);
    float sum  = vals[0];
#pragma unroll
    for (int l = 1; l < int(sizeof(int)); ++l) {
        amax = fmaxf(amax, fabsf(vals[l]));
        sum += vals[l];
    }
#pragma unroll
    for (int mask = QI8_1/2; mask > 0; mask >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, mask, 32));
        sum +=             __shfl_xor_sync(0xFFFFFFFF, sum,  mask, 32);
    }

    const float d = amax / 127;
    int q32 = 0;
    int8_t * q8 = (int8_t *) &q32;

    if (d != 0.0f) {
#pragma unroll
        for (int l = 0; l < int(sizeof(int)); ++l) {
            q8[l] = roundf(vals[l] / d);
        }
    }

    yq32[threadIdx.x] = q32;
    if (threadIdx.x % QI8_1 == 0 && (ni == WARP_SIZE || threadIdx.x < ni)) {
        if (std::is_same<Tds, half2>::value) {
            ((half2  *) yds)[threadIdx.x/QI8_1] =  make_half2(d, sum);
        } else {
            ((float2 *) yds)[threadIdx.x/QI8_1] = make_float2(d, sum);
        }
    }
}

typedef void (*dequantize_V_t)(const void *, void *, const int64_t);

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_f16(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    if constexpr (std::is_same_v<T, half>) {
        ggml_cuda_memcpy_1<ne*sizeof(half)>(dst, (const half *) vx + i0);
    } else if constexpr (std::is_same_v<T, float>) {
        static_assert(ne % 2 == 0, "bad ne");
        __align__(16) half2 tmp[ne/2];
        ggml_cuda_memcpy_1<ne*sizeof(half)>(tmp, (const half *) vx + i0);
        float2 * dst_f2 = (float2 *) dst;
#pragma unroll
        for (int l = 0; l < ne/2; ++l) {
            dst_f2[l] = __half22float2(tmp[l]);
        }
    } else {
        static_assert(std::is_same_v<T, void>, "unsupported type");
    }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_bf16(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    static_assert(std::is_same_v<T, float>, "BF16 V dequantization only supports float output");
    static_assert(ne % 2 == 0, "bad ne");
    __align__(16) nv_bfloat162 tmp[ne/2];
    ggml_cuda_memcpy_1<ne*sizeof(nv_bfloat16)>(tmp, (const nv_bfloat16 *) vx + i0);
    float2 * dst_f2 = (float2 *) dst;
#pragma unroll
    for (int l = 0; l < ne/2; ++l) {
        dst_f2[l] = ggml_cuda_cast<float2>(tmp[l]);
    }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_q4_0(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_q4_0 * x = (const block_q4_0 *) vx;

    const int64_t ib    =  i0          /  QK4_0;
    const int     iqs   =  i0          % (QK4_0/2);
    const int     shift = (i0 % QK4_0) / (QK4_0/2);

    int q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_cuda_memcpy_1<ne, 2>(&q, x[ib].qs + iqs);
    q >>= 4*shift;
    q &= 0x0F0F0F0F;
    q = __vsubss4(q, 0x08080808);

    const int8_t * q8 = (const int8_t *) &q;

#ifdef FP16_AVAILABLE
    if constexpr (std::is_same_v<T, half>) {
        const half2 d = __half2half2(x[ib].d);

#pragma unroll
        for (int l0 = 0; l0 < ne; l0 += 2) {
            ((half2 *) dst)[l0/2] = d * make_half2(q8[l0 + 0], q8[l0 + 1]);
        }
    } else
#endif // FP16_AVAILABLE
    if constexpr (std::is_same_v<T, float>) {
        const float d = x[ib].d;

#pragma unroll
        for (int l = 0; l < ne; ++l) {
            ((float *) dst)[l] = d * q8[l];
        }
    } else {
        static_assert(std::is_same_v<T, void>, "bad type");
    }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_q4_1(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_q4_1 * x = (const block_q4_1 *) vx;

    const int64_t ib    =  i0          /  QK4_1;
    const int     iqs   =  i0          % (QK4_1/2);
    const int     shift = (i0 % QK4_1) / (QK4_1/2);

    int q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_cuda_memcpy_1<ne>(&q, x[ib].qs + iqs);
    q >>= 4*shift;
    q &= 0x0F0F0F0F;

    const int8_t * q8 = (const int8_t *) &q;

#ifdef FP16_AVAILABLE
    if constexpr (std::is_same_v<T, half>) {
        const half2 dm = x[ib].dm;
        const half2 d  = __half2half2( __low2half(dm));
        const half2 m  = __half2half2(__high2half(dm));

#pragma unroll
        for (int l0 = 0; l0 < ne; l0 += 2) {
            ((half2 *) dst)[l0/2] = d * make_half2(q8[l0 + 0], q8[l0 + 1]) + m;
        }
    } else
#endif // FP16_AVAILABLE
    if constexpr (std::is_same_v<T, float>) {
        const float2 dm = __half22float2(x[ib].dm);

#pragma unroll
        for (int l = 0; l < ne; ++l) {
            ((float *) dst)[l] = dm.x * q8[l] + dm.y;
        }
    } else {
        static_assert(std::is_same_v<T, void>, "bad type");
    }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_q5_0(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_q5_0 * x = (const block_q5_0 *) vx;

    const int64_t ib    =  i0          /  QK5_0;
    const int     idq   =  i0          %  QK5_0;
    const int     iqs   =  i0          % (QK5_0/2);
    const int     shift = (i0 % QK5_0) / (QK5_0/2);

    int q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_cuda_memcpy_1<ne, 2>(&q, x[ib].qs + iqs);
    q >>= 4*shift;
    q &= 0x0F0F0F0F;

    {
        int qh;
        ggml_cuda_memcpy_1<ne, 2>(&qh, x[ib].qh);
#pragma unroll
        for (int l = 0; l < ne; ++l) {
            q |= ((qh >> (idq + l)) & 0x00000001) << (8*l + 4);
        }
    }

    q = __vsubss4(q, 0x10101010);

    const int8_t * q8 = (const int8_t *) &q;

#ifdef FP16_AVAILABLE
    if constexpr (std::is_same_v<T, half>) {
        const half2 d = __half2half2(x[ib].d);

#pragma unroll
        for (int l0 = 0; l0 < ne; l0 += 2) {
            ((half2 *) dst)[l0/2] = d * make_half2(q8[l0 + 0], q8[l0 + 1]);
        }
    } else
#endif // FP16_AVAILABLE
    if constexpr (std::is_same_v<T, float>) {
        const float d = x[ib].d;

#pragma unroll
        for (int l = 0; l < ne; ++l) {
            ((float *) dst)[l] = d * q8[l];
        }
    } else {
        static_assert(std::is_same_v<T, void>, "bad type");
    }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_q5_1(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_q5_1 * x = (const block_q5_1 *) vx;

    const int64_t ib    =  i0          /  QK5_1;
    const int     idq   =  i0          %  QK5_1;
    const int     iqs   =  i0          % (QK5_1/2);
    const int     shift = (i0 % QK5_1) / (QK5_1/2);

    int q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_cuda_memcpy_1<ne>(&q, x[ib].qs + iqs);
    q >>= 4*shift;
    q &= 0x0F0F0F0F;

    {
        int qh;
        ggml_cuda_memcpy_1<ne>(&qh, x[ib].qh);
#pragma unroll
        for (int l = 0; l < ne; ++l) {
            q |= ((qh >> (idq + l)) & 0x00000001) << (8*l + 4);
        }
    }

    const int8_t * q8 = (const int8_t *) &q;

#ifdef FP16_AVAILABLE
    if constexpr (std::is_same_v<T, half>) {
        const half2 dm = x[ib].dm;
        const half2 d  = __half2half2( __low2half(dm));
        const half2 m  = __half2half2(__high2half(dm));

#pragma unroll
        for (int l0 = 0; l0 < ne; l0 += 2) {
            ((half2 *) dst)[l0/2] = d * make_half2(q8[l0 + 0], q8[l0 + 1]) + m;
        }
    } else
#endif // FP16_AVAILABLE
    if constexpr (std::is_same_v<T, float>) {
        const float2 dm = __half22float2(x[ib].dm);

#pragma unroll
        for (int l = 0; l < ne; ++l) {
            ((float *) dst)[l] = dm.x * q8[l] + dm.y;
        }
    } else {
        static_assert(std::is_same_v<T, void>, "bad type");
    }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_q8_0(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_q8_0 * x = (const block_q8_0 *) vx;

    const int64_t ib  = i0 / QK8_0;
    const int     iqs = i0 % QK8_0;

    static_assert(ne % 2 == 0, "bad ne");
    int8_t qs[ne];
    ggml_cuda_memcpy_1<ne, 2>(qs, x[ib].qs + iqs);

#ifdef FP16_AVAILABLE
    if constexpr (std::is_same<T, half>::value) {
        const half2 d = __half2half2(x[ib].d);

#pragma unroll
        for (int l0 = 0; l0 < ne; l0 += 2) {
            ((half2 *) dst)[l0/2] = d * make_half2(qs[l0 + 0], qs[l0 + 1]);
        }
    } else
#endif // FP16_AVAILABLE
    if constexpr (std::is_same<T, float>::value) {
        const float d = x[ib].d;

#pragma unroll
        for (int l = 0; l < ne; ++l) {
            ((float *) dst)[l] = d * qs[l];
        }
    } else {
        static_assert(std::is_same_v<T, void>, "unsupported type");
    }
}

// TurboQuant V dequantization: centroid lookup in WHT domain (no IWHT here)
// IWHT is applied ONCE to the final attention output (after softmax*V sum)
// block_tbq4_0: d at offset 0, qs[128] at offset 2 → 2-byte aligned.
template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_tbq4_0(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_tbq4_0 * x = (const block_tbq4_0 *) vx;
    static constexpr float c4[16] = {
        -2.7326f,-2.0690f,-1.6180f,-1.2562f,-0.9424f,-0.6568f,-0.3881f,-0.1284f,
         0.1284f, 0.3881f, 0.6568f, 0.9424f, 1.2562f, 1.6180f, 2.0690f, 2.7326f,
    };

    const int64_t ib = i0 / QK_K;
    const int elem = i0 % QK_K;
    const float norm = __half2float(x[ib].d);

#pragma unroll
    for (int l = 0; l < ne; l += 2) {
        const int byte_idx = (elem + l) >> 1;
        int start_byte = byte_idx & ~1;
        if (start_byte > (int)(QK_K/2) - 4) start_byte = (int)(QK_K/2) - 4; // qs is 128 bytes
        int qs_word;
        ggml_cuda_memcpy_1<sizeof(int), 2>(&qs_word, &x[ib].qs[start_byte]);
        const int packed = (qs_word >> ((byte_idx - start_byte) * 8)) & 0xFF;
        const float c0 = c4[packed & 0xF] * norm;
        const float c1 = c4[packed >> 4] * norm;
        if constexpr (std::is_same_v<T, float>) {
            ((float *) dst)[l]   = c0;
            ((float *) dst)[l+1] = c1;
        }
#ifdef FP16_AVAILABLE
        else if constexpr (std::is_same_v<T, half>) {
            ((half *) dst)[l]   = __float2half(c0);
            ((half *) dst)[l+1] = __float2half(c1);
        }
#endif
    }
}

// block_tbq3_0: d at offset 0, qs[96] at offset 2 → 2-byte aligned.
template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_tbq3_0(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_tbq3_0 * x = (const block_tbq3_0 *) vx;
    static constexpr float c3[8] = {
        -2.1520f,-1.3440f,-0.7560f,-0.2451f, 0.2451f, 0.7560f, 1.3440f, 2.1520f,
    };

    const int64_t ib = i0 / QK_K;
    const int elem = i0 % QK_K;
    const float norm = __half2float(x[ib].d);

#pragma unroll
    for (int l = 0; l < ne; ++l) {
        const int e = elem + l;
        const int bp = e * 3;
        const int by = bp >> 3;
        int start_byte = by & ~1;
        if (start_byte > (int)(QK_K*3/8) - 4) start_byte = (int)(QK_K*3/8) - 4;
        int qs_word;
        ggml_cuda_memcpy_1<sizeof(int), 2>(&qs_word, &x[ib].qs[start_byte]);
        const uint32_t qs_word_u = (uint32_t) qs_word;
        const int bit_in_word = bp - (start_byte << 3);
        const float cent = c3[(qs_word_u >> bit_in_word) & 0x7] * norm;
        if constexpr (std::is_same_v<T, float>) {
            ((float *) dst)[l] = cent;
        }
#ifdef FP16_AVAILABLE
        else if constexpr (std::is_same_v<T, half>) {
            ((half *) dst)[l] = __float2half(cent);
        }
#endif
    }
}

// TurboQuant_prod: fused score = MSE_term + QJL_term
// Q_v = WHT(signs1*q)*scale/D (MSE query, in Q_reg)
// Q_ds_v = WHT(signs2*q)*sqrt(pi/2)/D (QJL query, in Q_ds -- independent random projection)
// score = sum(cent[idx]*Q_mse[j])*norm + sum(qjl_sign[j]*Q_qjl[j])*d_qjl
template <int D, int nthreads>
// TBQP3_0: block_tbqp3_0 has d at offset 0, d_qjl at 2, qs[64] at 4, qjl[32] at 68. All 4-aligned.
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_tbqp3_0(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {

    const block_tbqp3_0 * K = (const block_tbqp3_0 *) K_c;
    GGML_UNUSED(Q_q8);

    static constexpr float c2[4] = { -1.5104f, -0.4528f, 0.4528f, 1.5104f };

    float sum = 0.0f;

    #pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < D/2; k_KQ_0 += nthreads) {
        const int k = k_KQ_0 + (nthreads == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads);
        const int elem = k * 2;
        const int ib = elem / QK_K;
        const int elem_in_blk = elem - ib*QK_K;

        // Read both norms once (direct half struct field load)
        const float norm = __half2float(K[ib].d);
        const float d_qjl = __half2float(K[ib].d_qjl);

        // qs: 2-bit indices (4 elements per byte). Elem and elem+1 in same byte.
        // 4-byte int load via memcpy_1 (qs at struct offset 4, 4-aligned).
        const int qs_byte = elem_in_blk >> 2;
        const int qs_aligned = qs_byte & ~3;
        int qs_word;
        ggml_cuda_memcpy_1<sizeof(int), 4>(&qs_word, &K[ib].qs[qs_aligned]);
        const int qs_byte_val = (qs_word >> ((qs_byte & 3) * 8)) & 0xFF;
        const int bit_off = (elem_in_blk & 3) * 2;
        const float cent0 = c2[(qs_byte_val >> bit_off)       & 0x3];
        const float cent1 = c2[(qs_byte_val >> (bit_off + 2)) & 0x3];

        // qjl: 1-bit signs. 4-byte int load (qjl at offset 68, 4-aligned).
        const int qjl_byte = elem_in_blk >> 3;
        const int qjl_aligned = qjl_byte & ~3;
        int qjl_word;
        ggml_cuda_memcpy_1<sizeof(int), 4>(&qjl_word, &K[ib].qjl[qjl_aligned]);
        const int qjl_byte_val = (qjl_word >> ((qjl_byte & 3) * 8)) & 0xFF;
        const int sign_bit = elem_in_blk & 7;
        const int qjl0 = (qjl_byte_val >> sign_bit)       & 1;
        const int qjl1 = (qjl_byte_val >> (sign_bit + 1)) & 1;

        const float2 q_qjl = ((const float2 *) Q_ds_v)[k_KQ_0/nthreads];
        const float qc0 = qjl0 ? q_qjl.x : -q_qjl.x;
        const float qc1 = qjl1 ? q_qjl.y : -q_qjl.y;

#ifdef V_DOT2_F32_F16_AVAILABLE
        const float2 q_mse = __half22float2(((const half2 *) Q_v)[k_KQ_0/nthreads]);
#else
        const float2 q_mse = ((const float2 *) Q_v)[k_KQ_0/nthreads];
#endif

        sum += norm * (q_mse.x * cent0 + q_mse.y * cent1)
             + d_qjl * (qc0 + qc1);
    }

    return sum;
}

// TBQP4_0: block_tbqp4_0 has d at 0, d_qjl at 2, qs[96] at 4, qjl[32] at 100. All 4-aligned.
template <int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_tbqp4_0(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {

    const block_tbqp4_0 * K = (const block_tbqp4_0 *) K_c;
    GGML_UNUSED(Q_q8);

    static constexpr float c3[8] = {
        -2.1520f,-1.3440f,-0.7560f,-0.2451f, 0.2451f, 0.7560f, 1.3440f, 2.1520f,
    };

    float sum = 0.0f;

    #pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < D/2; k_KQ_0 += nthreads) {
        const int k = k_KQ_0 + (nthreads == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads);
        const int elem = k * 2;
        const int ib = elem / QK_K;
        const int elem_in_blk = elem - ib*QK_K;

        const float norm = __half2float(K[ib].d);
        const float d_qjl = __half2float(K[ib].d_qjl);

        // qs: 3-bit indices. 2-byte aligned 4-byte load (alignment=2 since we use byte_idx & ~1 base).
        // bit_in_word ≤ 15 (general) or ≤ 26 (clamped tail). qs is 96 bytes; clamp to 92.
        const int bp0 = elem_in_blk * 3;
        const int byte_idx0 = bp0 >> 3;
        int start_byte = byte_idx0 & ~1;
        if (start_byte > (int)(QK_K*3/8) - 4) start_byte = (int)(QK_K*3/8) - 4;
        int qs_word;
        ggml_cuda_memcpy_1<sizeof(int), 2>(&qs_word, &K[ib].qs[start_byte]);
        const uint32_t qs_word_u = (uint32_t) qs_word;
        const int bit_in_word = bp0 - (start_byte << 3);
        const float cent0 = c3[(qs_word_u >> bit_in_word)       & 0x7];
        const float cent1 = c3[(qs_word_u >> (bit_in_word + 3)) & 0x7];

        // qjl: 1-bit signs. 4-byte int load (qjl at offset 100, 4-aligned).
        const int qjl_byte = elem_in_blk >> 3;
        const int qjl_aligned = qjl_byte & ~3;
        int qjl_word;
        ggml_cuda_memcpy_1<sizeof(int), 4>(&qjl_word, &K[ib].qjl[qjl_aligned]);
        const int qjl_byte_val = (qjl_word >> ((qjl_byte & 3) * 8)) & 0xFF;
        const int sign_bit = elem_in_blk & 7;
        const int qjl0 = (qjl_byte_val >> sign_bit)       & 1;
        const int qjl1 = (qjl_byte_val >> (sign_bit + 1)) & 1;

        const float2 q_qjl = ((const float2 *) Q_ds_v)[k_KQ_0/nthreads];
        const float qc0 = qjl0 ? q_qjl.x : -q_qjl.x;
        const float qc1 = qjl1 ? q_qjl.y : -q_qjl.y;

#ifdef V_DOT2_F32_F16_AVAILABLE
        const float2 q_mse = __half22float2(((const half2 *) Q_v)[k_KQ_0/nthreads]);
#else
        const float2 q_mse = ((const float2 *) Q_v)[k_KQ_0/nthreads];
#endif

        sum += norm * (q_mse.x * cent0 + q_mse.y * cent1)
             + d_qjl * (qc0 + qc1);
    }

    return sum;
}

// TBQ4_0: block_tbq4_0 has d at offset 0, qs[128] at offset 2 → 2-byte aligned.
// score = sum_j( centroid[K_idx[j]] * Q_wht[j] ) * norm
// Note: scale/D already applied to Q_wht during query preprocessing
template <int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_tbq4_0(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {
    const block_tbq4_0 * K_tbq = (const block_tbq4_0 *) K_c;
    GGML_UNUSED(Q_q8);
    GGML_UNUSED(Q_ds_v);

    static constexpr float c4[16] = {
        -2.7326f, -2.0690f, -1.6180f, -1.2562f, -0.9424f, -0.6568f, -0.3881f, -0.1284f,
         0.1284f,  0.3881f,  0.6568f,  0.9424f,  1.2562f,  1.6180f,  2.0690f,  2.7326f,
    };

    // D=512: two 256-blocks, each with own norm. Sum both contributions.
    constexpr int n_blocks = D / 256;
    float total = 0.0f;

    for (int blk = 0; blk < n_blocks; blk++) {
        const float norm = __half2float(K_tbq[blk].d);
        float sum = 0.0f;

        #pragma unroll
        for (int k_KQ_0 = 0; k_KQ_0 < 128; k_KQ_0 += nthreads) {
            const int k = k_KQ_0 + (nthreads == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads);

            // qs: 4-bit packed, 2 elements per byte. 128 bytes per block.
            // 2-byte aligned 4-byte int load. Clamp to keep window in 128-byte qs.
            const int qs_byte = k;
            int start_byte = qs_byte & ~1;
            if (start_byte > 124) start_byte = 124;
            int qs_word;
            ggml_cuda_memcpy_1<sizeof(int), 2>(&qs_word, &K_tbq[blk].qs[start_byte]);
            const int packed = (qs_word >> ((qs_byte - start_byte) * 8)) & 0xFF;
            const float cent_lo = c4[packed & 0xF];
            const float cent_hi = c4[packed >> 4];

            const int q_idx = blk * (128/nthreads) + k_KQ_0/nthreads;
#ifdef V_DOT2_F32_F16_AVAILABLE
            const float2 q = __half22float2(((const half2 *) Q_v)[q_idx]);
#else
            const float2 q = ((const float2 *) Q_v)[q_idx];
#endif

            sum += q.x * cent_lo + q.y * cent_hi;
        }
        total += norm * sum;
    }

    return total;
}

// TBQ3_0: 3-bit fused attention score
// block_tbq3_0 has d at offset 0, qs[96] at offset 2 → 2-byte aligned.
template <int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_tbq3_0(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {
    const block_tbq3_0 * K_tbq = (const block_tbq3_0 *) K_c;
    GGML_UNUSED(Q_q8);
    GGML_UNUSED(Q_ds_v);

    static constexpr float c3[8] = {
        -2.1520f, -1.3440f, -0.7560f, -0.2451f,
         0.2451f,  0.7560f,  1.3440f,  2.1520f,
    };

    // D=512: two 256-blocks, each with own norm. Sum both contributions.
    constexpr int n_blocks = D / 256;
    float total = 0.0f;

    for (int blk = 0; blk < n_blocks; blk++) {
        const float norm = __half2float(K_tbq[blk].d);
        float sum = 0.0f;

        #pragma unroll
        for (int k_KQ_0 = 0; k_KQ_0 < 128; k_KQ_0 += nthreads) {
            const int k = k_KQ_0 + (nthreads == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads);
            const int elem = k * 2;

            // qs: 3-bit packed indices. 2-byte aligned 4-byte int load via memcpy_1.
            // qs is 96 bytes per block; clamp window start to 92.
            const int bp0 = elem * 3;
            const int byte_idx0 = bp0 >> 3;
            int start_byte = byte_idx0 & ~1;
            if (start_byte > (int)(QK_K*3/8) - 4) start_byte = (int)(QK_K*3/8) - 4;
            int qs_word;
            ggml_cuda_memcpy_1<sizeof(int), 2>(&qs_word, &K_tbq[blk].qs[start_byte]);
            const uint32_t qs_word_u = (uint32_t) qs_word;
            const int bit_in_word = bp0 - (start_byte << 3);
            const float cent0 = c3[(qs_word_u >> bit_in_word)       & 0x7];
            const float cent1 = c3[(qs_word_u >> (bit_in_word + 3)) & 0x7];

            const int q_idx = blk * (128/nthreads) + k_KQ_0/nthreads;
#ifdef V_DOT2_F32_F16_AVAILABLE
            const float2 q = __half22float2(((const half2 *) Q_v)[q_idx]);
#else
            const float2 q = ((const float2 *) Q_v)[q_idx];
#endif

            sum += q.x * cent0 + q.y * cent1;
        }
        total += norm * sum;
    }

    return total;
}

// ============================================================
// TurboQuant 128-block (_1) variants for head_dim=128
// ============================================================

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_tbq4_1(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_tbq4_1 * x = (const block_tbq4_1 *) vx;
    static constexpr float c4[16] = {
        -2.7326f,-2.0690f,-1.6180f,-1.2562f,-0.9424f,-0.6568f,-0.3881f,-0.1284f,
         0.1284f, 0.3881f, 0.6568f, 0.9424f, 1.2562f, 1.6180f, 2.0690f, 2.7326f,
    };

    const int64_t ib = i0 / TBQ_K128;
    const int elem = i0 % TBQ_K128;
    const float norm = __half2float(x[ib].d);

    // Memory pattern fix: use ggml_cuda_memcpy_1 for byte array reads (block_tbq4_1::qs at offset 2).
    // qs is 64 bytes (4-bit packed, 2 elements/byte). Clamp 4-byte window start to 60.
#pragma unroll
    for (int l = 0; l < ne; l += 2) {
        const int byte_idx = (elem + l) >> 1;
        int start_byte = byte_idx & ~1;
        if (start_byte > (int)(TBQ_K128/2) - 4) start_byte = (int)(TBQ_K128/2) - 4; // qs is 64 bytes
        int qs_word;
        ggml_cuda_memcpy_1<sizeof(int), 2>(&qs_word, &x[ib].qs[start_byte]);
        const int packed = (qs_word >> ((byte_idx - start_byte) * 8)) & 0xFF;
        const float c0 = c4[packed & 0xF] * norm;
        const float c1 = c4[packed >> 4] * norm;
        if constexpr (std::is_same_v<T, float>) {
            ((float *) dst)[l]   = c0;
            ((float *) dst)[l+1] = c1;
        }
#ifdef FP16_AVAILABLE
        else if constexpr (std::is_same_v<T, half>) {
            ((half *) dst)[l]   = __float2half(c0);
            ((half *) dst)[l+1] = __float2half(c1);
        }
#endif
    }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_tbq3_1(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_tbq3_1 * x = (const block_tbq3_1 *) vx;
    static constexpr float c3[8] = {
        -2.1520f,-1.3440f,-0.7560f,-0.2451f, 0.2451f, 0.7560f, 1.3440f, 2.1520f,
    };

    const int64_t ib = i0 / TBQ_K128;
    const int elem = i0 % TBQ_K128;
    const float norm = __half2float(x[ib].d);

    // Memory pattern fix: avoid raw byte indexing (causes GB10 freeze under sustained load).
    // Use ggml_cuda_memcpy_1 with 2-byte alignment (block_tbq3_1::qs is at struct offset 2).
    // 4-byte int load with 2-byte alignment → 2 short loads → coalesced across warp.
    // qs is 48 bytes, so clamp start to 44 to keep the 4-byte window in bounds.
#pragma unroll
    for (int l = 0; l < ne; ++l) {
        const int e = elem + l;
        const int bp = e * 3;
        const int by = bp >> 3;
        int start_byte = by & ~1;
        if (start_byte > (int)(TBQ_K128*3/8) - 4) start_byte = (int)(TBQ_K128*3/8) - 4; // qs is 48 bytes
        int qs_word;
        ggml_cuda_memcpy_1<sizeof(int), 2>(&qs_word, &x[ib].qs[start_byte]);
        const uint32_t qs_word_u = (uint32_t) qs_word;
        const int bit_in_word = bp - (start_byte << 3); // ≤ 26 (clamped tail) or ≤ 15 (general)
        const float cent = c3[(qs_word_u >> bit_in_word) & 0x7] * norm;
        if constexpr (std::is_same_v<T, float>) {
            ((float *) dst)[l] = cent;
        }
#ifdef FP16_AVAILABLE
        else if constexpr (std::is_same_v<T, half>) {
            ((half *) dst)[l] = __float2half(cent);
        }
#endif
    }
}

// TBQP3_1: fused MSE + Direct Sign score (128-block)
// Direct Sign: sign(residual) stored directly, no SRHT — uses Q_v (MSE query) instead of Q_ds
// Memory pattern: 4-byte aligned int loads via ggml_cuda_memcpy_1 (matches q4_0/q5_0 pattern)
// to avoid uncoalesced byte loads that accumulate driver state on GB10 unified memory.
template <int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_tbqp3_1(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {

    const block_tbqp3_1 * K = (const block_tbqp3_1 *) K_c;
    GGML_UNUSED(Q_q8);
    GGML_UNUSED(Q_ds_v); // Direct Sign uses Q_v directly, no separate QJL projection needed

    static constexpr float c2[4] = { -1.5104f, -0.4528f, 0.4528f, 1.5104f };

    // D=128: single block per token (ib==0). Hoist d/d_qjl as one 4-byte aligned int load.
    int dd_word;
    ggml_cuda_memcpy_1<sizeof(int), 4>(&dd_word, &K[0].d);
    const half2 dd_h2 = *reinterpret_cast<const half2 *>(&dd_word);
    const float2 dd = __half22float2(dd_h2);
    const float norm     = dd.x;
    const float d_direct = dd.y; // mean(|residual|) * ||k||

    float sum = 0.0f;

    #pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < D/2; k_KQ_0 += nthreads) {
        const int k = k_KQ_0 + (nthreads == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads);
        const int elem = k * 2;

        // qs: 2-bit indices, 4 elements per byte. Load 4 bytes (16 elements) as int.
        const int qs_byte = elem >> 2;          // byte index containing elem and elem+1 (elem even)
        const int qs_aligned = qs_byte & ~3;
        int qs_word;
        ggml_cuda_memcpy_1<sizeof(int), 4>(&qs_word, &K[0].qs[qs_aligned]);
        const int qs_byte_val = (qs_word >> ((qs_byte & 3) * 8)) & 0xFF;
        const int bit_off = (elem & 3) * 2;     // 0 or 4
        const float cent0 = c2[(qs_byte_val >> bit_off)       & 0x3];
        const float cent1 = c2[(qs_byte_val >> (bit_off + 2)) & 0x3];

        // qjl: 1-bit signs, 8 elements per byte. Load 4 bytes (32 elements) as int.
        const int qjl_byte = elem >> 3;
        const int qjl_aligned = qjl_byte & ~3;
        int qjl_word;
        ggml_cuda_memcpy_1<sizeof(int), 4>(&qjl_word, &K[0].qjl[qjl_aligned]);
        const int qjl_byte_val = (qjl_word >> ((qjl_byte & 3) * 8)) & 0xFF;
        const int sign_bit = elem & 7;           // 0,2,4,6 (elem always even)
        const int sign0 = (qjl_byte_val >> sign_bit)       & 1;
        const int sign1 = (qjl_byte_val >> (sign_bit + 1)) & 1;

#ifdef V_DOT2_F32_F16_AVAILABLE
        const float2 q = __half22float2(((const half2 *) Q_v)[k_KQ_0/nthreads]);
#else
        const float2 q = ((const float2 *) Q_v)[k_KQ_0/nthreads];
#endif

        // MSE: norm * centroid * q_wht + Direct Sign: d_direct * sign(r) * q_wht
        const float sc0 = sign0 ? q.x : -q.x;
        const float sc1 = sign1 ? q.y : -q.y;
        sum += norm * (q.x * cent0 + q.y * cent1)
             + d_direct * (sc0 + sc1);
    }

    return sum;
}

// TBQP4_1: fused MSE + Direct Sign score (128-block)
template <int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_tbqp4_1(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {

    const block_tbqp4_1 * K = (const block_tbqp4_1 *) K_c;
    GGML_UNUSED(Q_q8);
    GGML_UNUSED(Q_ds_v);

    static constexpr float c3[8] = {
        -2.1520f,-1.3440f,-0.7560f,-0.2451f, 0.2451f, 0.7560f, 1.3440f, 2.1520f,
    };

    // D=128: single block per token. Hoist d/d_qjl as one 4-byte aligned int load.
    // (block_tbqp4_1 has d at offset 0 and qs at offset 4 — 4-byte aligned.)
    int dd_word;
    ggml_cuda_memcpy_1<sizeof(int), 4>(&dd_word, &K[0].d);
    const half2 dd_h2 = *reinterpret_cast<const half2 *>(&dd_word);
    const float2 dd = __half22float2(dd_h2);
    const float norm     = dd.x;
    const float d_direct = dd.y;

    float sum = 0.0f;

    #pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < D/2; k_KQ_0 += nthreads) {
        const int k = k_KQ_0 + (nthreads == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads);
        const int elem = k * 2;

        // qs: 3-bit indices, packed across byte boundaries. Pair (elem,elem+1) needs 6 bits at bp0..bp0+5.
        // Use 2-byte aligned 4-byte load (start_byte even). bit_in_word ≤ 15 in normal case,
        // ≤ 26 in clamped tail case — both fit within 32-bit window (cent1 high bit ≤ 31).
        const int bp0 = elem * 3;
        const int byte_idx0 = bp0 >> 3;
        int start_byte = byte_idx0 & ~1;
        if (start_byte > (int)(TBQ_K128*3/8) - 4) start_byte = (int)(TBQ_K128*3/8) - 4; // qs is 48 bytes
        int qs_word;
        ggml_cuda_memcpy_1<sizeof(int), 2>(&qs_word, &K[0].qs[start_byte]);
        const uint32_t qs_word_u = (uint32_t) qs_word;
        const int bit_in_word = bp0 - (start_byte << 3);
        const float cent0 = c3[(qs_word_u >> bit_in_word)       & 0x7];
        const float cent1 = c3[(qs_word_u >> (bit_in_word + 3)) & 0x7];

        // qjl: 1-bit signs, qjl is 16 bytes, offset 52 in struct (4-byte aligned).
        const int qjl_byte = elem >> 3;
        const int qjl_aligned = qjl_byte & ~3;
        int qjl_word;
        ggml_cuda_memcpy_1<sizeof(int), 4>(&qjl_word, &K[0].qjl[qjl_aligned]);
        const int qjl_byte_val = (qjl_word >> ((qjl_byte & 3) * 8)) & 0xFF;
        const int sign_bit = elem & 7;
        const int sign0 = (qjl_byte_val >> sign_bit)       & 1;
        const int sign1 = (qjl_byte_val >> (sign_bit + 1)) & 1;

#ifdef V_DOT2_F32_F16_AVAILABLE
        const float2 q = __half22float2(((const half2 *) Q_v)[k_KQ_0/nthreads]);
#else
        const float2 q = ((const float2 *) Q_v)[k_KQ_0/nthreads];
#endif

        const float sc0 = sign0 ? q.x : -q.x;
        const float sc1 = sign1 ? q.y : -q.y;
        sum += norm * (q.x * cent0 + q.y * cent1)
             + d_direct * (sc0 + sc1);
    }

    return sum;
}

// TBQ4_1: 4-bit fused attention score (128-block)
// Memory pattern: block_tbq4_1 has d at offset 0, qs at offset 2 → 2-byte aligned.
// Use ggml_cuda_memcpy_1 with alignment=2 (matches q4_0/q5_0 pattern).
template <int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_tbq4_1(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {
    const block_tbq4_1 * K_tbq = (const block_tbq4_1 *) K_c;
    GGML_UNUSED(Q_q8);
    GGML_UNUSED(Q_ds_v);

    static constexpr float c4[16] = {
        -2.7326f, -2.0690f, -1.6180f, -1.2562f, -0.9424f, -0.6568f, -0.3881f, -0.1284f,
         0.1284f,  0.3881f,  0.6568f,  0.9424f,  1.2562f,  1.6180f,  2.0690f,  2.7326f,
    };

    const float norm = __half2float(K_tbq[0].d); // direct half struct field load (compiler-friendly)
    float sum = 0.0f;

    #pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < D/2; k_KQ_0 += nthreads) {
        const int k = k_KQ_0 + (nthreads == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads);

        // qs: 4-bit packed, 2 elements per byte. Each thread reads 1 byte (qs_byte=k).
        // Load 4 bytes (4 thread-bytes) starting from 2-byte aligned base.
        // qs is 64 bytes; clamp window start to 60 to keep load in bounds.
        const int qs_byte = k;
        int start_byte = qs_byte & ~1;
        if (start_byte > (int)(TBQ_K128/2) - 4) start_byte = (int)(TBQ_K128/2) - 4;
        int qs_word;
        ggml_cuda_memcpy_1<sizeof(int), 2>(&qs_word, &K_tbq[0].qs[start_byte]);
        const int packed = (qs_word >> ((qs_byte - start_byte) * 8)) & 0xFF;
        const float cent_lo = c4[packed & 0xF];
        const float cent_hi = c4[packed >> 4];

#ifdef V_DOT2_F32_F16_AVAILABLE
        const float2 q = __half22float2(((const half2 *) Q_v)[k_KQ_0/nthreads]);
#else
        const float2 q = ((const float2 *) Q_v)[k_KQ_0/nthreads];
#endif

        sum += q.x * cent_lo + q.y * cent_hi;
    }

    return norm * sum;
}

// TBQ3_1: 3-bit fused attention score (128-block)
// Memory pattern: block_tbq3_1 has d at offset 0, qs at offset 2 → 2-byte aligned.
template <int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_tbq3_1(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {
    const block_tbq3_1 * K_tbq = (const block_tbq3_1 *) K_c;
    GGML_UNUSED(Q_q8);
    GGML_UNUSED(Q_ds_v);

    static constexpr float c3[8] = {
        -2.1520f, -1.3440f, -0.7560f, -0.2451f,
         0.2451f,  0.7560f,  1.3440f,  2.1520f,
    };

    const float norm = __half2float(K_tbq[0].d); // direct half load
    float sum = 0.0f;

    #pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < D/2; k_KQ_0 += nthreads) {
        const int k = k_KQ_0 + (nthreads == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads);
        const int elem = k * 2;

        // qs: 3-bit indices. Pair (elem, elem+1) needs 6 bits at bp0..bp0+5 (elem even).
        // 2-byte aligned 4-byte int load → bit_in_word ≤ 15 (general) or ≤ 26 (clamped tail). Both fit in 32-bit.
        const int bp0 = elem * 3;
        const int byte_idx0 = bp0 >> 3;
        int start_byte = byte_idx0 & ~1;
        if (start_byte > (int)(TBQ_K128*3/8) - 4) start_byte = (int)(TBQ_K128*3/8) - 4; // qs is 48 bytes
        int qs_word;
        ggml_cuda_memcpy_1<sizeof(int), 2>(&qs_word, &K_tbq[0].qs[start_byte]);
        const uint32_t qs_word_u = (uint32_t) qs_word;
        const int bit_in_word = bp0 - (start_byte << 3);
        const float cent0 = c3[(qs_word_u >> bit_in_word)       & 0x7];
        const float cent1 = c3[(qs_word_u >> (bit_in_word + 3)) & 0x7];

#ifdef V_DOT2_F32_F16_AVAILABLE
        const float2 q = __half22float2(((const half2 *) Q_v)[k_KQ_0/nthreads]);
#else
        const float2 q = ((const float2 *) Q_v)[k_KQ_0/nthreads];
#endif

        sum += q.x * cent0 + q.y * cent1;
    }

    return norm * sum;
}

// TBQX3_1: Polar Derotate 3-bit attention score (head_dim=128).
// Block stores per-pair (r, φ_content) where φ_content = φ_post − pos·freq_i.
// r uses Rayleigh Lloyd-Max codebook with per-block σ scale.
// At read time: φ_full = φ_content + cell_pos·freq_i, reconstruct (kx,ky)
// via __sincosf, and dot with raw Q (Q_v is post-rope Q in pair layout where
// slot k holds (Q[k], Q[k + n_pairs])).
//
// freq_table is a shmem-resident fp32 array pre-computed once per kernel
// at the outer fattn-vec kernel start (s_tbqx_freq). pos·freq_i is computed
// in fp32 — for cell_pos ≤ ~2^23 (8M tokens) precision is exact integer.
template <int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_tbqx3_1_pos(
    const char * __restrict__ K_c, const void * __restrict__ Q_v,
    int cell_pos, const float * __restrict__ freq_table) {
    const block_tbqx3_1 * K_tbq = (const block_tbqx3_1 *) K_c;

    constexpr int n_pairs = D / 2;  // 64
    constexpr float TWO_PI     = 6.28318530717958647692f;
    constexpr float PI         = 3.14159265358979323846f;
    constexpr float INV_TWO_PI = 1.0f / TWO_PI;

    // Mirror of TBQX_R_CENT in cpy-utils.cuh — Rayleigh Lloyd-Max 8-level.
    static constexpr float r_cent[8] = {
        0.2400f, 0.6160f, 0.9420f, 1.2547f, 1.5685f, 1.8946f, 2.2520f, 2.6650f
    };

    const float sigma = __half2float(K_tbq[0].d_r);
    float sum = 0.0f;

    #pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < n_pairs; k_KQ_0 += nthreads) {
        const int p = k_KQ_0 + (nthreads == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads);

        // Unpack 3-bit r and phi indices for pair p.
        const int bp = p * 3;
        const int byte_idx = bp >> 3;
        int start_byte = byte_idx & ~1;
        if (start_byte > (int)(TBQ_K128*3/16) - 4) start_byte = (int)(TBQ_K128*3/16) - 4; // qr is 24 bytes
        int qr_word;
        ggml_cuda_memcpy_1<sizeof(int), 2>(&qr_word, &K_tbq[0].qr[start_byte]);
        int qphi_word;
        ggml_cuda_memcpy_1<sizeof(int), 2>(&qphi_word, &K_tbq[0].qphi[start_byte]);
        const int bit_in_word = bp - (start_byte << 3);
        const int r_idx   = ((uint32_t) qr_word   >> bit_in_word) & 0x7;
        const int phi_idx = ((uint32_t) qphi_word >> bit_in_word) & 0x7;

        // Dequant: r = sigma * Lloyd-Max centroid.
        // phi_content mid-point of 8 uniform bins on [-π, π).
        const float r           = sigma * r_cent[r_idx];
        const float phi_content = ((phi_idx + 0.5f) * (1.0f / 8.0f)) * TWO_PI - PI;

        // Re-rotate by pos·freq_i — fp32 only, freq_i broadcast-read from shmem.
        const float freq_i = freq_table[p];
        float theta = (float)cell_pos * freq_i;
        // Wrap into [-π, π) — manual modular reduction (cheap, no fp64).
        theta -= TWO_PI * floorf(theta * INV_TWO_PI + 0.5f);
        float phi_full = phi_content + theta;
        // Phi_content already in [-π,π); after add theta is in [-π,π); sum in
        // [-2π, 2π). One more wrap step.
        phi_full -= TWO_PI * floorf(phi_full * INV_TWO_PI + 0.5f);

        float sin_p, cos_p;
        __sincosf(phi_full, &sin_p, &cos_p);  // fast intrinsic
        float kx = r * cos_p;
        float ky = r * sin_p;

        // Tangent residual: analytical half-cell correction in the φ tangent direction.
        //   K_ref = K_polar + r·(π/16)·sign_dφ·(-sin φ_full, cos φ_full)
        // Reuses cos_p/sin_p — no extra sincos. sign bit stored at qtan bit p.
        constexpr float TANG_MAG = 0.19634954084936207f; // π/16
        const int tan_bit = (K_tbq[0].qtan[p >> 3] >> (p & 7)) & 1;
        const float tan_sign = tan_bit ? 1.0f : -1.0f;
        const float tan_scale = r * TANG_MAG * tan_sign;
        kx += tan_scale * (-sin_p);
        ky += tan_scale * ( cos_p);

#ifdef V_DOT2_F32_F16_AVAILABLE
        const float2 q = __half22float2(((const half2 *) Q_v)[k_KQ_0/nthreads]);
#else
        const float2 q = ((const float2 *) Q_v)[k_KQ_0/nthreads];
#endif

        // Q_v layout for TBQX3_1: slot p = (Q_post[p], Q_post[p + n_pairs])
        sum += q.x * kx + q.y * ky;
    }

    return sum;
}

// Stub matching the standard vec_dot_KQ_t signature so get_vec_dot_KQ
// dispatch still compiles. The actual TBQX3_1 path bypasses this.
template <int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_tbqx3_1(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {
    GGML_UNUSED(K_c);
    GGML_UNUSED(Q_v);
    GGML_UNUSED(Q_q8);
    GGML_UNUSED(Q_ds_v);
    return 0.0f;
}

// ============================================================
// TurboQuant 64-block (_2) variants for head_dim=64
// ============================================================

// block_tbq4_2: d at 0, qs[32] at offset 2 (2-byte aligned).
template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_tbq4_2(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_tbq4_2 * x = (const block_tbq4_2 *) vx;
    static constexpr float c4[16] = { -2.7326f,-2.0690f,-1.6180f,-1.2562f,-0.9424f,-0.6568f,-0.3881f,-0.1284f,0.1284f,0.3881f,0.6568f,0.9424f,1.2562f,1.6180f,2.0690f,2.7326f };
    const int64_t ib = i0 / TBQ_K64; const int elem = i0 % TBQ_K64; const float norm = __half2float(x[ib].d);
#pragma unroll
    for (int l = 0; l < ne; l += 2) {
        const int byte_idx = (elem + l) >> 1;
        int start_byte = byte_idx & ~1;
        if (start_byte > (int)(TBQ_K64/2) - 4) start_byte = (int)(TBQ_K64/2) - 4; // qs is 32 bytes
        int qs_word; ggml_cuda_memcpy_1<sizeof(int), 2>(&qs_word, &x[ib].qs[start_byte]);
        const int packed = (qs_word >> ((byte_idx - start_byte) * 8)) & 0xFF;
        const float c0 = c4[packed&0xF]*norm; const float c1 = c4[packed>>4]*norm;
        if constexpr (std::is_same_v<T,float>) { ((float*)dst)[l]=c0; ((float*)dst)[l+1]=c1; }
#ifdef FP16_AVAILABLE
        else if constexpr (std::is_same_v<T,half>) { ((half*)dst)[l]=__float2half(c0); ((half*)dst)[l+1]=__float2half(c1); }
#endif
    }
}

// block_tbq3_2: d at 0, qs[24] at offset 2 (2-byte aligned).
template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_tbq3_2(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_tbq3_2 * x = (const block_tbq3_2 *) vx;
    static constexpr float c3[8] = { -2.1520f,-1.3440f,-0.7560f,-0.2451f,0.2451f,0.7560f,1.3440f,2.1520f };
    const int64_t ib = i0 / TBQ_K64; const int elem = i0 % TBQ_K64; const float norm = __half2float(x[ib].d);
#pragma unroll
    for (int l = 0; l < ne; ++l) {
        const int e = elem+l; const int bp = e*3; const int by = bp >> 3;
        int start_byte = by & ~1;
        if (start_byte > (int)(TBQ_K64*3/8) - 4) start_byte = (int)(TBQ_K64*3/8) - 4; // qs is 24 bytes
        int qs_word; ggml_cuda_memcpy_1<sizeof(int), 2>(&qs_word, &x[ib].qs[start_byte]);
        const uint32_t qs_word_u = (uint32_t) qs_word;
        const int bit_in_word = bp - (start_byte << 3);
        const float cent = c3[(qs_word_u >> bit_in_word) & 0x7] * norm;
        if constexpr (std::is_same_v<T,float>) { ((float*)dst)[l]=cent; }
#ifdef FP16_AVAILABLE
        else if constexpr (std::is_same_v<T,half>) { ((half*)dst)[l]=__float2half(cent); }
#endif
    }
}

// TBQP3_2: block_tbqp3_2 has d at 0, d_qjl at 2, qs[16] at 4, qjl[8] at 20. All 4-aligned.
template <int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_tbqp3_2(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {
    const block_tbqp3_2 * K = (const block_tbqp3_2 *) K_c;
    GGML_UNUSED(Q_q8); GGML_UNUSED(Q_ds_v);
    static constexpr float c2[4] = { -1.5104f,-0.4528f,0.4528f,1.5104f }; float sum = 0.0f;
    #pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < D/2; k_KQ_0 += nthreads) {
        const int k = k_KQ_0+(nthreads==WARP_SIZE?threadIdx.x:threadIdx.x%nthreads);
        const int elem = k*2; const int ib = elem/TBQ_K64; const int e_in = elem - ib*TBQ_K64;
        const float norm = __half2float(K[ib].d); const float d_direct = __half2float(K[ib].d_qjl);
        // qs (16 bytes, 4-aligned): single 4-byte int load covers whole block
        int qs_word; ggml_cuda_memcpy_1<sizeof(int), 4>(&qs_word, &K[ib].qs[(e_in>>2) & ~3]);
        const int qs_byte = e_in >> 2;
        const int qs_byte_val = (qs_word >> ((qs_byte & 3) * 8)) & 0xFF;
        const int bit_off = (e_in & 3) * 2;
        const float cent0 = c2[(qs_byte_val >> bit_off)       & 0x3];
        const float cent1 = c2[(qs_byte_val >> (bit_off + 2)) & 0x3];
        // qjl (8 bytes, 4-aligned): clamp start to 4
        int qjl_byte = e_in >> 3;
        int qjl_aligned = qjl_byte & ~3;
        if (qjl_aligned > (int)(TBQ_K64/8) - 4) qjl_aligned = (int)(TBQ_K64/8) - 4;
        int qjl_word; ggml_cuda_memcpy_1<sizeof(int), 4>(&qjl_word, &K[ib].qjl[qjl_aligned]);
        const int qjl_byte_val = (qjl_word >> ((qjl_byte - qjl_aligned) * 8)) & 0xFF;
        const int sign_bit = e_in & 7;
        const int s0 = (qjl_byte_val >> sign_bit)       & 1;
        const int s1 = (qjl_byte_val >> (sign_bit + 1)) & 1;
#ifdef V_DOT2_F32_F16_AVAILABLE
        const float2 q = __half22float2(((const half2*)Q_v)[k_KQ_0/nthreads]);
#else
        const float2 q = ((const float2*)Q_v)[k_KQ_0/nthreads];
#endif
        sum += norm*(q.x*cent0+q.y*cent1) + d_direct*((s0?q.x:-q.x)+(s1?q.y:-q.y));
    }
    return sum;
}

// TBQP4_2: block_tbqp4_2 has d at 0, d_qjl at 2, qs[24] at 4, qjl[8] at 28. All 4-aligned.
template <int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_tbqp4_2(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {
    const block_tbqp4_2 * K = (const block_tbqp4_2 *) K_c;
    GGML_UNUSED(Q_q8); GGML_UNUSED(Q_ds_v);
    static constexpr float c3[8] = { -2.1520f,-1.3440f,-0.7560f,-0.2451f,0.2451f,0.7560f,1.3440f,2.1520f }; float sum = 0.0f;
    #pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < D/2; k_KQ_0 += nthreads) {
        const int k = k_KQ_0+(nthreads==WARP_SIZE?threadIdx.x:threadIdx.x%nthreads);
        const int elem = k*2; const int ib = elem/TBQ_K64; const int e_in = elem - ib*TBQ_K64;
        const float norm = __half2float(K[ib].d); const float d_direct = __half2float(K[ib].d_qjl);
        // qs (24 bytes, 4-aligned for tbqp): use 2-byte aligned 4-byte load for bit-window safety
        const int bp0 = e_in * 3;
        const int byte_idx0 = bp0 >> 3;
        int start_byte = byte_idx0 & ~1;
        if (start_byte > (int)(TBQ_K64*3/8) - 4) start_byte = (int)(TBQ_K64*3/8) - 4;
        int qs_word; ggml_cuda_memcpy_1<sizeof(int), 2>(&qs_word, &K[ib].qs[start_byte]);
        const uint32_t qs_word_u = (uint32_t) qs_word;
        const int bit_in_word = bp0 - (start_byte << 3);
        const float cent0 = c3[(qs_word_u >> bit_in_word)       & 0x7];
        const float cent1 = c3[(qs_word_u >> (bit_in_word + 3)) & 0x7];
        // qjl (8 bytes, 4-aligned)
        int qjl_byte = e_in >> 3;
        int qjl_aligned = qjl_byte & ~3;
        if (qjl_aligned > (int)(TBQ_K64/8) - 4) qjl_aligned = (int)(TBQ_K64/8) - 4;
        int qjl_word; ggml_cuda_memcpy_1<sizeof(int), 4>(&qjl_word, &K[ib].qjl[qjl_aligned]);
        const int qjl_byte_val = (qjl_word >> ((qjl_byte - qjl_aligned) * 8)) & 0xFF;
        const int sign_bit = e_in & 7;
        const int s0 = (qjl_byte_val >> sign_bit)       & 1;
        const int s1 = (qjl_byte_val >> (sign_bit + 1)) & 1;
#ifdef V_DOT2_F32_F16_AVAILABLE
        const float2 q = __half22float2(((const half2*)Q_v)[k_KQ_0/nthreads]);
#else
        const float2 q = ((const float2*)Q_v)[k_KQ_0/nthreads];
#endif
        sum += norm*(q.x*cent0+q.y*cent1) + d_direct*((s0?q.x:-q.x)+(s1?q.y:-q.y));
    }
    return sum;
}

// block_tbq4_2: d at 0, qs[32] at offset 2 (2-byte aligned).
template <int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_tbq4_2(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {
    const block_tbq4_2 * K_tbq = (const block_tbq4_2 *) K_c; GGML_UNUSED(Q_q8); GGML_UNUSED(Q_ds_v);
    static constexpr float c4[16] = { -2.7326f,-2.0690f,-1.6180f,-1.2562f,-0.9424f,-0.6568f,-0.3881f,-0.1284f,0.1284f,0.3881f,0.6568f,0.9424f,1.2562f,1.6180f,2.0690f,2.7326f };
    const float norm = __half2float(K_tbq[0].d); float sum = 0.0f;
    #pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < D/2; k_KQ_0 += nthreads) {
        const int k = k_KQ_0+(nthreads==WARP_SIZE?threadIdx.x:threadIdx.x%nthreads);
        const int qs_byte = k;
        int start_byte = qs_byte & ~1;
        if (start_byte > (int)(TBQ_K64/2) - 4) start_byte = (int)(TBQ_K64/2) - 4;
        int qs_word; ggml_cuda_memcpy_1<sizeof(int), 2>(&qs_word, &K_tbq[0].qs[start_byte]);
        const int packed = (qs_word >> ((qs_byte - start_byte) * 8)) & 0xFF;
#ifdef V_DOT2_F32_F16_AVAILABLE
        const float2 q = __half22float2(((const half2*)Q_v)[k_KQ_0/nthreads]);
#else
        const float2 q = ((const float2*)Q_v)[k_KQ_0/nthreads];
#endif
        sum += q.x*c4[packed&0xF] + q.y*c4[packed>>4];
    }
    return norm*sum;
}

// block_tbq3_2: d at 0, qs[24] at offset 2 (2-byte aligned).
template <int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_tbq3_2(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {
    const block_tbq3_2 * K_tbq = (const block_tbq3_2 *) K_c; GGML_UNUSED(Q_q8); GGML_UNUSED(Q_ds_v);
    static constexpr float c3[8] = { -2.1520f,-1.3440f,-0.7560f,-0.2451f,0.2451f,0.7560f,1.3440f,2.1520f };
    const float norm = __half2float(K_tbq[0].d); float sum = 0.0f;
    #pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < D/2; k_KQ_0 += nthreads) {
        const int k = k_KQ_0+(nthreads==WARP_SIZE?threadIdx.x:threadIdx.x%nthreads);
        const int elem = k*2;
        const int bp0 = elem * 3;
        const int byte_idx0 = bp0 >> 3;
        int start_byte = byte_idx0 & ~1;
        if (start_byte > (int)(TBQ_K64*3/8) - 4) start_byte = (int)(TBQ_K64*3/8) - 4;
        int qs_word; ggml_cuda_memcpy_1<sizeof(int), 2>(&qs_word, &K_tbq[0].qs[start_byte]);
        const uint32_t qs_word_u = (uint32_t) qs_word;
        const int bit_in_word = bp0 - (start_byte << 3);
        const float cent0 = c3[(qs_word_u >> bit_in_word)       & 0x7];
        const float cent1 = c3[(qs_word_u >> (bit_in_word + 3)) & 0x7];
#ifdef V_DOT2_F32_F16_AVAILABLE
        const float2 q = __half22float2(((const half2*)Q_v)[k_KQ_0/nthreads]);
#else
        const float2 q = ((const float2*)Q_v)[k_KQ_0/nthreads];
#endif
        sum += q.x*cent0 + q.y*cent1;
    }
    return norm*sum;
}

// ============================================================
// TurboQuant 576-block (_4) K dot product and V dequantize
// Split 256+256+64: sub-blocks processed independently
// ============================================================

// TBQP3_4: 2-bit Lloyd-Max + QJL(256) / DirectSign(64)
// block_tbqp3_4: qs1[64] at offset 4, qjl1[32] at 68, qs2[64] at 104, qjl2[32] at 168 — all 4-aligned.
template <int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_tbqp3_4(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {
    const block_tbqp3_4 * K = (const block_tbqp3_4 *) K_c;
    GGML_UNUSED(Q_q8);
    static constexpr float c2[4] = { -1.5104f,-0.4528f,0.4528f,1.5104f };
    float sum = 0.0f;
    #pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < D/2; k_KQ_0 += nthreads) {
        const int k = k_KQ_0 + (nthreads == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads);
        const int elem = k * 2;
        float norm, d_corr, cent0, cent1;
        int corr0, corr1;
        bool use_qjl;
        if (elem < 512) {
            const bool is_blk1 = (elem < 256);
            const int e = is_blk1 ? elem : (elem - 256);
            norm   = __half2float(is_blk1 ? K->d1     : K->d2);
            d_corr = __half2float(is_blk1 ? K->d1_qjl : K->d2_qjl);
            const uint8_t * qs_sub  = is_blk1 ? K->qs1  : K->qs2;
            const uint8_t * qjl_sub = is_blk1 ? K->qjl1 : K->qjl2;
            // qs (64 bytes, 4-aligned): single 4-byte int load
            const int qs_byte = e >> 2;
            const int qs_aligned = qs_byte & ~3;
            int qs_word; ggml_cuda_memcpy_1<sizeof(int), 4>(&qs_word, &qs_sub[qs_aligned]);
            const int qs_byte_val = (qs_word >> ((qs_byte & 3) * 8)) & 0xFF;
            const int bit_off = (e & 3) * 2;
            cent0 = c2[(qs_byte_val >> bit_off)       & 0x3];
            cent1 = c2[(qs_byte_val >> (bit_off + 2)) & 0x3];
            // qjl (32 bytes, 4-aligned): 4-byte int load
            const int qjl_byte = e >> 3;
            const int qjl_aligned = qjl_byte & ~3;
            int qjl_word; ggml_cuda_memcpy_1<sizeof(int), 4>(&qjl_word, &qjl_sub[qjl_aligned]);
            const int qjl_byte_val = (qjl_word >> ((qjl_byte & 3) * 8)) & 0xFF;
            const int sign_bit = e & 7;
            corr0 = (qjl_byte_val >> sign_bit)       & 1;
            corr1 = (qjl_byte_val >> (sign_bit + 1)) & 1;
            use_qjl = true;
        } else {
            // Sub-block 3: f16 passthrough (rope) — 2-byte aligned half loads, 2 elements
            const int e = elem - 512;
            int rope_pair_word;
            ggml_cuda_memcpy_1<sizeof(int), 2>(&rope_pair_word, &K->rope[e]);
            const half2 rope_h2 = *reinterpret_cast<const half2 *>(&rope_pair_word);
            const float2 rope_f2 = __half22float2(rope_h2);
            cent0 = rope_f2.x;
            cent1 = rope_f2.y;
            use_qjl = false;
            norm = 0.0f; d_corr = 0.0f;
            corr0 = 0; corr1 = 0;
        }
#ifdef V_DOT2_F32_F16_AVAILABLE
        const float2 q_mse = __half22float2(((const half2*)Q_v)[k_KQ_0/nthreads]);
#else
        const float2 q_mse = ((const float2*)Q_v)[k_KQ_0/nthreads];
#endif
        if (use_qjl) {
            const float2 q_qjl = ((const float2*)Q_ds_v)[k_KQ_0/nthreads];
            sum += norm*(q_mse.x*cent0 + q_mse.y*cent1)
                 + d_corr*((corr0?q_qjl.x:-q_qjl.x) + (corr1?q_qjl.y:-q_qjl.y));
        } else {
            sum += q_mse.x*cent0 + q_mse.y*cent1;
        }
    }
    return sum;
}

// TBQP4_4: 3-bit Lloyd-Max + QJL(256) / DirectSign(64)
// block_tbqp4_4: qs1[96] at offset 4, qjl1[32] at 100, qs2[96] at 136, qjl2[32] at 232 — all 4-aligned.
template <int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_tbqp4_4(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {
    const block_tbqp4_4 * K = (const block_tbqp4_4 *) K_c;
    GGML_UNUSED(Q_q8);
    static constexpr float c3[8] = { -2.1520f,-1.3440f,-0.7560f,-0.2451f,0.2451f,0.7560f,1.3440f,2.1520f };
    float sum = 0.0f;
    #pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < D/2; k_KQ_0 += nthreads) {
        const int k = k_KQ_0 + (nthreads == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads);
        const int elem = k * 2;
        float norm, d_corr, cent0, cent1;
        int corr0, corr1;
        bool use_qjl;
        if (elem < 512) {
            const bool is_blk1 = (elem < 256);
            const int e = is_blk1 ? elem : (elem - 256);
            norm   = __half2float(is_blk1 ? K->d1     : K->d2);
            d_corr = __half2float(is_blk1 ? K->d1_qjl : K->d2_qjl);
            const uint8_t * qs_sub  = is_blk1 ? K->qs1  : K->qs2;
            const uint8_t * qjl_sub = is_blk1 ? K->qjl1 : K->qjl2;
            // qs (96 bytes, 4-aligned). Use 2-byte aligned 4-byte int load (bit-window safe).
            const int bp0 = e * 3;
            const int byte_idx0 = bp0 >> 3;
            int start_byte = byte_idx0 & ~1;
            if (start_byte > (int)(QK_K*3/8) - 4) start_byte = (int)(QK_K*3/8) - 4;
            int qs_word; ggml_cuda_memcpy_1<sizeof(int), 2>(&qs_word, &qs_sub[start_byte]);
            const uint32_t qs_word_u = (uint32_t) qs_word;
            const int bit_in_word = bp0 - (start_byte << 3);
            cent0 = c3[(qs_word_u >> bit_in_word)       & 0x7];
            cent1 = c3[(qs_word_u >> (bit_in_word + 3)) & 0x7];
            // qjl (32 bytes, 4-aligned)
            const int qjl_byte = e >> 3;
            const int qjl_aligned = qjl_byte & ~3;
            int qjl_word; ggml_cuda_memcpy_1<sizeof(int), 4>(&qjl_word, &qjl_sub[qjl_aligned]);
            const int qjl_byte_val = (qjl_word >> ((qjl_byte & 3) * 8)) & 0xFF;
            const int sign_bit = e & 7;
            corr0 = (qjl_byte_val >> sign_bit)       & 1;
            corr1 = (qjl_byte_val >> (sign_bit + 1)) & 1;
            use_qjl = true;
        } else {
            // Sub-block 3: f16 passthrough (rope) — 4-byte aligned int load yields 2 halves
            const int e = elem - 512;
            int rope_pair_word;
            ggml_cuda_memcpy_1<sizeof(int), 2>(&rope_pair_word, &K->rope[e]);
            const half2 rope_h2 = *reinterpret_cast<const half2 *>(&rope_pair_word);
            const float2 rope_f2 = __half22float2(rope_h2);
            cent0 = rope_f2.x;
            cent1 = rope_f2.y;
            use_qjl = false;
            norm = 0.0f; d_corr = 0.0f;
            corr0 = 0; corr1 = 0;
        }
#ifdef V_DOT2_F32_F16_AVAILABLE
        const float2 q_mse = __half22float2(((const half2*)Q_v)[k_KQ_0/nthreads]);
#else
        const float2 q_mse = ((const float2*)Q_v)[k_KQ_0/nthreads];
#endif
        if (use_qjl) {
            const float2 q_qjl = ((const float2*)Q_ds_v)[k_KQ_0/nthreads];
            sum += norm*(q_mse.x*cent0 + q_mse.y*cent1)
                 + d_corr*((corr0?q_qjl.x:-q_qjl.x) + (corr1?q_qjl.y:-q_qjl.y));
        } else {
            sum += q_mse.x*cent0 + q_mse.y*cent1;
        }
    }
    return sum;
}

// TBQ3_4: 3-bit Lloyd-Max (no QJL), split 256+256+64
// block_tbq3_4: qs1[96] at offset 2 (2-aligned), qs2[96] at offset 100 (4-aligned).
// Use alignment=2 for both — strictly correct for both sub-blocks.
template <int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_tbq3_4(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {
    const block_tbq3_4 * K = (const block_tbq3_4 *) K_c;
    GGML_UNUSED(Q_q8); GGML_UNUSED(Q_ds_v);
    static constexpr float c3[8] = { -2.1520f,-1.3440f,-0.7560f,-0.2451f,0.2451f,0.7560f,1.3440f,2.1520f };
    float sum = 0.0f;
    #pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < D/2; k_KQ_0 += nthreads) {
        const int k = k_KQ_0 + (nthreads == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads);
        const int elem = k * 2;
        float cent0, cent1;
        float norm = 0.0f;
        bool is_rope = false;
        if (elem < 512) {
            const bool is_blk1 = (elem < 256);
            const int e = is_blk1 ? elem : (elem - 256);
            norm = __half2float(is_blk1 ? K->d1 : K->d2);
            const uint8_t * qs = is_blk1 ? K->qs1 : K->qs2;
            const int bp0 = e * 3;
            const int byte_idx0 = bp0 >> 3;
            int start_byte = byte_idx0 & ~1;
            if (start_byte > (int)(QK_K*3/8) - 4) start_byte = (int)(QK_K*3/8) - 4;
            int qs_word; ggml_cuda_memcpy_1<sizeof(int), 2>(&qs_word, &qs[start_byte]);
            const uint32_t qs_word_u = (uint32_t) qs_word;
            const int bit_in_word = bp0 - (start_byte << 3);
            cent0 = c3[(qs_word_u >> bit_in_word)       & 0x7];
            cent1 = c3[(qs_word_u >> (bit_in_word + 3)) & 0x7];
        } else {
            const int e = elem - 512;
            int rope_pair_word;
            ggml_cuda_memcpy_1<sizeof(int), 2>(&rope_pair_word, &K->rope[e]);
            const half2 rope_h2 = *reinterpret_cast<const half2 *>(&rope_pair_word);
            const float2 rope_f2 = __half22float2(rope_h2);
            cent0 = rope_f2.x;
            cent1 = rope_f2.y;
            is_rope = true;
        }
#ifdef V_DOT2_F32_F16_AVAILABLE
        const float2 q = __half22float2(((const half2*)Q_v)[k_KQ_0/nthreads]);
#else
        const float2 q = ((const float2*)Q_v)[k_KQ_0/nthreads];
#endif
        if (is_rope) {
            sum += q.x*cent0 + q.y*cent1;
        } else {
            sum += norm*(q.x*cent0 + q.y*cent1);
        }
    }
    return sum;
}

// TBQ4_4: 4-bit Lloyd-Max (no QJL), split 256+256+64
// block_tbq4_4: qs1[128] at offset 2 (2-aligned), qs2[128] at offset 132 (4-aligned).
template <int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_tbq4_4(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {
    const block_tbq4_4 * K = (const block_tbq4_4 *) K_c;
    GGML_UNUSED(Q_q8); GGML_UNUSED(Q_ds_v);
    static constexpr float c4[16] = { -2.7326f,-2.0690f,-1.6180f,-1.2562f,-0.9424f,-0.6568f,-0.3881f,-0.1284f,0.1284f,0.3881f,0.6568f,0.9424f,1.2562f,1.6180f,2.0690f,2.7326f };
    float sum = 0.0f;
    #pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < D/2; k_KQ_0 += nthreads) {
        const int k = k_KQ_0 + (nthreads == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads);
        const int elem = k * 2;
        float cent0, cent1;
        float norm = 0.0f;
        bool is_rope = false;
        if (elem < 512) {
            const bool is_blk1 = (elem < 256);
            const int e = is_blk1 ? elem : (elem - 256);
            norm = __half2float(is_blk1 ? K->d1 : K->d2);
            const uint8_t * qs = is_blk1 ? K->qs1 : K->qs2;
            // 4-bit packed: each thread reads 1 byte (2 elements). 2-aligned 4-byte int load.
            const int qs_byte = e >> 1;  // elem and elem+1 share same byte
            int start_byte = qs_byte & ~1;
            if (start_byte > (int)(QK_K/2) - 4) start_byte = (int)(QK_K/2) - 4;
            int qs_word; ggml_cuda_memcpy_1<sizeof(int), 2>(&qs_word, &qs[start_byte]);
            const int packed = (qs_word >> ((qs_byte - start_byte) * 8)) & 0xFF;
            cent0 = c4[packed & 0xF];
            cent1 = c4[packed >> 4];
        } else {
            const int e = elem - 512;
            int rope_pair_word;
            ggml_cuda_memcpy_1<sizeof(int), 2>(&rope_pair_word, &K->rope[e]);
            const half2 rope_h2 = *reinterpret_cast<const half2 *>(&rope_pair_word);
            const float2 rope_f2 = __half22float2(rope_h2);
            cent0 = rope_f2.x;
            cent1 = rope_f2.y;
            is_rope = true;
        }
#ifdef V_DOT2_F32_F16_AVAILABLE
        const float2 q = __half22float2(((const half2*)Q_v)[k_KQ_0/nthreads]);
#else
        const float2 q = ((const float2*)Q_v)[k_KQ_0/nthreads];
#endif
        if (is_rope) {
            sum += q.x*cent0 + q.y*cent1;
        } else {
            sum += norm*(q.x*cent0 + q.y*cent1);
        }
    }
    return sum;
}

// V dequantize for TBQ3_4 (576-block, 3-bit Lloyd-Max)
template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_tbq3_4(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_tbq3_4 * x = (const block_tbq3_4 *) vx;
    static constexpr float c3[8] = { -2.1520f,-1.3440f,-0.7560f,-0.2451f,0.2451f,0.7560f,1.3440f,2.1520f };
    const int64_t ib = i0 / TBQ_K576;
    const int elem = i0 % TBQ_K576;

    // Sub-block 3: f16 passthrough (rope) — use memcpy_1 for half loads
    if (elem >= 512) {
#pragma unroll
        for (int l = 0; l < ne; ++l) {
            half rope_h;
            ggml_cuda_memcpy_1<sizeof(half), 2>(&rope_h, &x[ib].rope[elem - 512 + l]);
            const float val = __half2float(rope_h);
            if constexpr (std::is_same_v<T,float>) { ((float*)dst)[l] = val; }
#ifdef FP16_AVAILABLE
            else if constexpr (std::is_same_v<T,half>) { ((half*)dst)[l] = __float2half(val); }
#endif
        }
        return;
    }

    const bool is_blk1 = (elem < 256);
    const float norm = __half2float(is_blk1 ? x[ib].d1 : x[ib].d2);
    const uint8_t * qs = is_blk1 ? x[ib].qs1 : x[ib].qs2;
    const int e_base = is_blk1 ? elem : (elem - 256);

#pragma unroll
    for (int l = 0; l < ne; ++l) {
        const int e = e_base + l;
        const int bp = e*3;
        const int by = bp >> 3;
        int start_byte = by & ~1;
        if (start_byte > (int)(QK_K*3/8) - 4) start_byte = (int)(QK_K*3/8) - 4;
        int qs_word; ggml_cuda_memcpy_1<sizeof(int), 2>(&qs_word, &qs[start_byte]);
        const uint32_t qs_word_u = (uint32_t) qs_word;
        const int bit_in_word = bp - (start_byte << 3);
        const float cent = c3[(qs_word_u >> bit_in_word) & 0x7] * norm;
        if constexpr (std::is_same_v<T,float>) { ((float*)dst)[l] = cent; }
#ifdef FP16_AVAILABLE
        else if constexpr (std::is_same_v<T,half>) { ((half*)dst)[l] = __float2half(cent); }
#endif
    }
}

// V dequantize for TBQ4_4 (576-block, 4-bit Lloyd-Max)
template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_tbq4_4(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_tbq4_4 * x = (const block_tbq4_4 *) vx;
    static constexpr float c4[16] = { -2.7326f,-2.0690f,-1.6180f,-1.2562f,-0.9424f,-0.6568f,-0.3881f,-0.1284f,0.1284f,0.3881f,0.6568f,0.9424f,1.2562f,1.6180f,2.0690f,2.7326f };
    const int64_t ib = i0 / TBQ_K576;
    const int elem = i0 % TBQ_K576;

    if (elem >= 512) {
#pragma unroll
        for (int l = 0; l < ne; ++l) {
            half rope_h;
            ggml_cuda_memcpy_1<sizeof(half), 2>(&rope_h, &x[ib].rope[elem - 512 + l]);
            const float val = __half2float(rope_h);
            if constexpr (std::is_same_v<T,float>) { ((float*)dst)[l] = val; }
#ifdef FP16_AVAILABLE
            else if constexpr (std::is_same_v<T,half>) { ((half*)dst)[l] = __float2half(val); }
#endif
        }
        return;
    }

    const bool is_blk1 = (elem < 256);
    const float norm = __half2float(is_blk1 ? x[ib].d1 : x[ib].d2);
    const uint8_t * qs = is_blk1 ? x[ib].qs1 : x[ib].qs2;
    const int e_base = is_blk1 ? elem : (elem - 256);

#pragma unroll
    for (int l = 0; l < ne; ++l) {
        const int e = e_base + l;
        const int qs_byte = e >> 1;
        int start_byte = qs_byte & ~1;
        if (start_byte > (int)(QK_K/2) - 4) start_byte = (int)(QK_K/2) - 4;
        int qs_word; ggml_cuda_memcpy_1<sizeof(int), 2>(&qs_word, &qs[start_byte]);
        const int packed = (qs_word >> ((qs_byte - start_byte) * 8)) & 0xFF;
        const float cent = c4[(packed >> ((e&1)*4)) & 0xF] * norm;
        if constexpr (std::is_same_v<T,float>) { ((float*)dst)[l] = cent; }
#ifdef FP16_AVAILABLE
        else if constexpr (std::is_same_v<T,half>) { ((half*)dst)[l] = __float2half(cent); }
#endif
    }
}

// V dequantize for TBQP3_4 (576-block, 2-bit Lloyd-Max + 1-bit QJL)
// block_tbqp3_4: qs1[64]/qs2[64] at struct offset 4/104 (4-aligned).
template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_tbqp3_4(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_tbqp3_4 * x = (const block_tbqp3_4 *) vx;
    static constexpr float c2[4] = { -1.5104f, -0.4528f, 0.4528f, 1.5104f };
    const int64_t ib = i0 / TBQ_K576;
    const int elem = i0 % TBQ_K576;

    if (elem >= 512) {
#pragma unroll
        for (int l = 0; l < ne; ++l) {
            half rope_h;
            ggml_cuda_memcpy_1<sizeof(half), 2>(&rope_h, &x[ib].rope[elem - 512 + l]);
            const float val = __half2float(rope_h);
            if constexpr (std::is_same_v<T, float>) { ((float *)dst)[l] = val; }
#ifdef FP16_AVAILABLE
            else if constexpr (std::is_same_v<T, half>) { ((half *)dst)[l] = __float2half(val); }
#endif
        }
        return;
    }

    // NOTE: QJL correction is NOT applied here — QJL is for K·Q dot product only.
    const bool is_blk1 = (elem < 256);
    const float norm  = __half2float(is_blk1 ? x[ib].d1 : x[ib].d2);
    const uint8_t * qs = is_blk1 ? x[ib].qs1 : x[ib].qs2;
    const int e_base = is_blk1 ? elem : (elem - 256);

#pragma unroll
    for (int l = 0; l < ne; ++l) {
        const int e = e_base + l;
        const int qs_byte = e >> 2;
        int start_byte = qs_byte & ~3;
        if (start_byte > (int)(QK_K/4) - 4) start_byte = (int)(QK_K/4) - 4; // qs is 64 bytes
        int qs_word; ggml_cuda_memcpy_1<sizeof(int), 4>(&qs_word, &qs[start_byte]);
        const int qs_byte_val = (qs_word >> ((qs_byte - start_byte) * 8)) & 0xFF;
        const float val = c2[(qs_byte_val >> ((e & 3) * 2)) & 0x3] * norm;
        if constexpr (std::is_same_v<T, float>) { ((float *)dst)[l] = val; }
#ifdef FP16_AVAILABLE
        else if constexpr (std::is_same_v<T, half>) { ((half *)dst)[l] = __float2half(val); }
#endif
    }
}

// V dequantize for TBQP4_4 (576-block, 3-bit Lloyd-Max + 1-bit QJL)
// block_tbqp4_4: qs1[96]/qs2[96] at struct offset 4/136 (4-aligned).
template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_tbqp4_4(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_tbqp4_4 * x = (const block_tbqp4_4 *) vx;
    static constexpr float c3[8] = { -2.1520f,-1.3440f,-0.7560f,-0.2451f,0.2451f,0.7560f,1.3440f,2.1520f };
    const int64_t ib = i0 / TBQ_K576;
    const int elem = i0 % TBQ_K576;

    if (elem >= 512) {
#pragma unroll
        for (int l = 0; l < ne; ++l) {
            half rope_h;
            ggml_cuda_memcpy_1<sizeof(half), 2>(&rope_h, &x[ib].rope[elem - 512 + l]);
            const float val = __half2float(rope_h);
            if constexpr (std::is_same_v<T, float>) { ((float *)dst)[l] = val; }
#ifdef FP16_AVAILABLE
            else if constexpr (std::is_same_v<T, half>) { ((half *)dst)[l] = __float2half(val); }
#endif
        }
        return;
    }

    const bool is_blk1 = (elem < 256);
    const float norm  = __half2float(is_blk1 ? x[ib].d1 : x[ib].d2);
    const uint8_t * qs = is_blk1 ? x[ib].qs1 : x[ib].qs2;
    const int e_base = is_blk1 ? elem : (elem - 256);

#pragma unroll
    for (int l = 0; l < ne; ++l) {
        const int e = e_base + l;
        const int bp = e*3;
        const int by = bp >> 3;
        int start_byte = by & ~1;
        if (start_byte > (int)(QK_K*3/8) - 4) start_byte = (int)(QK_K*3/8) - 4;
        int qs_word; ggml_cuda_memcpy_1<sizeof(int), 2>(&qs_word, &qs[start_byte]);
        const uint32_t qs_word_u = (uint32_t) qs_word;
        const int bit_in_word = bp - (start_byte << 3);
        const float val = c3[(qs_word_u >> bit_in_word) & 0x7] * norm;
        if constexpr (std::is_same_v<T, float>) { ((float *)dst)[l] = val; }
#ifdef FP16_AVAILABLE
        else if constexpr (std::is_same_v<T, half>) { ((half *)dst)[l] = __float2half(val); }
#endif
    }
}

template <ggml_type type_K, int D, int nthreads>
constexpr __device__ vec_dot_KQ_t get_vec_dot_KQ() {
    if constexpr (type_K == GGML_TYPE_F16) {
        return vec_dot_fattn_vec_KQ_f16<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_TBQ4_0) {
        return vec_dot_fattn_vec_KQ_tbq4_0<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_TBQ3_0) {
        return vec_dot_fattn_vec_KQ_tbq3_0<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_TBQP3_0) {
        return vec_dot_fattn_vec_KQ_tbqp3_0<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_TBQP4_0) {
        return vec_dot_fattn_vec_KQ_tbqp4_0<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_TBQ4_1) {
        return vec_dot_fattn_vec_KQ_tbq4_1<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_TBQ3_1) {
        return vec_dot_fattn_vec_KQ_tbq3_1<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_TBQX3_1) {
        return vec_dot_fattn_vec_KQ_tbqx3_1<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_TBQP3_1) {
        return vec_dot_fattn_vec_KQ_tbqp3_1<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_TBQP4_1) {
        return vec_dot_fattn_vec_KQ_tbqp4_1<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_TBQ4_2) {
        return vec_dot_fattn_vec_KQ_tbq4_2<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_TBQ3_2) {
        return vec_dot_fattn_vec_KQ_tbq3_2<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_TBQP3_2) {
        return vec_dot_fattn_vec_KQ_tbqp3_2<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_TBQP4_2) {
        return vec_dot_fattn_vec_KQ_tbqp4_2<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_TBQ3_3) {
        return vec_dot_fattn_vec_KQ_tbq3_2<D, nthreads>;  // base function (used for per-group scoring)
    } else if constexpr (type_K == GGML_TYPE_TBQ4_3) {
        return vec_dot_fattn_vec_KQ_tbq4_2<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_TBQP3_3) {
        return vec_dot_fattn_vec_KQ_tbqp3_2<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_TBQP4_3) {
        return vec_dot_fattn_vec_KQ_tbqp4_2<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_TBQ3_4) {
        return vec_dot_fattn_vec_KQ_tbq3_4<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_TBQ4_4) {
        return vec_dot_fattn_vec_KQ_tbq4_4<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_TBQP3_4) {
        return vec_dot_fattn_vec_KQ_tbqp3_4<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_TBQP4_4) {
        return vec_dot_fattn_vec_KQ_tbqp4_4<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_Q4_0) {
        return vec_dot_fattn_vec_KQ_q4_0<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_Q4_1) {
        return vec_dot_fattn_vec_KQ_q4_1<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_Q5_0) {
        return vec_dot_fattn_vec_KQ_q5_0<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_Q5_1) {
        return vec_dot_fattn_vec_KQ_q5_1<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_Q8_0) {
        return vec_dot_fattn_vec_KQ_q8_0<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_BF16) {
        return vec_dot_fattn_vec_KQ_bf16<D, nthreads>;
    } else {
        static_assert(type_K == -1, "bad type");
        return nullptr;
    }
}

template <ggml_type type_V, typename T, int ne>
constexpr __device__ dequantize_V_t get_dequantize_V() {
    if constexpr (type_V == GGML_TYPE_F16) {
        return dequantize_V_f16<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_Q4_0) {
        return dequantize_V_q4_0<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_Q4_1) {
        return dequantize_V_q4_1<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_Q5_0) {
        return dequantize_V_q5_0<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_Q5_1) {
        return dequantize_V_q5_1<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_Q8_0) {
        return dequantize_V_q8_0<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_BF16) {
        return dequantize_V_bf16<float, ne>;
    } else if constexpr (type_V == GGML_TYPE_TBQ4_0) {
        return dequantize_V_tbq4_0<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_TBQ3_0) {
        return dequantize_V_tbq3_0<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_TBQ4_1) {
        return dequantize_V_tbq4_1<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_TBQ3_1) {
        return dequantize_V_tbq3_1<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_TBQ4_2) {
        return dequantize_V_tbq4_2<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_TBQ3_2) {
        return dequantize_V_tbq3_2<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_TBQ4_3) {
        return dequantize_V_tbq4_2<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_TBQ3_3) {
        return dequantize_V_tbq3_2<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_TBQ3_4) {
        return dequantize_V_tbq3_4<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_TBQ4_4) {
        return dequantize_V_tbq4_4<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_TBQP3_4) {
        return dequantize_V_tbqp3_4<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_TBQP4_4) {
        return dequantize_V_tbqp4_4<T, ne>;
    } else {
        static_assert(type_V == -1, "bad type");
        return nullptr;
    }
}

template <int ncols1>
__launch_bounds__(FATTN_KQ_STRIDE/2, 1)
static __global__ void flash_attn_mask_to_KV_max(
        const half2 * __restrict__ mask, int * __restrict__ KV_max, const int ne30, const int s31, const int s33) {
    const int ne31     = gridDim.x;
    const int tid      = threadIdx.x;
    const int sequence = blockIdx.y;
    const int jt       = blockIdx.x;

    mask += sequence*s33 + jt*ncols1*s31;

    __shared__ int buf_iw[WARP_SIZE];
    if (tid < WARP_SIZE) {
        buf_iw[tid] = 1;
    }
    __syncthreads();

    int KV_max_sj = (ne30 - 1) * FATTN_KQ_STRIDE;
    for (; KV_max_sj >= 0; KV_max_sj -= FATTN_KQ_STRIDE) {
        int all_inf = 1;

#pragma unroll
        for (int j = 0; j < ncols1; ++j) {
            const float2 tmp = __half22float2(mask[j*s31 + KV_max_sj/2 + tid]);
            all_inf = all_inf && int(isinf(tmp.x)) && int(isinf(tmp.y));
        }

        all_inf = warp_reduce_all(all_inf);
        if (tid % WARP_SIZE == 0) {
            buf_iw[tid / WARP_SIZE] = all_inf;
        }
        __syncthreads();
        all_inf = buf_iw[tid % WARP_SIZE];
        __syncthreads();
        all_inf = warp_reduce_all(all_inf);

        if (!all_inf) {
            break;
        }
    }

    // If the break in the loop was not triggered, KV_max_sj is now -FATTN_KQ_STRIDE.
    // If the break was triggered it's the lower edge of the tile with the first non-masked values.
    // In either case, walk back the decrementation by FATTN_KQ_STRIDE.
    KV_max_sj += FATTN_KQ_STRIDE;

    if (threadIdx.x != 0) {
        return;
    }

    KV_max[sequence*ne31 + jt] = KV_max_sj;
}

template<int D, int ncols1, int ncols2> // D == head size
__launch_bounds__(D, 1)
static __global__ void flash_attn_stream_k_fixup_uniform(
        float * __restrict__ dst,
        const float2 * __restrict__ dst_fixup,
        const int ne01, const int ne02,
        const int ne12, const int nblocks_stream_k,
        const int gqa_ratio,
        const int blocks_per_tile,
        const uint3 fd_iter_j_z_ne12,
        const uint3 fd_iter_j_z,
        const uint3 fd_iter_j) {
    constexpr int ncols = ncols1*ncols2;

    const int tile_idx = blockIdx.x; // One block per output tile.
    const int j        = blockIdx.y;
    const int c        = blockIdx.z;
    const int jc       = j*ncols2 + c;
    const int tid      = threadIdx.x;

    // nblocks_stream_k is a multiple of ntiles_dst (== gridDim.x), so each tile gets the same number of blocks.
    const int b_first = tile_idx * blocks_per_tile;
    const int b_last  = b_first + blocks_per_tile - 1;

    const float * dst_fixup_data = ((const float *) dst_fixup) + nblocks_stream_k*(2*2*ncols);

    // z_KV == K/V head index, zt_gqa = Q head start index per K/V head, jt = token position start index
    const uint2 dm0 = fast_div_modulo(tile_idx, fd_iter_j_z_ne12);
    const uint2 dm1 = fast_div_modulo(dm0.y,    fd_iter_j_z);
    const uint2 dm2 = fast_div_modulo(dm1.y,    fd_iter_j);

    const int sequence = dm0.x;
    const int z_KV     = dm1.x;
    const int zt_gqa   = dm2.x;
    const int jt       = dm2.y;

    const int zt_Q = z_KV*gqa_ratio + zt_gqa*ncols2; // Global Q head start index.

    if (jt*ncols1 + j >= ne01 || zt_gqa*ncols2 + c >= gqa_ratio) {
        return;
    }

    dst += sequence*ne02*ne01*D + jt*ne02*(ncols1*D) + zt_Q*D + (j*ne02 + c)*D + tid;

    // Load the partial result that needs a fixup
    float dst_val = *dst;
    float max_val;
    float rowsum;
    {
        const float2 tmp = dst_fixup[b_last*ncols + jc];
        max_val = tmp.x;
        rowsum  = tmp.y;
    }

    // Combine with all previous blocks in this tile.
    for (int bidx = b_last - 1; bidx >= b_first; --bidx) {
        const float dst_add = dst_fixup_data[bidx*ncols*D + jc*D + tid];

        const float2 tmp = dst_fixup[(nblocks_stream_k + bidx)*ncols + jc];

        const float max_val_new = fmaxf(max_val, tmp.x);

        const float diff_val = max_val - max_val_new;
        const float diff_add = tmp.x   - max_val_new;

        const float scale_val = diff_val >= SOFTMAX_FTZ_THRESHOLD ? expf(diff_val) : 0.0f;
        const float scale_add = diff_add >= SOFTMAX_FTZ_THRESHOLD ? expf(diff_add) : 0.0f;

        dst_val = scale_val*dst_val + scale_add*dst_add;
        rowsum  = scale_val*rowsum  + scale_add*tmp.y;

        max_val = max_val_new;
    }

    // Write back final result:
    *dst = dst_val / rowsum;
}

// General fixup kernel for the case where the number of blocks per tile is not uniform across tiles
// (blocks_num.x not a multiple of ntiles_dst)
template <int D, int ncols1, int ncols2> // D == head size
__launch_bounds__(D, 1)
static __global__ void flash_attn_stream_k_fixup_general(
        float * __restrict__ dst,
        const float2 * __restrict__ dst_fixup,
        const int ne01, const int ne02,
        const int gqa_ratio,
        const int total_work,
        const uint3 fd_iter_k_j_z_ne12,
        const uint3 fd_iter_k_j_z,
        const uint3 fd_iter_k_j,
        const uint3 fd_iter_k) {
    constexpr int ncols = ncols1*ncols2;

    const int bidx0 = blockIdx.x;
    const int j     = blockIdx.y;
    const int c     = blockIdx.z;
    const int jc    = j*ncols2 + c;
    const int tid   = threadIdx.x;

    const float * dst_fixup_data = ((const float *) dst_fixup) + gridDim.x*(2*2*ncols);

    const int kbc0      = int64_t(bidx0 + 0)*total_work / gridDim.x;
    const int kbc0_stop = int64_t(bidx0 + 1)*total_work / gridDim.x;

    const bool did_not_have_any_data   = kbc0 == kbc0_stop;
    const bool wrote_beginning_of_tile = fastmodulo(kbc0, fd_iter_k) == 0;
    const bool did_not_write_last      = fastdiv(kbc0, fd_iter_k) == fastdiv(kbc0_stop, fd_iter_k) && fastmodulo(kbc0_stop, fd_iter_k) != 0;
    if (did_not_have_any_data || wrote_beginning_of_tile || did_not_write_last) {
        return;
    }

    // z_KV == K/V head index, zt_gqa = Q head start index per K/V head, jt = token position start index
    const uint2 dm0 = fast_div_modulo(kbc0, fd_iter_k_j_z_ne12);
    const uint2 dm1 = fast_div_modulo(dm0.y, fd_iter_k_j_z);
    const uint2 dm2 = fast_div_modulo(dm1.y, fd_iter_k_j);
    const uint2 dm3 = fast_div_modulo(dm2.y, fd_iter_k);

    const int sequence = dm0.x;
    const int z_KV     = dm1.x;
    const int zt_gqa   = dm2.x;
    const int jt       = dm3.x;

    const int zt_Q = z_KV*gqa_ratio + zt_gqa*ncols2; // Global Q head start index.

    if (jt*ncols1 + j >= ne01 || zt_gqa*ncols2 + c >= gqa_ratio) {
        return;
    }

    dst += sequence*ne02*ne01*D + jt*ne02*(ncols1*D) + zt_Q*D + (j*ne02 + c)*D + tid;

    // Load the partial result that needs a fixup:
    float dst_val = 0.0f;
    float max_val = 0.0f;
    float rowsum  = 0.0f;
    {
        dst_val = *dst;

        const float2 tmp = dst_fixup[bidx0*ncols + jc];
        max_val = tmp.x;
        rowsum  = tmp.y;
    }

    // Iterate over previous blocks and compute the combined results.
    // All CUDA blocks that get here must have a previous block that needs a fixup.
    const int tile_kbc0 = fastdiv(kbc0, fd_iter_k);
    int bidx = bidx0 - 1;
    int kbc_stop = kbc0;
    while(true) {
        const int kbc = int64_t(bidx)*total_work / gridDim.x;
        if (kbc == kbc_stop) { // Did not have any data.
            bidx--;
            kbc_stop = kbc;
            continue;
        }

        const float dst_add = dst_fixup_data[bidx*ncols*D + jc*D + tid];

        const float2 tmp = dst_fixup[(gridDim.x + bidx)*ncols + jc];

        // Scale the current and new value accumulators depending on the max. values.
        const float max_val_new = fmaxf(max_val, tmp.x);

        const float diff_val = max_val - max_val_new;
        const float diff_add = tmp.x   - max_val_new;

        const float scale_val = diff_val >= SOFTMAX_FTZ_THRESHOLD ? expf(diff_val) : 0.0f;
        const float scale_add = diff_add >= SOFTMAX_FTZ_THRESHOLD ? expf(diff_add) : 0.0f;

        dst_val = scale_val*dst_val + scale_add*dst_add;
        rowsum  = scale_val*rowsum  + scale_add*tmp.y;

        max_val = max_val_new;

        // If this block started in a previous tile we are done and don't need to combine additional partial results.
        if (fastmodulo(kbc, fd_iter_k) == 0 || fastdiv(kbc, fd_iter_k) < tile_kbc0) {
            break;
        }
        bidx--;
        kbc_stop = kbc;
    }

    // Write back final result:
    *dst = dst_val / rowsum;
}

template<int D> // D == head size
__launch_bounds__(D, 1)
static __global__ void flash_attn_combine_results(
        const float  * __restrict__ VKQ_parts,
        const float2 * __restrict__ VKQ_meta,
        float * __restrict__ dst,
        const int parallel_blocks) {
    // Dimension 0: threadIdx.x
    // Dimension 1: blockIdx.x
    // Dimension 2: blockIdx.y
    // Dimension 3: blockIdx.z
    // Memory layout is permuted with [0, 2, 1, 3]

    const int ne01 = gridDim.x;
    const int ne02 = gridDim.y;

    const int col      = blockIdx.x;
    const int head     = blockIdx.y;
    const int sequence = blockIdx.z;

    const int j_dst_unrolled = (sequence*ne01 + col)*ne02 + head;

    VKQ_parts += j_dst_unrolled * parallel_blocks*D;
    VKQ_meta  += j_dst_unrolled * parallel_blocks;
    dst       += j_dst_unrolled *                 D;

    const int tid = threadIdx.x;
    __builtin_assume(tid < D);

    extern __shared__ float2 meta[];
    for (int i = tid; i < 2*parallel_blocks; i += D) {
        ((float *) meta)[i] = ((const float *)VKQ_meta) [i];
    }

    __syncthreads();

    float kqmax = meta[0].x;
    for (int l = 1; l < parallel_blocks; ++l) {
        kqmax = max(kqmax, meta[l].x);
    }

    float VKQ_numerator   = 0.0f;
    float VKQ_denominator = 0.0f;
    for (int l = 0; l < parallel_blocks; ++l) {
        const float KQ_max_scale = expf(meta[l].x - kqmax);

        VKQ_numerator   += KQ_max_scale * VKQ_parts[l*D + tid];
        VKQ_denominator += KQ_max_scale * meta[l].y;
    }

    dst[tid] = VKQ_numerator / VKQ_denominator;
}

template <int DV, int ncols1, int ncols2>
void launch_fattn(
    ggml_backend_cuda_context & ctx, ggml_tensor * dst, fattn_kernel_t fattn_kernel, const int nwarps, const size_t nbytes_shared,
    const int nbatch_fa, const bool need_f16_K, const bool need_f16_V, const bool stream_k, const int warp_size = WARP_SIZE
) {
    constexpr int ncols = ncols1 * ncols2;

    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    const bool V_is_K_view = V->view_src && (V->view_src == K || (V->view_src == K->view_src && V->view_offs == K->view_offs));

    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];

    // TurboQuant side-channel tensors (optional, only set when relevant):
    //   src[5]: k_rope (f16) — decoupled rope slice for MLA `_4` types (GLM/DeepSeek)
    const ggml_tensor * k_rope     = dst->src[5];

    ggml_tensor * KQV = dst;

    GGML_ASSERT(Q->type == GGML_TYPE_F32);
    GGML_ASSERT(KQV->type == GGML_TYPE_F32);

    GGML_ASSERT(Q->nb[0] == ggml_element_size(Q));
    GGML_ASSERT(K->nb[0] == ggml_element_size(K));
    GGML_ASSERT(V->nb[0] == ggml_element_size(V));

    GGML_ASSERT(!mask || mask->type == GGML_TYPE_F16);

    ggml_cuda_pool & pool = ctx.pool();
    cudaStream_t main_stream = ctx.stream();
    const int id  = ggml_cuda_get_device();
    const int cc  = ggml_cuda_info().devices[id].cc;
    const int nsm = ggml_cuda_info().devices[id].nsm;

    ggml_cuda_pool_alloc<half>   K_f16(pool);
    ggml_cuda_pool_alloc<half>   V_f16(pool);
    ggml_cuda_pool_alloc<int>    KV_max(pool);
    ggml_cuda_pool_alloc<float>  dst_tmp(pool);
    ggml_cuda_pool_alloc<float2> dst_tmp_meta(pool);

    const char * K_data = (const char *) K->data;
    const char * K_data_orig = K_data;  // Preserved for TBQP V spatial dequant (before K→WHT conversion)
    size_t nb11 = K->nb[1];
    size_t nb12 = K->nb[2];
    size_t nb13 = K->nb[3];

    // TBQP WHT-domain MMA detection
    const bool tbqp_wht_mode = need_f16_K &&
        (K->type == GGML_TYPE_TBQP3_4 || K->type == GGML_TYPE_TBQP4_4);



    const char * V_data = (const char *) V->data;
    size_t nb21 = V->nb[1];
    size_t nb22 = V->nb[2];
    size_t nb23 = V->nb[3];

    if (need_f16_K && K->type != GGML_TYPE_F16) {
        const size_t bs = ggml_blck_size(K->type);
        const size_t ts = ggml_type_size(K->type);
        const int64_t k_elems = ggml_nelements(K);

        // TBQP: K_mse only (1x). QJL correction computed as scalar from raw block data.
        K_f16.alloc(k_elems);

        if (tbqp_wht_mode) {
            // TBQP: K_mse WHT f16 (no QJL). QJL applied as scalar correction in MMA kernel.
            // to_fp16 already registered as MSE-only WHT dequant for TBQP types.
            to_fp16_cuda_t to_fp16 = ggml_get_to_fp16_cuda(K->type);
            to_fp16(K_data, K_f16.ptr, k_elems, main_stream);
            nb11 = K->ne[0] * sizeof(half);
            nb12 = K->ne[1] * nb11;
            nb13 = K->ne[2] * nb12;
        } else if (ggml_is_contiguously_allocated(K)) {
            to_fp16_cuda_t to_fp16 = ggml_get_to_fp16_cuda(K->type);
            to_fp16(K_data, K_f16.ptr, k_elems, main_stream);

            nb11 = nb11*bs*sizeof(half)/ts;
            nb12 = nb12*bs*sizeof(half)/ts;
            nb13 = nb13*bs*sizeof(half)/ts;
        } else {
            GGML_ASSERT(K->nb[0] == ts);
            to_fp16_nc_cuda_t to_fp16 = ggml_get_to_fp16_nc_cuda(K->type);
            const int64_t s01 = nb11 / ts;
            const int64_t s02 = nb12 / ts;
            const int64_t s03 = nb13 / ts;
            to_fp16(K_data, K_f16.ptr, K->ne[0], K->ne[1], K->ne[2], K->ne[3], s01, s02, s03, main_stream);

            nb11 = K->ne[0] * sizeof(half);
            nb12 = K->ne[1] * nb11;
            nb13 = K->ne[2] * nb12;
        }

        K_data = (char *) K_f16.ptr;
    }

    if (need_f16_V && V->type != GGML_TYPE_F16) {
        if (V_is_K_view) {
            // MLA: V shares K's spatial f16 buffer. Both TBQ and TBQP.
            // TBQP: K is spatial (IWHT in dequant). V = K view = spatial. No output IWHT.
            V_data = K_data;
            nb21   = nb11;
            nb22   = nb12;
            nb23   = nb13;
        } else {
            const size_t bs = ggml_blck_size(V->type);
            const size_t ts = ggml_type_size(V->type);

            V_f16.alloc(ggml_nelements(V));
            if (ggml_is_contiguously_allocated(V)) {
                to_fp16_cuda_t to_fp16 = ggml_get_to_fp16_cuda(V->type);
                to_fp16(V_data, V_f16.ptr, ggml_nelements(V), main_stream);
                V_data = (char *) V_f16.ptr;

                nb21 = nb21*bs*sizeof(half)/ts;
                nb22 = nb22*bs*sizeof(half)/ts;
                nb23 = nb23*bs*sizeof(half)/ts;
            } else {
                GGML_ASSERT(V->nb[0] == ts);
                to_fp16_nc_cuda_t to_fp16 = ggml_get_to_fp16_nc_cuda(V->type);
                const int64_t s01 = nb21 / ts;
                const int64_t s02 = nb22 / ts;
                const int64_t s03 = nb23 / ts;
                to_fp16(V_data, V_f16.ptr, V->ne[0], V->ne[1], V->ne[2], V->ne[3], s01, s02, s03, main_stream);

                nb21 = V->ne[0] * sizeof(half);
                nb22 = V->ne[1] * nb21;
                nb23 = V->ne[2] * nb22;
            }
            V_data = (char *) V_f16.ptr;
        }
    }

    // TBQP: K is spatial. Q stays spatial (no WHT needed for dot product).
    // Only Q_wht2 needed for QJL scalar correction. Q_wht1 is intermediate.
    ggml_cuda_pool_alloc<float> q_wht_buf(pool);
    const char * q_wht2_ptr = nullptr;
    int32_t q_wht2_stride = 0;
    if (tbqp_wht_mode) {
        extern void tbq_q_wht12_cuda(const float *, float *, float *, int64_t, int64_t, int64_t, cudaStream_t);
        const size_t n_q_elems = ggml_nelements(Q);
        q_wht_buf.alloc(2 * n_q_elems);  // [Q_wht1 (temp) | Q_wht2 (for QJL)]
        float * q_wht1 = q_wht_buf.ptr;
        float * q_wht2 = q_wht_buf.ptr + n_q_elems;
        // Reads Q->data directly, computes Q_wht1 (temp) + Q_wht2 (for QJL). No cudaMemcpy.
        tbq_q_wht12_cuda((const float *)Q->data, q_wht1, q_wht2,
                         Q->ne[0], Q->ne[1]*Q->ne[2]*Q->ne[3], Q->ne[0], main_stream);
        q_wht2_ptr = (const char *)q_wht2;
        q_wht2_stride = Q->ne[0] * sizeof(float);
    }

    const int ntiles_x     = ((Q->ne[1] + ncols1 - 1) / ncols1);
    const int gqa_ratio    = Q->ne[2] / K->ne[2];
    const int ntiles_z_gqa = ((gqa_ratio + ncols2 - 1) / ncols2);
    const int ntiles_dst   = ntiles_x * ntiles_z_gqa * K->ne[2] * Q->ne[3];

    // Optional optimization where the mask is scanned to determine whether part of the calculation can be skipped.
    // Only worth the overhead if there is at lease one FATTN_KQ_STRIDE x FATTN_KQ_STRIDE square to be skipped or
    //     multiple sequences of possibly different lengths.
    if (mask && K->ne[1] % FATTN_KQ_STRIDE == 0 && (Q->ne[1] >= 1024 || Q->ne[3] > 1)) {
        const int s31 = mask->nb[1] / sizeof(half2);
        const int s33 = mask->nb[3] / sizeof(half2);

        const dim3 blocks_num_KV_max(ntiles_x, Q->ne[3], 1);
        const dim3 block_dim_KV_max(FATTN_KQ_STRIDE/2, 1, 1);

        const int ne_KV_max = blocks_num_KV_max.x*blocks_num_KV_max.y;
        const int iter_k = K->ne[1] / FATTN_KQ_STRIDE;

        KV_max.alloc(ne_KV_max);
        flash_attn_mask_to_KV_max<ncols1><<<blocks_num_KV_max, block_dim_KV_max, 0, main_stream>>>
            ((const half2 *) mask->data, KV_max.ptr, iter_k, s31, s33);
        CUDA_CHECK(cudaGetLastError());
    }

    const dim3 block_dim(warp_size, nwarps, 1);
    int max_blocks_per_sm = 1; // Max. number of active blocks limited by occupancy.
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm, fattn_kernel, block_dim.x * block_dim.y * block_dim.z, nbytes_shared));
    GGML_ASSERT(max_blocks_per_sm > 0);
    int parallel_blocks = max_blocks_per_sm;

    const int ntiles_KV = (K->ne[1] + nbatch_fa - 1) / nbatch_fa; // Max. number of parallel blocks limited by KV cache length.

    dim3 blocks_num;
    if (stream_k) {
        // For short contexts it can be faster to have the SMs work on whole tiles because this lets us skip the fixup.
        const int max_blocks = max_blocks_per_sm*nsm;
        const int tiles_nwaves = (ntiles_dst + max_blocks - 1) / max_blocks;
        const int tiles_efficiency_percent = 100 * ntiles_dst / (max_blocks*tiles_nwaves);

        const bool use_stream_k = cc >= GGML_CUDA_CC_ADA_LOVELACE || amd_wmma_available(cc) || tiles_efficiency_percent < 75;

        blocks_num.x = ntiles_dst;
        blocks_num.y = 1;
        blocks_num.z = 1;

        if(use_stream_k) {
            const int nblocks_stream_k_raw = std::min(max_blocks, ntiles_KV*ntiles_dst);
            // Round down to a multiple of ntiles_dst so that each output tile gets the same number of blocks (avoids fixup).
            // Only do this if the occupancy loss from rounding is acceptable.
            const int nblocks_stream_k_rounded = (nblocks_stream_k_raw / ntiles_dst) * ntiles_dst;
            const int max_efficiency_loss_percent = 5;
            const int efficiency_loss_percent = nblocks_stream_k_rounded > 0
                ? 100 * (nblocks_stream_k_raw - nblocks_stream_k_rounded) / nblocks_stream_k_raw
                : 100;
            const int nblocks_stream_k = efficiency_loss_percent <= max_efficiency_loss_percent
                ? nblocks_stream_k_rounded
                : nblocks_stream_k_raw;

            blocks_num.x = nblocks_stream_k;
        }

        if (ntiles_dst % blocks_num.x != 0) { // Fixup is only needed if the SMs work on fractional tiles.
            dst_tmp_meta.alloc((size_t(blocks_num.x) * ncols * (2 + DV/2)));
        }
    } else {
        // parallel_blocks must not be larger than what the tensor size allows:
        parallel_blocks = std::min(parallel_blocks, ntiles_KV);

        // If ntiles_total % blocks_per_wave != 0 then some efficiency is lost due to tail effects.
        // Test whether parallel_blocks can be set to a higher value for better efficiency.
        const int blocks_per_wave = nsm * max_blocks_per_sm;
        int nwaves_best = 0;
        int efficiency_percent_best = 0;
        for (int parallel_blocks_test = parallel_blocks; parallel_blocks_test <= ntiles_KV; ++parallel_blocks_test) {
            const int nblocks_total = ntiles_dst * parallel_blocks_test;
            const int nwaves = (nblocks_total + blocks_per_wave - 1) / blocks_per_wave;
            const int efficiency_percent = 100 * nblocks_total / (nwaves*blocks_per_wave);

            // Stop trying configurations with more waves if we already have good efficiency to avoid excessive overhead.
            if (efficiency_percent_best >= 95 && nwaves > nwaves_best) {
                break;
            }

            if (efficiency_percent > efficiency_percent_best) {
                nwaves_best = nwaves;
                efficiency_percent_best = efficiency_percent;
                parallel_blocks = parallel_blocks_test;
            }
        }

        blocks_num.x = ntiles_x;
        blocks_num.y = parallel_blocks;
        blocks_num.z = ntiles_z_gqa*K->ne[2]*Q->ne[3];

        if (parallel_blocks > 1) {
            dst_tmp.alloc(parallel_blocks*ggml_nelements(KQV));
            dst_tmp_meta.alloc(parallel_blocks*ggml_nrows(KQV));
        }
    }

    float scale         = 1.0f;
    float max_bias      = 0.0f;
    float logit_softcap = 0.0f;

    memcpy(&scale,         (const float *) KQV->op_params + 0, sizeof(float));
    memcpy(&max_bias,      (const float *) KQV->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));

    if (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }

    const uint32_t n_head      = Q->ne[2];
    const uint32_t n_head_log2 = 1u << uint32_t(floorf(log2f(float(n_head))));

    const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    // TODO other tensor dimensions after removal of WMMA kernel:
    const uint3 ne01 = init_fastdiv_values(Q->ne[1]);

    // TurboQuant side-channel pointers (nullptr unless the corresponding src slot is bound):
    const char * k_rope_data      = k_rope     ? (const char *)  k_rope->data     : nullptr;
    const int32_t k_rope_stride_b = k_rope     ? (int32_t) k_rope->nb[1]           : 0;

    GGML_ASSERT(block_dim.x % warp_size == 0);
    // TBQP: Q stays spatial (K is spatial too). V = K view (spatial). No hacks.
    fattn_kernel<<<blocks_num, block_dim, nbytes_shared, main_stream>>>(
        (const char *) Q->data,
        K_data,
        V_data,
        mask ? ((const char *) mask->data) : nullptr,
        sinks ? ((const char *) sinks->data) : nullptr,
        KV_max.ptr,
        !stream_k && parallel_blocks > 1 ? dst_tmp.ptr : (float *) KQV->data, dst_tmp_meta.ptr,
        scale, max_bias, m0, m1, n_head_log2, logit_softcap,
        Q->ne[0], ne01,     Q->ne[2], Q->ne[3], Q->nb[1], Q->nb[2], Q->nb[3],
        K->ne[0], K->ne[1], K->ne[2], K->ne[3], nb11, nb12, nb13,
        nb21, nb22, nb23,
        mask ? mask->ne[1] : 0, mask ? mask->ne[2] : 0, mask ? mask->ne[3] : 0,
        mask ? mask->nb[1] : 0, mask ? mask->nb[2] : 0, mask ? mask->nb[3] : 0,
        tbqp_wht_mode ? K_data_orig : nullptr,
        tbqp_wht_mode ? (int32_t)K->nb[1] : 0,
        q_wht2_ptr,
        q_wht2_stride,
        k_rope_data,
        k_rope_stride_b
    );
    CUDA_CHECK(cudaGetLastError());

    if (stream_k) {
        if ((int)blocks_num.x % ntiles_dst == 0 && (int)blocks_num.x > ntiles_dst) {
            // Optimized fixup: nblocks_stream_k is a multiple of ntiles_dst, launch one block per tile.
            const int nblocks_sk  = (int)blocks_num.x;
            const int bpt         = nblocks_sk / ntiles_dst;

            const uint3 fd0 = init_fastdiv_values(ntiles_x * ntiles_z_gqa * K->ne[2]);
            const uint3 fd1 = init_fastdiv_values(ntiles_x * ntiles_z_gqa);
            const uint3 fd2 = init_fastdiv_values(ntiles_x);

            const dim3 block_dim_combine(DV, 1, 1);
            const dim3 blocks_num_combine = {(unsigned)ntiles_dst, ncols1, ncols2};

            flash_attn_stream_k_fixup_uniform<DV, ncols1, ncols2>
                <<<blocks_num_combine, block_dim_combine, 0, main_stream>>>
                ((float *) KQV->data, dst_tmp_meta.ptr,
                 Q->ne[1], Q->ne[2], K->ne[2], nblocks_sk,
                 gqa_ratio, bpt, fd0, fd1, fd2);
        } else if (ntiles_dst % blocks_num.x != 0) {
            // General fixup for the cases where nblocks_stream_k < ntiles_dst.
            const int total_work = ntiles_KV * ntiles_dst;

            const uint3 fd_k_j_z_ne12 = init_fastdiv_values(ntiles_KV * ntiles_x * ntiles_z_gqa * K->ne[2]);
            const uint3 fd_k_j_z      = init_fastdiv_values(ntiles_KV * ntiles_x * ntiles_z_gqa);
            const uint3 fd_k_j        = init_fastdiv_values(ntiles_KV * ntiles_x);
            const uint3 fd_k          = init_fastdiv_values(ntiles_KV);

            const dim3 block_dim_combine(DV, 1, 1);
            const dim3 blocks_num_combine = {blocks_num.x, ncols1, ncols2};

            flash_attn_stream_k_fixup_general<DV, ncols1, ncols2>
                <<<blocks_num_combine, block_dim_combine, 0, main_stream>>>
                ((float *) KQV->data, dst_tmp_meta.ptr,
                 Q->ne[1], Q->ne[2], gqa_ratio, total_work,
                 fd_k_j_z_ne12, fd_k_j_z, fd_k_j, fd_k);
        }
    } else if (parallel_blocks > 1) {
        const dim3 block_dim_combine(DV, 1, 1);
        const dim3 blocks_num_combine(Q->ne[1], Q->ne[2], Q->ne[3]);
        const size_t nbytes_shared_combine = parallel_blocks*sizeof(float2);

        flash_attn_combine_results<DV>
            <<<blocks_num_combine, block_dim_combine, nbytes_shared_combine, main_stream>>>
            (dst_tmp.ptr, dst_tmp_meta.ptr, (float *) KQV->data, parallel_blocks);
    }
    CUDA_CHECK(cudaGetLastError());

    // TBQP: K and V are spatial (IWHT in K dequant). Output is spatial. No output IWHT.
    // Q WHT buffer is persistent (thread_local), no free needed per call.
}
