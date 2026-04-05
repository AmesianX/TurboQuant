#include "common.cuh"
#include "fattn-common.cuh"

static int ggml_cuda_fattn_vec_get_nthreads_host(const int cc) {
    return 128;
    GGML_UNUSED(cc);
}

static constexpr __device__ int ggml_cuda_fattn_vec_get_nthreads_device() {
    return 128;
}

// Currently llvm with the amdgcn target does not support unrolling loops
// that contain a break that can not be resolved at compile time.
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpass-failed"
#endif // __clang__
template<int D, int ncols, ggml_type type_K, ggml_type type_V, bool use_logit_softcap, int D_V = D> // D == K/Q head size, D_V == V head size
__launch_bounds__(ggml_cuda_fattn_vec_get_nthreads_device(), 1)
static __global__ void flash_attn_ext_vec(
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
        const char * __restrict__ raw_K_data, const int32_t raw_K_stride,
        const char * __restrict__ Q_wht2_data, const int32_t Q_wht2_stride) {
#ifdef FLASH_ATTN_AVAILABLE

    // Skip unused kernel variants for faster compilation:
    if (use_logit_softcap && !(D == 128 || D == 256)) {
        GGML_UNUSED_VARS(Q, K, V, mask, sinks, KV_max, dst, dst_meta, scale,
            max_bias, m0, m1, n_head_log2, logit_softcap,
            ne00, ne01, ne02, ne03,
                  nb01, nb02, nb03,
            ne10, ne11, ne12, ne13,
                  nb11, nb12, nb13,
                  nb21, nb22, nb23,
                  ne31, ne32, ne33,
                  nb31, nb32, nb33, raw_K_data, raw_K_stride, Q_wht2_data, Q_wht2_stride);
        NO_DEVICE_CODE;
        return;
    }

    //In this kernel Q, K, V are matrices while i, j, k are matrix indices.

    constexpr int cpy_nb = ggml_cuda_get_max_cpy_bytes();
    constexpr int cpy_ne = cpy_nb / 4;

#ifdef GGML_USE_HIP
#ifdef RDNA
    constexpr int nthreads_KQ_q = 2;
#else
    constexpr int nthreads_KQ_q = 4;
#endif // RDNA
    constexpr int nthreads_V_q  = (D/4 < 32 ? D/4 : 32);
#else
    constexpr int nthreads_KQ_q = (D/4 < 32 ? D/4 : 32);
    constexpr int nthreads_V_q  = (D/4 < 32 ? D/4 : 32);
#endif // GGML_USE_HIP

    constexpr int nthreads    = ggml_cuda_fattn_vec_get_nthreads_device();
    constexpr int nthreads_KQ = (type_K == GGML_TYPE_F16 || type_K == GGML_TYPE_BF16) ? 128 / cpy_nb : nthreads_KQ_q;
    constexpr int nthreads_V  = (type_V == GGML_TYPE_F16 || type_V == GGML_TYPE_BF16) ? 128 / cpy_nb : nthreads_V_q;

    static_assert(WARP_SIZE % nthreads_KQ == 0, "bad nthreads_K");
    static_assert(WARP_SIZE % nthreads_V  == 0, "bad nthreads_V");

    constexpr int V_rows_per_thread = (type_V == GGML_TYPE_F16 || type_V == GGML_TYPE_BF16) ? 2*cpy_ne : 4;
    constexpr int V_cols_per_iter   = WARP_SIZE / nthreads_V;

    constexpr vec_dot_KQ_t vec_dot_KQ = get_vec_dot_KQ<type_K, D, nthreads_KQ>();
    constexpr bool Q_q8_1 = type_K != GGML_TYPE_F16 && type_K != GGML_TYPE_BF16
        && type_K != GGML_TYPE_TBQ4_0 && type_K != GGML_TYPE_TBQ3_0
        && type_K != GGML_TYPE_TBQP3_0 && type_K != GGML_TYPE_TBQP4_0
        && type_K != GGML_TYPE_TBQ4_1 && type_K != GGML_TYPE_TBQ3_1
        && type_K != GGML_TYPE_TBQP3_1 && type_K != GGML_TYPE_TBQP4_1
        && type_K != GGML_TYPE_TBQ4_2 && type_K != GGML_TYPE_TBQ3_2
        && type_K != GGML_TYPE_TBQP3_2 && type_K != GGML_TYPE_TBQP4_2
        && type_K != GGML_TYPE_TBQ4_3 && type_K != GGML_TYPE_TBQ3_3
        && type_K != GGML_TYPE_TBQP3_3 && type_K != GGML_TYPE_TBQP4_3
        && type_K != GGML_TYPE_TBQ4_4 && type_K != GGML_TYPE_TBQ3_4
        && type_K != GGML_TYPE_TBQP3_4 && type_K != GGML_TYPE_TBQP4_4;
    constexpr bool is_cross_head_K = type_K == GGML_TYPE_TBQ3_3 || type_K == GGML_TYPE_TBQ4_3
        || type_K == GGML_TYPE_TBQP3_3 || type_K == GGML_TYPE_TBQP4_3;
    constexpr bool is_cross_head_V = type_V == GGML_TYPE_TBQ3_3 || type_V == GGML_TYPE_TBQ4_3;
#ifdef V_DOT2_F32_F16_AVAILABLE
    constexpr dequantize_V_t dequantize_V = get_dequantize_V<type_V, half,  V_rows_per_thread>();
#else
    constexpr dequantize_V_t dequantize_V = get_dequantize_V<type_V, float, V_rows_per_thread>();
#endif // V_DOT2_F32_F16_AVAILABLE

    const int ic0 = blockIdx.x * ncols; // Index of the Q/QKV column to work on.

    const int sequence = blockIdx.z / ne02;
    const int head = blockIdx.z - sequence*ne02;
    const int gqa_ratio = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.
    Q += nb03*sequence + nb02* head              + nb01*ic0;
    K += nb13*sequence + nb12*(head / gqa_ratio);
    V += nb23*sequence + nb22*(head / gqa_ratio);

    const half * maskh  = (const half  *) (mask + nb33*(sequence % ne33) + nb31*ic0);

    const float slope = get_alibi_slope(max_bias, head, n_head_log2, m0, m1);

    static_assert(D   % (2*WARP_SIZE) == 0, "D not divisible by 2*WARP_SIZE == 64.");
    static_assert(D_V % (2*WARP_SIZE) == 0, "D_V not divisible by 2*WARP_SIZE == 64.");
    constexpr int nwarps = nthreads / WARP_SIZE;
    const int tid = WARP_SIZE*threadIdx.y + threadIdx.x;
    __builtin_assume(tid < nthreads);

    constexpr int ne_KQ      = ncols*D;
    constexpr int ne_combine = nwarps*V_cols_per_iter*D_V;
#ifdef V_DOT2_F32_F16_AVAILABLE
    half2            VKQ[ncols][(D_V/2)/nthreads_V] = {{{0.0f, 0.0f}}};
    __shared__ half   KQ[ne_KQ > ne_combine ? ne_KQ : ne_combine];
#else
    float2           VKQ[ncols][(D_V/2)/nthreads_V] = {{{0.0f, 0.0f}}};
    __shared__ float  KQ[ne_KQ > ne_combine ? ne_KQ : ne_combine];
#endif // V_DOT2_F32_F16_AVAILABLE

    float KQ_max[ncols];
    float KQ_sum[ncols];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        KQ_max[j] = -FLT_MAX/2.0f;
        KQ_sum[j] = 0.0f;
    }

    // Convert Q to float2 (f16 K) or q8_1 (quantized K) and store in registers:
#ifdef V_DOT2_F32_F16_AVAILABLE
    half2  Q_reg[ncols][(D/2)/nthreads_KQ]; // Will be initialized completely.
#else
    __align__(16) float2 Q_reg[ncols][(D/2)/nthreads_KQ] = {{{0.0f, 0.0f}}}; // May be only partially initialized.
#endif // V_DOT2_F32_F16_AVAILABLE
    int    Q_i32[ncols][1 > D/(sizeof(int)*nthreads_KQ) ? 1 : D/(sizeof(int)*nthreads_KQ)];
    // For TBQP: Q_ds stores QJL WHT'd query (needs same size as Q_reg)
    constexpr int Q_ds_size = (type_K == GGML_TYPE_TBQP3_0 || type_K == GGML_TYPE_TBQP4_0
                            || type_K == GGML_TYPE_TBQP3_1 || type_K == GGML_TYPE_TBQP4_1
                            || type_K == GGML_TYPE_TBQP3_2 || type_K == GGML_TYPE_TBQP4_2
                            || type_K == GGML_TYPE_TBQP3_4 || type_K == GGML_TYPE_TBQP4_4)
        ? (D/2)/nthreads_KQ
        : (1 > D/(sizeof(int)*nthreads_KQ) ? 1 : D/(sizeof(int)*nthreads_KQ));
    float2  Q_ds[ncols][Q_ds_size];
    if constexpr (Q_q8_1) {
#pragma unroll
        for (int j0 = 0; j0 < ncols; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

            if (j0 + nwarps > ncols && j >= ncols) {
                break;
            }

            // Reuse KQ as temporary storage for converting Q to q8_1:
            int    * tmp_q_i32 = (int    *) &KQ[j*D];
            float2 * tmp_q_ds  = (float2 *) (tmp_q_i32 + D/sizeof(int));

            // Set memory to zero if out of bounds:
            if (ncols > 1 && ic0 + j >= int(ne01.z)) {
#pragma unroll
                for (int i0 = 0; i0 < int(D/sizeof(int)); i0 += WARP_SIZE) {
                    const int i = i0 + threadIdx.x;

                    if (i0 + WARP_SIZE <= int(D/sizeof(int)) || i < int(D/sizeof(int))) {
                        tmp_q_i32[i] = 0;
                    }
                }
                if (threadIdx.x < D/QK8_1) {
                    tmp_q_ds[threadIdx.x] = make_float2(0.0f, 0.0f);
                }
            } else {
                const float * Q_f = (const float *) (Q + j*nb01);
                constexpr int nthreads_quantize = D/sizeof(int) < WARP_SIZE ? D/sizeof(int) : WARP_SIZE;
#pragma unroll
                for (int i0 = 0; i0 < int(D/sizeof(int)); i0 += nthreads_quantize) {
                    quantize_q8_1_to_shared<float2, nthreads_quantize>
                        (Q_f + i0*sizeof(int), scale, tmp_q_i32 + i0, tmp_q_ds + i0/QI8_1);
                }
            }
        }

        __syncthreads();

#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            int    * tmp_q_i32 = (int    *) &KQ[j*D];
            float2 * tmp_q_ds  = (float2 *) (tmp_q_i32 + D/sizeof(int));

#pragma unroll
            for (int i0 = 0; i0 < int(D/sizeof(int)); i0 += nthreads_KQ) {
                const int i = i0 + (nthreads_KQ == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_KQ);

                Q_i32[j][i0/nthreads_KQ] = tmp_q_i32[i];
                Q_ds[j][i0/nthreads_KQ]  = tmp_q_ds[i/QI8_1];
            }
        }

        __syncthreads();
    } else if constexpr (type_K == GGML_TYPE_TBQ4_0 || type_K == GGML_TYPE_TBQ3_0
                      || type_K == GGML_TYPE_TBQP4_0 || type_K == GGML_TYPE_TBQP3_0) {
        // TurboQuant KV: pre-compute WHT(sign1*query) for MSE, store in Q_reg
        // For TBQP: also compute WHT(sign2*query) for QJL, store in Q_ds
        static constexpr uint8_t tbq_signs[32] = {
            0xa7,0x3b,0x91,0xf4,0x6d,0xc2,0x58,0x0e,
            0xb3,0x7f,0x24,0xd6,0x89,0x45,0xea,0x1c,
            0x63,0xaf,0xd8,0x52,0x97,0x0b,0xe1,0x3d,
            0x76,0xc4,0x19,0xfe,0x4a,0x85,0x2c,0xdb,
        };
        // Second sign pattern for QJL SRHT (must match quantize_f32_tbqp*_block)
        static constexpr uint8_t qjl_signs[32] = {
            0xd3,0x4e,0xa8,0x17,0x9c,0x5b,0xe6,0x31,
            0x72,0xb9,0x0d,0xf5,0x43,0x8a,0x6e,0xc7,
            0x58,0x2f,0x94,0xe1,0xb6,0x3d,0x0a,0x7c,
            0xc5,0x61,0xd8,0x4f,0xa3,0x97,0x1e,0x85,
        };

        // 512-element sign pattern for single-pass D=512 WHT
        static constexpr uint8_t tbq_signs_512[64] = {
            0xa7,0x3b,0x91,0xf4,0x6d,0xc2,0x58,0x0e,
            0xb3,0x7f,0x24,0xd6,0x89,0x45,0xea,0x1c,
            0x63,0xaf,0xd8,0x52,0x97,0x0b,0xe1,0x3d,
            0x76,0xc4,0x19,0xfe,0x4a,0x85,0x2c,0xdb,
            0xd3,0x4e,0xa8,0x17,0x9c,0x5b,0xe6,0x31,
            0x72,0xb9,0x0d,0xf5,0x43,0x8a,0x6e,0xc7,
            0x58,0x2f,0x94,0xe1,0xb6,0x3d,0x0a,0x7c,
            0xc5,0x61,0xd8,0x4f,0xa3,0x97,0x1e,0x85,
        };

        for (int j = 0; j < ncols; ++j) {
            float * Q_wht = (float *) &KQ[0];
            const float * Q_f = (const float *) (Q + j*nb01);

            if constexpr (D == 512) {
                // === D=512: Single-pass 512-point WHT ===
                // 128 threads handle 4 elements each: tid, tid+128, tid+256, tid+384
                float f0, f1, f2, f3;
                {
                    const uint8_t * signs = tbq_signs_512;
                    float sign0 = ((signs[(tid)      >> 3] >> ((tid)      & 7)) & 1) ? -1.0f : 1.0f;
                    float sign1 = ((signs[(tid+128)  >> 3] >> ((tid+128)  & 7)) & 1) ? -1.0f : 1.0f;
                    float sign2 = ((signs[(tid+256)  >> 3] >> ((tid+256)  & 7)) & 1) ? -1.0f : 1.0f;
                    float sign3 = ((signs[(tid+384)  >> 3] >> ((tid+384)  & 7)) & 1) ? -1.0f : 1.0f;
                    float q0 = (ncols > 1 && ic0 + j >= int(ne01.z)) ? 0.0f : Q_f[tid];
                    float q1 = (ncols > 1 && ic0 + j >= int(ne01.z)) ? 0.0f : Q_f[tid + 128];
                    float q2 = (ncols > 1 && ic0 + j >= int(ne01.z)) ? 0.0f : Q_f[tid + 256];
                    float q3 = (ncols > 1 && ic0 + j >= int(ne01.z)) ? 0.0f : Q_f[tid + 384];
                    f0 = q0 * sign0; f1 = q1 * sign1; f2 = q2 * sign2; f3 = q3 * sign3;
                }
                // Stage 8 (stride 256): register-local pairs (0,2) and (1,3)
                { float u = f0, v = f2; f0 = u + v; f2 = u - v; }
                { float u = f1, v = f3; f1 = u + v; f3 = u - v; }
                // Stage 7 (stride 128): register-local pairs (0,1) and (2,3)
                { float u = f0, v = f1; f0 = u + v; f1 = u - v; }
                { float u = f2, v = f3; f2 = u + v; f3 = u - v; }
                // Stages 0-4: warp shuffle on each of the 4 elements
                #pragma unroll
                for (int s = 0; s < 5; s++) {
                    float o0 = __shfl_xor_sync(0xffffffff, f0, 1 << s, WARP_SIZE);
                    float o1 = __shfl_xor_sync(0xffffffff, f1, 1 << s, WARP_SIZE);
                    float o2 = __shfl_xor_sync(0xffffffff, f2, 1 << s, WARP_SIZE);
                    float o3 = __shfl_xor_sync(0xffffffff, f3, 1 << s, WARP_SIZE);
                    if (tid & (1 << s)) { f0 = o0 - f0; f1 = o1 - f1; f2 = o2 - f2; f3 = o3 - f3; }
                    else                { f0 = f0 + o0; f1 = f1 + o1; f2 = f2 + o2; f3 = f3 + o3; }
                }
                // Stages 5-6: shared memory (need 512 floats)
                Q_wht[tid] = f0; Q_wht[tid+128] = f1; Q_wht[tid+256] = f2; Q_wht[tid+384] = f3;
                __syncthreads();
                { float u0 = Q_wht[tid],     v0 = Q_wht[tid     ^ 32];
                  float u1 = Q_wht[tid+128], v1 = Q_wht[(tid+128) ^ 32];
                  float u2 = Q_wht[tid+256], v2 = Q_wht[(tid+256) ^ 32];
                  float u3 = Q_wht[tid+384], v3 = Q_wht[(tid+384) ^ 32];
                  f0 = (tid & 32) ? v0 - u0 : u0 + v0; f1 = (tid & 32) ? v1 - u1 : u1 + v1;
                  f2 = (tid & 32) ? v2 - u2 : u2 + v2; f3 = (tid & 32) ? v3 - u3 : u3 + v3; }
                Q_wht[tid] = f0; Q_wht[tid+128] = f1; Q_wht[tid+256] = f2; Q_wht[tid+384] = f3;
                __syncthreads();
                { float u0 = Q_wht[tid],     v0 = Q_wht[tid     ^ 64];
                  float u1 = Q_wht[tid+128], v1 = Q_wht[(tid+128) ^ 64];
                  float u2 = Q_wht[tid+256], v2 = Q_wht[(tid+256) ^ 64];
                  float u3 = Q_wht[tid+384], v3 = Q_wht[(tid+384) ^ 64];
                  f0 = (tid & 64) ? v0 - u0 : u0 + v0; f1 = (tid & 64) ? v1 - u1 : u1 + v1;
                  f2 = (tid & 64) ? v2 - u2 : u2 + v2; f3 = (tid & 64) ? v3 - u3 : u3 + v3; }
                Q_wht[tid] = f0; Q_wht[tid+128] = f1; Q_wht[tid+256] = f2; Q_wht[tid+384] = f3;
                __syncthreads();

                // Store all 512 WHT'd values in Q_reg (scale/512 applied — single 512-block norm)
                for (int i0 = 0; i0 < D/2; i0 += nthreads_KQ) {
                    const int i = i0 + (nthreads_KQ == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_KQ);
                    const float v0 = Q_wht[2*i] * scale / 512.0f;
                    const float v1 = Q_wht[2*i+1] * scale / 512.0f;
#ifdef V_DOT2_F32_F16_AVAILABLE
                    Q_reg[j][i0/nthreads_KQ] = make_half2(__float2half(v0), __float2half(v1));
#else
                    Q_reg[j][i0/nthreads_KQ] = make_float2(v0, v1);
#endif
                }
                __syncthreads();
            } else {
                // === D=256: Original single-pass 256-point WHT ===
                {
                    float f0, f1;
                    {
                        float sign0 = ((tbq_signs[tid >> 3] >> (tid & 7)) & 1) ? -1.0f : 1.0f;
                        float sign1 = ((tbq_signs[(tid+128) >> 3] >> ((tid+128) & 7)) & 1) ? -1.0f : 1.0f;
                        float q0 = (ncols > 1 && ic0 + j >= int(ne01.z)) ? 0.0f : Q_f[tid];
                        float q1 = (ncols > 1 && ic0 + j >= int(ne01.z)) ? 0.0f : Q_f[tid + 128];
                        f0 = q0 * sign0;
                        f1 = q1 * sign1;
                    }
                    { float u = f0, v = f1; f0 = u + v; f1 = u - v; }
                    #pragma unroll
                    for (int s = 0; s < 5; s++) {
                        float o0 = __shfl_xor_sync(0xffffffff, f0, 1 << s, WARP_SIZE);
                        float o1 = __shfl_xor_sync(0xffffffff, f1, 1 << s, WARP_SIZE);
                        if (tid & (1 << s)) { f0 = o0 - f0; f1 = o1 - f1; }
                        else                { f0 = f0 + o0; f1 = f1 + o1; }
                    }
                    Q_wht[tid] = f0; Q_wht[tid + 128] = f1;
                    __syncthreads();
                    { float u0 = Q_wht[tid], v0 = Q_wht[tid ^ 32];
                      float u1 = Q_wht[tid+128], v1 = Q_wht[(tid+128) ^ 32];
                      f0 = (tid & 32) ? (v0 - u0) : (u0 + v0);
                      f1 = (tid & 32) ? (v1 - u1) : (u1 + v1); }
                    Q_wht[tid] = f0; Q_wht[tid + 128] = f1;
                    __syncthreads();
                    { float u0 = Q_wht[tid], v0 = Q_wht[tid ^ 64];
                      float u1 = Q_wht[tid+128], v1 = Q_wht[(tid+128) ^ 64];
                      f0 = (tid & 64) ? (v0 - u0) : (u0 + v0);
                      f1 = (tid & 64) ? (v1 - u1) : (u1 + v1); }
                    Q_wht[tid] = f0; Q_wht[tid + 128] = f1;
                    __syncthreads();
                }

                for (int i0 = 0; i0 < D/2; i0 += nthreads_KQ) {
                    const int i = i0 + (nthreads_KQ == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_KQ);
                    const float v0 = Q_wht[2*i] * scale / (float)D;
                    const float v1 = Q_wht[2*i+1] * scale / (float)D;
#ifdef V_DOT2_F32_F16_AVAILABLE
                    Q_reg[j][i0/nthreads_KQ] = make_half2(__float2half(v0), __float2half(v1));
#else
                    Q_reg[j][i0/nthreads_KQ] = make_float2(v0, v1);
#endif
                }
                __syncthreads();
            }

            // === WHT 2: QJL part (signs2) -- for TBQP types, D=256 only ===
            // Note: D=512 TBQP auto-downgraded to TBQ in common.cpp (QJL ineffective at D=512)
            if constexpr ((type_K == GGML_TYPE_TBQP3_0 || type_K == GGML_TYPE_TBQP4_0) && D <= 256) {
                {
                    // D=256: original single-block QJL WHT
                    constexpr float qjl_factor = 1.2533f;
                    const float sf = scale * qjl_factor / (256.0f * 256.0f);

                    // Redo MSE WHT
                    {
                        float f0, f1;
                        {
                            float sign0 = ((tbq_signs[tid >> 3] >> (tid & 7)) & 1) ? -1.0f : 1.0f;
                            float sign1 = ((tbq_signs[(tid+128) >> 3] >> ((tid+128) & 7)) & 1) ? -1.0f : 1.0f;
                            float q0 = (ncols > 1 && ic0 + j >= int(ne01.z)) ? 0.0f : Q_f[tid];
                            float q1 = (ncols > 1 && ic0 + j >= int(ne01.z)) ? 0.0f : Q_f[tid + 128];
                            f0 = q0 * sign0; f1 = q1 * sign1;
                        }
                        { float u = f0, v = f1; f0 = u + v; f1 = u - v; }
                        #pragma unroll
                        for (int s = 0; s < 5; s++) {
                            float o0 = __shfl_xor_sync(0xffffffff, f0, 1 << s, WARP_SIZE);
                            float o1 = __shfl_xor_sync(0xffffffff, f1, 1 << s, WARP_SIZE);
                            if (tid & (1 << s)) { f0 = o0 - f0; f1 = o1 - f1; }
                            else                { f0 = f0 + o0; f1 = f1 + o1; }
                        }
                        Q_wht[tid] = f0; Q_wht[tid + 128] = f1;
                        __syncthreads();
                        { float u0 = Q_wht[tid], v0 = Q_wht[tid ^ 32]; float u1 = Q_wht[tid+128], v1 = Q_wht[(tid+128) ^ 32];
                          f0 = (tid & 32) ? (v0 - u0) : (u0 + v0); f1 = (tid & 32) ? (v1 - u1) : (u1 + v1); }
                        Q_wht[tid] = f0; Q_wht[tid + 128] = f1;
                        __syncthreads();
                        { float u0 = Q_wht[tid], v0 = Q_wht[tid ^ 64]; float u1 = Q_wht[tid+128], v1 = Q_wht[(tid+128) ^ 64];
                          f0 = (tid & 64) ? (v0 - u0) : (u0 + v0); f1 = (tid & 64) ? (v1 - u1) : (u1 + v1); }
                        Q_wht[tid] = f0; Q_wht[tid + 128] = f1;
                        __syncthreads();
                    }

                    // Apply QJL signs2 WHT
                    {
                        float f0, f1;
                        {
                            float sign0 = ((qjl_signs[tid >> 3] >> (tid & 7)) & 1) ? -1.0f : 1.0f;
                            float sign1 = ((qjl_signs[(tid+128) >> 3] >> ((tid+128) & 7)) & 1) ? -1.0f : 1.0f;
                            f0 = Q_wht[tid] * sign0;
                            f1 = Q_wht[tid + 128] * sign1;
                        }
                        { float u = f0, v = f1; f0 = u + v; f1 = u - v; }
                        #pragma unroll
                        for (int s = 0; s < 5; s++) {
                            float o0 = __shfl_xor_sync(0xffffffff, f0, 1 << s, WARP_SIZE);
                            float o1 = __shfl_xor_sync(0xffffffff, f1, 1 << s, WARP_SIZE);
                            if (tid & (1 << s)) { f0 = o0 - f0; f1 = o1 - f1; }
                            else                { f0 = f0 + o0; f1 = f1 + o1; }
                        }
                        Q_wht[tid] = f0; Q_wht[tid + 128] = f1;
                        __syncthreads();
                        { float u0 = Q_wht[tid], v0 = Q_wht[tid ^ 32]; float u1 = Q_wht[tid+128], v1 = Q_wht[(tid+128) ^ 32];
                          f0 = (tid & 32) ? (v0 - u0) : (u0 + v0); f1 = (tid & 32) ? (v1 - u1) : (u1 + v1); }
                        Q_wht[tid] = f0; Q_wht[tid + 128] = f1;
                        __syncthreads();
                        { float u0 = Q_wht[tid], v0 = Q_wht[tid ^ 64]; float u1 = Q_wht[tid+128], v1 = Q_wht[(tid+128) ^ 64];
                          f0 = (tid & 64) ? (v0 - u0) : (u0 + v0); f1 = (tid & 64) ? (v1 - u1) : (u1 + v1); }
                        Q_wht[tid] = f0; Q_wht[tid + 128] = f1;
                        __syncthreads();
                    }

                    for (int i0 = 0; i0 < D/2; i0 += nthreads_KQ) {
                        const int i = i0 + (nthreads_KQ == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_KQ);
                        Q_ds[j][i0/nthreads_KQ] = make_float2(Q_wht[2*i] * sf, Q_wht[2*i+1] * sf);
                    }
                    __syncthreads();
                }
            }
    } else if constexpr (type_K == GGML_TYPE_TBQ4_1 || type_K == GGML_TYPE_TBQ3_1
                      || type_K == GGML_TYPE_TBQP4_1 || type_K == GGML_TYPE_TBQP3_1) {
        // TurboQuant 128-block: D=128, nthreads=128, 1:1 thread-element mapping
        // WHT has 7 stages: stages 0-4 warp shuffle, stages 5-6 shared memory
        // No stage 7 (stride 128) needed — each thread handles exactly 1 element
        static constexpr uint8_t tbq_signs[16] = {
            0xa7,0x3b,0x91,0xf4,0x6d,0xc2,0x58,0x0e,
            0xb3,0x7f,0x24,0xd6,0x89,0x45,0xea,0x1c,
        };

        for (int j = 0; j < ncols; ++j) {
            float * Q_wht = (float *) &KQ[0];
            const float * Q_f = (const float *) (Q + j*nb01);

            // === WHT 1: MSE part (signs1) — FP32 butterfly ===
            {
                float f0;
                {
                    float sign0 = ((tbq_signs[tid >> 3] >> (tid & 7)) & 1) ? -1.0f : 1.0f;
                    float q0 = (ncols > 1 && ic0 + j >= int(ne01.z)) ? 0.0f : Q_f[tid];
                    f0 = q0 * sign0;
                }
                // Stages 0-4: warp shuffle
                #pragma unroll
                for (int s = 0; s < 5; s++) {
                    float o0 = __shfl_xor_sync(0xffffffff, f0, 1 << s, WARP_SIZE);
                    if (tid & (1 << s)) { f0 = o0 - f0; }
                    else                { f0 = f0 + o0; }
                }
                // Stage 5 (stride 32): shared memory
                Q_wht[tid] = f0;
                __syncthreads();
                { float u0 = Q_wht[tid], v0 = Q_wht[tid ^ 32]; f0 = (tid & 32) ? (v0 - u0) : (u0 + v0); }
                // Stage 6 (stride 64): shared memory
                Q_wht[tid] = f0;
                __syncthreads();
                { float u0 = Q_wht[tid], v0 = Q_wht[tid ^ 64]; f0 = (tid & 64) ? (v0 - u0) : (u0 + v0); }
                Q_wht[tid] = f0;
                __syncthreads();
            }

            // Store MSE WHT'd query in Q_reg (scale/D applied)
            for (int i0 = 0; i0 < D/2; i0 += nthreads_KQ) {
                const int i = i0 + (nthreads_KQ == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_KQ);
                const float v0 = Q_wht[2*i] * scale / (float)D;
                const float v1 = Q_wht[2*i+1] * scale / (float)D;
#ifdef V_DOT2_F32_F16_AVAILABLE
                Q_reg[j][i0/nthreads_KQ] = make_half2(__float2half(v0), __float2half(v1));
#else
                Q_reg[j][i0/nthreads_KQ] = make_float2(v0, v1);
#endif
            }

            __syncthreads(); // ensure all warps finished Q_reg load before next j overwrites KQ

            // Direct Sign: no second WHT needed for _1 TBQP types
            // vec_dot reuses Q_reg (MSE WHT'd query) directly for sign correction
        }
    } else if constexpr (type_K == GGML_TYPE_TBQ4_2 || type_K == GGML_TYPE_TBQ3_2
                      || type_K == GGML_TYPE_TBQP4_2 || type_K == GGML_TYPE_TBQP3_2
                      || type_K == GGML_TYPE_TBQ4_3 || type_K == GGML_TYPE_TBQ3_3
                      || type_K == GGML_TYPE_TBQP4_3 || type_K == GGML_TYPE_TBQP3_3) {
        // TurboQuant 64-block: D=64, nthreads=128, tid 0-63 participate in WHT
        // WHT has 6 stages: stages 0-4 warp shuffle, stage 5 shared memory
        // For cross-head _3: use kv_head-specific 64-bit sign slice from 512-bit pattern
        static constexpr uint8_t tbq_signs_full[64] = {
            0xa7,0x3b,0x91,0xf4,0x6d,0xc2,0x58,0x0e,  // head 0 (same as _2)
            0xb3,0x7f,0x24,0xd6,0x89,0x45,0xea,0x1c,  // head 1
            0x63,0xaf,0xd8,0x52,0x97,0x0b,0xe1,0x3d,  // head 2
            0x76,0xc4,0x19,0xfe,0x4a,0x85,0x2c,0xdb,  // head 3
            0xd3,0x4e,0xa8,0x17,0x9c,0x5b,0xe6,0x31,  // head 4
            0x72,0xb9,0x0d,0xf5,0x43,0x8a,0x6e,0xc7,  // head 5
            0x58,0x2f,0x94,0xe1,0xb6,0x3d,0x0a,0x7c,  // head 6
            0xc5,0x61,0xd8,0x4f,0xa3,0x97,0x1e,0x85,  // head 7
        };
        // For _2 types: always use head 0's signs. For _3: use kv_head-specific slice.
        const int q_signs_offset = (is_cross_head_K) ? ((head / gqa_ratio) % 8) * 8 : 0;
        const uint8_t * tbq_signs = tbq_signs_full + q_signs_offset;

        for (int j = 0; j < ncols; ++j) {
            float * Q_wht = (float *) &KQ[0];
            const float * Q_f = (const float *) (Q + j*nb01);

            // === WHT 1: MSE part — FP32 butterfly ===
            {
                float f0 = 0.0f;
                if (tid < D) {
                    float sign0 = ((tbq_signs[tid >> 3] >> (tid & 7)) & 1) ? -1.0f : 1.0f;
                    float q0 = (ncols > 1 && ic0 + j >= int(ne01.z)) ? 0.0f : Q_f[tid];
                    f0 = q0 * sign0;
                }
                // Stages 0-4: warp shuffle (tid >= 64 carries f0=0, no data corruption)
                #pragma unroll
                for (int s = 0; s < 5; s++) {
                    float o0 = __shfl_xor_sync(0xffffffff, f0, 1 << s, WARP_SIZE);
                    if (tid & (1 << s)) { f0 = o0 - f0; }
                    else                { f0 = f0 + o0; }
                }
                // Stage 5 (stride 32): shared memory
                if (tid < D) { Q_wht[tid] = f0; }
                __syncthreads();
                if (tid < D) {
                    float u0 = Q_wht[tid], v0 = Q_wht[tid ^ 32];
                    f0 = (tid & 32) ? (v0 - u0) : (u0 + v0);
                }
                if (tid < D) { Q_wht[tid] = f0; }
                __syncthreads();
            }

            for (int i0 = 0; i0 < D/2; i0 += nthreads_KQ) {
                const int i = i0 + (nthreads_KQ == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_KQ);
                const float v0 = Q_wht[2*i] * scale / (float)D;
                const float v1 = Q_wht[2*i+1] * scale / (float)D;
#ifdef V_DOT2_F32_F16_AVAILABLE
                Q_reg[j][i0/nthreads_KQ] = make_half2(__float2half(v0), __float2half(v1));
#else
                Q_reg[j][i0/nthreads_KQ] = make_float2(v0, v1);
#endif
            }

            __syncthreads(); // ensure all warps finished Q_reg load before next j overwrites KQ

            // Direct Sign: no second WHT needed for _2 TBQP types
        }
    } else if constexpr (type_K == GGML_TYPE_TBQ4_4 || type_K == GGML_TYPE_TBQ3_4
                      || type_K == GGML_TYPE_TBQP4_4 || type_K == GGML_TYPE_TBQP3_4) {
        // TurboQuant 576-block: D=576, split 256+256+64
        // Three sub-block WHTs applied sequentially, each reuses smem
        static constexpr uint8_t signs_256[32] = {
            0xa7,0x3b,0x91,0xf4,0x6d,0xc2,0x58,0x0e,0xb3,0x7f,0x24,0xd6,0x89,0x45,0xea,0x1c,
            0x63,0xaf,0xd8,0x52,0x97,0x0b,0xe1,0x3d,0x76,0xc4,0x19,0xfe,0x4a,0x85,0x2c,0xdb,
        };
        static constexpr uint8_t qjl_signs_256[32] = {
            0xd3,0x4e,0xa8,0x17,0x9c,0x5b,0xe6,0x31,0x72,0xb9,0x0d,0xf5,0x43,0x8a,0x6e,0xc7,
            0x58,0x2f,0x94,0xe1,0xb6,0x3d,0x0a,0x7c,0xc5,0x61,0xd8,0x4f,0xa3,0x97,0x1e,0x85,
        };
        for (int j = 0; j < ncols; ++j) {
            float * Q_wht = (float *) &KQ[0]; // 576 floats workspace
            const float * Q_f = (const float *) (Q + j*nb01);

            // === Sub-block 1: WHT_256 on Q[0:256] ===
            {
                float f0, f1;
                {
                    float sign0 = ((signs_256[tid >> 3] >> (tid & 7)) & 1) ? -1.0f : 1.0f;
                    float sign1 = ((signs_256[(tid+128) >> 3] >> ((tid+128) & 7)) & 1) ? -1.0f : 1.0f;
                    float q0 = (ncols > 1 && ic0 + j >= int(ne01.z)) ? 0.0f : Q_f[tid];
                    float q1 = (ncols > 1 && ic0 + j >= int(ne01.z)) ? 0.0f : Q_f[tid + 128];
                    f0 = q0 * sign0; f1 = q1 * sign1;
                }
                { float u = f0, v = f1; f0 = u + v; f1 = u - v; }
                #pragma unroll
                for (int s = 0; s < 5; s++) {
                    float o0 = __shfl_xor_sync(0xffffffff, f0, 1 << s, WARP_SIZE);
                    float o1 = __shfl_xor_sync(0xffffffff, f1, 1 << s, WARP_SIZE);
                    if (tid & (1 << s)) { f0 = o0 - f0; f1 = o1 - f1; }
                    else                { f0 = f0 + o0; f1 = f1 + o1; }
                }
                Q_wht[tid] = f0; Q_wht[tid + 128] = f1;
                __syncthreads();
                { float u0 = Q_wht[tid], v0 = Q_wht[tid ^ 32]; float u1 = Q_wht[tid+128], v1 = Q_wht[(tid+128) ^ 32];
                  f0 = (tid & 32) ? v0 - u0 : u0 + v0; f1 = (tid & 32) ? v1 - u1 : u1 + v1; }
                Q_wht[tid] = f0; Q_wht[tid + 128] = f1;
                __syncthreads();
                { float u0 = Q_wht[tid], v0 = Q_wht[tid ^ 64]; float u1 = Q_wht[tid+128], v1 = Q_wht[(tid+128) ^ 64];
                  f0 = (tid & 64) ? v0 - u0 : u0 + v0; f1 = (tid & 64) ? v1 - u1 : u1 + v1; }
                __syncthreads();
                Q_wht[tid] = f0; Q_wht[tid + 128] = f1;
                __syncthreads();
            }
            // Store sub-block 1 in Q_reg[0..3] (256/2/nthreads_KQ elements, scale/256)
            for (int i0 = 0; i0 < 128; i0 += nthreads_KQ) {
                const int i = i0 + (nthreads_KQ == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_KQ);
                const float v0 = Q_wht[2*i] * scale / 256.0f;
                const float v1 = Q_wht[2*i+1] * scale / 256.0f;
#ifdef V_DOT2_F32_F16_AVAILABLE
                Q_reg[j][i0/nthreads_KQ] = make_half2(__float2half(v0), __float2half(v1));
#else
                Q_reg[j][i0/nthreads_KQ] = make_float2(v0, v1);
#endif
            }

            __syncthreads(); // Barrier: all warps must finish reading Q_wht before sub-block 2 overwrites it

            // QJL WHT for sub-block 1 (TBQP only)
            if constexpr (type_K == GGML_TYPE_TBQP3_4 || type_K == GGML_TYPE_TBQP4_4) {
                float f0, f1;
                { float sign0 = ((qjl_signs_256[tid >> 3] >> (tid & 7)) & 1) ? -1.0f : 1.0f;
                  float sign1 = ((qjl_signs_256[(tid+128) >> 3] >> ((tid+128) & 7)) & 1) ? -1.0f : 1.0f;
                  f0 = Q_wht[tid] * sign0; f1 = Q_wht[tid + 128] * sign1; }
                { float u = f0, v = f1; f0 = u + v; f1 = u - v; }
                #pragma unroll
                for (int s = 0; s < 5; s++) {
                    float o0 = __shfl_xor_sync(0xffffffff, f0, 1 << s, WARP_SIZE);
                    float o1 = __shfl_xor_sync(0xffffffff, f1, 1 << s, WARP_SIZE);
                    if (tid & (1 << s)) { f0 = o0 - f0; f1 = o1 - f1; }
                    else                { f0 = f0 + o0; f1 = f1 + o1; }
                }
                Q_wht[tid] = f0; Q_wht[tid + 128] = f1;
                __syncthreads();
                { float u0 = Q_wht[tid], v0 = Q_wht[tid ^ 32]; float u1 = Q_wht[tid+128], v1 = Q_wht[(tid+128) ^ 32];
                  f0 = (tid & 32) ? v0 - u0 : u0 + v0; f1 = (tid & 32) ? v1 - u1 : u1 + v1; }
                Q_wht[tid] = f0; Q_wht[tid + 128] = f1;
                __syncthreads();
                { float u0 = Q_wht[tid], v0 = Q_wht[tid ^ 64]; float u1 = Q_wht[tid+128], v1 = Q_wht[(tid+128) ^ 64];
                  f0 = (tid & 64) ? v0 - u0 : u0 + v0; f1 = (tid & 64) ? v1 - u1 : u1 + v1; }
                __syncthreads();
                constexpr float qjl_factor = 1.2533f;
                const float sf = scale * qjl_factor / (256.0f * 256.0f);
                Q_wht[tid] = f0; Q_wht[tid + 128] = f1;
                __syncthreads();
                for (int i0 = 0; i0 < 128; i0 += nthreads_KQ) {
                    const int i = i0 + (nthreads_KQ == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_KQ);
                    Q_ds[j][i0/nthreads_KQ] = make_float2(Q_wht[2*i] * sf, Q_wht[2*i+1] * sf);
                }
                __syncthreads();
            }

            // === Sub-block 2: WHT_256 on Q[256:512] ===
            {
                float f0, f1;
                {
                    float sign0 = ((signs_256[tid >> 3] >> (tid & 7)) & 1) ? -1.0f : 1.0f;
                    float sign1 = ((signs_256[(tid+128) >> 3] >> ((tid+128) & 7)) & 1) ? -1.0f : 1.0f;
                    float q0 = (ncols > 1 && ic0 + j >= int(ne01.z)) ? 0.0f : Q_f[256 + tid];
                    float q1 = (ncols > 1 && ic0 + j >= int(ne01.z)) ? 0.0f : Q_f[256 + tid + 128];
                    f0 = q0 * sign0; f1 = q1 * sign1;
                }
                { float u = f0, v = f1; f0 = u + v; f1 = u - v; }
                #pragma unroll
                for (int s = 0; s < 5; s++) {
                    float o0 = __shfl_xor_sync(0xffffffff, f0, 1 << s, WARP_SIZE);
                    float o1 = __shfl_xor_sync(0xffffffff, f1, 1 << s, WARP_SIZE);
                    if (tid & (1 << s)) { f0 = o0 - f0; f1 = o1 - f1; }
                    else                { f0 = f0 + o0; f1 = f1 + o1; }
                }
                Q_wht[tid] = f0; Q_wht[tid + 128] = f1;
                __syncthreads();
                { float u0 = Q_wht[tid], v0 = Q_wht[tid ^ 32]; float u1 = Q_wht[tid+128], v1 = Q_wht[(tid+128) ^ 32];
                  f0 = (tid & 32) ? v0 - u0 : u0 + v0; f1 = (tid & 32) ? v1 - u1 : u1 + v1; }
                Q_wht[tid] = f0; Q_wht[tid + 128] = f1;
                __syncthreads();
                { float u0 = Q_wht[tid], v0 = Q_wht[tid ^ 64]; float u1 = Q_wht[tid+128], v1 = Q_wht[(tid+128) ^ 64];
                  f0 = (tid & 64) ? v0 - u0 : u0 + v0; f1 = (tid & 64) ? v1 - u1 : u1 + v1; }
                __syncthreads();
                Q_wht[tid] = f0; Q_wht[tid + 128] = f1;
                __syncthreads();
            }
            // Store sub-block 2 in Q_reg[4..7]
            for (int i0 = 0; i0 < 128; i0 += nthreads_KQ) {
                const int i = i0 + (nthreads_KQ == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_KQ);
                const float v0 = Q_wht[2*i] * scale / 256.0f;
                const float v1 = Q_wht[2*i+1] * scale / 256.0f;
#ifdef V_DOT2_F32_F16_AVAILABLE
                Q_reg[j][128/nthreads_KQ + i0/nthreads_KQ] = make_half2(__float2half(v0), __float2half(v1));
#else
                Q_reg[j][128/nthreads_KQ + i0/nthreads_KQ] = make_float2(v0, v1);
#endif
            }

            __syncthreads(); // Barrier: all warps must finish reading Q_wht before sub-block 3 overwrites it

            // QJL WHT for sub-block 2 (TBQP only)
            if constexpr (type_K == GGML_TYPE_TBQP3_4 || type_K == GGML_TYPE_TBQP4_4) {
                float f0, f1;
                { float sign0 = ((qjl_signs_256[tid >> 3] >> (tid & 7)) & 1) ? -1.0f : 1.0f;
                  float sign1 = ((qjl_signs_256[(tid+128) >> 3] >> ((tid+128) & 7)) & 1) ? -1.0f : 1.0f;
                  f0 = Q_wht[tid] * sign0; f1 = Q_wht[tid + 128] * sign1; }
                { float u = f0, v = f1; f0 = u + v; f1 = u - v; }
                #pragma unroll
                for (int s = 0; s < 5; s++) {
                    float o0 = __shfl_xor_sync(0xffffffff, f0, 1 << s, WARP_SIZE);
                    float o1 = __shfl_xor_sync(0xffffffff, f1, 1 << s, WARP_SIZE);
                    if (tid & (1 << s)) { f0 = o0 - f0; f1 = o1 - f1; }
                    else                { f0 = f0 + o0; f1 = f1 + o1; }
                }
                Q_wht[tid] = f0; Q_wht[tid + 128] = f1;
                __syncthreads();
                { float u0 = Q_wht[tid], v0 = Q_wht[tid ^ 32]; float u1 = Q_wht[tid+128], v1 = Q_wht[(tid+128) ^ 32];
                  f0 = (tid & 32) ? v0 - u0 : u0 + v0; f1 = (tid & 32) ? v1 - u1 : u1 + v1; }
                Q_wht[tid] = f0; Q_wht[tid + 128] = f1;
                __syncthreads();
                { float u0 = Q_wht[tid], v0 = Q_wht[tid ^ 64]; float u1 = Q_wht[tid+128], v1 = Q_wht[(tid+128) ^ 64];
                  f0 = (tid & 64) ? v0 - u0 : u0 + v0; f1 = (tid & 64) ? v1 - u1 : u1 + v1; }
                __syncthreads();
                constexpr float qjl_factor = 1.2533f;
                const float sf = scale * qjl_factor / (256.0f * 256.0f);
                Q_wht[tid] = f0; Q_wht[tid + 128] = f1;
                __syncthreads();
                for (int i0 = 0; i0 < 128; i0 += nthreads_KQ) {
                    const int i = i0 + (nthreads_KQ == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_KQ);
                    Q_ds[j][128/nthreads_KQ + i0/nthreads_KQ] = make_float2(Q_wht[2*i] * sf, Q_wht[2*i+1] * sf);
                }
                __syncthreads();
            }

            // === Sub-block 3: f16 passthrough (rope, no WHT) ===
            for (int i0 = 0; i0 < 32; i0 += nthreads_KQ) {
                const int i = i0 + (nthreads_KQ == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_KQ);
                if (i < 32) {
                    const float v0 = ((ncols > 1 && ic0 + j >= int(ne01.z)) ? 0.0f : Q_f[512 + 2*i]) * scale;
                    const float v1 = ((ncols > 1 && ic0 + j >= int(ne01.z)) ? 0.0f : Q_f[512 + 2*i + 1]) * scale;
#ifdef V_DOT2_F32_F16_AVAILABLE
                    Q_reg[j][256/nthreads_KQ + i0/nthreads_KQ] = make_half2(__float2half(v0), __float2half(v1));
#else
                    Q_reg[j][256/nthreads_KQ + i0/nthreads_KQ] = make_float2(v0, v1);
#endif
                }
            }
            // Sub-block 3: f16 passthrough, no QJL needed (even for TBQP types)

            __syncthreads();
        }
    } else {
#ifdef V_DOT2_F32_F16_AVAILABLE
        const half2 scale_h2 = make_half2(scale, scale);
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            const float2 * Q_j = (const float2 *) (Q + j*nb01);
#pragma unroll
            for (int i0 = 0; i0 < D/2; i0 += nthreads_KQ*cpy_ne) {
                const int i = i0 + (nthreads_KQ == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_KQ)*cpy_ne;

                __align__(16) float2 tmp[cpy_ne] = {{0.0f, 0.0f}};
                if (ncols == 1 || ic0 + j < int(ne01.z)) {
                    ggml_cuda_memcpy_1<cpy_nb>(tmp,            &Q_j[i]);
                    ggml_cuda_memcpy_1<cpy_nb>(tmp + cpy_ne/2, &Q_j[i + cpy_ne/2]);
                }
#pragma unroll
                for (int i1 = 0; i1 < cpy_ne; ++i1) {
                    Q_reg[j][i0/nthreads_KQ + i1] = make_half2(tmp[i1].x, tmp[i1].y);
                }
            }
#pragma unroll
            for (int k = 0; k < (D/2)/nthreads_KQ; ++k) {
                Q_reg[j][k] *= scale_h2;
            }
        }
#else
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            const float2 * Q_j = (const float2 *) (Q + j*nb01);
#pragma unroll
            for (int i0 = 0; i0 < D/2; i0 += nthreads_KQ*cpy_ne) {
                const int i = i0 + (nthreads_KQ == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_KQ)*cpy_ne;
                if (ncols == 1 || ic0 + j < int(ne01.z)) {
                    ggml_cuda_memcpy_1<cpy_nb>(&Q_reg[j][i0/nthreads_KQ],            &Q_j[i]);
                    ggml_cuda_memcpy_1<cpy_nb>(&Q_reg[j][i0/nthreads_KQ + cpy_ne/2], &Q_j[i + cpy_ne/2]);
                }
            }
#pragma unroll
            for (int k = 0; k < (D/2)/nthreads_KQ; ++k) {
                Q_reg[j][k].x *= scale;
                Q_reg[j][k].y *= scale;
            }
        }
#endif // V_DOT2_F32_F16_AVAILABLE
    }

    // Cross-head WHT: kv_head_idx within group of 8 for H_8 sign computation
    [[maybe_unused]] const int kv_head_idx = (is_cross_head_K || is_cross_head_V) ? ((head / gqa_ratio) % 8) : 0;

    const int k_VKQ_max = KV_max ? KV_max[sequence*gridDim.x + blockIdx.x] : ne11;
    K     += blockIdx.y*nthreads * nb11;
    V     += blockIdx.y*nthreads * nb21;
    maskh += blockIdx.y*nthreads;
    for (int k_VKQ_0 = blockIdx.y*nthreads; k_VKQ_0 < k_VKQ_max; k_VKQ_0 += gridDim.y*nthreads,
             // Increment pointers after each loop:
             K += gridDim.y*nthreads*nb11, V += gridDim.y*nthreads*nb21, maskh += gridDim.y*nthreads) {

        // Calculate KQ tile and keep track of new maximum KQ values:
        float KQ_reg[ncols]; // KQ in registers.

        float KQ_max_new[ncols];
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            KQ_max_new[j] = KQ_max[j];
        }

#pragma unroll
        for (int i_KQ_0 = 0; i_KQ_0 < nthreads_KQ; ++i_KQ_0) {
            const int i_KQ = threadIdx.y*WARP_SIZE + (nthreads_KQ == WARP_SIZE ? 0 : (threadIdx.x & ~(nthreads_KQ-1))) + i_KQ_0;

#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                float sum;
                if constexpr (is_cross_head_K) {
                    // Cross-head scoring: combine 8 groups with H_8 signs
                    // H_512 = H_8 ⊗ H_64 → score = (1/8)Σ_g H_8[h][g] · vec_dot(K_g, Q_wht)
                    sum = 0.0f;
                    #pragma unroll
                    for (int g = 0; g < 8; g++) {
                        const char * K_g = K + (int64_t)(g - kv_head_idx) * nb12 + (int64_t)i_KQ * nb11;
                        float partial = vec_dot_KQ(K_g, Q_reg[j], Q_i32[j], Q_ds[j]);
                        const int h8_sign = (__popc(kv_head_idx & g) & 1) ? -1 : 1;
                        sum += h8_sign * partial;
                    }
                    sum = warp_reduce_sum<nthreads_KQ>(sum);
                    sum *= 0.125f; // 1/8
                } else {
                    sum = vec_dot_KQ(K + i_KQ*nb11, Q_reg[j], Q_i32[j], Q_ds[j]);
                    sum = warp_reduce_sum<nthreads_KQ>(sum);
                }
                if (use_logit_softcap) {
                    sum = logit_softcap*tanhf(sum);
                }

                if (mask && (ncols == 1 || ic0 + j < int(ne01.z))) {
                    sum += slope*__half2float(maskh[j*ne11 + i_KQ]);
                }

                KQ_max_new[j] = fmaxf(KQ_max_new[j], sum + FATTN_KQ_MAX_OFFSET);

                if ((nthreads_KQ == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_KQ) == uint32_t(i_KQ_0)) {
                    KQ_reg[j] = sum;
                }
            }
        }

#pragma unroll
        for (int j = 0; j < ncols; ++j) {
#pragma unroll
            for (int offset = nthreads_KQ; offset < WARP_SIZE; offset <<= 1) {
                KQ_max_new[j] = fmaxf(KQ_max_new[j], __shfl_xor_sync(0xFFFFFFFF, KQ_max_new[j], offset, WARP_SIZE));
            }
            const float KQ_max_scale = expf(KQ_max[j] - KQ_max_new[j]);
            KQ_max[j] = KQ_max_new[j];

            KQ_reg[j] = expf(KQ_reg[j] - KQ_max[j]);
            KQ_sum[j] = KQ_sum[j]*KQ_max_scale + KQ_reg[j];
            KQ[j*nthreads + tid] = KQ_reg[j];

#ifdef V_DOT2_F32_F16_AVAILABLE
            const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale, KQ_max_scale);
#pragma unroll
            for (int i_VKQ_0 = 0; i_VKQ_0 < D_V/2; i_VKQ_0 += nthreads_V) {
                VKQ[j][i_VKQ_0/nthreads_V] *= KQ_max_scale_h2;
            }
#else
#pragma unroll
            for (int i_VKQ_0 = 0; i_VKQ_0 < D_V/2; i_VKQ_0 += nthreads_V) {
                VKQ[j][i_VKQ_0/nthreads_V].x *= KQ_max_scale;
                VKQ[j][i_VKQ_0/nthreads_V].y *= KQ_max_scale;
            }
#endif // V_DOT2_F32_F16_AVAILABLE
        }

#ifndef GGML_USE_HIP
        __syncwarp();
#endif // GGML_USE_HIP

#pragma unroll
        for (int k0 = 0; k0 < WARP_SIZE; k0 += V_cols_per_iter) {
            const int k = threadIdx.y*WARP_SIZE + k0 + (nthreads_V == WARP_SIZE ? 0 : threadIdx.x / nthreads_V);

#ifdef V_DOT2_F32_F16_AVAILABLE
            half2 KQ_k[ncols];
#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                KQ_k[j] = __half2half2(KQ[j*nthreads + k]);
            }
#pragma unroll
            for (int i_VKQ_0 = 0; i_VKQ_0 < D_V/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
                half2 tmp[V_rows_per_thread/2];
                if constexpr (type_V == GGML_TYPE_BF16) {
                    float2 tmp_f[V_rows_per_thread/2];
                    dequantize_V(V + k*nb21, tmp_f,
                        2*i_VKQ_0 + (nthreads_V == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_V)*V_rows_per_thread);
#pragma unroll
                    for (int i_VKQ_1 = 0; i_VKQ_1 < V_rows_per_thread/2; ++i_VKQ_1) {
                        tmp[i_VKQ_1] = __float22half2_rn(tmp_f[i_VKQ_1]);
                    }
                } else {
                    dequantize_V(V + k*nb21, tmp,
                        2*i_VKQ_0 + (nthreads_V == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_V)*V_rows_per_thread);
                }
#pragma unroll
                for (int i_VKQ_1 = 0; i_VKQ_1 < V_rows_per_thread/2; ++i_VKQ_1) {
#pragma unroll
                    for (int j = 0; j < ncols; ++j) {
                        VKQ[j][i_VKQ_0/nthreads_V + i_VKQ_1] += tmp[i_VKQ_1]*KQ_k[j];
                    }
                }
            }
#else
            float KQ_k[ncols];
#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                KQ_k[j] = KQ[j*nthreads + k];
            }
#pragma unroll
            for (int i_VKQ_0 = 0; i_VKQ_0 < D_V/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
                float2 tmp[V_rows_per_thread/2];
                const int v_elem = 2*i_VKQ_0 + (nthreads_V == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_V)*V_rows_per_thread;
                if constexpr (is_cross_head_V) {
                    // Cross-head V: load 8 V blocks, combine with (1/8)H_8 inverse
#pragma unroll
                    for (int iv = 0; iv < V_rows_per_thread/2; iv++) tmp[iv] = make_float2(0.0f, 0.0f);
#pragma unroll
                    for (int g = 0; g < 8; g++) {
                        float2 v_g[V_rows_per_thread/2];
                        dequantize_V(V + (int64_t)(g - kv_head_idx) * nb22 + k*nb21, v_g, v_elem);
                        const float sign = (__popc(kv_head_idx & g) & 1) ? -0.125f : 0.125f;
#pragma unroll
                        for (int iv = 0; iv < V_rows_per_thread/2; iv++) {
                            tmp[iv].x += sign * v_g[iv].x;
                            tmp[iv].y += sign * v_g[iv].y;
                        }
                    }
                } else {
                    dequantize_V(V + k*nb21, tmp, v_elem);
                }
#pragma unroll
                for (int i_VKQ_1 = 0; i_VKQ_1 < V_rows_per_thread/2; ++i_VKQ_1) {
#pragma unroll
                    for (int j = 0; j < ncols; ++j) {
                        VKQ[j][i_VKQ_0/nthreads_V + i_VKQ_1].x += tmp[i_VKQ_1].x*KQ_k[j];
                        VKQ[j][i_VKQ_0/nthreads_V + i_VKQ_1].y += tmp[i_VKQ_1].y*KQ_k[j];
                    }
                }
            }
#endif // V_DOT2_F32_F16_AVAILABLE
        }
    }

    if (sinks && blockIdx.y == 0) {
        const float sink = ((const float *) sinks)[head];

#pragma unroll
        for (int j0 = 0; j0 < ncols; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

            if (j0 + nwarps > ncols && j >= ncols) {
                break;
            }

            const float kqmax_new_j = fmaxf(sink, KQ_max[j]);
            const float KQ_max_scale = expf(KQ_max[j] - kqmax_new_j);
            KQ_max[j] = kqmax_new_j;

            KQ_sum[j] = KQ_sum[j]*KQ_max_scale + (threadIdx.x == 0 ? expf(sink - KQ_max[j]) : 0.0f);

#ifdef V_DOT2_F32_F16_AVAILABLE
            const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale, KQ_max_scale);
#pragma unroll
            for (int i_VKQ_0 = 0; i_VKQ_0 < D_V/2; i_VKQ_0 += nthreads_V) {
                VKQ[j][i_VKQ_0/nthreads_V] *= KQ_max_scale_h2;
            }
#else
#pragma unroll
            for (int i_VKQ_0 = 0; i_VKQ_0 < D_V/2; i_VKQ_0 += nthreads_V) {
                VKQ[j][i_VKQ_0/nthreads_V].x *= KQ_max_scale;
                VKQ[j][i_VKQ_0/nthreads_V].y *= KQ_max_scale;
            }
#endif // V_DOT2_F32_F16_AVAILABLE
        }
    }

    __shared__ float KQ_max_shared[ncols][WARP_SIZE];
    __shared__ float KQ_sum_shared[ncols][WARP_SIZE];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        if (threadIdx.y == 0) {
            KQ_max_shared[j][threadIdx.x] = -FLT_MAX/2.0f;
            KQ_sum_shared[j][threadIdx.x] = 0.0f;
        }
    }

    __syncthreads();

#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        if (threadIdx.x == 0) {
            KQ_max_shared[j][threadIdx.y] = KQ_max[j];
        }
    }
    __syncthreads();

#pragma unroll
    for (int j_VKQ = 0; j_VKQ < ncols; ++j_VKQ) {
        if (ncols > 1 && ic0 + j_VKQ >= int(ne01.z)) {
            break;
        }

        float kqmax_new = KQ_max_shared[j_VKQ][threadIdx.x];
        kqmax_new = warp_reduce_max(kqmax_new);
        const float kqmax_scale = expf(KQ_max[j_VKQ] - kqmax_new);
        KQ_max[j_VKQ] = kqmax_new;

#ifdef V_DOT2_F32_F16_AVAILABLE
        half2 * VKQ_tmp = (half2 *) KQ + threadIdx.y*(V_cols_per_iter*D_V/2)
            + (nthreads_V == WARP_SIZE ? 0 : threadIdx.x / nthreads_V)*(D_V/2);

        const half2 kqmax_scale_h2 = make_half2(kqmax_scale, kqmax_scale);
#pragma unroll
        for (int i_VKQ_0 = 0; i_VKQ_0 < D_V/2; i_VKQ_0 += nthreads_V) {
            VKQ[j_VKQ][i_VKQ_0/nthreads_V] *= kqmax_scale_h2;
        }
#pragma unroll
        for (int i_VKQ_0 = 0; i_VKQ_0 < D_V/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
            const int i_VKQ = i_VKQ_0 + (nthreads_V == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_V)*(V_rows_per_thread/2);

            ggml_cuda_memcpy_1<V_rows_per_thread*sizeof(half)>(VKQ_tmp + i_VKQ, &VKQ[j_VKQ][i_VKQ_0/nthreads_V]);
        }
#else
        float2 * VKQ_tmp = (float2 *) KQ + threadIdx.y*(V_cols_per_iter*D_V/2)
            + (nthreads_V == WARP_SIZE ? 0 : threadIdx.x / nthreads_V)*(D_V/2);

#pragma unroll
        for (int i_VKQ_0 = 0; i_VKQ_0 < D_V/2; i_VKQ_0 += nthreads_V) {
            VKQ[j_VKQ][i_VKQ_0/nthreads_V].x *= kqmax_scale;
            VKQ[j_VKQ][i_VKQ_0/nthreads_V].y *= kqmax_scale;
        }
#pragma unroll
        for (int i_VKQ_0 = 0; i_VKQ_0 < D_V/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
            const int i_VKQ = i_VKQ_0 + (nthreads_V == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads_V)*(V_rows_per_thread/2);

            ggml_cuda_memcpy_1<V_rows_per_thread/2*sizeof(float)>(VKQ_tmp + i_VKQ,                       &VKQ[j_VKQ][i_VKQ_0/nthreads_V]);
            ggml_cuda_memcpy_1<V_rows_per_thread/2*sizeof(float)>(VKQ_tmp + i_VKQ + V_rows_per_thread/4, &VKQ[j_VKQ][i_VKQ_0/nthreads_V + V_rows_per_thread/4]);
        }
#endif // V_DOT2_F32_F16_AVAILABLE

        KQ_sum[j_VKQ] *= kqmax_scale;
        KQ_sum[j_VKQ] = warp_reduce_sum(KQ_sum[j_VKQ]);
        if (threadIdx.x == 0) {
            KQ_sum_shared[j_VKQ][threadIdx.y] = KQ_sum[j_VKQ];
        }

        __syncthreads();

        if (nthreads <= D_V || tid < D_V) {
            KQ_sum[j_VKQ] = KQ_sum_shared[j_VKQ][threadIdx.x];
            KQ_sum[j_VKQ] = warp_reduce_sum(KQ_sum[j_VKQ]);

#pragma unroll
            for (int i0 = 0; i0 < D_V; i0 += nthreads) {
                float dst_val = 0;
#pragma unroll
                for (int w = 0; w < nwarps; ++w) {
#pragma unroll
                    for (int v = 0; v < V_cols_per_iter; ++v) {
                        dst_val += float(KQ[w*V_cols_per_iter*D_V + v*D_V + i0 + tid]);
                    }
                }
                if (gridDim.y == 1) {
                    dst_val /= KQ_sum[j_VKQ];
                }
                // For TBQ V: store to shared memory for IWHT post-processing
                if constexpr (type_V == GGML_TYPE_TBQ4_0 || type_V == GGML_TYPE_TBQ3_0
                           || type_V == GGML_TYPE_TBQ4_1 || type_V == GGML_TYPE_TBQ3_1
                           || type_V == GGML_TYPE_TBQ4_2 || type_V == GGML_TYPE_TBQ3_2
                           || type_V == GGML_TYPE_TBQ4_3 || type_V == GGML_TYPE_TBQ3_3
                           || type_V == GGML_TYPE_TBQ4_4 || type_V == GGML_TYPE_TBQ3_4
                           || type_V == GGML_TYPE_TBQP4_4 || type_V == GGML_TYPE_TBQP3_4) {
                    KQ[i0 + tid] = dst_val; // reuse KQ shared memory
                } else {
                    dst[(((sequence*int(ne01.z) + ic0 + j_VKQ)*ne02 + head)*gridDim.y + blockIdx.y)*D_V + i0 + tid] = dst_val;
                }
            }

            // TBQ V post-processing: IWHT on the WHT-domain weighted sum
            // Always use 256-block independent IWHT (even for D_V=512)
            // because V blocks have per-block norms — mixing them in 512-IWHT corrupts scales
            if constexpr (type_V == GGML_TYPE_TBQ4_0 || type_V == GGML_TYPE_TBQ3_0) {
                static constexpr uint8_t tbq_signs_v[32] = {
                    0xa7,0x3b,0x91,0xf4,0x6d,0xc2,0x58,0x0e,
                    0xb3,0x7f,0x24,0xd6,0x89,0x45,0xea,0x1c,
                    0x63,0xaf,0xd8,0x52,0x97,0x0b,0xe1,0x3d,
                    0x76,0xc4,0x19,0xfe,0x4a,0x85,0x2c,0xdb,
                };
                __syncthreads();

                float * buf_base = (float *) &KQ[0];
                const int dst_base = (((sequence*int(ne01.z) + ic0 + j_VKQ)*ne02 + head)*gridDim.y + blockIdx.y)*D_V;

                // Process each 256-block independently (D_V/256 passes)
                for (int blk = 0; blk < D_V; blk += 256) {
                    float * buf = buf_base + blk;

                    {
                        float f0 = buf[tid], f1 = buf[tid + 128];
                        __syncthreads();
                        { float u = f0, v = f1; f0 = u + v; f1 = u - v; }
                        #pragma unroll
                        for (int s = 0; s < 5; s++) {
                            float o0 = __shfl_xor_sync(0xffffffff, f0, 1 << s, WARP_SIZE);
                            float o1 = __shfl_xor_sync(0xffffffff, f1, 1 << s, WARP_SIZE);
                            if (tid & (1 << s)) { f0 = o0 - f0; f1 = o1 - f1; }
                            else                { f0 = f0 + o0; f1 = f1 + o1; }
                        }
                        buf[tid] = f0; buf[tid + 128] = f1;
                        __syncthreads();
                        { float u0 = buf[tid], v0 = buf[tid ^ 32];
                          float u1 = buf[tid+128], v1 = buf[(tid+128) ^ 32];
                          f0 = (tid & 32) ? (v0 - u0) : (u0 + v0);
                          f1 = (tid & 32) ? (v1 - u1) : (u1 + v1); }
                        buf[tid] = f0; buf[tid + 128] = f1;
                        __syncthreads();
                        { float u0 = buf[tid], v0 = buf[tid ^ 64];
                          float u1 = buf[tid+128], v1 = buf[(tid+128) ^ 64];
                          f0 = (tid & 64) ? (v0 - u0) : (u0 + v0);
                          f1 = (tid & 64) ? (v1 - u1) : (u1 + v1); }
                        __syncthreads();
                        float sign0 = ((tbq_signs_v[tid >> 3] >> (tid & 7)) & 1) ? -1.0f : 1.0f;
                        float sign1 = ((tbq_signs_v[(tid+128) >> 3] >> ((tid+128) & 7)) & 1) ? -1.0f : 1.0f;
                        dst[dst_base + blk + tid]       = f0 * sign0 / 256.0f;
                        dst[dst_base + blk + tid + 128] = f1 * sign1 / 256.0f;
                    }

                    if (blk + 256 < D_V) {
                        __syncthreads();
                    }
                }
            }

            // TBQ V 128-block IWHT: D=128, 1:1 thread mapping, no stage 7
            if constexpr (type_V == GGML_TYPE_TBQ4_1 || type_V == GGML_TYPE_TBQ3_1) {
                static constexpr uint8_t tbq_signs_v[16] = {
                    0xa7,0x3b,0x91,0xf4,0x6d,0xc2,0x58,0x0e,
                    0xb3,0x7f,0x24,0xd6,0x89,0x45,0xea,0x1c,
                };
                __syncthreads();

                // FP32 IWHT (FP16 causes precision loss with accumulated V values)
                {
                    float * buf = (float *) &KQ[0];
                    float f0 = buf[tid];
                    __syncthreads();
                    // Stages 0-4: warp shuffle
                    #pragma unroll
                    for (int s = 0; s < 5; s++) {
                        float o0 = __shfl_xor_sync(0xffffffff, f0, 1 << s, WARP_SIZE);
                        if (tid & (1 << s)) { f0 = o0 - f0; }
                        else                { f0 = f0 + o0; }
                    }
                    // Stage 5 (stride 32)
                    buf[tid] = f0;
                    __syncthreads();
                    { float u0 = buf[tid], v0 = buf[tid ^ 32]; f0 = (tid & 32) ? (v0 - u0) : (u0 + v0); }
                    // Stage 6 (stride 64)
                    buf[tid] = f0;
                    __syncthreads();
                    { float u0 = buf[tid], v0 = buf[tid ^ 64]; f0 = (tid & 64) ? (v0 - u0) : (u0 + v0); }
                    __syncthreads();
                    float sign0 = ((tbq_signs_v[tid >> 3] >> (tid & 7)) & 1) ? -1.0f : 1.0f;
                    buf[tid] = f0 * sign0 / (float)D_V;
                }
                __syncthreads();

                for (int i0 = 0; i0 < D_V; i0 += nthreads) {
                    if (i0 + tid < D_V) {
                        dst[(((sequence*int(ne01.z) + ic0 + j_VKQ)*ne02 + head)*gridDim.y + blockIdx.y)*D_V + i0 + tid] =
                            ((float *)&KQ[0])[i0 + tid];
                    }
                }
            }

            // TBQ V 576-block IWHT: split 256+256+64, three passes
            if constexpr (type_V == GGML_TYPE_TBQ4_4 || type_V == GGML_TYPE_TBQ3_4
                       || type_V == GGML_TYPE_TBQP4_4 || type_V == GGML_TYPE_TBQP3_4) {
                static constexpr uint8_t signs_256[32] = {
                    0xa7,0x3b,0x91,0xf4,0x6d,0xc2,0x58,0x0e,0xb3,0x7f,0x24,0xd6,0x89,0x45,0xea,0x1c,
                    0x63,0xaf,0xd8,0x52,0x97,0x0b,0xe1,0x3d,0x76,0xc4,0x19,0xfe,0x4a,0x85,0x2c,0xdb,
                };
                __syncthreads();

                float * buf = (float *) &KQ[0];
                const int dst_base = (((sequence*int(ne01.z) + ic0 + j_VKQ)*ne02 + head)*gridDim.y + blockIdx.y)*D_V;

                // === FP32 IWHT pass 1: sub-block 1, buf[0:255] → dst[0:255] ===
                {
                    float f0 = buf[tid], f1 = buf[tid + 128];
                    __syncthreads();
                    { float u = f0, v = f1; f0 = u + v; f1 = u - v; }
                    #pragma unroll
                    for (int s = 0; s < 5; s++) {
                        float o0 = __shfl_xor_sync(0xffffffff, f0, 1 << s, WARP_SIZE);
                        float o1 = __shfl_xor_sync(0xffffffff, f1, 1 << s, WARP_SIZE);
                        if (tid & (1 << s)) { f0 = o0 - f0; f1 = o1 - f1; }
                        else                { f0 = f0 + o0; f1 = f1 + o1; }
                    }
                    buf[tid] = f0; buf[tid + 128] = f1;
                    __syncthreads();
                    { float u0 = buf[tid], v0 = buf[tid ^ 32]; float u1 = buf[tid+128], v1 = buf[(tid+128) ^ 32];
                      f0 = (tid & 32) ? (v0 - u0) : (u0 + v0); f1 = (tid & 32) ? (v1 - u1) : (u1 + v1); }
                    buf[tid] = f0; buf[tid + 128] = f1;
                    __syncthreads();
                    { float u0 = buf[tid], v0 = buf[tid ^ 64]; float u1 = buf[tid+128], v1 = buf[(tid+128) ^ 64];
                      f0 = (tid & 64) ? (v0 - u0) : (u0 + v0); f1 = (tid & 64) ? (v1 - u1) : (u1 + v1); }
                    __syncthreads();
                    float s0 = ((signs_256[tid >> 3] >> (tid & 7)) & 1) ? -1.0f : 1.0f;
                    float s1 = ((signs_256[(tid+128) >> 3] >> ((tid+128) & 7)) & 1) ? -1.0f : 1.0f;
                    dst[dst_base + tid]       = f0 * s0 / 256.0f;
                    dst[dst_base + tid + 128] = f1 * s1 / 256.0f;
                }

                // === FP32 IWHT pass 2: sub-block 2, buf[256:511] → dst[256:511] ===
                {
                    float f0 = buf[256 + tid], f1 = buf[256 + tid + 128];
                    __syncthreads();
                    { float u = f0, v = f1; f0 = u + v; f1 = u - v; }
                    #pragma unroll
                    for (int s = 0; s < 5; s++) {
                        float o0 = __shfl_xor_sync(0xffffffff, f0, 1 << s, WARP_SIZE);
                        float o1 = __shfl_xor_sync(0xffffffff, f1, 1 << s, WARP_SIZE);
                        if (tid & (1 << s)) { f0 = o0 - f0; f1 = o1 - f1; }
                        else                { f0 = f0 + o0; f1 = f1 + o1; }
                    }
                    buf[tid] = f0; buf[tid + 128] = f1;
                    __syncthreads();
                    { float u0 = buf[tid], v0 = buf[tid ^ 32]; float u1 = buf[tid+128], v1 = buf[(tid+128) ^ 32];
                      f0 = (tid & 32) ? (v0 - u0) : (u0 + v0); f1 = (tid & 32) ? (v1 - u1) : (u1 + v1); }
                    buf[tid] = f0; buf[tid + 128] = f1;
                    __syncthreads();
                    { float u0 = buf[tid], v0 = buf[tid ^ 64]; float u1 = buf[tid+128], v1 = buf[(tid+128) ^ 64];
                      f0 = (tid & 64) ? (v0 - u0) : (u0 + v0); f1 = (tid & 64) ? (v1 - u1) : (u1 + v1); }
                    __syncthreads();
                    float s0 = ((signs_256[tid >> 3] >> (tid & 7)) & 1) ? -1.0f : 1.0f;
                    float s1 = ((signs_256[(tid+128) >> 3] >> ((tid+128) & 7)) & 1) ? -1.0f : 1.0f;
                    dst[dst_base + 256 + tid]       = f0 * s0 / 256.0f;
                    dst[dst_base + 256 + tid + 128] = f1 * s1 / 256.0f;
                }

                // === Pass 3: sub-block 3 (rope, f16 passthrough) → dst[512:575] ===
                // No IWHT needed — values are already in spatial domain.
                // Only needed when D_V > 512 (full 576-block); skipped for V view (D_V=512)
                if constexpr (D_V > 512) {
                    if (tid < 64) {
                        dst[dst_base + 512 + tid] = buf[512 + tid];
                    }
                }
            }

            // TBQ V 64-block IWHT: D=64, tid 0-63 participate
            if constexpr (type_V == GGML_TYPE_TBQ4_2 || type_V == GGML_TYPE_TBQ3_2
                      || type_V == GGML_TYPE_TBQ4_3 || type_V == GGML_TYPE_TBQ3_3) {
                // Full 512-bit sign pattern: _2 uses head 0 (offset 0), _3 uses head-specific slice
                static constexpr uint8_t tbq_signs_v_full[64] = {
                    0xa7,0x3b,0x91,0xf4,0x6d,0xc2,0x58,0x0e,  // head 0
                    0xb3,0x7f,0x24,0xd6,0x89,0x45,0xea,0x1c,  // head 1
                    0x63,0xaf,0xd8,0x52,0x97,0x0b,0xe1,0x3d,  // head 2
                    0x76,0xc4,0x19,0xfe,0x4a,0x85,0x2c,0xdb,  // head 3
                    0xd3,0x4e,0xa8,0x17,0x9c,0x5b,0xe6,0x31,  // head 4
                    0x72,0xb9,0x0d,0xf5,0x43,0x8a,0x6e,0xc7,  // head 5
                    0x58,0x2f,0x94,0xe1,0xb6,0x3d,0x0a,0x7c,  // head 6
                    0xc5,0x61,0xd8,0x4f,0xa3,0x97,0x1e,0x85,  // head 7
                };
                const int v_signs_offset = is_cross_head_V ? ((head / gqa_ratio) % 8) * 8 : 0;
                __syncthreads();
                // FP32 IWHT (FP16 causes precision loss with accumulated V values)
                {
                    float * buf = (float *) &KQ[0];
                    float f0 = (tid < D_V) ? buf[tid] : 0.0f;
                    __syncthreads();
                    #pragma unroll
                    for (int s = 0; s < 5; s++) {
                        float o0 = __shfl_xor_sync(0xffffffff, f0, 1 << s, WARP_SIZE);
                        if (tid & (1 << s)) { f0 = o0 - f0; }
                        else                { f0 = f0 + o0; }
                    }
                    if (tid < D_V) { buf[tid] = f0; }
                    __syncthreads();
                    if (tid < D_V) {
                        float u0 = buf[tid], v0 = buf[tid ^ 32];
                        f0 = (tid & 32) ? (v0 - u0) : (u0 + v0);
                    }
                    __syncthreads();
                    if (tid < D_V) {
                        float sign0 = ((tbq_signs_v_full[v_signs_offset + (tid >> 3)] >> (tid & 7)) & 1) ? -1.0f : 1.0f;
                        buf[tid] = f0 * sign0 / (float)D_V;
                    }
                }
                __syncthreads();
                for (int i0 = 0; i0 < D_V; i0 += nthreads) {
                    if (i0 + tid < D_V) {
                        dst[(((sequence*int(ne01.z) + ic0 + j_VKQ)*ne02 + head)*gridDim.y + blockIdx.y)*D_V + i0 + tid] =
                            ((float *)&KQ[0])[i0 + tid];
                    }
                }
            }
        }

        if (j_VKQ < ncols-1) {
            __syncthreads();
        }

    }

    if (gridDim.y != 1 && tid < ncols && (ncols == 1 || ic0 + tid < int(ne01.z))) {
        dst_meta[((sequence*int(ne01.z) + ic0 + tid)*ne02 + head)*gridDim.y + blockIdx.y] = make_float2(KQ_max[tid], KQ_sum[tid]);
    }
#else
    GGML_UNUSED_VARS(Q, K, V, mask, sinks, KV_max, dst, dst_meta, scale,
        max_bias, m0, m1, n_head_log2, logit_softcap,
        ne00, ne01, ne02, ne03,
              nb01, nb02, nb03,
        ne10, ne11, ne12, ne13,
              nb11, nb12, nb13,
              nb21, nb22, nb23,
              ne31, ne32, ne33,
              nb31, nb32, nb33, raw_K_data, raw_K_stride, Q_wht2_data, Q_wht2_stride);
    NO_DEVICE_CODE;
#endif // FLASH_ATTN_AVAILABLE
}
#ifdef __clang__
#pragma clang diagnostic pop
#endif // __clang__

template <int D, int cols_per_block, ggml_type type_K, ggml_type type_V, bool use_logit_softcap, int D_V = D>
void ggml_cuda_flash_attn_ext_vec_case_impl(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;

    const int nthreads = ggml_cuda_fattn_vec_get_nthreads_host(cc);
    const int nwarps   = nthreads / WARP_SIZE;
    fattn_kernel_t fattn_kernel = flash_attn_ext_vec<D, cols_per_block, type_K, type_V, use_logit_softcap, D_V>;
    const bool need_f16_K = type_K == GGML_TYPE_F16;
    const bool need_f16_V = type_V == GGML_TYPE_F16;
    constexpr size_t nbytes_shared = 0;
    launch_fattn<D_V, cols_per_block, 1>(ctx, dst, fattn_kernel, nwarps, nbytes_shared, D, need_f16_K, need_f16_V, false);
}

template <int D, ggml_type type_K, ggml_type type_V>
void ggml_cuda_flash_attn_ext_vec_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV = dst;
    const ggml_tensor * Q   = dst->src[0];

    float logit_softcap;
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));

    if (Q->ne[1] == 1) {
        constexpr int cols_per_block = 1;
        if (logit_softcap == 0.0f) {
            constexpr bool use_logit_softcap = false;
            ggml_cuda_flash_attn_ext_vec_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
        } else {
            constexpr bool use_logit_softcap = true;
            ggml_cuda_flash_attn_ext_vec_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
        }
        return;
    }

    constexpr int cols_per_block = 2;
    if (logit_softcap == 0.0f) {
        constexpr bool use_logit_softcap = false;
        ggml_cuda_flash_attn_ext_vec_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
    } else {
        constexpr bool use_logit_softcap = true;
        ggml_cuda_flash_attn_ext_vec_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
    }
}

// Asymmetric K/V: D = K/Q head dim, D_V = V head dim (e.g. GLM-4.7-Flash: D=576, D_V=512)
template <int D, int D_V, ggml_type type_K, ggml_type type_V>
void ggml_cuda_flash_attn_ext_vec_case_asym(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV = dst;
    const ggml_tensor * Q   = dst->src[0];

    float logit_softcap;
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));

    if (Q->ne[1] == 1) {
        constexpr int cols_per_block = 1;
        if (logit_softcap == 0.0f) {
            ggml_cuda_flash_attn_ext_vec_case_impl<D, cols_per_block, type_K, type_V, false, D_V>(ctx, dst);
        } else {
            ggml_cuda_flash_attn_ext_vec_case_impl<D, cols_per_block, type_K, type_V, true, D_V>(ctx, dst);
        }
        return;
    }

    constexpr int cols_per_block = 2;
    if (logit_softcap == 0.0f) {
        ggml_cuda_flash_attn_ext_vec_case_impl<D, cols_per_block, type_K, type_V, false, D_V>(ctx, dst);
    } else {
        ggml_cuda_flash_attn_ext_vec_case_impl<D, cols_per_block, type_K, type_V, true, D_V>(ctx, dst);
    }
}

#define DECL_FATTN_VEC_CASE(D, type_K, type_V)                              \
    template void ggml_cuda_flash_attn_ext_vec_case                         \
    <D, type_K, type_V>(ggml_backend_cuda_context & ctx, ggml_tensor * dst) \

#define DECL_FATTN_VEC_CASE_ASYM(D, D_V, type_K, type_V)                        \
    template void ggml_cuda_flash_attn_ext_vec_case_asym                         \
    <D, D_V, type_K, type_V>(ggml_backend_cuda_context & ctx, ggml_tensor * dst) \

#define EXTERN_DECL_FATTN_VEC_CASES(D, type_K)             \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_F16);  \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q4_0); \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q4_1); \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q5_0); \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q5_1); \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q8_0); \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_BF16); \

EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_F16)
EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_Q4_0)
EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_Q4_1)
EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_Q5_0)
EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_Q5_1)
EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_Q8_0)
EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_BF16)

EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_F16)
EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_Q4_0)
EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_Q4_1)
EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_Q5_0)
EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_Q5_1)
EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_Q8_0)
EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_BF16)

EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_F16)
EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_Q4_0)
EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_Q4_1)
EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_Q5_0)
EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_Q5_1)
EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_Q8_0)
EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_BF16)

// TurboQuant 256-block (_0) extern declarations: D=256 only
#define EXTERN_DECL_FATTN_VEC_TBQ0(type_K)                           \
    extern DECL_FATTN_VEC_CASE(256, type_K, GGML_TYPE_F16);          \
    extern DECL_FATTN_VEC_CASE(256, type_K, GGML_TYPE_Q8_0);         \
    extern DECL_FATTN_VEC_CASE(256, type_K, GGML_TYPE_TBQ3_0);       \
    extern DECL_FATTN_VEC_CASE(256, type_K, GGML_TYPE_TBQ4_0);       \

EXTERN_DECL_FATTN_VEC_TBQ0(GGML_TYPE_TBQ3_0)
EXTERN_DECL_FATTN_VEC_TBQ0(GGML_TYPE_TBQ4_0)
EXTERN_DECL_FATTN_VEC_TBQ0(GGML_TYPE_TBQP3_0)
EXTERN_DECL_FATTN_VEC_TBQ0(GGML_TYPE_TBQP4_0)

// TurboQuant 128-block (_1) extern declarations: D=128 only
#define EXTERN_DECL_FATTN_VEC_TBQ1(type_K)                           \
    extern DECL_FATTN_VEC_CASE(128, type_K, GGML_TYPE_F16);          \
    extern DECL_FATTN_VEC_CASE(128, type_K, GGML_TYPE_Q8_0);         \
    extern DECL_FATTN_VEC_CASE(128, type_K, GGML_TYPE_TBQ3_1);       \
    extern DECL_FATTN_VEC_CASE(128, type_K, GGML_TYPE_TBQ4_1);       \

EXTERN_DECL_FATTN_VEC_TBQ1(GGML_TYPE_TBQ3_1)
EXTERN_DECL_FATTN_VEC_TBQ1(GGML_TYPE_TBQ4_1)
EXTERN_DECL_FATTN_VEC_TBQ1(GGML_TYPE_TBQP3_1)
EXTERN_DECL_FATTN_VEC_TBQ1(GGML_TYPE_TBQP4_1)

// TurboQuant 64-block (_2) extern declarations: D=64 only
#define EXTERN_DECL_FATTN_VEC_TBQ2(type_K)                            \
    extern DECL_FATTN_VEC_CASE(64, type_K, GGML_TYPE_F16);            \
    extern DECL_FATTN_VEC_CASE(64, type_K, GGML_TYPE_Q8_0);           \
    extern DECL_FATTN_VEC_CASE(64, type_K, GGML_TYPE_TBQ3_2);         \
    extern DECL_FATTN_VEC_CASE(64, type_K, GGML_TYPE_TBQ4_2);         \

EXTERN_DECL_FATTN_VEC_TBQ2(GGML_TYPE_TBQ3_2)
EXTERN_DECL_FATTN_VEC_TBQ2(GGML_TYPE_TBQ4_2)
EXTERN_DECL_FATTN_VEC_TBQ2(GGML_TYPE_TBQP3_2)
EXTERN_DECL_FATTN_VEC_TBQ2(GGML_TYPE_TBQP4_2)

// TurboQuant cross-head (_3) extern declarations: D=64, K=_3, V=_2/_3 or standard
#define EXTERN_DECL_FATTN_VEC_TBQ3(type_K)                            \
    extern DECL_FATTN_VEC_CASE(64, type_K, GGML_TYPE_F16);            \
    extern DECL_FATTN_VEC_CASE(64, type_K, GGML_TYPE_Q8_0);           \
    extern DECL_FATTN_VEC_CASE(64, type_K, GGML_TYPE_TBQ3_2);         \
    extern DECL_FATTN_VEC_CASE(64, type_K, GGML_TYPE_TBQ4_2);         \
    extern DECL_FATTN_VEC_CASE(64, type_K, GGML_TYPE_TBQ3_3);         \
    extern DECL_FATTN_VEC_CASE(64, type_K, GGML_TYPE_TBQ4_3);         \

EXTERN_DECL_FATTN_VEC_TBQ3(GGML_TYPE_TBQ3_3)
EXTERN_DECL_FATTN_VEC_TBQ3(GGML_TYPE_TBQ4_3)
EXTERN_DECL_FATTN_VEC_TBQ3(GGML_TYPE_TBQP3_3)
EXTERN_DECL_FATTN_VEC_TBQ3(GGML_TYPE_TBQP4_3)

// Asymmetric: standard K + TBQ V extern declarations
// D=256: only _0 V types
extern DECL_FATTN_VEC_CASE(256, GGML_TYPE_F16,  GGML_TYPE_TBQ3_0);
extern DECL_FATTN_VEC_CASE(256, GGML_TYPE_F16,  GGML_TYPE_TBQ4_0);
extern DECL_FATTN_VEC_CASE(256, GGML_TYPE_Q8_0, GGML_TYPE_TBQ3_0);
extern DECL_FATTN_VEC_CASE(256, GGML_TYPE_Q8_0, GGML_TYPE_TBQ4_0);
extern DECL_FATTN_VEC_CASE(256, GGML_TYPE_Q4_0, GGML_TYPE_TBQ3_0);
extern DECL_FATTN_VEC_CASE(256, GGML_TYPE_Q4_0, GGML_TYPE_TBQ4_0);
// D=128: only _1 V types
extern DECL_FATTN_VEC_CASE(128, GGML_TYPE_F16,  GGML_TYPE_TBQ3_1);
extern DECL_FATTN_VEC_CASE(128, GGML_TYPE_F16,  GGML_TYPE_TBQ4_1);
extern DECL_FATTN_VEC_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_TBQ3_1);
extern DECL_FATTN_VEC_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_TBQ4_1);
extern DECL_FATTN_VEC_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_TBQ3_1);
extern DECL_FATTN_VEC_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_TBQ4_1);
// D=64: only _2 V types
extern DECL_FATTN_VEC_CASE( 64, GGML_TYPE_F16,  GGML_TYPE_TBQ3_2);
extern DECL_FATTN_VEC_CASE( 64, GGML_TYPE_F16,  GGML_TYPE_TBQ4_2);
extern DECL_FATTN_VEC_CASE( 64, GGML_TYPE_Q8_0, GGML_TYPE_TBQ3_2);
extern DECL_FATTN_VEC_CASE( 64, GGML_TYPE_Q8_0, GGML_TYPE_TBQ4_2);
extern DECL_FATTN_VEC_CASE( 64, GGML_TYPE_Q4_0, GGML_TYPE_TBQ3_2);
extern DECL_FATTN_VEC_CASE( 64, GGML_TYPE_Q4_0, GGML_TYPE_TBQ4_2);

// TurboQuant 576-block (_4) extern declarations: D=576 only
#define EXTERN_DECL_FATTN_VEC_TBQ4_576(type_K)                            \
    extern DECL_FATTN_VEC_CASE(576, type_K, GGML_TYPE_F16);               \
    extern DECL_FATTN_VEC_CASE(576, type_K, GGML_TYPE_Q8_0);              \
    extern DECL_FATTN_VEC_CASE(576, type_K, GGML_TYPE_TBQ3_4);            \
    extern DECL_FATTN_VEC_CASE(576, type_K, GGML_TYPE_TBQ4_4);            \

EXTERN_DECL_FATTN_VEC_TBQ4_576(GGML_TYPE_TBQ3_4)
EXTERN_DECL_FATTN_VEC_TBQ4_576(GGML_TYPE_TBQ4_4)
EXTERN_DECL_FATTN_VEC_TBQ4_576(GGML_TYPE_TBQP3_4)
EXTERN_DECL_FATTN_VEC_TBQ4_576(GGML_TYPE_TBQP4_4)

// Asymmetric: standard K + TBQ_4 V (D=576)
extern DECL_FATTN_VEC_CASE(576, GGML_TYPE_F16,  GGML_TYPE_TBQ3_4);
extern DECL_FATTN_VEC_CASE(576, GGML_TYPE_F16,  GGML_TYPE_TBQ4_4);
extern DECL_FATTN_VEC_CASE(576, GGML_TYPE_Q8_0, GGML_TYPE_TBQ3_4);
extern DECL_FATTN_VEC_CASE(576, GGML_TYPE_Q8_0, GGML_TYPE_TBQ4_4);

// GLM asymmetric: K=576 (_4 type), V=512 (_0 type or same-type view)
#define EXTERN_DECL_FATTN_VEC_ASYM_576x512(type_K)                                 \
    extern DECL_FATTN_VEC_CASE_ASYM(576, 512, type_K, GGML_TYPE_TBQ3_0);           \
    extern DECL_FATTN_VEC_CASE_ASYM(576, 512, type_K, GGML_TYPE_TBQ4_0);           \
    extern DECL_FATTN_VEC_CASE_ASYM(576, 512, type_K, GGML_TYPE_F16);              \
    extern DECL_FATTN_VEC_CASE_ASYM(576, 512, type_K, GGML_TYPE_Q8_0);             \
    extern DECL_FATTN_VEC_CASE_ASYM(576, 512, type_K, type_K);

EXTERN_DECL_FATTN_VEC_ASYM_576x512(GGML_TYPE_TBQ3_4)
EXTERN_DECL_FATTN_VEC_ASYM_576x512(GGML_TYPE_TBQ4_4)
EXTERN_DECL_FATTN_VEC_ASYM_576x512(GGML_TYPE_TBQP3_4)
EXTERN_DECL_FATTN_VEC_ASYM_576x512(GGML_TYPE_TBQP4_4)
