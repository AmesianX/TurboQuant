#include "common.cuh"
#include "fattn-common.cuh"
#include "fattn-mma-f16.cuh"
#include "fattn-tile.cuh"
#include "fattn-vec.cuh"
#include "fattn-wmma-f16.cuh"
#include "fattn.cuh"

template <int DKQ, int DV, int ncols2>
static void ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * Q = dst->src[0];

    if constexpr (ncols2 <= 8) {
        if (turing_mma_available(cc) && Q->ne[1] <= 8/ncols2) {
            ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 8/ncols2, ncols2>(ctx, dst);
            return;
        }
    }

    if constexpr (ncols2 <= 16) {
        if ((turing_mma_available(cc) || amd_wmma_available(cc)) && Q->ne[1] <= 16/ncols2) {
            ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 16/ncols2, ncols2>(ctx, dst);
            return;
        }
    }

    if (ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_TURING || amd_wmma_available(cc) || Q->ne[1] <= 32/ncols2) {
        ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 32/ncols2, ncols2>(ctx, dst);
        return;
    }

    ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 64/ncols2, ncols2>(ctx, dst);
}

template <int DKQ, int DV>
static void ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * KQV  = dst;
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

    // Edge cases like no mask, ALiBi, unpadded K/V, or misaligned addresses for large data transfers
    //     are put into the template specialization without GQA optimizations.
    bool use_gqa_opt = mask && max_bias == 0.0f && K->ne[1] % FATTN_KQ_STRIDE == 0;
    for (const ggml_tensor * t : {Q, K, V, mask}) {
        if (t == nullptr || ggml_is_quantized(t->type)) {
            continue;
        }
        for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
            if (t->nb[i] % 16 != 0) {
                use_gqa_opt = false;
                break;
            }
        }
    }

    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
    const int gqa_ratio = Q->ne[2] / K->ne[2];

    // On Volta the GQA optimizations aren't as impactful vs. minimizing wasted compute:
    if (cc == GGML_CUDA_CC_VOLTA) {
        if (use_gqa_opt && gqa_ratio % 8 == 0) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 8>(ctx, dst);
            return;
        }

        if (use_gqa_opt && gqa_ratio % 4 == 0) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 4>(ctx, dst);
            return;
        }

        if (use_gqa_opt && gqa_ratio % 2 == 0) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 2>(ctx, dst);
            return;
        }

        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 1>(ctx, dst);
        return;
    }

    if (use_gqa_opt && gqa_ratio > 4) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 8>(ctx, dst);
        return;
    }

    if (use_gqa_opt && gqa_ratio > 2) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 4>(ctx, dst);
        return;
    }

    if (use_gqa_opt && gqa_ratio > 1) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 2>(ctx, dst);
        return;
    }

    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 1>(ctx, dst);
}

static void ggml_cuda_flash_attn_ext_mma_f16(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * KQV  = dst;
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    switch (Q->ne[0]) {
        case 64:
            GGML_ASSERT(V->ne[0] == 64);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 64,  64>(ctx, dst);
            break;
        case 80:
            GGML_ASSERT(V->ne[0] == 80);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 80,  80>(ctx, dst);
            break;
        case 96:
            GGML_ASSERT(V->ne[0] == 96);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 96,  96>(ctx, dst);
            break;
        case 112:
            GGML_ASSERT(V->ne[0] == 112);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<112, 112>(ctx, dst);
            break;
        case 128:
            GGML_ASSERT(V->ne[0] == 128);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<128, 128>(ctx, dst);
            break;
        case 256:
            GGML_ASSERT(V->ne[0] == 256);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<256, 256>(ctx, dst);
            break;
        case 512:
            GGML_ASSERT(V->ne[0] == 512);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<512, 512>(ctx, dst);
            break;
        case 576: {
            // For Deepseek, go straight to the ncols1 switch to avoid compiling unnecessary kernels.
            GGML_ASSERT(V->ne[0] == 512);
            float max_bias = 0.0f;
            memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

            const bool use_gqa_opt = mask && max_bias == 0.0f;
            GGML_ASSERT(use_gqa_opt);

            GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
            const int gqa_ratio = Q->ne[2] / K->ne[2];
            if (gqa_ratio == 20) { // GLM 4.7 Flash
                if (cc >= GGML_CUDA_CC_DGX_SPARK) {
                    if (Q->ne[1] <= 8) {
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                if (cc >= GGML_CUDA_CC_BLACKWELL) {
                    if (Q->ne[1] <= 4 && K->ne[1] >= 65536) {
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                if (cc >= GGML_CUDA_CC_ADA_LOVELACE) {
                    if (Q->ne[1] <= 4) {
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                if (cc >= GGML_CUDA_CC_TURING) {
                    if (Q->ne[1] <= 4) {
                        if (K->ne[1] <= 16384) {
                            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                            break;
                        }
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 32>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                // Volta:
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
            } else if (gqa_ratio % 16 == 0) {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
            } else {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512,  4>(ctx, dst);
            }
        } break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

#define FATTN_VEC_CASE(D, type_K, type_V)                                                                        \
    {                                                                                                            \
        const bool type_K_okay = K->type == (type_K) || (K->type == GGML_TYPE_F32 && (type_K) == GGML_TYPE_F16); \
        const bool type_V_okay = V->type == (type_V) || (V->type == GGML_TYPE_F32 && (type_V) == GGML_TYPE_F16); \
        if (Q->ne[0] == (D) && type_K_okay && type_V_okay) {                                                     \
            ggml_cuda_flash_attn_ext_vec_case<D, type_K, type_V>(ctx, dst);                                      \
            return;                                                                                              \
        }                                                                                                        \
    }                                                                                                            \

#define FATTN_VEC_CASES_ALL_D(type_K, type_V) \
    FATTN_VEC_CASE( 64, type_K, type_V)       \
    FATTN_VEC_CASE(128, type_K, type_V)       \
    FATTN_VEC_CASE(256, type_K, type_V)       \

#define FATTN_VEC_CASE_ASYM(D, D_V, type_K, type_V)                                                             \
    {                                                                                                            \
        const bool type_K_okay = K->type == (type_K) || (K->type == GGML_TYPE_F32 && (type_K) == GGML_TYPE_F16); \
        const bool type_V_okay = V->type == (type_V) || (V->type == GGML_TYPE_F32 && (type_V) == GGML_TYPE_F16); \
        if (Q->ne[0] == (D) && V->ne[0] == (D_V) && type_K_okay && type_V_okay) {                               \
            ggml_cuda_flash_attn_ext_vec_case_asym<D, D_V, type_K, type_V>(ctx, dst);                            \
            return;                                                                                              \
        }                                                                                                        \
    }                                                                                                            \


static void ggml_cuda_flash_attn_ext_vec(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_tensor * Q = dst->src[0];
    ggml_tensor * K = dst->src[1];
    ggml_tensor * V = dst->src[2];

#ifdef GGML_CUDA_FA_TBQ_TUNING
    // TBQ tuning + SWA f16 + QJL support + K/V split testing
    FATTN_VEC_CASE(256, GGML_TYPE_TBQ3_0, GGML_TYPE_TBQ3_0)
    FATTN_VEC_CASE(512, GGML_TYPE_TBQ3_0, GGML_TYPE_TBQ3_0)
    FATTN_VEC_CASE(512, GGML_TYPE_TBQP3_0, GGML_TYPE_TBQ3_0)
    FATTN_VEC_CASE(256, GGML_TYPE_TBQ3_0, GGML_TYPE_F16)
    FATTN_VEC_CASE(512, GGML_TYPE_TBQ3_0, GGML_TYPE_F16)
    FATTN_VEC_CASE(512, GGML_TYPE_TBQP3_0, GGML_TYPE_F16)
    FATTN_VEC_CASE(512, GGML_TYPE_F16, GGML_TYPE_TBQ3_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_F16)
    // D=64 per-head _2 (single WHT)
    FATTN_VEC_CASE(64, GGML_TYPE_TBQ3_2, GGML_TYPE_TBQ3_2)
    // D=64 double WHT _3
    FATTN_VEC_CASE(64, GGML_TYPE_TBQ3_3, GGML_TYPE_TBQ3_2)
    FATTN_VEC_CASE(64, GGML_TYPE_TBQP3_3, GGML_TYPE_TBQ3_2)
    // D=64 4-bit K
    FATTN_VEC_CASE(64, GGML_TYPE_TBQ4_2, GGML_TYPE_TBQ3_2)
    FATTN_VEC_CASE(64, GGML_TYPE_TBQ4_3, GGML_TYPE_TBQ3_2)
    FATTN_VEC_CASE(64, GGML_TYPE_TBQP4_3, GGML_TYPE_TBQ3_2)
    // D=64 V=f16 variants
    FATTN_VEC_CASE(64, GGML_TYPE_TBQ3_2, GGML_TYPE_F16)
    FATTN_VEC_CASE(64, GGML_TYPE_TBQ3_3, GGML_TYPE_F16)
    FATTN_VEC_CASE(64, GGML_TYPE_TBQP3_3, GGML_TYPE_F16)
    FATTN_VEC_CASE(64, GGML_TYPE_TBQP3_3, GGML_TYPE_Q8_0)
    FATTN_VEC_CASE(64, GGML_TYPE_TBQ4_2, GGML_TYPE_F16)
    FATTN_VEC_CASE(64, GGML_TYPE_TBQ4_3, GGML_TYPE_F16)
    FATTN_VEC_CASE(64, GGML_TYPE_TBQP4_3, GGML_TYPE_F16)
    // D=64 f16/q8_0 K + TBQ V (for comparison testing)
    FATTN_VEC_CASE(64, GGML_TYPE_F16, GGML_TYPE_TBQ3_2)
    FATTN_VEC_CASE(64, GGML_TYPE_Q8_0, GGML_TYPE_TBQ3_2)

    // GLM-4.7-Flash / DeepSeek-V2/V3 MLA: K=576 (TBQP3_4/TBQ3_4), V=512
    // V-as-K-view (MLA absorption): V->type == K->type at dispatch time (TBQP3_4 + TBQP3_4, etc.)
    // Quality experiment: force VEC path for MLA to apply v1.5.2 improvements.
    // Kept inside TBQ_TUNING so builds with tuning=ON can still test GLM.
    FATTN_VEC_CASE_ASYM(576, 512, GGML_TYPE_TBQP3_4, GGML_TYPE_TBQP3_4)
    FATTN_VEC_CASE_ASYM(576, 512, GGML_TYPE_TBQ3_4,  GGML_TYPE_TBQ3_4)
    FATTN_VEC_CASE_ASYM(576, 512, GGML_TYPE_TBQP4_4, GGML_TYPE_TBQP4_4)
    FATTN_VEC_CASE_ASYM(576, 512, GGML_TYPE_TBQ4_4,  GGML_TYPE_TBQ4_4)
    // Also support asymmetric cases (non-absorption / f16 V fallback)
    FATTN_VEC_CASE_ASYM(576, 512, GGML_TYPE_TBQP3_4, GGML_TYPE_TBQ3_0)
    FATTN_VEC_CASE_ASYM(576, 512, GGML_TYPE_TBQ3_4,  GGML_TYPE_TBQ3_0)
    FATTN_VEC_CASE_ASYM(576, 512, GGML_TYPE_TBQP4_4, GGML_TYPE_TBQ4_0)
    FATTN_VEC_CASE_ASYM(576, 512, GGML_TYPE_TBQ4_4,  GGML_TYPE_TBQ4_0)
    FATTN_VEC_CASE_ASYM(576, 512, GGML_TYPE_TBQP3_4, GGML_TYPE_F16)
    FATTN_VEC_CASE_ASYM(576, 512, GGML_TYPE_TBQ3_4,  GGML_TYPE_F16)
#elif defined(GGML_CUDA_FA_ALL_QUANTS)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_F16)
    FATTN_VEC_CASE(256, GGML_TYPE_TBQ3_0, GGML_TYPE_F16)
    FATTN_VEC_CASE(256, GGML_TYPE_TBQ4_0, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_F16)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_Q4_0)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_Q4_1)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_Q5_0)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_Q5_1)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_Q8_0)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_BF16)

    // TurboQuant 256-block (_0): D=256 and D=512 (256-block × 2)
    FATTN_VEC_CASE(256, GGML_TYPE_TBQ3_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASE(256, GGML_TYPE_TBQ4_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASE(256, GGML_TYPE_TBQ3_0, GGML_TYPE_TBQ3_0)
    FATTN_VEC_CASE(256, GGML_TYPE_TBQ4_0, GGML_TYPE_TBQ4_0)
    FATTN_VEC_CASE(256, GGML_TYPE_TBQ4_0, GGML_TYPE_TBQ3_0)
    FATTN_VEC_CASE(256, GGML_TYPE_TBQP3_0, GGML_TYPE_TBQ3_0)
    FATTN_VEC_CASE(256, GGML_TYPE_TBQP4_0, GGML_TYPE_TBQ4_0)
    FATTN_VEC_CASE(256, GGML_TYPE_TBQP3_0, GGML_TYPE_F16)
    FATTN_VEC_CASE(256, GGML_TYPE_TBQP4_0, GGML_TYPE_F16)

    // TurboQuant 256-block (_0) at D=512 (Gemma 4 global attention)
    FATTN_VEC_CASE(512, GGML_TYPE_TBQ3_0, GGML_TYPE_F16)
    FATTN_VEC_CASE(512, GGML_TYPE_TBQ4_0, GGML_TYPE_F16)
    FATTN_VEC_CASE(512, GGML_TYPE_TBQ3_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASE(512, GGML_TYPE_TBQ4_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASE(512, GGML_TYPE_TBQ3_0, GGML_TYPE_TBQ3_0)
    FATTN_VEC_CASE(512, GGML_TYPE_TBQ4_0, GGML_TYPE_TBQ4_0)
    FATTN_VEC_CASE(512, GGML_TYPE_TBQ4_0, GGML_TYPE_TBQ3_0)
    FATTN_VEC_CASE(512, GGML_TYPE_TBQP3_0, GGML_TYPE_TBQ3_0)
    FATTN_VEC_CASE(512, GGML_TYPE_TBQP4_0, GGML_TYPE_TBQ4_0)
    FATTN_VEC_CASE(512, GGML_TYPE_TBQP3_0, GGML_TYPE_F16)
    FATTN_VEC_CASE(512, GGML_TYPE_TBQP4_0, GGML_TYPE_F16)
    FATTN_VEC_CASE(512, GGML_TYPE_TBQP3_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASE(512, GGML_TYPE_TBQP4_0, GGML_TYPE_Q8_0)

    // TurboQuant 128-block (_1): D=128 only
    FATTN_VEC_CASE(128, GGML_TYPE_TBQ3_1, GGML_TYPE_F16)
    FATTN_VEC_CASE(128, GGML_TYPE_TBQ4_1, GGML_TYPE_F16)
    FATTN_VEC_CASE(128, GGML_TYPE_TBQ3_1, GGML_TYPE_Q8_0)
    FATTN_VEC_CASE(128, GGML_TYPE_TBQ4_1, GGML_TYPE_Q8_0)
    FATTN_VEC_CASE(128, GGML_TYPE_TBQ3_1, GGML_TYPE_TBQ3_1)
    FATTN_VEC_CASE(128, GGML_TYPE_TBQ4_1, GGML_TYPE_TBQ4_1)
    FATTN_VEC_CASE(128, GGML_TYPE_TBQ4_1, GGML_TYPE_TBQ3_1)
    FATTN_VEC_CASE(128, GGML_TYPE_TBQ3_1, GGML_TYPE_TBQ4_1)
    FATTN_VEC_CASE(128, GGML_TYPE_TBQP3_1, GGML_TYPE_TBQ3_1)
    FATTN_VEC_CASE(128, GGML_TYPE_TBQP3_1, GGML_TYPE_TBQ4_1)
    FATTN_VEC_CASE(128, GGML_TYPE_TBQP4_1, GGML_TYPE_TBQ3_1)
    FATTN_VEC_CASE(128, GGML_TYPE_TBQP4_1, GGML_TYPE_TBQ4_1)
    FATTN_VEC_CASE(128, GGML_TYPE_TBQP3_1, GGML_TYPE_F16)
    FATTN_VEC_CASE(128, GGML_TYPE_TBQP4_1, GGML_TYPE_F16)
    FATTN_VEC_CASE(128, GGML_TYPE_TBQP3_1, GGML_TYPE_Q8_0)
    FATTN_VEC_CASE(128, GGML_TYPE_TBQP4_1, GGML_TYPE_Q8_0)

    // TurboQuant 64-block (_2): D=64 only
    FATTN_VEC_CASE(64, GGML_TYPE_TBQ3_2, GGML_TYPE_F16)
    FATTN_VEC_CASE(64, GGML_TYPE_TBQ4_2, GGML_TYPE_F16)
    FATTN_VEC_CASE(64, GGML_TYPE_TBQ3_2, GGML_TYPE_Q8_0)
    FATTN_VEC_CASE(64, GGML_TYPE_TBQ4_2, GGML_TYPE_Q8_0)
    FATTN_VEC_CASE(64, GGML_TYPE_TBQ3_2, GGML_TYPE_TBQ3_2)
    FATTN_VEC_CASE(64, GGML_TYPE_TBQ4_2, GGML_TYPE_TBQ4_2)
    FATTN_VEC_CASE(64, GGML_TYPE_TBQ4_2, GGML_TYPE_TBQ3_2)
    FATTN_VEC_CASE(64, GGML_TYPE_TBQ3_2, GGML_TYPE_TBQ4_2)
    FATTN_VEC_CASE(64, GGML_TYPE_TBQP3_2, GGML_TYPE_TBQ3_2)
    FATTN_VEC_CASE(64, GGML_TYPE_TBQP3_2, GGML_TYPE_TBQ4_2)
    FATTN_VEC_CASE(64, GGML_TYPE_TBQP4_2, GGML_TYPE_TBQ3_2)
    FATTN_VEC_CASE(64, GGML_TYPE_TBQP4_2, GGML_TYPE_TBQ4_2)
    FATTN_VEC_CASE(64, GGML_TYPE_TBQP3_2, GGML_TYPE_F16)
    FATTN_VEC_CASE(64, GGML_TYPE_TBQP4_2, GGML_TYPE_F16)
    FATTN_VEC_CASE(64, GGML_TYPE_TBQP3_2, GGML_TYPE_Q8_0)
    FATTN_VEC_CASE(64, GGML_TYPE_TBQP4_2, GGML_TYPE_Q8_0)

    // Asymmetric: standard K + TBQ V (ALL_QUANTS path)
    FATTN_VEC_CASE(256, GGML_TYPE_F16,  GGML_TYPE_TBQ3_0)
    FATTN_VEC_CASE(256, GGML_TYPE_F16,  GGML_TYPE_TBQ4_0)
    FATTN_VEC_CASE(256, GGML_TYPE_Q8_0, GGML_TYPE_TBQ3_0)
    FATTN_VEC_CASE(256, GGML_TYPE_Q8_0, GGML_TYPE_TBQ4_0)
    FATTN_VEC_CASE(256, GGML_TYPE_Q4_0, GGML_TYPE_TBQ3_0)
    FATTN_VEC_CASE(256, GGML_TYPE_Q4_0, GGML_TYPE_TBQ4_0)
    // _1 (D=128 only)
    FATTN_VEC_CASE(128, GGML_TYPE_F16,  GGML_TYPE_TBQ3_1)
    FATTN_VEC_CASE(128, GGML_TYPE_F16,  GGML_TYPE_TBQ4_1)
    FATTN_VEC_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_TBQ3_1)
    FATTN_VEC_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_TBQ4_1)
    FATTN_VEC_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_TBQ3_1)
    FATTN_VEC_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_TBQ4_1)
    // _2 (D=64 only)
    FATTN_VEC_CASE( 64, GGML_TYPE_F16,  GGML_TYPE_TBQ3_2)
    FATTN_VEC_CASE( 64, GGML_TYPE_F16,  GGML_TYPE_TBQ4_2)
    FATTN_VEC_CASE( 64, GGML_TYPE_Q8_0, GGML_TYPE_TBQ3_2)
    FATTN_VEC_CASE( 64, GGML_TYPE_Q8_0, GGML_TYPE_TBQ4_2)
    FATTN_VEC_CASE( 64, GGML_TYPE_Q4_0, GGML_TYPE_TBQ3_2)
    FATTN_VEC_CASE( 64, GGML_TYPE_Q4_0, GGML_TYPE_TBQ4_2)

    // Cross-head _3 (ALL_QUANTS path)
    FATTN_VEC_CASE(64, GGML_TYPE_TBQ3_3, GGML_TYPE_F16)
    FATTN_VEC_CASE(64, GGML_TYPE_TBQ4_3, GGML_TYPE_F16)
    FATTN_VEC_CASE(64, GGML_TYPE_TBQ3_3, GGML_TYPE_Q8_0)
    FATTN_VEC_CASE(64, GGML_TYPE_TBQ4_3, GGML_TYPE_Q8_0)
    FATTN_VEC_CASE(64, GGML_TYPE_TBQ3_3, GGML_TYPE_TBQ3_2)
    FATTN_VEC_CASE(64, GGML_TYPE_TBQ4_3, GGML_TYPE_TBQ4_2)
    FATTN_VEC_CASE(64, GGML_TYPE_TBQ4_3, GGML_TYPE_TBQ3_2)
    FATTN_VEC_CASE(64, GGML_TYPE_TBQ3_3, GGML_TYPE_TBQ4_2)
    FATTN_VEC_CASE(64, GGML_TYPE_TBQP3_3, GGML_TYPE_TBQ3_2)
    FATTN_VEC_CASE(64, GGML_TYPE_TBQP3_3, GGML_TYPE_TBQ4_2)
    FATTN_VEC_CASE(64, GGML_TYPE_TBQP4_3, GGML_TYPE_TBQ3_2)
    FATTN_VEC_CASE(64, GGML_TYPE_TBQP4_3, GGML_TYPE_TBQ4_2)
    FATTN_VEC_CASE(64, GGML_TYPE_TBQP3_3, GGML_TYPE_F16)
    FATTN_VEC_CASE(64, GGML_TYPE_TBQP4_3, GGML_TYPE_F16)
    FATTN_VEC_CASE(64, GGML_TYPE_TBQP3_3, GGML_TYPE_Q8_0)
    FATTN_VEC_CASE(64, GGML_TYPE_TBQP4_3, GGML_TYPE_Q8_0)
    FATTN_VEC_CASE(64, GGML_TYPE_TBQ3_3,  GGML_TYPE_TBQ3_3)
    FATTN_VEC_CASE(64, GGML_TYPE_TBQ3_3,  GGML_TYPE_TBQ4_3)
    FATTN_VEC_CASE(64, GGML_TYPE_TBQ4_3,  GGML_TYPE_TBQ3_3)
    FATTN_VEC_CASE(64, GGML_TYPE_TBQ4_3,  GGML_TYPE_TBQ4_3)
    FATTN_VEC_CASE(64, GGML_TYPE_TBQP3_3, GGML_TYPE_TBQ3_3)
    FATTN_VEC_CASE(64, GGML_TYPE_TBQP3_3, GGML_TYPE_TBQ4_3)
    FATTN_VEC_CASE(64, GGML_TYPE_TBQP4_3, GGML_TYPE_TBQ3_3)
    FATTN_VEC_CASE(64, GGML_TYPE_TBQP4_3, GGML_TYPE_TBQ4_3)

    // GLM asymmetric MUST come before symmetric 576 cases (ASYM checks V->ne[0]==512, symmetric doesn't)
    FATTN_VEC_CASE_ASYM(576, 512, GGML_TYPE_TBQ3_4,  GGML_TYPE_TBQ3_0)
    FATTN_VEC_CASE_ASYM(576, 512, GGML_TYPE_TBQ3_4,  GGML_TYPE_TBQ4_0)
    FATTN_VEC_CASE_ASYM(576, 512, GGML_TYPE_TBQ4_4,  GGML_TYPE_TBQ3_0)
    FATTN_VEC_CASE_ASYM(576, 512, GGML_TYPE_TBQ4_4,  GGML_TYPE_TBQ4_0)
    FATTN_VEC_CASE_ASYM(576, 512, GGML_TYPE_TBQP3_4, GGML_TYPE_TBQ3_0)
    FATTN_VEC_CASE_ASYM(576, 512, GGML_TYPE_TBQP3_4, GGML_TYPE_TBQ4_0)
    FATTN_VEC_CASE_ASYM(576, 512, GGML_TYPE_TBQP4_4, GGML_TYPE_TBQ3_0)
    FATTN_VEC_CASE_ASYM(576, 512, GGML_TYPE_TBQP4_4, GGML_TYPE_TBQ4_0)
    FATTN_VEC_CASE_ASYM(576, 512, GGML_TYPE_TBQ3_4,  GGML_TYPE_F16)
    FATTN_VEC_CASE_ASYM(576, 512, GGML_TYPE_TBQ4_4,  GGML_TYPE_F16)
    FATTN_VEC_CASE_ASYM(576, 512, GGML_TYPE_TBQP3_4, GGML_TYPE_F16)
    FATTN_VEC_CASE_ASYM(576, 512, GGML_TYPE_TBQP4_4, GGML_TYPE_F16)
    FATTN_VEC_CASE_ASYM(576, 512, GGML_TYPE_TBQ3_4,  GGML_TYPE_Q8_0)
    FATTN_VEC_CASE_ASYM(576, 512, GGML_TYPE_TBQ4_4,  GGML_TYPE_Q8_0)
    FATTN_VEC_CASE_ASYM(576, 512, GGML_TYPE_TBQP3_4, GGML_TYPE_Q8_0)
    FATTN_VEC_CASE_ASYM(576, 512, GGML_TYPE_TBQP4_4, GGML_TYPE_Q8_0)
    FATTN_VEC_CASE_ASYM(576, 512, GGML_TYPE_TBQP3_4, GGML_TYPE_TBQP3_4)
    FATTN_VEC_CASE_ASYM(576, 512, GGML_TYPE_TBQP4_4, GGML_TYPE_TBQP4_4)
    FATTN_VEC_CASE_ASYM(576, 512, GGML_TYPE_TBQ3_4,  GGML_TYPE_TBQ3_4)
    FATTN_VEC_CASE_ASYM(576, 512, GGML_TYPE_TBQ4_4,  GGML_TYPE_TBQ4_4)

    // TurboQuant 576-block (_4): D=576 symmetric (ALL_QUANTS path)
    FATTN_VEC_CASE(576, GGML_TYPE_TBQ3_4,  GGML_TYPE_F16)
    FATTN_VEC_CASE(576, GGML_TYPE_TBQ4_4,  GGML_TYPE_F16)
    FATTN_VEC_CASE(576, GGML_TYPE_TBQ3_4,  GGML_TYPE_Q8_0)
    FATTN_VEC_CASE(576, GGML_TYPE_TBQ4_4,  GGML_TYPE_Q8_0)
    FATTN_VEC_CASE(576, GGML_TYPE_TBQ3_4,  GGML_TYPE_TBQ3_4)
    FATTN_VEC_CASE(576, GGML_TYPE_TBQ4_4,  GGML_TYPE_TBQ4_4)
    FATTN_VEC_CASE(576, GGML_TYPE_TBQ4_4,  GGML_TYPE_TBQ3_4)
    FATTN_VEC_CASE(576, GGML_TYPE_TBQ3_4,  GGML_TYPE_TBQ4_4)
    FATTN_VEC_CASE(576, GGML_TYPE_TBQP3_4, GGML_TYPE_TBQ3_4)
    FATTN_VEC_CASE(576, GGML_TYPE_TBQP3_4, GGML_TYPE_TBQ4_4)
    FATTN_VEC_CASE(576, GGML_TYPE_TBQP4_4, GGML_TYPE_TBQ3_4)
    FATTN_VEC_CASE(576, GGML_TYPE_TBQP4_4, GGML_TYPE_TBQ4_4)
    FATTN_VEC_CASE(576, GGML_TYPE_TBQP3_4, GGML_TYPE_F16)
    FATTN_VEC_CASE(576, GGML_TYPE_TBQP4_4, GGML_TYPE_F16)
    FATTN_VEC_CASE(576, GGML_TYPE_TBQP3_4, GGML_TYPE_Q8_0)
    FATTN_VEC_CASE(576, GGML_TYPE_TBQP4_4, GGML_TYPE_Q8_0)
    FATTN_VEC_CASE(576, GGML_TYPE_F16,     GGML_TYPE_TBQ3_4)
    FATTN_VEC_CASE(576, GGML_TYPE_F16,     GGML_TYPE_TBQ4_4)
    FATTN_VEC_CASE(576, GGML_TYPE_Q8_0,    GGML_TYPE_TBQ3_4)
    FATTN_VEC_CASE(576, GGML_TYPE_Q8_0,    GGML_TYPE_TBQ4_4)
#else
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_BF16)
    // TurboQuant 256-block (_0): D=256 only
    FATTN_VEC_CASE(256, GGML_TYPE_TBQ3_0, GGML_TYPE_F16)
    FATTN_VEC_CASE(256, GGML_TYPE_TBQ4_0, GGML_TYPE_F16)
    FATTN_VEC_CASE(256, GGML_TYPE_TBQ3_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASE(256, GGML_TYPE_TBQ4_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASE(256, GGML_TYPE_TBQ3_0, GGML_TYPE_TBQ3_0)
    FATTN_VEC_CASE(256, GGML_TYPE_TBQ4_0, GGML_TYPE_TBQ4_0)
    FATTN_VEC_CASE(256, GGML_TYPE_TBQ3_0, GGML_TYPE_TBQ4_0)
    FATTN_VEC_CASE(256, GGML_TYPE_TBQ4_0, GGML_TYPE_TBQ3_0)
    FATTN_VEC_CASE(256, GGML_TYPE_TBQP3_0, GGML_TYPE_TBQ3_0)
    FATTN_VEC_CASE(256, GGML_TYPE_TBQP3_0, GGML_TYPE_TBQ4_0)
    FATTN_VEC_CASE(256, GGML_TYPE_TBQP4_0, GGML_TYPE_TBQ3_0)
    FATTN_VEC_CASE(256, GGML_TYPE_TBQP4_0, GGML_TYPE_TBQ4_0)
    FATTN_VEC_CASE(256, GGML_TYPE_TBQP3_0, GGML_TYPE_F16)
    FATTN_VEC_CASE(256, GGML_TYPE_TBQP4_0, GGML_TYPE_F16)
    FATTN_VEC_CASE(256, GGML_TYPE_TBQP3_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASE(256, GGML_TYPE_TBQP4_0, GGML_TYPE_Q8_0)
    // TurboQuant 256-block (_0) at D=512 (Gemma 4 global attention)
    FATTN_VEC_CASE(512, GGML_TYPE_TBQ3_0, GGML_TYPE_F16)
    FATTN_VEC_CASE(512, GGML_TYPE_TBQ4_0, GGML_TYPE_F16)
    FATTN_VEC_CASE(512, GGML_TYPE_TBQ3_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASE(512, GGML_TYPE_TBQ4_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASE(512, GGML_TYPE_TBQ3_0, GGML_TYPE_TBQ3_0)
    FATTN_VEC_CASE(512, GGML_TYPE_TBQ4_0, GGML_TYPE_TBQ4_0)
    FATTN_VEC_CASE(512, GGML_TYPE_TBQ4_0, GGML_TYPE_TBQ3_0)
    FATTN_VEC_CASE(512, GGML_TYPE_TBQP3_0, GGML_TYPE_TBQ3_0)
    FATTN_VEC_CASE(512, GGML_TYPE_TBQP4_0, GGML_TYPE_TBQ4_0)
    FATTN_VEC_CASE(512, GGML_TYPE_TBQP3_0, GGML_TYPE_F16)
    FATTN_VEC_CASE(512, GGML_TYPE_TBQP4_0, GGML_TYPE_F16)
    FATTN_VEC_CASE(512, GGML_TYPE_TBQP3_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASE(512, GGML_TYPE_TBQP4_0, GGML_TYPE_Q8_0)
    // TurboQuant 128-block (_1): D=128 only
    FATTN_VEC_CASE(128, GGML_TYPE_TBQ3_1, GGML_TYPE_F16)
    FATTN_VEC_CASE(128, GGML_TYPE_TBQ4_1, GGML_TYPE_F16)
    FATTN_VEC_CASE(128, GGML_TYPE_TBQ3_1, GGML_TYPE_Q8_0)
    FATTN_VEC_CASE(128, GGML_TYPE_TBQ4_1, GGML_TYPE_Q8_0)
    FATTN_VEC_CASE(128, GGML_TYPE_TBQ3_1, GGML_TYPE_TBQ3_1)
    FATTN_VEC_CASE(128, GGML_TYPE_TBQ4_1, GGML_TYPE_TBQ4_1)
    FATTN_VEC_CASE(128, GGML_TYPE_TBQ4_1, GGML_TYPE_TBQ3_1)
    FATTN_VEC_CASE(128, GGML_TYPE_TBQ3_1, GGML_TYPE_TBQ4_1)
    FATTN_VEC_CASE(128, GGML_TYPE_TBQP3_1, GGML_TYPE_TBQ3_1)
    FATTN_VEC_CASE(128, GGML_TYPE_TBQP3_1, GGML_TYPE_TBQ4_1)
    FATTN_VEC_CASE(128, GGML_TYPE_TBQP4_1, GGML_TYPE_TBQ3_1)
    FATTN_VEC_CASE(128, GGML_TYPE_TBQP4_1, GGML_TYPE_TBQ4_1)
    FATTN_VEC_CASE(128, GGML_TYPE_TBQP3_1, GGML_TYPE_F16)
    FATTN_VEC_CASE(128, GGML_TYPE_TBQP4_1, GGML_TYPE_F16)
    FATTN_VEC_CASE(128, GGML_TYPE_TBQP3_1, GGML_TYPE_Q8_0)
    FATTN_VEC_CASE(128, GGML_TYPE_TBQP4_1, GGML_TYPE_Q8_0)
    // TurboQuant 64-block (_2): D=64 only
    FATTN_VEC_CASE(64, GGML_TYPE_TBQ3_2, GGML_TYPE_F16)
    FATTN_VEC_CASE(64, GGML_TYPE_TBQ4_2, GGML_TYPE_F16)
    FATTN_VEC_CASE(64, GGML_TYPE_TBQ3_2, GGML_TYPE_Q8_0)
    FATTN_VEC_CASE(64, GGML_TYPE_TBQ4_2, GGML_TYPE_Q8_0)
    FATTN_VEC_CASE(64, GGML_TYPE_TBQ3_2, GGML_TYPE_TBQ3_2)
    FATTN_VEC_CASE(64, GGML_TYPE_TBQ4_2, GGML_TYPE_TBQ4_2)
    FATTN_VEC_CASE(64, GGML_TYPE_TBQ4_2, GGML_TYPE_TBQ3_2)
    FATTN_VEC_CASE(64, GGML_TYPE_TBQ3_2, GGML_TYPE_TBQ4_2)
    FATTN_VEC_CASE(64, GGML_TYPE_TBQP3_2, GGML_TYPE_TBQ3_2)
    FATTN_VEC_CASE(64, GGML_TYPE_TBQP3_2, GGML_TYPE_TBQ4_2)
    FATTN_VEC_CASE(64, GGML_TYPE_TBQP4_2, GGML_TYPE_TBQ3_2)
    FATTN_VEC_CASE(64, GGML_TYPE_TBQP4_2, GGML_TYPE_TBQ4_2)
    FATTN_VEC_CASE(64, GGML_TYPE_TBQP3_2, GGML_TYPE_F16)
    FATTN_VEC_CASE(64, GGML_TYPE_TBQP4_2, GGML_TYPE_F16)
    FATTN_VEC_CASE(64, GGML_TYPE_TBQP3_2, GGML_TYPE_Q8_0)
    FATTN_VEC_CASE(64, GGML_TYPE_TBQP4_2, GGML_TYPE_Q8_0)
    // TurboQuant cross-head (_3): D=64
    FATTN_VEC_CASE(64, GGML_TYPE_TBQ3_3, GGML_TYPE_F16)
    FATTN_VEC_CASE(64, GGML_TYPE_TBQ4_3, GGML_TYPE_F16)
    FATTN_VEC_CASE(64, GGML_TYPE_TBQ3_3, GGML_TYPE_Q8_0)
    FATTN_VEC_CASE(64, GGML_TYPE_TBQ4_3, GGML_TYPE_Q8_0)
    FATTN_VEC_CASE(64, GGML_TYPE_TBQ3_3, GGML_TYPE_TBQ3_2)
    FATTN_VEC_CASE(64, GGML_TYPE_TBQ4_3, GGML_TYPE_TBQ4_2)
    FATTN_VEC_CASE(64, GGML_TYPE_TBQ4_3, GGML_TYPE_TBQ3_2)
    FATTN_VEC_CASE(64, GGML_TYPE_TBQ3_3, GGML_TYPE_TBQ4_2)
    FATTN_VEC_CASE(64, GGML_TYPE_TBQP3_3, GGML_TYPE_TBQ3_2)
    FATTN_VEC_CASE(64, GGML_TYPE_TBQP3_3, GGML_TYPE_TBQ4_2)
    FATTN_VEC_CASE(64, GGML_TYPE_TBQP4_3, GGML_TYPE_TBQ3_2)
    FATTN_VEC_CASE(64, GGML_TYPE_TBQP4_3, GGML_TYPE_TBQ4_2)
    FATTN_VEC_CASE(64, GGML_TYPE_TBQP3_3, GGML_TYPE_F16)
    FATTN_VEC_CASE(64, GGML_TYPE_TBQP4_3, GGML_TYPE_F16)
    FATTN_VEC_CASE(64, GGML_TYPE_TBQP3_3, GGML_TYPE_Q8_0)
    FATTN_VEC_CASE(64, GGML_TYPE_TBQP4_3, GGML_TYPE_Q8_0)
    FATTN_VEC_CASE(64, GGML_TYPE_TBQ3_3,  GGML_TYPE_TBQ3_3)
    FATTN_VEC_CASE(64, GGML_TYPE_TBQ3_3,  GGML_TYPE_TBQ4_3)
    FATTN_VEC_CASE(64, GGML_TYPE_TBQ4_3,  GGML_TYPE_TBQ3_3)
    FATTN_VEC_CASE(64, GGML_TYPE_TBQ4_3,  GGML_TYPE_TBQ4_3)
    FATTN_VEC_CASE(64, GGML_TYPE_TBQP3_3, GGML_TYPE_TBQ3_3)
    FATTN_VEC_CASE(64, GGML_TYPE_TBQP3_3, GGML_TYPE_TBQ4_3)
    FATTN_VEC_CASE(64, GGML_TYPE_TBQP4_3, GGML_TYPE_TBQ3_3)
    FATTN_VEC_CASE(64, GGML_TYPE_TBQP4_3, GGML_TYPE_TBQ4_3)
    // Asymmetric: standard K + TBQ V (e.g. q8_0 K + tbq3 V)
    // D=256 (_0)
    FATTN_VEC_CASE(256, GGML_TYPE_F16,  GGML_TYPE_TBQ3_0)
    FATTN_VEC_CASE(256, GGML_TYPE_F16,  GGML_TYPE_TBQ4_0)
    FATTN_VEC_CASE(256, GGML_TYPE_Q8_0, GGML_TYPE_TBQ3_0)
    FATTN_VEC_CASE(256, GGML_TYPE_Q8_0, GGML_TYPE_TBQ4_0)
    FATTN_VEC_CASE(256, GGML_TYPE_Q4_0, GGML_TYPE_TBQ3_0)
    FATTN_VEC_CASE(256, GGML_TYPE_Q4_0, GGML_TYPE_TBQ4_0)
    // D=128 (_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_TBQ3_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_TBQ4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_TBQ3_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_TBQ4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_TBQ3_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_TBQ4_1)
    // D=64 (_2)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_TBQ3_2)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_TBQ4_2)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_TBQ3_2)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_TBQ4_2)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_TBQ3_2)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_TBQ4_2)
    // GLM asymmetric MUST come before symmetric 576 cases (ASYM checks V->ne[0]==512, symmetric doesn't)
    FATTN_VEC_CASE_ASYM(576, 512, GGML_TYPE_TBQ3_4,  GGML_TYPE_TBQ3_0)
    FATTN_VEC_CASE_ASYM(576, 512, GGML_TYPE_TBQ3_4,  GGML_TYPE_TBQ4_0)
    FATTN_VEC_CASE_ASYM(576, 512, GGML_TYPE_TBQ4_4,  GGML_TYPE_TBQ3_0)
    FATTN_VEC_CASE_ASYM(576, 512, GGML_TYPE_TBQ4_4,  GGML_TYPE_TBQ4_0)
    FATTN_VEC_CASE_ASYM(576, 512, GGML_TYPE_TBQP3_4, GGML_TYPE_TBQ3_0)
    FATTN_VEC_CASE_ASYM(576, 512, GGML_TYPE_TBQP3_4, GGML_TYPE_TBQ4_0)
    FATTN_VEC_CASE_ASYM(576, 512, GGML_TYPE_TBQP4_4, GGML_TYPE_TBQ3_0)
    FATTN_VEC_CASE_ASYM(576, 512, GGML_TYPE_TBQP4_4, GGML_TYPE_TBQ4_0)
    FATTN_VEC_CASE_ASYM(576, 512, GGML_TYPE_TBQ3_4,  GGML_TYPE_F16)
    FATTN_VEC_CASE_ASYM(576, 512, GGML_TYPE_TBQ4_4,  GGML_TYPE_F16)
    FATTN_VEC_CASE_ASYM(576, 512, GGML_TYPE_TBQP3_4, GGML_TYPE_F16)
    FATTN_VEC_CASE_ASYM(576, 512, GGML_TYPE_TBQP4_4, GGML_TYPE_F16)
    FATTN_VEC_CASE_ASYM(576, 512, GGML_TYPE_TBQ3_4,  GGML_TYPE_Q8_0)
    FATTN_VEC_CASE_ASYM(576, 512, GGML_TYPE_TBQ4_4,  GGML_TYPE_Q8_0)
    FATTN_VEC_CASE_ASYM(576, 512, GGML_TYPE_TBQP3_4, GGML_TYPE_Q8_0)
    FATTN_VEC_CASE_ASYM(576, 512, GGML_TYPE_TBQP4_4, GGML_TYPE_Q8_0)
    FATTN_VEC_CASE_ASYM(576, 512, GGML_TYPE_TBQP3_4, GGML_TYPE_TBQP3_4)
    FATTN_VEC_CASE_ASYM(576, 512, GGML_TYPE_TBQP4_4, GGML_TYPE_TBQP4_4)
    FATTN_VEC_CASE_ASYM(576, 512, GGML_TYPE_TBQ3_4,  GGML_TYPE_TBQ3_4)
    FATTN_VEC_CASE_ASYM(576, 512, GGML_TYPE_TBQ4_4,  GGML_TYPE_TBQ4_4)
    // TurboQuant 576-block (_4): D=576 symmetric
    FATTN_VEC_CASE(576, GGML_TYPE_TBQ3_4,  GGML_TYPE_F16)
    FATTN_VEC_CASE(576, GGML_TYPE_TBQ4_4,  GGML_TYPE_F16)
    FATTN_VEC_CASE(576, GGML_TYPE_TBQ3_4,  GGML_TYPE_Q8_0)
    FATTN_VEC_CASE(576, GGML_TYPE_TBQ4_4,  GGML_TYPE_Q8_0)
    FATTN_VEC_CASE(576, GGML_TYPE_TBQ3_4,  GGML_TYPE_TBQ3_4)
    FATTN_VEC_CASE(576, GGML_TYPE_TBQ4_4,  GGML_TYPE_TBQ4_4)
    FATTN_VEC_CASE(576, GGML_TYPE_TBQ4_4,  GGML_TYPE_TBQ3_4)
    FATTN_VEC_CASE(576, GGML_TYPE_TBQ3_4,  GGML_TYPE_TBQ4_4)
    FATTN_VEC_CASE(576, GGML_TYPE_TBQP3_4, GGML_TYPE_TBQ3_4)
    FATTN_VEC_CASE(576, GGML_TYPE_TBQP3_4, GGML_TYPE_TBQ4_4)
    FATTN_VEC_CASE(576, GGML_TYPE_TBQP4_4, GGML_TYPE_TBQ3_4)
    FATTN_VEC_CASE(576, GGML_TYPE_TBQP4_4, GGML_TYPE_TBQ4_4)
    FATTN_VEC_CASE(576, GGML_TYPE_TBQP3_4, GGML_TYPE_F16)
    FATTN_VEC_CASE(576, GGML_TYPE_TBQP4_4, GGML_TYPE_F16)
    FATTN_VEC_CASE(576, GGML_TYPE_TBQP3_4, GGML_TYPE_Q8_0)
    FATTN_VEC_CASE(576, GGML_TYPE_TBQP4_4, GGML_TYPE_Q8_0)
    FATTN_VEC_CASE(576, GGML_TYPE_F16,     GGML_TYPE_TBQ3_4)
    FATTN_VEC_CASE(576, GGML_TYPE_F16,     GGML_TYPE_TBQ4_4)
    FATTN_VEC_CASE(576, GGML_TYPE_Q8_0,    GGML_TYPE_TBQ3_4)
    FATTN_VEC_CASE(576, GGML_TYPE_Q8_0,    GGML_TYPE_TBQ4_4)
#endif // GGML_CUDA_FA_ALL_QUANTS

    GGML_ABORT("fatal error");
}

// Best FlashAttention kernel for a specific GPU:
enum best_fattn_kernel {
    BEST_FATTN_KERNEL_NONE     =   0,
    BEST_FATTN_KERNEL_TILE     = 200,
    BEST_FATTN_KERNEL_VEC      = 100,
    BEST_FATTN_KERNEL_WMMA_F16 = 300,
    BEST_FATTN_KERNEL_MMA_F16  = 400,
};

static best_fattn_kernel ggml_cuda_get_best_fattn_kernel(const int device, const ggml_tensor * dst) {
#ifndef FLASH_ATTN_AVAILABLE
    GGML_UNUSED(device); GGML_UNUSED(dst);
    return BEST_FATTN_KERNEL_NONE;
#endif// FLASH_ATTN_AVAILABLE

    const ggml_tensor * KQV   = dst;
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];

    const int gqa_ratio = Q->ne[2] / K->ne[2];
    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);

    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

    // The effective batch size for the kernel can be increased by gqa_ratio.
    // The kernel versions without this optimization are also used for ALiBi, if there is no mask, or if the KV cache is not padded,
    bool gqa_opt_applies = gqa_ratio >= 2 && mask && max_bias == 0.0f && K->ne[1] % FATTN_KQ_STRIDE == 0;
    for (const ggml_tensor * t : {Q, K, V, mask}) {
        if (t == nullptr || ggml_is_quantized(t->type)) {
            continue;
        }
        for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
            if (t->nb[i] % 16 != 0) {
                gqa_opt_applies = false;
                break;
            }
        }
    }

    const int cc = ggml_cuda_info().devices[device].cc;

    switch (K->ne[0]) {
        case  40:
        case  64:
        case  72:
        case  80:
        case  96:
        case 128:
        case 112:
        case 256:
            if (V->ne[0] != K->ne[0]) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 512:
            if (V->ne[0] != K->ne[0]) {
                return BEST_FATTN_KERNEL_NONE;
            }
            // TBQ vec kernel doesn't require GQA opt, only MMA does
            if (!gqa_opt_applies && !ggml_is_quantized(K->type) &&
                K->type != GGML_TYPE_TBQ3_0 && K->type != GGML_TYPE_TBQ4_0 &&
                K->type != GGML_TYPE_TBQP3_0 && K->type != GGML_TYPE_TBQP4_0) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 576:
            if (V->ne[0] != 512) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (!gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        default:
            return BEST_FATTN_KERNEL_NONE;
    }

#ifndef GGML_CUDA_FA_ALL_QUANTS
    if (K->type != V->type) {
        // Allow mixed K/V types for TBQ: TBQ K + standard V, or standard K + TBQ V
        const bool tbq_k = K->type == GGML_TYPE_TBQ3_0 || K->type == GGML_TYPE_TBQ4_0
                        || K->type == GGML_TYPE_TBQP3_0 || K->type == GGML_TYPE_TBQP4_0
                        || K->type == GGML_TYPE_TBQ3_1 || K->type == GGML_TYPE_TBQ4_1
                        || K->type == GGML_TYPE_TBQP3_1 || K->type == GGML_TYPE_TBQP4_1
                        || K->type == GGML_TYPE_TBQ3_2 || K->type == GGML_TYPE_TBQ4_2
                        || K->type == GGML_TYPE_TBQP3_2 || K->type == GGML_TYPE_TBQP4_2
                        || K->type == GGML_TYPE_TBQ3_3 || K->type == GGML_TYPE_TBQ4_3
                        || K->type == GGML_TYPE_TBQP3_3 || K->type == GGML_TYPE_TBQP4_3
                        || K->type == GGML_TYPE_TBQ3_4 || K->type == GGML_TYPE_TBQ4_4
                        || K->type == GGML_TYPE_TBQP3_4 || K->type == GGML_TYPE_TBQP4_4;
        const bool tbq_v = V->type == GGML_TYPE_TBQ3_0 || V->type == GGML_TYPE_TBQ4_0
                        || V->type == GGML_TYPE_TBQ3_1 || V->type == GGML_TYPE_TBQ4_1
                        || V->type == GGML_TYPE_TBQ3_2 || V->type == GGML_TYPE_TBQ4_2
                        || V->type == GGML_TYPE_TBQ3_3 || V->type == GGML_TYPE_TBQ4_3
                        || V->type == GGML_TYPE_TBQ3_4 || V->type == GGML_TYPE_TBQ4_4;
        const bool std_k = K->type == GGML_TYPE_F16 || K->type == GGML_TYPE_Q8_0
                        || K->type == GGML_TYPE_Q4_0 || K->type == GGML_TYPE_BF16;
        const bool std_v = V->type == GGML_TYPE_F16 || V->type == GGML_TYPE_Q8_0;
        // TBQ K + standard/TBQ V, or standard K + TBQ V
        if (!(tbq_k && (std_v || tbq_v)) && !(std_k && tbq_v)) {
            return BEST_FATTN_KERNEL_NONE;
        }
    }
#endif // GGML_CUDA_FA_ALL_QUANTS

    switch (K->type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
            break;
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
#ifndef GGML_CUDA_FA_ALL_QUANTS
            return BEST_FATTN_KERNEL_NONE;
#endif // GGML_CUDA_FA_ALL_QUANTS
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_BF16:
        case GGML_TYPE_TBQ3_0:
        case GGML_TYPE_TBQ4_0:
        case GGML_TYPE_TBQP3_0:
        case GGML_TYPE_TBQP4_0:
        case GGML_TYPE_TBQ3_1:
        case GGML_TYPE_TBQ4_1:
        case GGML_TYPE_TBQP3_1:
        case GGML_TYPE_TBQP4_1:
        case GGML_TYPE_TBQ3_2:
        case GGML_TYPE_TBQ4_2:
        case GGML_TYPE_TBQP3_2:
        case GGML_TYPE_TBQP4_2:
        case GGML_TYPE_TBQ3_3:
        case GGML_TYPE_TBQ4_3:
        case GGML_TYPE_TBQP3_3:
        case GGML_TYPE_TBQP4_3:
        case GGML_TYPE_TBQ3_4:
        case GGML_TYPE_TBQ4_4:
        case GGML_TYPE_TBQP3_4:
        case GGML_TYPE_TBQP4_4:
            break;
        default:
            return BEST_FATTN_KERNEL_NONE;
    }

    if (mask && mask->ne[2] != 1) {
        return BEST_FATTN_KERNEL_NONE;
    }

    // TBQ kernel support (as of v1.5.3):
    //   - D≤512 (Gemma 4 512, Qwen3.5 256, Llama/Qwen3 128, GPT-OSS 64): VEC kernel only
    //   - D=576 asymmetric (GLM-4.7-Flash, DeepSeek-V2/V3 MLA): MMA kernel (v1.4.2+) with VEC fallback
    //   - Tile/WMMA: not implemented
    // NOTE: Gemma 4 / Qwen3.5 / others fall back from MMA (upstream default) to VEC when TBQ is used,
    //       which means those models accept a speed regression vs f16. Quality work (v1.5.2) has been
    //       applied only to the VEC path so far — the MMA path (MLA only) is stuck at v1.4.2 quality.
    const bool tbq_k_type = K->type == GGML_TYPE_TBQ4_0 || K->type == GGML_TYPE_TBQ3_0
     || K->type == GGML_TYPE_TBQP4_0 || K->type == GGML_TYPE_TBQP3_0
     || K->type == GGML_TYPE_TBQ4_1 || K->type == GGML_TYPE_TBQ3_1
     || K->type == GGML_TYPE_TBQP4_1 || K->type == GGML_TYPE_TBQP3_1
     || K->type == GGML_TYPE_TBQ4_2 || K->type == GGML_TYPE_TBQ3_2
     || K->type == GGML_TYPE_TBQP4_2 || K->type == GGML_TYPE_TBQP3_2
     || K->type == GGML_TYPE_TBQ4_3 || K->type == GGML_TYPE_TBQ3_3
     || K->type == GGML_TYPE_TBQP4_3 || K->type == GGML_TYPE_TBQP3_3
     || K->type == GGML_TYPE_TBQ4_4 || K->type == GGML_TYPE_TBQ3_4
     || K->type == GGML_TYPE_TBQP4_4 || K->type == GGML_TYPE_TBQP3_4;
    const bool tbq_v_type = V->type == GGML_TYPE_TBQ4_0 || V->type == GGML_TYPE_TBQ3_0
     || V->type == GGML_TYPE_TBQ4_1 || V->type == GGML_TYPE_TBQ3_1
     || V->type == GGML_TYPE_TBQ4_2 || V->type == GGML_TYPE_TBQ3_2
     || V->type == GGML_TYPE_TBQ4_3 || V->type == GGML_TYPE_TBQ3_3
     || V->type == GGML_TYPE_TBQ4_4 || V->type == GGML_TYPE_TBQ3_4;
    if (tbq_k_type || tbq_v_type) {
        // EXPERIMENT (post-v1.5.3): route GLM MLA (D=576) to VEC kernel as well, to apply
        // v1.5.2 quality improvements (precision fix, sharpening, V IWHT float staging, etc.)
        // that only exist in the VEC path today. Once quality is validated on VEC, the
        // improvements will be ported back to the MMA kernel for the speed gain.
        //
        // Historical: v1.4.1 started MLA on VEC (30 t/s, quality-first); v1.4.2 ported to
        // MMA (49 t/s). We're temporarily reverting to the v1.4.1 path for quality work.
        if (Q->ne[0] <= 576 && Q->ne[0] % 64 == 0 && K->ne[1] % FATTN_KQ_STRIDE == 0) {
            return BEST_FATTN_KERNEL_VEC;
        }
        // (Disabled during experiment — will be re-enabled after quality port.)
        // if (Q->ne[0] == 576 && V->ne[0] == 512 && K->ne[1] % FATTN_KQ_STRIDE == 0) {
        //     if (turing_mma_available(cc) || volta_mma_available(cc)) {
        //         return BEST_FATTN_KERNEL_MMA_F16;
        //     }
        //     return BEST_FATTN_KERNEL_VEC;
        // }
        return BEST_FATTN_KERNEL_NONE;
    }

    // For small batch sizes the vector kernel may be preferable over the kernels optimized for large batch sizes:
    const bool can_use_vector_kernel = Q->ne[0] <= 256 && Q->ne[0] % 64 == 0 && K->ne[1] % FATTN_KQ_STRIDE == 0;

    // If Turing tensor cores are available, use them:
    if (turing_mma_available(cc) && Q->ne[0] != 40 && Q->ne[0] != 72) {
        if (can_use_vector_kernel) {
            if (!ggml_is_quantized(K->type) && !ggml_is_quantized(V->type)) {
                if (cc >= GGML_CUDA_CC_ADA_LOVELACE && Q->ne[1] == 1 && Q->ne[3] == 1 && !(gqa_ratio > 4 && K->ne[1] >= 8192)) {
                    return BEST_FATTN_KERNEL_VEC;
                }
            } else {
                if (cc >= GGML_CUDA_CC_ADA_LOVELACE) {
                    if (Q->ne[1] <= 2) {
                        return BEST_FATTN_KERNEL_VEC;
                    }
                } else {
                    if (Q->ne[1] == 1) {
                        return BEST_FATTN_KERNEL_VEC;
                    }
                }
            }
            if (!gqa_opt_applies && Q->ne[1] == 1) {
                return BEST_FATTN_KERNEL_VEC;
            }
        }
        return BEST_FATTN_KERNEL_MMA_F16;
    }

    if (volta_mma_available(cc) && Q->ne[0] != 40 && Q->ne[0] != 72) {
        int gqa_ratio_eff = 1;
        const int ncols2_max = Q->ne[0] == 576 ? 16 : 8;
        while (gqa_ratio % (2*gqa_ratio_eff) == 0 && gqa_ratio_eff < ncols2_max) {
            gqa_ratio_eff *= 2;
        }
        if (can_use_vector_kernel && Q->ne[1] * gqa_ratio_eff <= 2) {
            return BEST_FATTN_KERNEL_VEC;
        }
        if (Q->ne[1] * gqa_ratio_eff <= 16) {
            return BEST_FATTN_KERNEL_TILE; // On Volta tensor cores are only faster for sufficiently large matrices.
        }
        return BEST_FATTN_KERNEL_MMA_F16;
    }

    // Use the WMMA kernel if possible:
    if (ggml_cuda_should_use_wmma_fattn(cc) && K->ne[1] % FATTN_KQ_STRIDE == 0 && Q->ne[0] != 40 && Q->ne[0] != 72 && Q->ne[0] != 512 && Q->ne[0] != 576) {
        if (can_use_vector_kernel && Q->ne[1] <= 2) {
            return BEST_FATTN_KERNEL_VEC;
        }
        return BEST_FATTN_KERNEL_WMMA_F16;
    }

    if (amd_wmma_available(cc) && GGML_CUDA_CC_IS_RDNA4(cc) && gqa_opt_applies && Q->ne[0] <= 128 && Q->ne[0] != 40 && Q->ne[0] != 72) {
        if (can_use_vector_kernel) {
            if (!ggml_is_quantized(K->type) && !ggml_is_quantized(V->type)) {
                if (Q->ne[1] == 1) {
                    if (!gqa_opt_applies) {
                        return BEST_FATTN_KERNEL_VEC;
                    }
                }
            } else {
                if (Q->ne[1] <= 2) {
                    return BEST_FATTN_KERNEL_VEC;
                }
            }
        }
        int gqa_ratio_eff = 1;
        const int ncols2_max = Q->ne[0] == 576 ? 16 : 8;
        while (gqa_ratio % (2*gqa_ratio_eff) == 0 && gqa_ratio_eff < ncols2_max) {
            gqa_ratio_eff *= 2;
        }
        if (Q->ne[1] * gqa_ratio_eff <= 8) {
            return BEST_FATTN_KERNEL_TILE; // AMD WMMA is only faster if the full tile width of 16 can be utilized.
        }
        return BEST_FATTN_KERNEL_MMA_F16;
    }

    // Use MFMA flash attention for CDNA (MI100+):
    if (amd_mfma_available(cc) && Q->ne[0] != 40 && Q->ne[0] != 72 && Q->ne[0] != 256 && Q->ne[0] != 512 && Q->ne[0] != 576) {
        const int64_t eff_nq = Q->ne[1] * (gqa_opt_applies ? gqa_ratio : 1);
        // MMA vs tile crossover benchmarked on MI300X @ d32768:
        //   hsk=64  (gqa=4): MMA wins at eff >= 128 (+11%)
        //   hsk=128 (gqa=4): MMA wins at eff >= 128 (+4%)
        if (eff_nq >= (GGML_CUDA_CC_IS_CDNA1(cc) && Q->ne[0] == 64 ? 64 : 128)) {
            return BEST_FATTN_KERNEL_MMA_F16;
        }
        // Fall through to tile kernel for small effective batch sizes.
    }

    // If there are no tensor cores available, use the generic tile kernel:
    if (can_use_vector_kernel) {
        if (!ggml_is_quantized(K->type) && !ggml_is_quantized(V->type)) {
            if (Q->ne[1] == 1) {
                if (!gqa_opt_applies) {
                    return BEST_FATTN_KERNEL_VEC;
                }
            }
        } else {
            if (Q->ne[1] <= 2) {
                return BEST_FATTN_KERNEL_VEC;
            }
        }
    }
    return BEST_FATTN_KERNEL_TILE;
}

void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_set_device(ctx.device);
    switch (ggml_cuda_get_best_fattn_kernel(ggml_cuda_get_device(), dst)) {
        case BEST_FATTN_KERNEL_NONE:
            GGML_ABORT("fatal error");
        case BEST_FATTN_KERNEL_TILE:
            ggml_cuda_flash_attn_ext_tile(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_VEC:
            ggml_cuda_flash_attn_ext_vec(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_WMMA_F16:
            ggml_cuda_flash_attn_ext_wmma_f16(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_MMA_F16:
            ggml_cuda_flash_attn_ext_mma_f16(ctx, dst);
            break;
    }
}

bool ggml_cuda_flash_attn_ext_supported(int device, const ggml_tensor * dst) {
    return ggml_cuda_get_best_fattn_kernel(device, dst) != BEST_FATTN_KERNEL_NONE;
}
