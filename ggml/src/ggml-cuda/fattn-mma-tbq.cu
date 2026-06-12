// TurboQuant: typed (quantized-KV) instantiations of the MMA flash-attention kernel.
// The tile loader dequantizes TBQ blocks global->smem (per-tile, NOT the full-cache to_fp16
// that froze GB10) — see turboquant/FATTN_MMA_TBQ_PORT.md.
//
// This TU lives in the ggml-cuda root on purpose: template-instances/*.cu are wiped by
// generate_cu_files.py, and keeping the dispatcher next to the instantiations lets them be
// implicit (no extern-template bookkeeping).
//
// S2a scope: D=128, K=F16, V in {TBQ3_1, TBQV3_1, AMXV3_1} (V-side dequant + output IWHT).
// K-side (Q-WHT + outlier correction) lands in S2b.

#include "common.cuh"
#include "fattn-common.cuh"
#include "fattn-mma-f16.cuh"

template <ggml_type type_V, int ncols2>
static void ggml_cuda_fattn_mma_tbq_switch_ncols1(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * Q = dst->src[0];

    constexpr int DKQ = 128;
    constexpr int DV  = 128;

    if constexpr (ncols2 <= 8) {
        if (turing_mma_available(cc) && Q->ne[1] <= 8/ncols2) {
            ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 8/ncols2, ncols2, GGML_TYPE_F16, type_V>(ctx, dst);
            return;
        }
    }

    if (Q->ne[1] <= 16/ncols2) {
        ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 16/ncols2, ncols2, GGML_TYPE_F16, type_V>(ctx, dst);
        return;
    }

    if (Q->ne[1] <= 32/ncols2 || (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_TURING)) {
        ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 32/ncols2, ncols2, GGML_TYPE_F16, type_V>(ctx, dst);
        return;
    }

    ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 64/ncols2, ncols2, GGML_TYPE_F16, type_V>(ctx, dst);
}

template <ggml_type type_V>
static void ggml_cuda_fattn_mma_tbq_switch_ncols2(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV  = dst;
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * mask = dst->src[3];

    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

    // same gating as the f16 switch: GQA specializations need a mask, no ALiBi and padded K
    // (quantized tensors are exempt from the 16B-alignment requirement, like upstream)
    bool use_gqa_opt = mask && max_bias == 0.0f && K->ne[1] % FATTN_KQ_STRIDE == 0;
    for (const ggml_tensor * t : {dst->src[0], dst->src[1], dst->src[2], dst->src[3]}) {
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

    const int gqa_ratio = Q->ne[2] / K->ne[2];

    if (use_gqa_opt && gqa_ratio % 8 == 0) {
        ggml_cuda_fattn_mma_tbq_switch_ncols1<type_V, 8>(ctx, dst);
        return;
    }
    if (use_gqa_opt && gqa_ratio % 4 == 0) {
        ggml_cuda_fattn_mma_tbq_switch_ncols1<type_V, 4>(ctx, dst);
        return;
    }
    if (use_gqa_opt && gqa_ratio % 2 == 0) {
        ggml_cuda_fattn_mma_tbq_switch_ncols1<type_V, 2>(ctx, dst);
        return;
    }
    ggml_cuda_fattn_mma_tbq_switch_ncols1<type_V, 1>(ctx, dst);
}

// Returns true when a typed MMA instantiation exists for this op (used by kernel selection —
// anything unsupported keeps falling back to the vec kernel).
bool ggml_cuda_fattn_mma_tbq_supported(const ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    if (Q->ne[0] != 128 || V->ne[0] != 128) {
        return false;
    }
    if (K->type != GGML_TYPE_F16) {
        return false; // S2b: TBQ K (Q-WHT + outlier correction) not ported yet
    }
    return V->type == GGML_TYPE_TBQ3_1 || V->type == GGML_TYPE_TBQV3_1 || V->type == GGML_TYPE_AMXV3_1;
}

void ggml_cuda_flash_attn_ext_mma_f16_tbq(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * V = dst->src[2];

    GGML_ASSERT(ggml_cuda_fattn_mma_tbq_supported(dst));

    switch (V->type) {
        case GGML_TYPE_TBQ3_1:
            ggml_cuda_fattn_mma_tbq_switch_ncols2<GGML_TYPE_TBQ3_1>(ctx, dst);
            break;
        case GGML_TYPE_TBQV3_1:
            ggml_cuda_fattn_mma_tbq_switch_ncols2<GGML_TYPE_TBQV3_1>(ctx, dst);
            break;
        case GGML_TYPE_AMXV3_1:
            ggml_cuda_fattn_mma_tbq_switch_ncols2<GGML_TYPE_AMXV3_1>(ctx, dst);
            break;
        default:
            GGML_ABORT("fattn-mma-tbq: unsupported V type %s", ggml_type_name(V->type));
    }
}
