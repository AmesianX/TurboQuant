#pragma once

#include "ggml-common.h"
#include "convert.cuh"

static __device__ __forceinline__ int best_index_int8(int n, const int8_t * val, float x) {
    if (x <= val[0]) return 0;
    if (x >= val[n-1]) return n-1;
    int ml = 0, mu = n-1;
    while (mu-ml > 1) {
        int mav = (ml+mu)/2;
        if (x < val[mav]) mu = mav; else ml = mav;
    }
    return x - val[mu-1] < val[mu] - x ? mu-1 : mu;
}

static __device__ void quantize_f32_q4_0_block(const float * __restrict__ x, block_q4_0 * __restrict__ y) {
    float amax = 0.0f;
    float vmax = 0.0f;

    for (int j = 0; j < QK4_0; ++j) {
        const float v = x[j];
        if (amax < fabsf(v)) {
            amax = fabsf(v);
            vmax = v;
        }
    }

    const float d  = vmax / -8;
    const float id = d ? 1.0f/d : 0.0f;

    y->d = d;

    for (int j = 0; j < QK4_0/2; ++j) {
        const float x0 = x[0       + j]*id;
        const float x1 = x[QK4_0/2 + j]*id;

        const uint8_t xi0 = min(15, (int8_t)(x0 + 8.5f));
        const uint8_t xi1 = min(15, (int8_t)(x1 + 8.5f));

        y->qs[j]  = xi0;
        y->qs[j] |= xi1 << 4;
    }
}

static __device__ void quantize_f32_q4_1_block(const float * __restrict__ x, block_q4_1 * __restrict__ y) {
    float vmin = FLT_MAX;
    float vmax = -FLT_MAX;

    for (int j = 0; j < QK4_1; ++j) {
        const float v = x[j];
        if (v < vmin) vmin = v;
        if (v > vmax) vmax = v;
    }

    const float d  = (vmax - vmin) / ((1 << 4) - 1);
    const float id = d ? 1.0f/d : 0.0f;

    y->dm.x = d;
    y->dm.y = vmin;

    for (int j = 0; j < QK4_1/2; ++j) {
        const float x0 = (x[0       + j] - vmin)*id;
        const float x1 = (x[QK4_1/2 + j] - vmin)*id;

        const uint8_t xi0 = min(15, (int8_t)(x0 + 0.5f));
        const uint8_t xi1 = min(15, (int8_t)(x1 + 0.5f));

        y->qs[j]  = xi0;
        y->qs[j] |= xi1 << 4;
    }
}

static __device__ void quantize_f32_q5_0_block(const float * __restrict__ x, block_q5_0 * __restrict__ y) {
    float amax = 0.0f;
    float vmax = 0.0f;

    for (int j = 0; j < QK5_0; ++j) {
        const float v = x[j];
        if (amax < fabsf(v)) {
            amax = fabsf(v);
            vmax = v;
        }
    }

    const float d  = vmax / -16;
    const float id = d ? 1.0f/d : 0.0f;

    y->d = d;

    uint32_t qh = 0;
    for (int j = 0; j < QK5_0/2; ++j) {
        const float x0 = x[0       + j]*id;
        const float x1 = x[QK5_0/2 + j]*id;

        const uint8_t xi0 = min(31, (int8_t)(x0 + 16.5f));
        const uint8_t xi1 = min(31, (int8_t)(x1 + 16.5f));

        y->qs[j]  = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_0/2);
    }
    memcpy(y->qh, &qh, sizeof(qh));
}

static __device__ void quantize_f32_q5_1_block(const float * __restrict__ x, block_q5_1 * __restrict__ y) {
    float min = x[0];
    float max = x[0];

    for (int j = 1; j < QK5_1; ++j) {
        const float v = x[j];
        min = v < min ? v : min;
        max = v > max ? v : max;
    }

    const float d  = (max - min) / 31;
    const float id = d ? 1.0f/d : 0.0f;

    y->dm.x = d;
    y->dm.y = min;

    uint32_t qh = 0;
    for (int j = 0; j < QK5_1/2; ++j) {
        const float x0 = (x[0       + j] - min)*id;
        const float x1 = (x[QK5_1/2 + j] - min)*id;

        const uint8_t xi0 = (uint8_t)(x0 + 0.5f);
        const uint8_t xi1 = (uint8_t)(x1 + 0.5f);

        y->qs[j]  = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_1/2);
    }
    memcpy(y->qh, &qh, sizeof(qh));
}

static __device__ void quantize_f32_q8_0_block(const float * __restrict__ x, block_q8_0 * __restrict__ y) {
    float amax = 0.0f; // absolute max

    for (int j = 0; j < QK8_0; j++) {
        const float v = x[j];
        amax = fmaxf(amax, fabsf(v));
    }

    const float d = amax / ((1 << 7) - 1);
    const float id = d ? 1.0f/d : 0.0f;

    y->d = d;

    for (int j = 0; j < QK8_0; ++j) {
        const float x0 = x[j]*id;
        y->qs[j] = roundf(x0);
    }
}

// TurboQuant 3-bit: L2 norm + random signs + serial WHT + Lloyd-Max quantization
static __device__ void quantize_f32_tbq3_0_block(const float * __restrict__ x, block_tbq3_0 * __restrict__ y) {
    static constexpr uint8_t tbq_signs[32] = {
        0xa7,0x3b,0x91,0xf4,0x6d,0xc2,0x58,0x0e,
        0xb3,0x7f,0x24,0xd6,0x89,0x45,0xea,0x1c,
        0x63,0xaf,0xd8,0x52,0x97,0x0b,0xe1,0x3d,
        0x76,0xc4,0x19,0xfe,0x4a,0x85,0x2c,0xdb,
    };
    static constexpr float tbq3_boundaries[7] = {
        -1.7480f, -1.0500f, -0.5006f, 0.0f, 0.5006f, 1.0500f, 1.7480f,
    };

    float tmp[QK_K];
    float sum_sq = 0.0f;
    for (int j = 0; j < QK_K; j++) sum_sq += x[j] * x[j];
    float norm = sqrtf(sum_sq);
    y->d = __float2half(norm);

    if (norm < 1e-10f) {
        for (int j = 0; j < QK_K*3/8; j++) y->qs[j] = 0;
        return;
    }

    float inv_norm = 1.0f / norm;
    for (int j = 0; j < QK_K; j++) {
        int sign = ((tbq_signs[j >> 3] >> (j & 7)) & 1) ? -1 : 1;
        tmp[j] = x[j] * inv_norm * sign;
    }

    // Serial WHT
    for (int len = 1; len < QK_K; len *= 2)
        for (int i = 0; i < QK_K; i += 2*len)
            for (int j = 0; j < len; j++) {
                float u = tmp[i+j], v = tmp[i+j+len];
                tmp[i+j] = u+v; tmp[i+j+len] = u-v;
            }

    // 3-bit quantization + packing
    int bit_pos = 0;
    for (int j = 0; j < QK_K*3/8; j++) y->qs[j] = 0;
    for (int j = 0; j < QK_K; j++) {
        uint8_t idx = 7;
        for (int b = 0; b < 7; b++) { if (tmp[j] < tbq3_boundaries[b]) { idx = b; break; } }
        int byte_idx = bit_pos / 8;
        int bit_off = bit_pos % 8;
        y->qs[byte_idx] |= (uint8_t)((idx & 0x7) << bit_off);
        if (bit_off > 5) y->qs[byte_idx + 1] |= (uint8_t)((idx & 0x7) >> (8 - bit_off));
        bit_pos += 3;
    }
}

// TurboQuant 4-bit: L2 norm + random signs + serial WHT + Lloyd-Max quantization
// Single-thread implementation for set_rows (called once per token, performance uncritical)
static __device__ void quantize_f32_tbq4_0_block(const float * __restrict__ x, block_tbq4_0 * __restrict__ y) {
    // Random sign pattern (must match CPU and turboquant.cu exactly)
    static constexpr uint8_t tbq_signs[32] = {
        0xa7,0x3b,0x91,0xf4,0x6d,0xc2,0x58,0x0e,
        0xb3,0x7f,0x24,0xd6,0x89,0x45,0xea,0x1c,
        0x63,0xaf,0xd8,0x52,0x97,0x0b,0xe1,0x3d,
        0x76,0xc4,0x19,0xfe,0x4a,0x85,0x2c,0xdb,
    };
    // Lloyd-Max decision boundaries for N(0,1), 4-bit (15 boundaries)
    static constexpr float tbq4_boundaries[15] = {
        -2.4008f,-1.8435f,-1.4371f,-1.0993f,-0.7996f,-0.5225f,-0.2583f,
         0.0f,
         0.2583f, 0.5225f, 0.7996f, 1.0993f, 1.4371f, 1.8435f, 2.4008f,
    };

    float tmp[QK_K];

    // Step 1: L2 norm
    float sum_sq = 0.0f;
    for (int j = 0; j < QK_K; j++) sum_sq += x[j] * x[j];
    float norm = sqrtf(sum_sq);
    y->d = __float2half(norm);

    if (norm < 1e-10f) {
        for (int j = 0; j < QK_K/2; j++) y->qs[j] = 0;
        return;
    }

    // Step 2: Normalize + random signs
    float inv_norm = 1.0f / norm;
    for (int j = 0; j < QK_K; j++) {
        int sign = ((tbq_signs[j >> 3] >> (j & 7)) & 1) ? -1 : 1;
        tmp[j] = x[j] * inv_norm * sign;
    }

    // Step 3: Serial WHT (single thread, QK_K=256)
    for (int len = 1; len < QK_K; len *= 2) {
        for (int i = 0; i < QK_K; i += 2 * len) {
            for (int j = 0; j < len; j++) {
                float u = tmp[i + j], v = tmp[i + j + len];
                tmp[i + j] = u + v;
                tmp[i + j + len] = u - v;
            }
        }
    }

    // Step 4: Lloyd-Max 4-bit quantization + packing
    for (int j = 0; j < QK_K/2; j++) {
        uint8_t idx0 = 15, idx1 = 15;
        for (int b = 0; b < 15; b++) { if (tmp[2*j]   < tbq4_boundaries[b]) { idx0 = b; break; } }
        for (int b = 0; b < 15; b++) { if (tmp[2*j+1] < tbq4_boundaries[b]) { idx1 = b; break; } }
        y->qs[j] = idx0 | (idx1 << 4);
    }
}

// TurboQuant_prod 3-bit: 2-bit Lloyd-Max + 1-bit QJL residual correction
// QJL uses SRHT: S = D2 * H * D1 (two sign patterns + WHT)
static __device__ void quantize_f32_tbqp3_0_block(const float * __restrict__ x, block_tbqp3_0 * __restrict__ y) {
    static constexpr uint8_t tbq_signs[32] = {
        0xa7,0x3b,0x91,0xf4,0x6d,0xc2,0x58,0x0e,0xb3,0x7f,0x24,0xd6,0x89,0x45,0xea,0x1c,
        0x63,0xaf,0xd8,0x52,0x97,0x0b,0xe1,0x3d,0x76,0xc4,0x19,0xfe,0x4a,0x85,0x2c,0xdb,
    };
    // Second sign pattern for QJL SRHT (different from WHT signs)
    static constexpr uint8_t qjl_signs[32] = {
        0xd3,0x4e,0xa8,0x17,0x9c,0x5b,0xe6,0x31,0x72,0xb9,0x0d,0xf5,0x43,0x8a,0x6e,0xc7,
        0x58,0x2f,0x94,0xe1,0xb6,0x3d,0x0a,0x7c,0xc5,0x61,0xd8,0x4f,0xa3,0x97,0x1e,0x85,
    };
    // 2-bit Lloyd-Max centroids for N(0,1)
    static constexpr float c2[4] = { -1.5104f, -0.4528f, 0.4528f, 1.5104f };
    static constexpr float b2[3] = { -0.9816f, 0.0f, 0.9816f };

    float tmp[QK_K], recon[QK_K];

    // Step 1: L2 norm + normalize + signs + WHT
    float sum_sq = 0.0f;
    for (int j = 0; j < QK_K; j++) sum_sq += x[j] * x[j];
    float norm = sqrtf(sum_sq);
    y->d = __float2half(norm);
    if (norm < 1e-10f) {
        for (int j = 0; j < QK_K/4; j++) y->qs[j] = 0;
        for (int j = 0; j < QK_K/8; j++) y->qjl[j] = 0;
        y->d_qjl = __float2half(0.0f);
        return;
    }
    float inv_norm = 1.0f / norm;
    for (int j = 0; j < QK_K; j++) {
        int sign = ((tbq_signs[j >> 3] >> (j & 7)) & 1) ? -1 : 1;
        tmp[j] = x[j] * inv_norm * sign;
    }
    for (int len = 1; len < QK_K; len *= 2)
        for (int i = 0; i < QK_K; i += 2*len)
            for (int j = 0; j < len; j++) {
                float u = tmp[i+j], v = tmp[i+j+len];
                tmp[i+j] = u+v; tmp[i+j+len] = u-v;
            }

    // Step 2: 2-bit Lloyd-Max quantization
    for (int j = 0; j < QK_K/4; j++) {
        uint8_t packed = 0;
        for (int k = 0; k < 4; k++) {
            uint8_t idx = 3;
            for (int b = 0; b < 3; b++) { if (tmp[j*4+k] < b2[b]) { idx = b; break; } }
            packed |= (idx & 0x3) << (k*2);
            recon[j*4+k] = c2[idx]; // reconstructed centroid for residual
        }
        y->qs[j] = packed;
    }

    // Step 3: Compute residual in WHT domain
    float res[QK_K];
    float res_sq = 0.0f;
    for (int j = 0; j < QK_K; j++) {
        res[j] = tmp[j] - recon[j];
        res_sq += res[j] * res[j];
    }
    float gamma = sqrtf(res_sq);
    y->d_qjl = __float2half(gamma * norm); // store gamma*norm for direct use

    // Step 4: QJL = sign(SRHT(residual)) where SRHT = D2 * H * D1
    // Apply D1 (first sign pattern for QJL)
    for (int j = 0; j < QK_K; j++) {
        int sign = ((qjl_signs[j >> 3] >> (j & 7)) & 1) ? -1 : 1;
        res[j] *= sign;
    }
    // WHT
    for (int len = 1; len < QK_K; len *= 2)
        for (int i = 0; i < QK_K; i += 2*len)
            for (int j = 0; j < len; j++) {
                float u = res[i+j], v = res[i+j+len];
                res[i+j] = u+v; res[i+j+len] = u-v;
            }
    // Store sign bits
    for (int j = 0; j < QK_K/8; j++) y->qjl[j] = 0;
    for (int j = 0; j < QK_K; j++) {
        if (res[j] >= 0.0f) y->qjl[j/8] |= (1 << (j%8));
    }
}

// TurboQuant_prod 4-bit: 3-bit Lloyd-Max + 1-bit QJL
static __device__ void quantize_f32_tbqp4_0_block(const float * __restrict__ x, block_tbqp4_0 * __restrict__ y) {
    static constexpr uint8_t tbq_signs[32] = {
        0xa7,0x3b,0x91,0xf4,0x6d,0xc2,0x58,0x0e,0xb3,0x7f,0x24,0xd6,0x89,0x45,0xea,0x1c,
        0x63,0xaf,0xd8,0x52,0x97,0x0b,0xe1,0x3d,0x76,0xc4,0x19,0xfe,0x4a,0x85,0x2c,0xdb,
    };
    static constexpr uint8_t qjl_signs[32] = {
        0xd3,0x4e,0xa8,0x17,0x9c,0x5b,0xe6,0x31,0x72,0xb9,0x0d,0xf5,0x43,0x8a,0x6e,0xc7,
        0x58,0x2f,0x94,0xe1,0xb6,0x3d,0x0a,0x7c,0xc5,0x61,0xd8,0x4f,0xa3,0x97,0x1e,0x85,
    };
    static constexpr float c3[8] = {
        -2.1520f,-1.3440f,-0.7560f,-0.2451f, 0.2451f, 0.7560f, 1.3440f, 2.1520f,
    };
    static constexpr float b3[7] = {
        -1.7480f, -1.0500f, -0.5006f, 0.0f, 0.5006f, 1.0500f, 1.7480f,
    };

    float tmp[QK_K], recon[QK_K];
    float sum_sq = 0.0f;
    for (int j = 0; j < QK_K; j++) sum_sq += x[j] * x[j];
    float norm = sqrtf(sum_sq);
    y->d = __float2half(norm);
    if (norm < 1e-10f) {
        for (int j = 0; j < QK_K*3/8; j++) y->qs[j] = 0;
        for (int j = 0; j < QK_K/8; j++) y->qjl[j] = 0;
        y->d_qjl = __float2half(0.0f);
        return;
    }
    float inv_norm = 1.0f / norm;
    for (int j = 0; j < QK_K; j++) {
        int sign = ((tbq_signs[j >> 3] >> (j & 7)) & 1) ? -1 : 1;
        tmp[j] = x[j] * inv_norm * sign;
    }
    for (int len = 1; len < QK_K; len *= 2)
        for (int i = 0; i < QK_K; i += 2*len)
            for (int j = 0; j < len; j++) {
                float u = tmp[i+j], v = tmp[i+j+len];
                tmp[i+j] = u+v; tmp[i+j+len] = u-v;
            }

    // 3-bit Lloyd-Max + 3-bit packing
    int bit_pos = 0;
    for (int j = 0; j < QK_K*3/8; j++) y->qs[j] = 0;
    for (int j = 0; j < QK_K; j++) {
        uint8_t idx = 7;
        for (int b = 0; b < 7; b++) { if (tmp[j] < b3[b]) { idx = b; break; } }
        int byte_idx = bit_pos / 8, bit_off = bit_pos % 8;
        y->qs[byte_idx] |= (uint8_t)((idx & 0x7) << bit_off);
        if (bit_off > 5) y->qs[byte_idx + 1] |= (uint8_t)((idx & 0x7) >> (8 - bit_off));
        recon[j] = c3[idx];
        bit_pos += 3;
    }

    // Residual + QJL
    float res[QK_K], res_sq = 0.0f;
    for (int j = 0; j < QK_K; j++) { res[j] = tmp[j] - recon[j]; res_sq += res[j]*res[j]; }
    float gamma = sqrtf(res_sq);
    y->d_qjl = __float2half(gamma * norm);
    for (int j = 0; j < QK_K; j++) {
        int sign = ((qjl_signs[j >> 3] >> (j & 7)) & 1) ? -1 : 1;
        res[j] *= sign;
    }
    for (int len = 1; len < QK_K; len *= 2)
        for (int i = 0; i < QK_K; i += 2*len)
            for (int j = 0; j < len; j++) {
                float u = res[i+j], v = res[i+j+len];
                res[i+j] = u+v; res[i+j+len] = u-v;
            }
    for (int j = 0; j < QK_K/8; j++) y->qjl[j] = 0;
    for (int j = 0; j < QK_K; j++) {
        if (res[j] >= 0.0f) y->qjl[j/8] |= (1 << (j%8));
    }
}

static __device__ void quantize_f32_iq4_nl_block(const float * __restrict__ x, block_iq4_nl * __restrict__ y) {
    float amax = 0.0f;
    float vmax = 0.0f;

    for (int j = 0; j < QK4_NL; ++j) {
        const float v = x[j];
        if (amax < fabsf(v)) {
            amax = fabsf(v);
            vmax = v;
        }
    }

    float d = vmax / kvalues_iq4nl[0];
    const float id = d ? 1.0f/d : 0.0f;

    float sumqx = 0, sumq2 = 0;
    for (int j = 0; j < QK4_NL/2; ++j) {
        const float x0 = x[0        + j]*id;
        const float x1 = x[QK4_NL/2 + j]*id;
        const uint8_t xi0 = best_index_int8(16, kvalues_iq4nl, x0);
        const uint8_t xi1 = best_index_int8(16, kvalues_iq4nl, x1);
        y->qs[j] = xi0 | (xi1 << 4);
        const float v0 = kvalues_iq4nl[xi0];
        const float v1 = kvalues_iq4nl[xi1];
        const float w0 = x[0        + j]*x[0        + j];
        const float w1 = x[QK4_NL/2 + j]*x[QK4_NL/2 + j];
        sumqx += w0*v0*x[j] + w1*v1*x[QK4_NL/2 + j];
        sumq2 += w0*v0*v0 + w1*v1*v1;
    }

    y->d = sumq2 > 0 ? sumqx/sumq2 : d;
}

// Wrapper functions for cpy.cu compatibility
static __device__ void cpy_blck_f32_q4_0(const char * cxi, char * cdsti) {
    quantize_f32_q4_0_block((const float *)cxi, (block_q4_0 *)cdsti);
}

static __device__ void cpy_blck_f32_q4_1(const char * cxi, char * cdsti) {
    quantize_f32_q4_1_block((const float *)cxi, (block_q4_1 *)cdsti);
}

static __device__ void cpy_blck_f32_q5_0(const char * cxi, char * cdsti) {
    quantize_f32_q5_0_block((const float *)cxi, (block_q5_0 *)cdsti);
}

static __device__ void cpy_blck_f32_q5_1(const char * cxi, char * cdsti) {
    quantize_f32_q5_1_block((const float *)cxi, (block_q5_1 *)cdsti);
}

static __device__ void cpy_blck_f32_q8_0(const char * cxi, char * cdsti) {
    quantize_f32_q8_0_block((const float *)cxi, (block_q8_0 *)cdsti);
}

static __device__ void cpy_blck_f32_iq4_nl(const char * cxi, char * cdsti) {
    quantize_f32_iq4_nl_block((const float *)cxi, (block_iq4_nl *)cdsti);
}

template<typename src_t, typename dst_t>
static __device__ void cpy_1_scalar(const char * cxi, char * cdsti) {
    *(dst_t *) cdsti = ggml_cuda_cast<dst_t>(*(const src_t *) cxi);
}
