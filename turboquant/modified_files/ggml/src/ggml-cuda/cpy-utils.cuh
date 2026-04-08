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

// TurboQuant 3-bit: 512-point single-pass WHT, then split into 2 × 256-block for storage
// Used when D=512 (Gemma 4 global attention). Better decorrelation than 2 × independent 256-WHT.
static __device__ void quantize_f32_tbq3_0_block_512(const float * __restrict__ x, block_tbq3_0 * __restrict__ y) {
    // 512-element sign pattern (64 bytes)
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
    static constexpr float tbq3_boundaries[7] = {
        -1.7480f, -1.0500f, -0.5006f, 0.0f, 0.5006f, 1.0500f, 1.7480f,
    };

    float tmp[512];

    // Apply signs on raw data
    for (int j = 0; j < 512; j++) {
        int sign = ((tbq_signs_512[j >> 3] >> (j & 7)) & 1) ? -1 : 1;
        tmp[j] = x[j] * sign;
    }

    // 512-point serial WHT (9 stages)
    for (int len = 1; len < 512; len *= 2)
        for (int i = 0; i < 512; i += 2*len)
            for (int j = 0; j < len; j++) {
                float u = tmp[i+j], v = tmp[i+j+len];
                tmp[i+j] = u+v; tmp[i+j+len] = u-v;
            }

    // Per-block norm: each 256-half gets its own norm after 512-WHT
    for (int blk = 0; blk < 2; blk++) {
        float * blk_data = tmp + blk * 256;

        float blk_sq = 0.0f;
        for (int j = 0; j < 256; j++) blk_sq += blk_data[j] * blk_data[j];
        float blk_norm = sqrtf(blk_sq / 256.0f);

        if (blk_norm < 1e-10f) {
            y[blk].d = __float2half(0.0f);
            for (int j = 0; j < QK_K*3/8; j++) y[blk].qs[j] = 0;
            continue;
        }

        float inv_norm = 1.0f / blk_norm;
        y[blk].d = __float2half(blk_norm);

        int bit_pos = 0;
        for (int j = 0; j < QK_K*3/8; j++) y[blk].qs[j] = 0;
        for (int j = 0; j < 256; j++) {
            float val = blk_data[j] * inv_norm;
            uint8_t idx = 7;
            for (int b = 0; b < 7; b++) { if (val < tbq3_boundaries[b]) { idx = b; break; } }
            int byte_idx = bit_pos / 8;
            int bit_off = bit_pos % 8;
            y[blk].qs[byte_idx] |= (uint8_t)((idx & 0x7) << bit_off);
            if (bit_off > 5) y[blk].qs[byte_idx + 1] |= (uint8_t)((idx & 0x7) >> (8 - bit_off));
            bit_pos += 3;
        }
    }
}

// TurboQuant_prod 3-bit: 512-point WHT + 2-bit Lloyd-Max + 1-bit QJL, global norm
// Note: uses 6KB stack (tmp[512]+recon[512]+res[512]) — called once per token via set_rows,
// NOT on the attention hot path. Do not attempt to "optimize" by reducing stack usage.
static __device__ void quantize_f32_tbqp3_0_block_512(const float * __restrict__ x, block_tbqp3_0 * __restrict__ y) {
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
    static constexpr uint8_t qjl_signs_512[64] = {
        0x21,0x4e,0x75,0x8d,0x12,0xa1,0x04,0x88,
        0x6c,0x5d,0x2c,0xb3,0x8c,0xe2,0x00,0xd4,
        0x30,0xc2,0x15,0x38,0x2b,0xb0,0xa5,0x32,
        0xf8,0xbe,0x8a,0x1d,0x43,0x86,0xf3,0x6f,
        0xbc,0x9b,0xdd,0xcb,0x05,0x8a,0x09,0xf3,
        0x2f,0x39,0x17,0x3c,0x6f,0xb8,0x75,0x78,
        0x74,0x44,0x6f,0x2a,0x6a,0x23,0x25,0x0d,
        0x61,0x4f,0x35,0xbb,0x04,0x7b,0xbc,0x3d,
    };
    static constexpr float c2[4] = { -1.5104f, -0.4528f, 0.4528f, 1.5104f };
    static constexpr float b2[3] = { -0.9816f, 0.0f, 0.9816f };

    float tmp[512], recon[512];

    for (int j = 0; j < 512; j++) {
        int sign = ((tbq_signs_512[j >> 3] >> (j & 7)) & 1) ? -1 : 1;
        tmp[j] = x[j] * sign;
    }
    for (int len = 1; len < 512; len *= 2)
        for (int i = 0; i < 512; i += 2*len)
            for (int j = 0; j < len; j++) {
                float u = tmp[i+j], v = tmp[i+j+len];
                tmp[i+j] = u+v; tmp[i+j+len] = u-v;
            }

    // Global norm for TBQP3: QJL residual uses cross-block 512-WHT,
    // so all elements must be in the same normalized space
    float global_sq = 0.0f;
    for (int j = 0; j < 512; j++) global_sq += tmp[j] * tmp[j];
    float global_norm = sqrtf(global_sq / 512.0f);

    if (global_norm < 1e-10f) {
        for (int blk = 0; blk < 2; blk++) {
            y[blk].d = __float2half(0.0f);
            y[blk].d_qjl = __float2half(0.0f);
            for (int j = 0; j < QK_K/4; j++) y[blk].qs[j] = 0;
            for (int j = 0; j < QK_K/8; j++) y[blk].qjl[j] = 0;
        }
        return;
    }

    float inv_norm = 1.0f / global_norm;
    y[0].d = __float2half(global_norm);
    y[1].d = __float2half(global_norm);

    float res[512];
    float res_sq = 0.0f;

    for (int blk = 0; blk < 2; blk++) {
        float * blk_data = tmp + blk * 256;
        for (int j = 0; j < 256/4; j++) {
            uint8_t packed = 0;
            for (int k = 0; k < 4; k++) {
                float val = blk_data[j*4+k] * inv_norm;
                uint8_t idx = 3;
                for (int b = 0; b < 3; b++) { if (val < b2[b]) { idx = b; break; } }
                packed |= (idx & 0x3) << (k*2);
                recon[blk*256 + j*4+k] = c2[idx];
            }
            y[blk].qs[j] = packed;
        }
    }

    for (int j = 0; j < 512; j++) {
        res[j] = tmp[j] * inv_norm - recon[j];
        res_sq += res[j] * res[j];
    }
    float gamma = sqrtf(res_sq);
    float d_qjl = gamma * global_norm;
    y[0].d_qjl = __float2half(d_qjl);
    y[1].d_qjl = __float2half(d_qjl);

    for (int j = 0; j < 512; j++) {
        int sign = ((qjl_signs_512[j >> 3] >> (j & 7)) & 1) ? -1 : 1;
        res[j] *= sign;
    }
    for (int len = 1; len < 512; len *= 2)
        for (int i = 0; i < 512; i += 2*len)
            for (int j = 0; j < len; j++) {
                float u = res[i+j], v = res[i+j+len];
                res[i+j] = u+v; res[i+j+len] = u-v;
            }
    for (int blk = 0; blk < 2; blk++) {
        for (int j = 0; j < 256/8; j++) y[blk].qjl[j] = 0;
        for (int j = 0; j < 256; j++) {
            if (res[blk*256 + j] >= 0.0f) y[blk].qjl[j/8] |= (1 << (j%8));
        }
    }
}

// TurboQuant 4-bit D=512: 512-point WHT + per-block norm + 4-bit Lloyd-Max
static __device__ void quantize_f32_tbq4_0_block_512(const float * __restrict__ x, block_tbq4_0 * __restrict__ y) {
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
    static constexpr float tbq4_boundaries[15] = {
        -2.4008f,-1.8435f,-1.4371f,-1.0993f,-0.7996f,-0.5225f,-0.2583f,
         0.0f,
         0.2583f, 0.5225f, 0.7996f, 1.0993f, 1.4371f, 1.8435f, 2.4008f,
    };

    float tmp[512];

    // Apply signs on raw data
    for (int j = 0; j < 512; j++) {
        int sign = ((tbq_signs_512[j >> 3] >> (j & 7)) & 1) ? -1 : 1;
        tmp[j] = x[j] * sign;
    }

    // 512-point serial WHT (9 stages)
    for (int len = 1; len < 512; len *= 2)
        for (int i = 0; i < 512; i += 2*len)
            for (int j = 0; j < len; j++) {
                float u = tmp[i+j], v = tmp[i+j+len];
                tmp[i+j] = u+v; tmp[i+j+len] = u-v;
            }

    // Per-block norm: each 256-half gets its own norm after 512-WHT
    for (int blk = 0; blk < 2; blk++) {
        float * blk_data = tmp + blk * 256;

        float blk_sq = 0.0f;
        for (int j = 0; j < 256; j++) blk_sq += blk_data[j] * blk_data[j];
        float blk_norm = sqrtf(blk_sq / 256.0f);

        if (blk_norm < 1e-10f) {
            y[blk].d = __float2half(0.0f);
            for (int j = 0; j < QK_K/2; j++) y[blk].qs[j] = 0;
            continue;
        }

        float inv_norm = 1.0f / blk_norm;
        y[blk].d = __float2half(blk_norm);

        for (int j = 0; j < 256/2; j++) {
            uint8_t idx0 = 15, idx1 = 15;
            float val0 = blk_data[2*j]   * inv_norm;
            float val1 = blk_data[2*j+1] * inv_norm;
            for (int b = 0; b < 15; b++) { if (val0 < tbq4_boundaries[b]) { idx0 = b; break; } }
            for (int b = 0; b < 15; b++) { if (val1 < tbq4_boundaries[b]) { idx1 = b; break; } }
            y[blk].qs[j] = idx0 | (idx1 << 4);
        }
    }
}

// TurboQuant_prod 4-bit D=512: 512-point WHT + 3-bit Lloyd-Max + 1-bit QJL, global norm
static __device__ void quantize_f32_tbqp4_0_block_512(const float * __restrict__ x, block_tbqp4_0 * __restrict__ y) {
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
    static constexpr uint8_t qjl_signs_512[64] = {
        0x21,0x4e,0x75,0x8d,0x12,0xa1,0x04,0x88,
        0x6c,0x5d,0x2c,0xb3,0x8c,0xe2,0x00,0xd4,
        0x30,0xc2,0x15,0x38,0x2b,0xb0,0xa5,0x32,
        0xf8,0xbe,0x8a,0x1d,0x43,0x86,0xf3,0x6f,
        0xbc,0x9b,0xdd,0xcb,0x05,0x8a,0x09,0xf3,
        0x2f,0x39,0x17,0x3c,0x6f,0xb8,0x75,0x78,
        0x74,0x44,0x6f,0x2a,0x6a,0x23,0x25,0x0d,
        0x61,0x4f,0x35,0xbb,0x04,0x7b,0xbc,0x3d,
    };
    static constexpr float c3[8] = {
        -2.1520f,-1.3440f,-0.7560f,-0.2451f, 0.2451f, 0.7560f, 1.3440f, 2.1520f,
    };
    static constexpr float b3[7] = {
        -1.7480f, -1.0500f, -0.5006f, 0.0f, 0.5006f, 1.0500f, 1.7480f,
    };

    float tmp[512], recon[512];

    for (int j = 0; j < 512; j++) {
        int sign = ((tbq_signs_512[j >> 3] >> (j & 7)) & 1) ? -1 : 1;
        tmp[j] = x[j] * sign;
    }
    for (int len = 1; len < 512; len *= 2)
        for (int i = 0; i < 512; i += 2*len)
            for (int j = 0; j < len; j++) {
                float u = tmp[i+j], v = tmp[i+j+len];
                tmp[i+j] = u+v; tmp[i+j+len] = u-v;
            }

    // Global norm for TBQP4: QJL residual uses cross-block 512-WHT
    float global_sq = 0.0f;
    for (int j = 0; j < 512; j++) global_sq += tmp[j] * tmp[j];
    float global_norm = sqrtf(global_sq / 512.0f);

    if (global_norm < 1e-10f) {
        for (int blk = 0; blk < 2; blk++) {
            y[blk].d = __float2half(0.0f);
            y[blk].d_qjl = __float2half(0.0f);
            for (int j = 0; j < QK_K*3/8; j++) y[blk].qs[j] = 0;
            for (int j = 0; j < QK_K/8; j++) y[blk].qjl[j] = 0;
        }
        return;
    }

    float inv_norm = 1.0f / global_norm;
    y[0].d = __float2half(global_norm);
    y[1].d = __float2half(global_norm);

    float res[512];
    float res_sq = 0.0f;

    for (int blk = 0; blk < 2; blk++) {
        float * blk_data = tmp + blk * 256;
        int bit_pos = 0;
        for (int j = 0; j < QK_K*3/8; j++) y[blk].qs[j] = 0;
        for (int j = 0; j < 256; j++) {
            float val = blk_data[j] * inv_norm;
            uint8_t idx = 7;
            for (int b = 0; b < 7; b++) { if (val < b3[b]) { idx = b; break; } }
            int byte_idx = bit_pos / 8, bit_off = bit_pos % 8;
            y[blk].qs[byte_idx] |= (uint8_t)((idx & 0x7) << bit_off);
            if (bit_off > 5) y[blk].qs[byte_idx + 1] |= (uint8_t)((idx & 0x7) >> (8 - bit_off));
            recon[blk*256 + j] = c3[idx];
            bit_pos += 3;
        }
    }

    for (int j = 0; j < 512; j++) {
        res[j] = tmp[j] * inv_norm - recon[j];
        res_sq += res[j] * res[j];
    }
    float gamma = sqrtf(res_sq);
    float d_qjl = gamma * global_norm;
    y[0].d_qjl = __float2half(d_qjl);
    y[1].d_qjl = __float2half(d_qjl);

    for (int j = 0; j < 512; j++) {
        int sign = ((qjl_signs_512[j >> 3] >> (j & 7)) & 1) ? -1 : 1;
        res[j] *= sign;
    }
    for (int len = 1; len < 512; len *= 2)
        for (int i = 0; i < 512; i += 2*len)
            for (int j = 0; j < len; j++) {
                float u = res[i+j], v = res[i+j+len];
                res[i+j] = u+v; res[i+j+len] = u-v;
            }
    for (int blk = 0; blk < 2; blk++) {
        for (int j = 0; j < 256/8; j++) y[blk].qjl[j] = 0;
        for (int j = 0; j < 256; j++) {
            if (res[blk*256 + j] >= 0.0f) y[blk].qjl[j/8] |= (1 << (j%8));
        }
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

// ============================================================
// TurboQuant 128-element (head_dim=128) quantization functions
// ============================================================

// TBQ3_1: 128-element WHT + 3-bit Lloyd-Max
static __device__ void quantize_f32_tbq3_1_block(const float * __restrict__ x, block_tbq3_1 * __restrict__ y) {
    static constexpr uint8_t tbq_signs[16] = {
        0xa7,0x3b,0x91,0xf4,0x6d,0xc2,0x58,0x0e,
        0xb3,0x7f,0x24,0xd6,0x89,0x45,0xea,0x1c,
    };
    static constexpr float tbq3_boundaries[7] = {
        -1.7480f, -1.0500f, -0.5006f, 0.0f, 0.5006f, 1.0500f, 1.7480f,
    };

    float tmp[TBQ_K128];
    float sum_sq = 0.0f;
    for (int j = 0; j < TBQ_K128; j++) sum_sq += x[j] * x[j];
    float norm = sqrtf(sum_sq);
    y->d = __float2half(norm);

    if (norm < 1e-10f) {
        for (int j = 0; j < TBQ_K128*3/8; j++) y->qs[j] = 0;
        return;
    }

    float inv_norm = 1.0f / norm;
    for (int j = 0; j < TBQ_K128; j++) {
        int sign = ((tbq_signs[j >> 3] >> (j & 7)) & 1) ? -1 : 1;
        tmp[j] = x[j] * inv_norm * sign;
    }

    // Serial WHT (128 elements, 7 stages)
    for (int len = 1; len < TBQ_K128; len *= 2)
        for (int i = 0; i < TBQ_K128; i += 2*len)
            for (int j = 0; j < len; j++) {
                float u = tmp[i+j], v = tmp[i+j+len];
                tmp[i+j] = u+v; tmp[i+j+len] = u-v;
            }

    // 3-bit quantization + packing
    int bit_pos = 0;
    for (int j = 0; j < TBQ_K128*3/8; j++) y->qs[j] = 0;
    for (int j = 0; j < TBQ_K128; j++) {
        uint8_t idx = 7;
        for (int b = 0; b < 7; b++) { if (tmp[j] < tbq3_boundaries[b]) { idx = b; break; } }
        int byte_idx = bit_pos / 8;
        int bit_off = bit_pos % 8;
        y->qs[byte_idx] |= (uint8_t)((idx & 0x7) << bit_off);
        if (bit_off > 5) y->qs[byte_idx + 1] |= (uint8_t)((idx & 0x7) >> (8 - bit_off));
        bit_pos += 3;
    }
}

// TBQ4_1: 128-element WHT + 4-bit Lloyd-Max
static __device__ void quantize_f32_tbq4_1_block(const float * __restrict__ x, block_tbq4_1 * __restrict__ y) {
    static constexpr uint8_t tbq_signs[16] = {
        0xa7,0x3b,0x91,0xf4,0x6d,0xc2,0x58,0x0e,
        0xb3,0x7f,0x24,0xd6,0x89,0x45,0xea,0x1c,
    };
    static constexpr float tbq4_boundaries[15] = {
        -2.4008f,-1.8435f,-1.4371f,-1.0993f,-0.7996f,-0.5225f,-0.2583f,
         0.0f,
         0.2583f, 0.5225f, 0.7996f, 1.0993f, 1.4371f, 1.8435f, 2.4008f,
    };

    float tmp[TBQ_K128];
    float sum_sq = 0.0f;
    for (int j = 0; j < TBQ_K128; j++) sum_sq += x[j] * x[j];
    float norm = sqrtf(sum_sq);
    y->d = __float2half(norm);

    if (norm < 1e-10f) {
        for (int j = 0; j < TBQ_K128/2; j++) y->qs[j] = 0;
        return;
    }

    float inv_norm = 1.0f / norm;
    for (int j = 0; j < TBQ_K128; j++) {
        int sign = ((tbq_signs[j >> 3] >> (j & 7)) & 1) ? -1 : 1;
        tmp[j] = x[j] * inv_norm * sign;
    }

    // Serial WHT (128 elements, 7 stages)
    for (int len = 1; len < TBQ_K128; len *= 2)
        for (int i = 0; i < TBQ_K128; i += 2*len)
            for (int j = 0; j < len; j++) {
                float u = tmp[i+j], v = tmp[i+j+len];
                tmp[i+j] = u+v; tmp[i+j+len] = u-v;
            }

    // 4-bit quantization + packing
    for (int j = 0; j < TBQ_K128/2; j++) {
        uint8_t idx0 = 15, idx1 = 15;
        for (int b = 0; b < 15; b++) { if (tmp[2*j]   < tbq4_boundaries[b]) { idx0 = b; break; } }
        for (int b = 0; b < 15; b++) { if (tmp[2*j+1] < tbq4_boundaries[b]) { idx1 = b; break; } }
        y->qs[j] = idx0 | (idx1 << 4);
    }
}

// TBQP3_1: 128-element 2-bit Lloyd-Max + 1-bit QJL
static __device__ void quantize_f32_tbqp3_1_block(const float * __restrict__ x, block_tbqp3_1 * __restrict__ y) {
    static constexpr uint8_t tbq_signs[16] = {
        0xa7,0x3b,0x91,0xf4,0x6d,0xc2,0x58,0x0e,
        0xb3,0x7f,0x24,0xd6,0x89,0x45,0xea,0x1c,
    };
    static constexpr float c2[4] = { -1.5104f, -0.4528f, 0.4528f, 1.5104f };
    static constexpr float b2[3] = { -0.9816f, 0.0f, 0.9816f };

    float tmp[TBQ_K128], recon[TBQ_K128];

    float sum_sq = 0.0f;
    for (int j = 0; j < TBQ_K128; j++) sum_sq += x[j] * x[j];
    float norm = sqrtf(sum_sq);
    y->d = __float2half(norm);
    if (norm < 1e-10f) {
        for (int j = 0; j < TBQ_K128/4; j++) y->qs[j] = 0;
        for (int j = 0; j < TBQ_K128/8; j++) y->qjl[j] = 0;
        y->d_qjl = __float2half(0.0f);
        return;
    }
    float inv_norm = 1.0f / norm;
    for (int j = 0; j < TBQ_K128; j++) {
        int sign = ((tbq_signs[j >> 3] >> (j & 7)) & 1) ? -1 : 1;
        tmp[j] = x[j] * inv_norm * sign;
    }
    for (int len = 1; len < TBQ_K128; len *= 2)
        for (int i = 0; i < TBQ_K128; i += 2*len)
            for (int j = 0; j < len; j++) {
                float u = tmp[i+j], v = tmp[i+j+len];
                tmp[i+j] = u+v; tmp[i+j+len] = u-v;
            }

    // 2-bit Lloyd-Max quantization
    for (int j = 0; j < TBQ_K128/4; j++) {
        uint8_t packed = 0;
        for (int k = 0; k < 4; k++) {
            uint8_t idx = 3;
            for (int b = 0; b < 3; b++) { if (tmp[j*4+k] < b2[b]) { idx = b; break; } }
            packed |= (idx & 0x3) << (k*2);
            recon[j*4+k] = c2[idx];
        }
        y->qs[j] = packed;
    }

    // Residual + Direct Sign (no SRHT — lower variance than QJL for d≤128)
    float res[TBQ_K128], res_abs_sum = 0.0f;
    for (int j = 0; j < TBQ_K128; j++) {
        res[j] = tmp[j] - recon[j];
        res_abs_sum += fabsf(res[j]);
    }
    float gamma = res_abs_sum / TBQ_K128; // mean(|residual|)
    y->d_qjl = __float2half(gamma * norm);

    // Store sign(residual) directly — no random rotation needed
    for (int j = 0; j < TBQ_K128/8; j++) y->qjl[j] = 0;
    for (int j = 0; j < TBQ_K128; j++) {
        if (res[j] >= 0.0f) y->qjl[j/8] |= (1 << (j%8));
    }
}

// TBQP4_1: 128-element 3-bit Lloyd-Max + 1-bit QJL
static __device__ void quantize_f32_tbqp4_1_block(const float * __restrict__ x, block_tbqp4_1 * __restrict__ y) {
    static constexpr uint8_t tbq_signs[16] = {
        0xa7,0x3b,0x91,0xf4,0x6d,0xc2,0x58,0x0e,
        0xb3,0x7f,0x24,0xd6,0x89,0x45,0xea,0x1c,
    };
    static constexpr float c3[8] = {
        -2.1520f,-1.3440f,-0.7560f,-0.2451f, 0.2451f, 0.7560f, 1.3440f, 2.1520f,
    };
    static constexpr float b3[7] = {
        -1.7480f, -1.0500f, -0.5006f, 0.0f, 0.5006f, 1.0500f, 1.7480f,
    };

    float tmp[TBQ_K128], recon[TBQ_K128];
    float sum_sq = 0.0f;
    for (int j = 0; j < TBQ_K128; j++) sum_sq += x[j] * x[j];
    float norm = sqrtf(sum_sq);
    y->d = __float2half(norm);
    if (norm < 1e-10f) {
        for (int j = 0; j < TBQ_K128*3/8; j++) y->qs[j] = 0;
        for (int j = 0; j < TBQ_K128/8; j++) y->qjl[j] = 0;
        y->d_qjl = __float2half(0.0f);
        return;
    }
    float inv_norm = 1.0f / norm;
    for (int j = 0; j < TBQ_K128; j++) {
        int sign = ((tbq_signs[j >> 3] >> (j & 7)) & 1) ? -1 : 1;
        tmp[j] = x[j] * inv_norm * sign;
    }
    for (int len = 1; len < TBQ_K128; len *= 2)
        for (int i = 0; i < TBQ_K128; i += 2*len)
            for (int j = 0; j < len; j++) {
                float u = tmp[i+j], v = tmp[i+j+len];
                tmp[i+j] = u+v; tmp[i+j+len] = u-v;
            }

    // 3-bit Lloyd-Max + packing
    int bit_pos = 0;
    for (int j = 0; j < TBQ_K128*3/8; j++) y->qs[j] = 0;
    for (int j = 0; j < TBQ_K128; j++) {
        uint8_t idx = 7;
        for (int b = 0; b < 7; b++) { if (tmp[j] < b3[b]) { idx = b; break; } }
        int byte_idx = bit_pos / 8, bit_off = bit_pos % 8;
        y->qs[byte_idx] |= (uint8_t)((idx & 0x7) << bit_off);
        if (bit_off > 5) y->qs[byte_idx + 1] |= (uint8_t)((idx & 0x7) >> (8 - bit_off));
        recon[j] = c3[idx];
        bit_pos += 3;
    }

    // Residual + Direct Sign (no SRHT — lower variance than QJL for d≤128)
    float res[TBQ_K128], res_abs_sum = 0.0f;
    for (int j = 0; j < TBQ_K128; j++) { res[j] = tmp[j] - recon[j]; res_abs_sum += fabsf(res[j]); }
    float gamma = res_abs_sum / TBQ_K128;
    y->d_qjl = __float2half(gamma * norm);
    for (int j = 0; j < TBQ_K128/8; j++) y->qjl[j] = 0;
    for (int j = 0; j < TBQ_K128; j++) {
        if (res[j] >= 0.0f) y->qjl[j/8] |= (1 << (j%8));
    }
}

// ============================================================
// TurboQuant 64-element (head_dim=64) quantization functions
// ============================================================

static __device__ void quantize_f32_tbq3_2_block(const float * __restrict__ x, block_tbq3_2 * __restrict__ y) {
    static constexpr uint8_t tbq_signs[8] = { 0xa7,0x3b,0x91,0xf4,0x6d,0xc2,0x58,0x0e };
    static constexpr float tbq3_boundaries[7] = { -1.7480f,-1.0500f,-0.5006f,0.0f,0.5006f,1.0500f,1.7480f };
    float tmp[TBQ_K64];
    float sum_sq = 0.0f;
    for (int j = 0; j < TBQ_K64; j++) sum_sq += x[j] * x[j];
    float norm = sqrtf(sum_sq);
    y->d = __float2half(norm);
    if (norm < 1e-10f) { for (int j = 0; j < TBQ_K64*3/8; j++) y->qs[j] = 0; return; }
    float inv_norm = 1.0f / norm;
    for (int j = 0; j < TBQ_K64; j++) {
        int sign = ((tbq_signs[j >> 3] >> (j & 7)) & 1) ? -1 : 1;
        tmp[j] = x[j] * inv_norm * sign;
    }
    for (int len = 1; len < TBQ_K64; len *= 2)
        for (int i = 0; i < TBQ_K64; i += 2*len)
            for (int j = 0; j < len; j++) { float u = tmp[i+j], v = tmp[i+j+len]; tmp[i+j] = u+v; tmp[i+j+len] = u-v; }
    int bit_pos = 0;
    for (int j = 0; j < TBQ_K64*3/8; j++) y->qs[j] = 0;
    for (int j = 0; j < TBQ_K64; j++) {
        uint8_t idx = 7;
        for (int b = 0; b < 7; b++) { if (tmp[j] < tbq3_boundaries[b]) { idx = b; break; } }
        int byte_idx = bit_pos / 8, bit_off = bit_pos % 8;
        y->qs[byte_idx] |= (uint8_t)((idx & 0x7) << bit_off);
        if (bit_off > 5) y->qs[byte_idx + 1] |= (uint8_t)((idx & 0x7) >> (8 - bit_off));
        bit_pos += 3;
    }
}

static __device__ void quantize_f32_tbq4_2_block(const float * __restrict__ x, block_tbq4_2 * __restrict__ y) {
    static constexpr uint8_t tbq_signs[8] = { 0xa7,0x3b,0x91,0xf4,0x6d,0xc2,0x58,0x0e };
    static constexpr float tbq4_boundaries[15] = { -2.4008f,-1.8435f,-1.4371f,-1.0993f,-0.7996f,-0.5225f,-0.2583f,0.0f,0.2583f,0.5225f,0.7996f,1.0993f,1.4371f,1.8435f,2.4008f };
    float tmp[TBQ_K64];
    float sum_sq = 0.0f;
    for (int j = 0; j < TBQ_K64; j++) sum_sq += x[j] * x[j];
    float norm = sqrtf(sum_sq);
    y->d = __float2half(norm);
    if (norm < 1e-10f) { for (int j = 0; j < TBQ_K64/2; j++) y->qs[j] = 0; return; }
    float inv_norm = 1.0f / norm;
    for (int j = 0; j < TBQ_K64; j++) {
        int sign = ((tbq_signs[j >> 3] >> (j & 7)) & 1) ? -1 : 1;
        tmp[j] = x[j] * inv_norm * sign;
    }
    for (int len = 1; len < TBQ_K64; len *= 2)
        for (int i = 0; i < TBQ_K64; i += 2*len)
            for (int j = 0; j < len; j++) { float u = tmp[i+j], v = tmp[i+j+len]; tmp[i+j] = u+v; tmp[i+j+len] = u-v; }
    for (int j = 0; j < TBQ_K64/2; j++) {
        uint8_t idx0 = 15, idx1 = 15;
        for (int b = 0; b < 15; b++) { if (tmp[2*j]   < tbq4_boundaries[b]) { idx0 = b; break; } }
        for (int b = 0; b < 15; b++) { if (tmp[2*j+1] < tbq4_boundaries[b]) { idx1 = b; break; } }
        y->qs[j] = idx0 | (idx1 << 4);
    }
}

static __device__ void quantize_f32_tbqp3_2_block(const float * __restrict__ x, block_tbqp3_2 * __restrict__ y) {
    static constexpr uint8_t tbq_signs[8] = { 0xa7,0x3b,0x91,0xf4,0x6d,0xc2,0x58,0x0e };
    static constexpr float c2[4] = { -1.5104f,-0.4528f,0.4528f,1.5104f };
    static constexpr float b2[3] = { -0.9816f,0.0f,0.9816f };
    float tmp[TBQ_K64], recon[TBQ_K64];
    float sum_sq = 0.0f;
    for (int j = 0; j < TBQ_K64; j++) sum_sq += x[j] * x[j];
    float norm = sqrtf(sum_sq);
    y->d = __float2half(norm);
    if (norm < 1e-10f) { for (int j = 0; j < TBQ_K64/4; j++) y->qs[j] = 0; for (int j = 0; j < TBQ_K64/8; j++) y->qjl[j] = 0; y->d_qjl = __float2half(0.0f); return; }
    float inv_norm = 1.0f / norm;
    for (int j = 0; j < TBQ_K64; j++) { int sign = ((tbq_signs[j>>3]>>(j&7))&1)?-1:1; tmp[j] = x[j]*inv_norm*sign; }
    for (int len = 1; len < TBQ_K64; len *= 2) for (int i = 0; i < TBQ_K64; i += 2*len) for (int j = 0; j < len; j++) { float u = tmp[i+j], v = tmp[i+j+len]; tmp[i+j] = u+v; tmp[i+j+len] = u-v; }
    for (int j = 0; j < TBQ_K64/4; j++) { uint8_t packed = 0; for (int k = 0; k < 4; k++) { uint8_t idx = 3; for (int b = 0; b < 3; b++) { if (tmp[j*4+k] < b2[b]) { idx = b; break; } } packed |= (idx&0x3)<<(k*2); recon[j*4+k] = c2[idx]; } y->qs[j] = packed; }
    float res[TBQ_K64], res_abs = 0.0f;
    for (int j = 0; j < TBQ_K64; j++) { res[j] = tmp[j]-recon[j]; res_abs += fabsf(res[j]); }
    y->d_qjl = __float2half((res_abs/TBQ_K64)*norm);
    for (int j = 0; j < TBQ_K64/8; j++) y->qjl[j] = 0;
    for (int j = 0; j < TBQ_K64; j++) { if (res[j] >= 0.0f) y->qjl[j/8] |= (1<<(j%8)); }
}

static __device__ void quantize_f32_tbqp4_2_block(const float * __restrict__ x, block_tbqp4_2 * __restrict__ y) {
    static constexpr uint8_t tbq_signs[8] = { 0xa7,0x3b,0x91,0xf4,0x6d,0xc2,0x58,0x0e };
    static constexpr float c3[8] = { -2.1520f,-1.3440f,-0.7560f,-0.2451f,0.2451f,0.7560f,1.3440f,2.1520f };
    static constexpr float b3[7] = { -1.7480f,-1.0500f,-0.5006f,0.0f,0.5006f,1.0500f,1.7480f };
    float tmp[TBQ_K64], recon[TBQ_K64];
    float sum_sq = 0.0f;
    for (int j = 0; j < TBQ_K64; j++) sum_sq += x[j] * x[j];
    float norm = sqrtf(sum_sq);
    y->d = __float2half(norm);
    if (norm < 1e-10f) { for (int j = 0; j < TBQ_K64*3/8; j++) y->qs[j] = 0; for (int j = 0; j < TBQ_K64/8; j++) y->qjl[j] = 0; y->d_qjl = __float2half(0.0f); return; }
    float inv_norm = 1.0f / norm;
    for (int j = 0; j < TBQ_K64; j++) { int sign = ((tbq_signs[j>>3]>>(j&7))&1)?-1:1; tmp[j] = x[j]*inv_norm*sign; }
    for (int len = 1; len < TBQ_K64; len *= 2) for (int i = 0; i < TBQ_K64; i += 2*len) for (int j = 0; j < len; j++) { float u = tmp[i+j], v = tmp[i+j+len]; tmp[i+j] = u+v; tmp[i+j+len] = u-v; }
    int bit_pos = 0;
    for (int j = 0; j < TBQ_K64*3/8; j++) y->qs[j] = 0;
    for (int j = 0; j < TBQ_K64; j++) { uint8_t idx = 7; for (int b = 0; b < 7; b++) { if (tmp[j] < b3[b]) { idx = b; break; } } int byte_idx = bit_pos/8, bit_off = bit_pos%8; y->qs[byte_idx] |= (uint8_t)((idx&0x7)<<bit_off); if (bit_off > 5) y->qs[byte_idx+1] |= (uint8_t)((idx&0x7)>>(8-bit_off)); recon[j] = c3[idx]; bit_pos += 3; }
    float res[TBQ_K64], res_abs = 0.0f;
    for (int j = 0; j < TBQ_K64; j++) { res[j] = tmp[j]-recon[j]; res_abs += fabsf(res[j]); }
    y->d_qjl = __float2half((res_abs/TBQ_K64)*norm);
    for (int j = 0; j < TBQ_K64/8; j++) y->qjl[j] = 0;
    for (int j = 0; j < TBQ_K64; j++) { if (res[j] >= 0.0f) y->qjl[j/8] |= (1<<(j%8)); }
}

// ============================================================
// TurboQuant 576-element (head_dim=576) quantization functions
// Split: 256+256+64, each sub-block WHT'd independently
// Sub-blocks 1,2 (256): reuse _0 signs/logic
// Sub-block 3 (64): reuse _2 signs/logic
// ============================================================

static __device__ void quantize_f32_tbq3_4_block(const float * __restrict__ x, block_tbq3_4 * __restrict__ y) {
    static constexpr uint8_t signs_256[32] = {
        0xa7,0x3b,0x91,0xf4,0x6d,0xc2,0x58,0x0e,0xb3,0x7f,0x24,0xd6,0x89,0x45,0xea,0x1c,
        0x63,0xaf,0xd8,0x52,0x97,0x0b,0xe1,0x3d,0x76,0xc4,0x19,0xfe,0x4a,0x85,0x2c,0xdb,
    };
    static constexpr float tbq3_boundaries[7] = { -1.7480f,-1.0500f,-0.5006f,0.0f,0.5006f,1.0500f,1.7480f };

    // Helper: quantize a sub-block with 3-bit Lloyd-Max
    auto quant_sub = [&](const float * src, int dim, const uint8_t * signs, ggml_half * d_out, uint8_t * qs_out) {
        float tmp[QK_K]; // max sub-block size is 256
        float sum_sq = 0.0f;
        for (int j = 0; j < dim; j++) sum_sq += src[j] * src[j];
        float norm = sqrtf(sum_sq);
        *d_out = __float2half(norm);
        if (norm < 1e-10f) { for (int j = 0; j < dim*3/8; j++) qs_out[j] = 0; return; }
        float inv_norm = 1.0f / norm;
        for (int j = 0; j < dim; j++) {
            int sign = ((signs[j>>3]>>(j&7))&1) ? -1 : 1;
            tmp[j] = src[j] * inv_norm * sign;
        }
        for (int len = 1; len < dim; len *= 2)
            for (int i = 0; i < dim; i += 2*len)
                for (int j = 0; j < len; j++) { float u = tmp[i+j], v = tmp[i+j+len]; tmp[i+j] = u+v; tmp[i+j+len] = u-v; }
        int bit_pos = 0;
        for (int j = 0; j < dim*3/8; j++) qs_out[j] = 0;
        for (int j = 0; j < dim; j++) {
            uint8_t idx = 7;
            for (int b = 0; b < 7; b++) { if (tmp[j] < tbq3_boundaries[b]) { idx = b; break; } }
            int byte_idx = bit_pos/8, bit_off = bit_pos%8;
            qs_out[byte_idx] |= (uint8_t)((idx&0x7)<<bit_off);
            if (bit_off > 5) qs_out[byte_idx+1] |= (uint8_t)((idx&0x7)>>(8-bit_off));
            bit_pos += 3;
        }
    };

    quant_sub(x,       QK_K,    signs_256, &y->d1, y->qs1);
    quant_sub(x + 256, QK_K,    signs_256, &y->d2, y->qs2);
    for (int j = 0; j < TBQ_K64; j++) y->rope[j] = __float2half(x[512 + j]);
}

static __device__ void quantize_f32_tbq4_4_block(const float * __restrict__ x, block_tbq4_4 * __restrict__ y) {
    static constexpr uint8_t signs_256[32] = {
        0xa7,0x3b,0x91,0xf4,0x6d,0xc2,0x58,0x0e,0xb3,0x7f,0x24,0xd6,0x89,0x45,0xea,0x1c,
        0x63,0xaf,0xd8,0x52,0x97,0x0b,0xe1,0x3d,0x76,0xc4,0x19,0xfe,0x4a,0x85,0x2c,0xdb,
    };
    static constexpr float tbq4_boundaries[15] = {
        -2.4008f,-1.8435f,-1.4371f,-1.0993f,-0.7996f,-0.5225f,-0.2583f,0.0f,
        0.2583f,0.5225f,0.7996f,1.0993f,1.4371f,1.8435f,2.4008f,
    };

    auto quant_sub = [&](const float * src, int dim, const uint8_t * signs, ggml_half * d_out, uint8_t * qs_out) {
        float tmp[QK_K];
        float sum_sq = 0.0f;
        for (int j = 0; j < dim; j++) sum_sq += src[j] * src[j];
        float norm = sqrtf(sum_sq);
        *d_out = __float2half(norm);
        if (norm < 1e-10f) { for (int j = 0; j < dim/2; j++) qs_out[j] = 0; return; }
        float inv_norm = 1.0f / norm;
        for (int j = 0; j < dim; j++) {
            int sign = ((signs[j>>3]>>(j&7))&1) ? -1 : 1;
            tmp[j] = src[j] * inv_norm * sign;
        }
        for (int len = 1; len < dim; len *= 2)
            for (int i = 0; i < dim; i += 2*len)
                for (int j = 0; j < len; j++) { float u = tmp[i+j], v = tmp[i+j+len]; tmp[i+j] = u+v; tmp[i+j+len] = u-v; }
        for (int j = 0; j < dim/2; j++) {
            uint8_t idx0 = 15, idx1 = 15;
            for (int b = 0; b < 15; b++) { if (tmp[2*j]   < tbq4_boundaries[b]) { idx0 = b; break; } }
            for (int b = 0; b < 15; b++) { if (tmp[2*j+1] < tbq4_boundaries[b]) { idx1 = b; break; } }
            qs_out[j] = idx0 | (idx1 << 4);
        }
    };

    quant_sub(x,       QK_K,    signs_256, &y->d1, y->qs1);
    quant_sub(x + 256, QK_K,    signs_256, &y->d2, y->qs2);
    // Sub-block 3: f16 passthrough (rope)
    for (int j = 0; j < TBQ_K64; j++) y->rope[j] = __float2half(x[512 + j]);
}

static __device__ void quantize_f32_tbqp3_4_block(const float * __restrict__ x, block_tbqp3_4 * __restrict__ y) {
    static constexpr uint8_t signs_256[32] = {
        0xa7,0x3b,0x91,0xf4,0x6d,0xc2,0x58,0x0e,0xb3,0x7f,0x24,0xd6,0x89,0x45,0xea,0x1c,
        0x63,0xaf,0xd8,0x52,0x97,0x0b,0xe1,0x3d,0x76,0xc4,0x19,0xfe,0x4a,0x85,0x2c,0xdb,
    };
    static constexpr uint8_t qjl_signs_256[32] = {
        0xd3,0x4e,0xa8,0x17,0x9c,0x5b,0xe6,0x31,0x72,0xb9,0x0d,0xf5,0x43,0x8a,0x6e,0xc7,
        0x58,0x2f,0x94,0xe1,0xb6,0x3d,0x0a,0x7c,0xc5,0x61,0xd8,0x4f,0xa3,0x97,0x1e,0x85,
    };
    static constexpr float c2[4] = { -1.5104f,-0.4528f,0.4528f,1.5104f };
    static constexpr float b2[3] = { -0.9816f,0.0f,0.9816f };

    // Sub-block 1,2: 256-element, 2-bit Lloyd-Max + QJL SRHT correction
    auto quant_sub_qjl = [&](const float * src, const uint8_t * signs, const uint8_t * qjl_s,
                              ggml_half * d_out, ggml_half * d_qjl_out, uint8_t * qs_out, uint8_t * qjl_out) {
        float tmp[QK_K], recon[QK_K];
        float sum_sq = 0.0f;
        for (int j = 0; j < QK_K; j++) sum_sq += src[j] * src[j];
        float norm = sqrtf(sum_sq);
        *d_out = __float2half(norm);
        if (norm < 1e-10f) {
            for (int j = 0; j < QK_K/4; j++) qs_out[j] = 0;
            for (int j = 0; j < QK_K/8; j++) qjl_out[j] = 0;
            *d_qjl_out = __float2half(0.0f); return;
        }
        float inv_norm = 1.0f / norm;
        for (int j = 0; j < QK_K; j++) {
            int sign = ((signs[j>>3]>>(j&7))&1) ? -1 : 1;
            tmp[j] = src[j] * inv_norm * sign;
        }
        for (int len = 1; len < QK_K; len *= 2)
            for (int i = 0; i < QK_K; i += 2*len)
                for (int j = 0; j < len; j++) { float u = tmp[i+j], v = tmp[i+j+len]; tmp[i+j] = u+v; tmp[i+j+len] = u-v; }
        for (int j = 0; j < QK_K/4; j++) {
            uint8_t packed = 0;
            for (int k = 0; k < 4; k++) {
                uint8_t idx = 3;
                for (int b = 0; b < 3; b++) { if (tmp[j*4+k] < b2[b]) { idx = b; break; } }
                packed |= (idx&0x3)<<(k*2);
                recon[j*4+k] = c2[idx];
            }
            qs_out[j] = packed;
        }
        float res[QK_K], res_sq = 0.0f;
        for (int j = 0; j < QK_K; j++) { res[j] = tmp[j]-recon[j]; res_sq += res[j]*res[j]; }
        float gamma = sqrtf(res_sq);
        *d_qjl_out = __float2half(gamma * norm);
        for (int j = 0; j < QK_K; j++) { int sign = ((qjl_s[j>>3]>>(j&7))&1)?-1:1; res[j] *= sign; }
        for (int len = 1; len < QK_K; len *= 2)
            for (int i = 0; i < QK_K; i += 2*len)
                for (int j = 0; j < len; j++) { float u = res[i+j], v = res[i+j+len]; res[i+j] = u+v; res[i+j+len] = u-v; }
        for (int j = 0; j < QK_K/8; j++) qjl_out[j] = 0;
        for (int j = 0; j < QK_K; j++) { if (res[j] >= 0.0f) qjl_out[j/8] |= (1<<(j%8)); }
    };

    quant_sub_qjl(x,       signs_256, qjl_signs_256, &y->d1, &y->d1_qjl, y->qs1, y->qjl1);
    quant_sub_qjl(x + 256, signs_256, qjl_signs_256, &y->d2, &y->d2_qjl, y->qs2, y->qjl2);
    // Sub-block 3: f16 passthrough (rope)
    for (int j = 0; j < TBQ_K64; j++) y->rope[j] = __float2half(x[512 + j]);
}

static __device__ void quantize_f32_tbqp4_4_block(const float * __restrict__ x, block_tbqp4_4 * __restrict__ y) {
    static constexpr uint8_t signs_256[32] = {
        0xa7,0x3b,0x91,0xf4,0x6d,0xc2,0x58,0x0e,0xb3,0x7f,0x24,0xd6,0x89,0x45,0xea,0x1c,
        0x63,0xaf,0xd8,0x52,0x97,0x0b,0xe1,0x3d,0x76,0xc4,0x19,0xfe,0x4a,0x85,0x2c,0xdb,
    };
    static constexpr uint8_t qjl_signs_256[32] = {
        0xd3,0x4e,0xa8,0x17,0x9c,0x5b,0xe6,0x31,0x72,0xb9,0x0d,0xf5,0x43,0x8a,0x6e,0xc7,
        0x58,0x2f,0x94,0xe1,0xb6,0x3d,0x0a,0x7c,0xc5,0x61,0xd8,0x4f,0xa3,0x97,0x1e,0x85,
    };
    static constexpr float c3[8] = { -2.1520f,-1.3440f,-0.7560f,-0.2451f,0.2451f,0.7560f,1.3440f,2.1520f };
    static constexpr float b3[7] = { -1.7480f,-1.0500f,-0.5006f,0.0f,0.5006f,1.0500f,1.7480f };

    // Sub-block 1,2: 256-element, 3-bit Lloyd-Max + QJL SRHT correction
    auto quant_sub_qjl = [&](const float * src, const uint8_t * signs, const uint8_t * qjl_s,
                              ggml_half * d_out, ggml_half * d_qjl_out, uint8_t * qs_out, uint8_t * qjl_out) {
        float tmp[QK_K], recon[QK_K];
        float sum_sq = 0.0f;
        for (int j = 0; j < QK_K; j++) sum_sq += src[j] * src[j];
        float norm = sqrtf(sum_sq);
        *d_out = __float2half(norm);
        if (norm < 1e-10f) {
            for (int j = 0; j < QK_K*3/8; j++) qs_out[j] = 0;
            for (int j = 0; j < QK_K/8; j++) qjl_out[j] = 0;
            *d_qjl_out = __float2half(0.0f); return;
        }
        float inv_norm = 1.0f / norm;
        for (int j = 0; j < QK_K; j++) {
            int sign = ((signs[j>>3]>>(j&7))&1) ? -1 : 1;
            tmp[j] = src[j] * inv_norm * sign;
        }
        for (int len = 1; len < QK_K; len *= 2)
            for (int i = 0; i < QK_K; i += 2*len)
                for (int j = 0; j < len; j++) { float u = tmp[i+j], v = tmp[i+j+len]; tmp[i+j] = u+v; tmp[i+j+len] = u-v; }
        int bit_pos = 0;
        for (int j = 0; j < QK_K*3/8; j++) qs_out[j] = 0;
        for (int j = 0; j < QK_K; j++) {
            uint8_t idx = 7;
            for (int b = 0; b < 7; b++) { if (tmp[j] < b3[b]) { idx = b; break; } }
            int byte_idx = bit_pos/8, bit_off = bit_pos%8;
            qs_out[byte_idx] |= (uint8_t)((idx&0x7)<<bit_off);
            if (bit_off > 5) qs_out[byte_idx+1] |= (uint8_t)((idx&0x7)>>(8-bit_off));
            recon[j] = c3[idx]; bit_pos += 3;
        }
        float res[QK_K], res_sq = 0.0f;
        for (int j = 0; j < QK_K; j++) { res[j] = tmp[j]-recon[j]; res_sq += res[j]*res[j]; }
        float gamma = sqrtf(res_sq);
        *d_qjl_out = __float2half(gamma * norm);
        for (int j = 0; j < QK_K; j++) { int sign = ((qjl_s[j>>3]>>(j&7))&1)?-1:1; res[j] *= sign; }
        for (int len = 1; len < QK_K; len *= 2)
            for (int i = 0; i < QK_K; i += 2*len)
                for (int j = 0; j < len; j++) { float u = res[i+j], v = res[i+j+len]; res[i+j] = u+v; res[i+j+len] = u-v; }
        for (int j = 0; j < QK_K/8; j++) qjl_out[j] = 0;
        for (int j = 0; j < QK_K; j++) { if (res[j] >= 0.0f) qjl_out[j/8] |= (1<<(j%8)); }
    };

    quant_sub_qjl(x,       signs_256, qjl_signs_256, &y->d1, &y->d1_qjl, y->qs1, y->qjl1);
    quant_sub_qjl(x + 256, signs_256, qjl_signs_256, &y->d2, &y->d2_qjl, y->qs2, y->qjl2);
    // Sub-block 3: f16 passthrough (rope)
    for (int j = 0; j < TBQ_K64; j++) y->rope[j] = __float2half(x[512 + j]);
}

// ============================================================
// TurboQuant cross-head WHT (8 heads × 64 = 512-element WHT)
// H_512 = H_8 ⊗ H_64 — CLT convergence at d=512 for head_dim=64
// ============================================================

// 512-bit random sign pattern (first 8 bytes match tbq_signs for _2)
static __device__ constexpr uint8_t tbq_signs_512[64] = {
    0xa7,0x3b,0x91,0xf4,0x6d,0xc2,0x58,0x0e,  // head 0
    0xb3,0x7f,0x24,0xd6,0x89,0x45,0xea,0x1c,  // head 1
    0x63,0xaf,0xd8,0x52,0x97,0x0b,0xe1,0x3d,  // head 2
    0x76,0xc4,0x19,0xfe,0x4a,0x85,0x2c,0xdb,  // head 3
    0xd3,0x4e,0xa8,0x17,0x9c,0x5b,0xe6,0x31,  // head 4
    0x72,0xb9,0x0d,0xf5,0x43,0x8a,0x6e,0xc7,  // head 5
    0x58,0x2f,0x94,0xe1,0xb6,0x3d,0x0a,0x7c,  // head 6
    0xc5,0x61,0xd8,0x4f,0xa3,0x97,0x1e,0x85,  // head 7
};

// Cross-head TBQ3: 8×64=512-element WHT, 3-bit Lloyd-Max
static __device__ void quantize_f32_tbq3_3_xhead(
    const float * __restrict__ x0, const float * __restrict__ x1,
    const float * __restrict__ x2, const float * __restrict__ x3,
    const float * __restrict__ x4, const float * __restrict__ x5,
    const float * __restrict__ x6, const float * __restrict__ x7,
    block_tbq3_3 * __restrict__ y0, block_tbq3_3 * __restrict__ y1,
    block_tbq3_3 * __restrict__ y2, block_tbq3_3 * __restrict__ y3,
    block_tbq3_3 * __restrict__ y4, block_tbq3_3 * __restrict__ y5,
    block_tbq3_3 * __restrict__ y6, block_tbq3_3 * __restrict__ y7) {

    static constexpr float b3[7] = { -1.7480f,-1.0500f,-0.5006f,0.0f,0.5006f,1.0500f,1.7480f };
    const float * heads[8] = { x0,x1,x2,x3,x4,x5,x6,x7 };
    block_tbq3_3 * outs[8] = { y0,y1,y2,y3,y4,y5,y6,y7 };

    float tmp[512];
    for (int h = 0; h < 8; h++)
        for (int j = 0; j < TBQ_K64; j++)
            tmp[h*TBQ_K64 + j] = heads[h][j];

    // Apply signs + 512-point WHT
    for (int j = 0; j < 512; j++) {
        int sign = ((tbq_signs_512[j >> 3] >> (j & 7)) & 1) ? -1 : 1;
        tmp[j] *= sign;
    }
    for (int len = 1; len < 512; len *= 2)
        for (int i = 0; i < 512; i += 2*len)
            for (int j = 0; j < len; j++) {
                float u = tmp[i+j], v = tmp[i+j+len];
                tmp[i+j] = u+v; tmp[i+j+len] = u-v;
            }

    // Per-head norm after 512-WHT (same approach as per-block norm for D=512)
    // Global norm destroys small heads — per-head norm preserves each head's scale
    for (int h = 0; h < 8; h++) {
        float * blk = tmp + h * TBQ_K64;
        float blk_sq = 0.0f;
        for (int j = 0; j < TBQ_K64; j++) blk_sq += blk[j] * blk[j];
        float blk_norm = sqrtf(blk_sq / (float)TBQ_K64);

        if (blk_norm < 1e-10f) {
            outs[h]->d = __float2half(0.0f);
            for (int j = 0; j < TBQ_K64*3/8; j++) outs[h]->qs[j] = 0;
            continue;
        }

        float inv_norm = 1.0f / blk_norm;
        outs[h]->d = __float2half(blk_norm);

        int bit_pos = 0;
        for (int j = 0; j < TBQ_K64*3/8; j++) outs[h]->qs[j] = 0;
        for (int j = 0; j < TBQ_K64; j++) {
            float val = blk[j] * inv_norm;
            uint8_t idx = 7;
            for (int b = 0; b < 7; b++) { if (val < b3[b]) { idx = b; break; } }
            int byte_idx = bit_pos / 8, bit_off = bit_pos % 8;
            outs[h]->qs[byte_idx] |= (uint8_t)((idx & 0x7) << bit_off);
            if (bit_off > 5) outs[h]->qs[byte_idx + 1] |= (uint8_t)((idx & 0x7) >> (8 - bit_off));
            bit_pos += 3;
        }
    }
}

// Cross-head TBQ4: 8×64=512-element WHT, 4-bit Lloyd-Max
static __device__ void quantize_f32_tbq4_3_xhead(
    const float * __restrict__ x0, const float * __restrict__ x1,
    const float * __restrict__ x2, const float * __restrict__ x3,
    const float * __restrict__ x4, const float * __restrict__ x5,
    const float * __restrict__ x6, const float * __restrict__ x7,
    block_tbq4_3 * __restrict__ y0, block_tbq4_3 * __restrict__ y1,
    block_tbq4_3 * __restrict__ y2, block_tbq4_3 * __restrict__ y3,
    block_tbq4_3 * __restrict__ y4, block_tbq4_3 * __restrict__ y5,
    block_tbq4_3 * __restrict__ y6, block_tbq4_3 * __restrict__ y7) {

    static constexpr float b4[15] = { -2.4008f,-1.8435f,-1.4371f,-1.0993f,-0.7996f,-0.5225f,-0.2583f,0.0f,0.2583f,0.5225f,0.7996f,1.0993f,1.4371f,1.8435f,2.4008f };
    const float * heads[8] = { x0,x1,x2,x3,x4,x5,x6,x7 };
    block_tbq4_3 * outs[8] = { y0,y1,y2,y3,y4,y5,y6,y7 };

    float tmp[512];
    for (int h = 0; h < 8; h++)
        for (int j = 0; j < TBQ_K64; j++)
            tmp[h*TBQ_K64+j] = heads[h][j];

    // Apply signs + 512-point WHT
    for (int j = 0; j < 512; j++) { int sign = ((tbq_signs_512[j>>3]>>(j&7))&1)?-1:1; tmp[j] *= sign; }
    for (int len = 1; len < 512; len *= 2)
        for (int i = 0; i < 512; i += 2*len)
            for (int j = 0; j < len; j++) { float u = tmp[i+j], v = tmp[i+j+len]; tmp[i+j] = u+v; tmp[i+j+len] = u-v; }

    // Per-head norm after 512-WHT
    for (int h = 0; h < 8; h++) {
        float * blk = tmp + h * TBQ_K64;
        float blk_sq = 0.0f;
        for (int j = 0; j < TBQ_K64; j++) blk_sq += blk[j] * blk[j];
        float blk_norm = sqrtf(blk_sq / (float)TBQ_K64);

        if (blk_norm < 1e-10f) {
            outs[h]->d = __float2half(0.0f);
            for (int j = 0; j < TBQ_K64/2; j++) outs[h]->qs[j] = 0;
            continue;
        }

        float inv_norm = 1.0f / blk_norm;
        outs[h]->d = __float2half(blk_norm);

        for (int j = 0; j < TBQ_K64/2; j++) {
            uint8_t idx0 = 15, idx1 = 15;
            float val0 = blk[2*j]   * inv_norm;
            float val1 = blk[2*j+1] * inv_norm;
            for (int b = 0; b < 15; b++) { if (val0 < b4[b]) { idx0 = b; break; } }
            for (int b = 0; b < 15; b++) { if (val1 < b4[b]) { idx1 = b; break; } }
            outs[h]->qs[j] = idx0 | (idx1 << 4);
        }
    }
}

// Cross-head TBQP3: 8×64=512-element WHT, 2-bit Lloyd-Max + 1-bit Direct Sign
static __device__ void quantize_f32_tbqp3_3_xhead(
    const float * __restrict__ x0, const float * __restrict__ x1,
    const float * __restrict__ x2, const float * __restrict__ x3,
    const float * __restrict__ x4, const float * __restrict__ x5,
    const float * __restrict__ x6, const float * __restrict__ x7,
    block_tbqp3_3 * __restrict__ y0, block_tbqp3_3 * __restrict__ y1,
    block_tbqp3_3 * __restrict__ y2, block_tbqp3_3 * __restrict__ y3,
    block_tbqp3_3 * __restrict__ y4, block_tbqp3_3 * __restrict__ y5,
    block_tbqp3_3 * __restrict__ y6, block_tbqp3_3 * __restrict__ y7) {

    static constexpr float c2[4] = { -1.5104f,-0.4528f,0.4528f,1.5104f };
    static constexpr float b2[3] = { -0.9816f,0.0f,0.9816f };
    const float * heads[8] = { x0,x1,x2,x3,x4,x5,x6,x7 };
    block_tbqp3_3 * outs[8] = { y0,y1,y2,y3,y4,y5,y6,y7 };

    float tmp[512];
    for (int h = 0; h < 8; h++)
        for (int j = 0; j < TBQ_K64; j++)
            tmp[h*TBQ_K64+j] = heads[h][j];

    // Apply signs + 512-point WHT
    for (int j = 0; j < 512; j++) { int sign = ((tbq_signs_512[j>>3]>>(j&7))&1)?-1:1; tmp[j] *= sign; }
    for (int len = 1; len < 512; len *= 2)
        for (int i = 0; i < 512; i += 2*len)
            for (int j = 0; j < len; j++) { float u = tmp[i+j], v = tmp[i+j+len]; tmp[i+j] = u+v; tmp[i+j+len] = u-v; }

    // Per-head norm + quantize: 2-bit centroid + direct sign
    for (int h = 0; h < 8; h++) {
        float * seg = &tmp[h*TBQ_K64];
        float blk_sq = 0.0f;
        for (int j = 0; j < TBQ_K64; j++) blk_sq += seg[j] * seg[j];
        float blk_norm = sqrtf(blk_sq / (float)TBQ_K64);

        if (blk_norm < 1e-10f) {
            outs[h]->d = __float2half(0.0f); outs[h]->d_qjl = __float2half(0.0f);
            for (int j = 0; j < TBQ_K64/4; j++) outs[h]->qs[j] = 0;
            for (int j = 0; j < TBQ_K64/8; j++) outs[h]->qjl[j] = 0;
            continue;
        }

        float inv_norm = 1.0f / blk_norm;
        outs[h]->d = __float2half(blk_norm);

        float recon[TBQ_K64];
        for (int j = 0; j < TBQ_K64/4; j++) {
            uint8_t packed = 0;
            for (int k = 0; k < 4; k++) {
                float val = seg[j*4+k] * inv_norm;
                uint8_t idx = 3;
                for (int b = 0; b < 3; b++) { if (val < b2[b]) { idx = b; break; } }
                packed |= (idx & 0x3) << (k*2);
                recon[j*4+k] = c2[idx];
            }
            outs[h]->qs[j] = packed;
        }
        // Residual + Direct Sign (d_qjl relative to per-head norm)
        float res_abs = 0.0f;
        for (int j = 0; j < TBQ_K64; j++) res_abs += fabsf(seg[j] * inv_norm - recon[j]);
        outs[h]->d_qjl = __float2half((res_abs / TBQ_K64) * blk_norm);
        for (int j = 0; j < TBQ_K64/8; j++) outs[h]->qjl[j] = 0;
        for (int j = 0; j < TBQ_K64; j++) {
            if (seg[j] * inv_norm - recon[j] >= 0.0f) outs[h]->qjl[j/8] |= (1 << (j%8));
        }
    }
}

// Cross-head TBQP4: 8×64=512-element WHT, 3-bit Lloyd-Max + 1-bit Direct Sign
static __device__ void quantize_f32_tbqp4_3_xhead(
    const float * __restrict__ x0, const float * __restrict__ x1,
    const float * __restrict__ x2, const float * __restrict__ x3,
    const float * __restrict__ x4, const float * __restrict__ x5,
    const float * __restrict__ x6, const float * __restrict__ x7,
    block_tbqp4_3 * __restrict__ y0, block_tbqp4_3 * __restrict__ y1,
    block_tbqp4_3 * __restrict__ y2, block_tbqp4_3 * __restrict__ y3,
    block_tbqp4_3 * __restrict__ y4, block_tbqp4_3 * __restrict__ y5,
    block_tbqp4_3 * __restrict__ y6, block_tbqp4_3 * __restrict__ y7) {

    static constexpr float c3[8] = { -2.1520f,-1.3440f,-0.7560f,-0.2451f,0.2451f,0.7560f,1.3440f,2.1520f };
    static constexpr float b3[7] = { -1.7480f,-1.0500f,-0.5006f,0.0f,0.5006f,1.0500f,1.7480f };
    const float * heads[8] = { x0,x1,x2,x3,x4,x5,x6,x7 };
    block_tbqp4_3 * outs[8] = { y0,y1,y2,y3,y4,y5,y6,y7 };

    float tmp[512];
    for (int h = 0; h < 8; h++)
        for (int j = 0; j < TBQ_K64; j++)
            tmp[h*TBQ_K64+j] = heads[h][j];

    // Apply signs + 512-point WHT
    for (int j = 0; j < 512; j++) { int sign = ((tbq_signs_512[j>>3]>>(j&7))&1)?-1:1; tmp[j] *= sign; }
    for (int len = 1; len < 512; len *= 2)
        for (int i = 0; i < 512; i += 2*len)
            for (int j = 0; j < len; j++) { float u = tmp[i+j], v = tmp[i+j+len]; tmp[i+j] = u+v; tmp[i+j+len] = u-v; }

    // Per-head norm + quantize: 3-bit centroid + direct sign
    for (int h = 0; h < 8; h++) {
        float * seg = &tmp[h*TBQ_K64];
        float blk_sq = 0.0f;
        for (int j = 0; j < TBQ_K64; j++) blk_sq += seg[j] * seg[j];
        float blk_norm = sqrtf(blk_sq / (float)TBQ_K64);

        if (blk_norm < 1e-10f) {
            outs[h]->d = __float2half(0.0f); outs[h]->d_qjl = __float2half(0.0f);
            for (int j = 0; j < TBQ_K64*3/8; j++) outs[h]->qs[j] = 0;
            for (int j = 0; j < TBQ_K64/8; j++) outs[h]->qjl[j] = 0;
            continue;
        }

        float inv_norm = 1.0f / blk_norm;
        outs[h]->d = __float2half(blk_norm);

        float recon[TBQ_K64];
        int bit_pos = 0;
        for (int j = 0; j < TBQ_K64*3/8; j++) outs[h]->qs[j] = 0;
        for (int j = 0; j < TBQ_K64; j++) {
            float val = seg[j] * inv_norm;
            uint8_t idx = 7;
            for (int b = 0; b < 7; b++) { if (val < b3[b]) { idx = b; break; } }
            int byte_idx = bit_pos/8, bit_off = bit_pos%8;
            outs[h]->qs[byte_idx] |= (uint8_t)((idx&0x7)<<bit_off);
            if (bit_off > 5) outs[h]->qs[byte_idx+1] |= (uint8_t)((idx&0x7)>>(8-bit_off));
            recon[j] = c3[idx]; bit_pos += 3;
        }
        float res_abs = 0.0f;
        for (int j = 0; j < TBQ_K64; j++) res_abs += fabsf(seg[j] * inv_norm - recon[j]);
        outs[h]->d_qjl = __float2half((res_abs/TBQ_K64)*blk_norm);
        for (int j = 0; j < TBQ_K64/8; j++) outs[h]->qjl[j] = 0;
        for (int j = 0; j < TBQ_K64; j++) { if (seg[j] * inv_norm - recon[j] >= 0.0f) outs[h]->qjl[j/8] |= (1<<(j%8)); }
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
