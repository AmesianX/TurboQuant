// Prevent Windows min/max macros from conflicting with std::min/std::max
#define NOMINMAX

#include "common.cuh"
#include "dsv4.cuh"

#include <cuda_fp8.h>
#include <cuda_bf16.h>
#if defined(GGML_USE_HIP)
#else
#include <mma.h>
#endif

#ifndef M_PI_F
#define M_PI_F 3.141592653589793238462643383279502884f
#endif

// bf16 16x16x16 WMMA fragments require sm80+ (Ampere). Our target is sm121 (GB10).
#if !defined(GGML_USE_HIP) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#define DSV4_WMMA_AVAILABLE
#endif

namespace {

constexpr int DSV4_HC_MAX = 16;

static __device__ __forceinline__ float dsv4_e4m3fn_dequant(float x) {
    // round-to-nearest-even + saturate-to-448 — identical semantics to the
    // old 127-entry linear search (even-tie == RTNE on the e4m3fn grid), but
    // a single hardware cvt instead of a 126-iteration loop per element.
    const __nv_fp8_storage_t r = __nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3);
    const __half_raw h = __nv_cvt_fp8_to_halfraw(r, __NV_E4M3);
    return __half2float(h);
}

static __device__ __forceinline__ float rope_yarn_ramp(const float low, const float high, const int i0) {
    const float y = (i0 / 2 - low) / max(0.001f, high - low);
    return 1.0f - min(1.0f, max(0.0f, y));
}

static __device__ __forceinline__ float rope_yarn_corr_factor(int n_dims, int n_ctx_orig, float n_rot, float base) {
    return n_dims * logf(n_ctx_orig / (n_rot * 2 * M_PI_F)) / (2 * logf(base));
}

static __device__ __forceinline__ void rope_yarn_corr_dims(int n_dims, int n_ctx_orig, float freq_base, float beta_fast, float beta_slow, float dims[2]) {
    dims[0] = max(0.0f,         floorf(rope_yarn_corr_factor(n_dims, n_ctx_orig, beta_fast, freq_base)));
    dims[1] = min(n_dims - 1.0f, ceilf(rope_yarn_corr_factor(n_dims, n_ctx_orig, beta_slow, freq_base)));
}

static __device__ __forceinline__ void rope_yarn(float theta_extrap, float freq_scale, float corr_dims[2], int i0, float ext_factor, float mscale, float * cos_theta, float * sin_theta) {
    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    if (ext_factor != 0.0f) {
        float ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], i0) * ext_factor;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    *cos_theta = cosf(theta) * mscale;
    *sin_theta = sinf(theta) * mscale;
}

struct ggml_cuda_kargs_dsv4_hc_split_sinkhorn {
    int32_t  n_hc;
    int32_t  sinkhorn_iters;
    int64_t  n_rows;
    int64_t  mix_hc;
    uint64_t nb01;
    uint64_t nb1;
    float    eps;
};

struct ggml_cuda_kargs_dsv4_hc_expand {
    int64_t  n_embd;
    int64_t  n_hc;
    int64_t  n_tokens;
    uint64_t nb_block0;
    uint64_t nb_block1;
    uint64_t nb_res0;
    uint64_t nb_res1;
    uint64_t nb_res2;
    uint64_t nb_post0;
    uint64_t nb_post1;
    uint64_t nb_comb0;
    uint64_t nb_comb1;
    uint64_t nb_comb2;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
};

struct ggml_cuda_kargs_dsv4_fp8_kv_quantize {
    int64_t  ne00;
    int64_t  ne01;
    int64_t  ne02;
    int64_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
    int32_t  n_rot;
};

struct ggml_cuda_kargs_dsv4_hc_weighted_sum {
    int64_t  n_embd;
    int64_t  n_hc;
    int64_t  n_tokens;
    uint64_t nb_x0;
    uint64_t nb_x1;
    uint64_t nb_x2;
    uint64_t nb_w0;
    uint64_t nb_w1;
    uint64_t nb0;
    uint64_t nb1;
};

struct ggml_cuda_kargs_dsv4_indexer_logits {
    int64_t  head_dim;   // ne00 of k/q
    int64_t  n_comp;     // k->ne[1]
    int64_t  n_tokens;   // q->ne[1]
    int64_t  n_head;     // q->ne[2]
    // k strides (bytes)
    uint64_t nb_k0;
    uint64_t nb_k1;
    // q strides (bytes)
    uint64_t nb_q0;
    uint64_t nb_q1;
    uint64_t nb_q2;
    // weights strides (bytes) [n_head, n_tokens]
    uint64_t nb_w0;
    uint64_t nb_w1;
    // dst strides (bytes) [n_comp, n_tokens]
    uint64_t nb0;
    uint64_t nb1;
};

struct ggml_cuda_kargs_dsv4_rope_tail {
    int64_t  ne00;
    int64_t  ne01;
    int64_t  ne02;
    int64_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
    int32_t  n_dims;
    int32_t  mode;
    int32_t  n_ctx_orig;
    int32_t  inverse;
    float    freq_base;
    float    freq_scale;
    float    ext_factor;
    float    attn_factor;
    float    beta_fast;
    float    beta_slow;
    bool     src2;
};

// per-row split + sinkhorn body, shared by the standalone kernel and the fused
// split+weighted-sum kernel
static __device__ void dsv4_hc_split_sinkhorn_one(
        const ggml_cuda_kargs_dsv4_hc_split_sinkhorn & args,
        const float * mix,
        const float * scale,
        const float * base,
        float * out) {
    const int HC = args.n_hc;

    const float epsv       = args.eps;
    const float pre_scale  = scale[0];
    const float post_scale = scale[1];
    const float comb_scale = scale[2];

    for (int i = 0; i < HC; ++i) {
        const float z = mix[i] * pre_scale + base[i];
        out[i] = 1.0f / (1.0f + expf(-z)) + epsv;
    }

    for (int i = 0; i < HC; ++i) {
        const int off = HC + i;
        const float z = mix[off] * post_scale + base[off];
        out[off] = 2.0f / (1.0f + expf(-z));
    }

    float c[DSV4_HC_MAX * DSV4_HC_MAX];

    for (int dst_hc = 0; dst_hc < HC; ++dst_hc) {
        float row_max = -INFINITY;
        for (int src_hc = 0; src_hc < HC; ++src_hc) {
            const int idx = src_hc + dst_hc * HC;
            const int off = 2 * HC + idx;
            const float v = mix[off] * comb_scale + base[off];
            c[idx] = v;
            row_max = fmaxf(row_max, v);
        }

        float row_sum = 0.0f;
        for (int src_hc = 0; src_hc < HC; ++src_hc) {
            const int idx = src_hc + dst_hc * HC;
            const float v = expf(c[idx] - row_max);
            c[idx] = v;
            row_sum += v;
        }

        const float inv_sum = 1.0f / row_sum;
        for (int src_hc = 0; src_hc < HC; ++src_hc) {
            const int idx = src_hc + dst_hc * HC;
            c[idx] = c[idx] * inv_sum + epsv;
        }
    }

    for (int src_hc = 0; src_hc < HC; ++src_hc) {
        float sum = 0.0f;
        for (int dst_hc = 0; dst_hc < HC; ++dst_hc) {
            sum += c[src_hc + dst_hc * HC];
        }

        const float inv_denom = 1.0f / (sum + epsv);
        for (int dst_hc = 0; dst_hc < HC; ++dst_hc) {
            c[src_hc + dst_hc * HC] *= inv_denom;
        }
    }

    for (int iter = 1; iter < args.sinkhorn_iters; ++iter) {
        for (int dst_hc = 0; dst_hc < HC; ++dst_hc) {
            float sum = 0.0f;
            for (int src_hc = 0; src_hc < HC; ++src_hc) {
                sum += c[src_hc + dst_hc * HC];
            }

            const float inv_denom = 1.0f / (sum + epsv);
            for (int src_hc = 0; src_hc < HC; ++src_hc) {
                c[src_hc + dst_hc * HC] *= inv_denom;
            }
        }

        for (int src_hc = 0; src_hc < HC; ++src_hc) {
            float sum = 0.0f;
            for (int dst_hc = 0; dst_hc < HC; ++dst_hc) {
                sum += c[src_hc + dst_hc * HC];
            }

            const float inv_denom = 1.0f / (sum + epsv);
            for (int dst_hc = 0; dst_hc < HC; ++dst_hc) {
                c[src_hc + dst_hc * HC] *= inv_denom;
            }
        }
    }

    for (int i = 0; i < HC * HC; ++i) {
        out[2 * HC + i] = c[i];
    }
}

static __global__ void kernel_dsv4_hc_split_sinkhorn(
        const ggml_cuda_kargs_dsv4_hc_split_sinkhorn args,
        const float * mixes,
        const float * scale,
        const float * base,
        float * dst) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if ((int64_t) tid >= args.n_rows) {
        return;
    }
    if (args.n_hc <= 0 || args.n_hc > DSV4_HC_MAX) {
        return;
    }
    dsv4_hc_split_sinkhorn_one(args,
        mixes + ((int64_t) tid) * args.mix_hc, scale, base,
        dst   + ((int64_t) tid) * args.mix_hc);
}

static __global__ void kernel_dsv4_hc_expand(
        const ggml_cuda_kargs_dsv4_hc_expand args,
        const char * block_out,
        const char * residual,
        const char * post,
        const char * comb,
        char * dst) {
    const int64_t n_elem = args.n_embd * args.n_hc * args.n_tokens;
    const int64_t gid = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n_elem) {
        return;
    }

    const int64_t d      = ((int64_t) gid) % args.n_embd;
    const int64_t tmp    = ((int64_t) gid) / args.n_embd;
    const int64_t dst_hc = tmp % args.n_hc;
    const int64_t t      = tmp / args.n_hc;

    const float block_v = *((const float *) (block_out + d * args.nb_block0 + t * args.nb_block1));
    const float post_v  = *((const float *) (post      + dst_hc * args.nb_post0 + t * args.nb_post1));

    float acc = block_v * post_v;
    for (int64_t src_hc = 0; src_hc < args.n_hc; ++src_hc) {
        const float comb_v = *((const float *) (comb     + dst_hc * args.nb_comb0 + src_hc * args.nb_comb1 + t * args.nb_comb2));
        const float res_v  = *((const float *) (residual + d       * args.nb_res0  + src_hc * args.nb_res1  + t * args.nb_res2));
        acc += comb_v * res_v;
    }

    *((float *) (dst + d * args.nb0 + dst_hc * args.nb1 + t * args.nb2)) = acc;
}

static __global__ void kernel_dsv4_fp8_kv_quantize(
        const ggml_cuda_kargs_dsv4_fp8_kv_quantize args,
        const char * src0,
        char * dst) {
    __shared__ float scratch[64];

    const int64_t n_rows = args.ne01 * args.ne02 * args.ne03;
    const int row = blockIdx.x;
    if ((int64_t) row >= n_rows) {
        return;
    }

    const int tid = threadIdx.x;

    const int64_t i1 = row % args.ne01;
    const int64_t i2 = (row / args.ne01) % args.ne02;
    const int64_t i3 = row / (args.ne01 * args.ne02);

    const char * src_base = src0 + i1 * args.nb01 + i2 * args.nb02 + i3 * args.nb03;
    char * dst_base = dst  + i1 * args.nb1  + i2 * args.nb2  + i3 * args.nb3;

    const int64_t n_nope = args.ne00 - args.n_rot;

    for (int64_t off = 0; off < n_nope; off += 64) {
        // guard the tail chunk: when n_nope % 64 != 0, unguarded lanes would read rope-region
        // (or out-of-row) values and contaminate the per-chunk amax/scale
        const bool lane_ok = off + tid < n_nope;
        float v = 0.0f;
        if (lane_ok) {
            v = *((const float *) (src_base + (off + tid) * args.nb00));
        }
        scratch[tid] = lane_ok ? fabsf(v) : 0.0f;
        __syncthreads();

        for (uint32_t stride = 32; stride > 0; stride >>= 1) {
            if (tid < stride) {
                scratch[tid] = fmaxf(scratch[tid], scratch[tid + stride]);
            }
            __syncthreads();
        }

        const float amax = fmaxf(scratch[0], 1.0e-4f);
        const float scale = exp2f(ceilf(log2f(amax / 448.0f)));
        if (lane_ok) {
            const float q = dsv4_e4m3fn_dequant(fminf(fmaxf(v / scale, -448.0f), 448.0f)) * scale;
            *((float *) (dst_base + (off + tid) * args.nb0)) = q;
        }
        __syncthreads();
    }

    for (int64_t i = n_nope + tid; i < args.ne00; i += 64) {
        *((float *) (dst_base + i * args.nb0)) = *((const float *) (src_base + i * args.nb00));
    }
}

// ============================================================================
// [Fusion 3] DSV4_NORM_ROPE — RMS-norm [* w] -> RoPE tail -> optional FP8 round.
//
// The Q/KV preparation used to be five separate launches per layer per token
//   (rms_norm, rope_tail) for Q and (rms_norm, mul w, rope_tail, fp8_quantize) for KV
// each of which is a per-ROW kernel over a 576-wide row: for KV that is ONE BLOCK, i.e. one of the
// 48 SMs, for a few microseconds, five times, 41 layers deep. The three stages touch the same row
// and DISJOINT slices of it (RoPE only the trailing n_dims, the FP8 round only the leading nope
// region), so one block-per-row kernel does all of them out of shared memory and the intermediates
// never see HBM.
// ============================================================================

#define DSV4_NR_NT 64   // threads/block — matches the FP8 chunk width (see the amax reduction)

struct ggml_cuda_kargs_dsv4_norm_rope {
    int64_t  ne00, ne01, ne02, ne03;
    uint64_t nb00, nb01, nb02, nb03;
    uint64_t nb0,  nb1,  nb2,  nb3;
    int32_t  n_dims, mode, n_ctx_orig, fp8_kv;
    float    eps, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
    bool     has_w;
};

static __global__ void kernel_dsv4_norm_rope_f32(
        const ggml_cuda_kargs_dsv4_norm_rope args,
        const char * __restrict__ src0,
        const char * __restrict__ src1,   // pos, I32
        const char * __restrict__ src2,   // norm weight, F32 [ne00] (may be null)
        char * __restrict__ dst) {
    extern __shared__ float row[];
    __shared__ float red[DSV4_NR_NT];

    const int64_t n_rows = args.ne01 * args.ne02 * args.ne03;
    const int     rid    = blockIdx.x;
    if ((int64_t) rid >= n_rows) {
        return;
    }

    const int tid = threadIdx.x;
    const int ne0 = (int) args.ne00;

    const int64_t i1 = rid % args.ne01;
    const int64_t i2 = (rid / args.ne01) % args.ne02;
    const int64_t i3 = rid / (args.ne01 * args.ne02);

    const char * sb = src0 + i1*args.nb01 + i2*args.nb02 + i3*args.nb03;
    char       * db = dst  + i1*args.nb1  + i2*args.nb2  + i3*args.nb3;

    // ---- 1) load the row into shared memory + sum of squares -----------------
    float ss = 0.0f;
    for (int i = tid; i < ne0; i += DSV4_NR_NT) {
        const float v = *((const float *) (sb + i*args.nb00));
        row[i] = v;
        ss += v*v;
    }
    red[tid] = ss;
    __syncthreads();
    for (int s = DSV4_NR_NT/2; s > 0; s >>= 1) {
        if (tid < s) {
            red[tid] += red[tid + s];
        }
        __syncthreads();
    }
    const float nscale = rsqrtf(red[0] / (float) ne0 + args.eps);
    __syncthreads();

    // ---- 2) RMS scale (+ the per-dim norm weight, when there is one) ---------
    const float * w = args.has_w ? (const float *) src2 : nullptr;
    for (int i = tid; i < ne0; i += DSV4_NR_NT) {
        row[i] = row[i] * nscale * (w ? w[i] : 1.0f);
    }
    __syncthreads();

    // ---- 3) RoPE the trailing n_dims ----------------------------------------
    const int n_nope = ne0 - args.n_dims;
    if (n_nope >= 0) {
        const int32_t * pos = (const int32_t *) src1;

        float corr_dims[2];
        rope_yarn_corr_dims(args.n_dims, args.n_ctx_orig, args.freq_base, args.beta_fast, args.beta_slow, corr_dims);

        const float theta_base = (float) pos[i2];
        const float inv_ndims  = -1.0f / args.n_dims;
        const int   n_half     = args.n_dims / 2;
        const bool  is_neox    = args.mode == 2;

        for (int ic = tid; ic < n_half; ic += DSV4_NR_NT) {
            // neox pairs (ic, ic + n_half); normal pairs (2*ic, 2*ic + 1)
            const int rel_i0 = is_neox ? 2*ic : 2*ic;
            const int j0     = is_neox ? n_nope + ic          : n_nope + 2*ic;
            const int j1     = is_neox ? n_nope + ic + n_half : n_nope + 2*ic + 1;

            const float theta = theta_base * powf(args.freq_base, inv_ndims * rel_i0);

            float ct, st;
            rope_yarn(theta, args.freq_scale, corr_dims, rel_i0, args.ext_factor, args.attn_factor, &ct, &st);

            const float x0 = row[j0];
            const float x1 = row[j1];
            row[j0] = x0*ct - x1*st;
            row[j1] = x0*st + x1*ct;
        }
        __syncthreads();
    }

    // ---- 4) optional FP8-E4M3 round of the NOPE region, 64-wide chunks -------
    // (byte-identical to kernel_dsv4_fp8_kv_quantize: per-chunk amax -> power-of-two scale)
    if (args.fp8_kv) {
        for (int off = 0; off < n_nope; off += DSV4_NR_NT) {
            const bool  ok = off + tid < n_nope;
            const float v  = ok ? row[off + tid] : 0.0f;

            red[tid] = ok ? fabsf(v) : 0.0f;
            __syncthreads();
            for (uint32_t s = 32; s > 0; s >>= 1) {
                if (tid < (int) s) {
                    red[tid] = fmaxf(red[tid], red[tid + s]);
                }
                __syncthreads();
            }

            const float amax  = fmaxf(red[0], 1.0e-4f);
            const float qscale = exp2f(ceilf(log2f(amax / 448.0f)));
            if (ok) {
                row[off + tid] = dsv4_e4m3fn_dequant(fminf(fmaxf(v / qscale, -448.0f), 448.0f)) * qscale;
            }
            __syncthreads();
        }
    }

    // ---- 5) store -----------------------------------------------------------
    for (int i = tid; i < ne0; i += DSV4_NR_NT) {
        *((float *) (db + i*args.nb0)) = row[i];
    }
}

static __global__ void kernel_dsv4_rope_tail_f32(
        const ggml_cuda_kargs_dsv4_rope_tail args,
        const char * src0,
        const char * src1,
        const char * src2,
        char * dst) {
    const int i1 = blockIdx.z;
    const int i2 = blockIdx.y;
    const int i3 = blockIdx.x;

    const int tid = threadIdx.x;

    const int n_nope = args.ne00 - args.n_dims;
    if (n_nope < 0) {
        return;
    }

    const int32_t * pos = (const int32_t *) src1;

    float corr_dims[2];
    rope_yarn_corr_dims(args.n_dims, args.n_ctx_orig, args.freq_base, args.beta_fast, args.beta_slow, corr_dims);

    const float theta_base = (float) pos[i2];
    const float inv_ndims = -1.0f / args.n_dims;
    const bool is_neox = args.mode == 2;

    for (int i0 = tid; i0 < args.ne00; i0 += blockDim.x) {
        const char * src_base = src0 + i3 * args.nb03 + i2 * args.nb02 + i1 * args.nb01;
        char * dst_base = dst  + i3 * args.nb3  + i2 * args.nb2  + i1 * args.nb1;

        if (i0 < n_nope) {
            *((float *) (dst_base + i0 * args.nb0)) = *((const float *) (src_base + i0 * args.nb00));
            continue;
        }

        const int r = i0 - n_nope;
        if (is_neox) {
            const int n_half = args.n_dims / 2;
            if (r >= n_half) {
                continue;
            }

            const int ic = r;
            const int rel_i0 = 2 * ic;
            const float theta = theta_base * powf(args.freq_base, inv_ndims * rel_i0);
            const float freq_factor = args.src2 ? ((const float *) src2)[ic] : 1.0f;

            float cos_theta;
            float sin_theta;
            rope_yarn(theta / freq_factor, args.freq_scale, corr_dims, rel_i0, args.ext_factor, args.attn_factor, &cos_theta, &sin_theta);
            if (args.inverse) {
                sin_theta = -sin_theta;
            }

            const int j0 = n_nope + ic;
            const int j1 = n_nope + ic + n_half;
            const float x0 = *((const float *) (src_base + j0 * args.nb00));
            const float x1 = *((const float *) (src_base + j1 * args.nb00));

            *((float *) (dst_base + j0 * args.nb0)) = x0 * cos_theta - x1 * sin_theta;
            *((float *) (dst_base + j1 * args.nb0)) = x0 * sin_theta + x1 * cos_theta;
        } else {
            if ((r & 1) != 0) {
                continue;
            }

            const int ic = r / 2;
            const float theta = theta_base * powf(args.freq_base, inv_ndims * r);
            const float freq_factor = args.src2 ? ((const float *) src2)[ic] : 1.0f;

            float cos_theta;
            float sin_theta;
            rope_yarn(theta / freq_factor, args.freq_scale, corr_dims, r, args.ext_factor, args.attn_factor, &cos_theta, &sin_theta);
            if (args.inverse) {
                sin_theta = -sin_theta;
            }

            const int j0 = n_nope + r;
            const int j1 = j0 + 1;
            const float x0 = *((const float *) (src_base + j0 * args.nb00));
            const float x1 = *((const float *) (src_base + j1 * args.nb00));

            *((float *) (dst_base + j0 * args.nb0)) = x0 * cos_theta - x1 * sin_theta;
            *((float *) (dst_base + j1 * args.nb0)) = x0 * sin_theta + x1 * cos_theta;
        }
    }
}

static __global__ void kernel_dsv4_hc_weighted_sum(
        const ggml_cuda_kargs_dsv4_hc_weighted_sum args,
        const char * x,
        const char * weights,
        char * dst) {
    const int64_t n_elem = args.n_embd * args.n_tokens;
    const int64_t gid = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n_elem) {
        return;
    }

    const int64_t d = ((int64_t) gid) % args.n_embd;
    const int64_t t = ((int64_t) gid) / args.n_embd;

    float acc = 0.0f;
    for (int64_t h = 0; h < args.n_hc; ++h) {
        const float xv = *((const float *) (x     + d * args.nb_x0 + h * args.nb_x1 + t * args.nb_x2));
        const float wv = *((const float *) (weights + h * args.nb_w0 + t * args.nb_w1));
        acc += xv * wv;
    }

    *((float *) (dst + d * args.nb0 + t * args.nb1)) = acc;
}

// ---- Fused DSA lightning-indexer logits ----------------------------------------------
// logits[c, t] = sum_h weights[h, t] * relu( sum_d q[d, t, h] * k[d, c] )
// Replaces the mul_mat([n_comp,ub,n_head]) + relu + mul(weights) + cont(transpose) +
// sum_rows chain, so the O(n_comp*ub*n_head) score tensor and its transpose are NEVER
// materialized (the O(ub^2) prefill wall). One block computes a TILE of comp rows for one
// token; blockDim.x = head_dim lanes do the per-head dot via a block reduction.
//
// Templated on the k/q element read so F16 (cache) and F32 (fresh chunk) both work; the
// dot product accumulates in F32 (numerically equivalent to the F32/BF16 mul_mat the
// indexer fed into a top-k SELECTION — reduced input precision is tolerant).
template <typename T>
static __device__ __forceinline__ float dsv4_idx_load(const char * p) {
    return (float) (*((const T *) p));
}

template <typename kT, typename qT, int HEAD_DIM>
static __global__ void kernel_dsv4_indexer_logits(
        const ggml_cuda_kargs_dsv4_indexer_logits args,
        const char * __restrict__ k,
        const char * __restrict__ q,
        const char * __restrict__ weights,
        char * __restrict__ dst) {
    // grid.x = token, grid.y = comp tile. blockDim.x == HEAD_DIM (== head_dim lanes).
    const int64_t t = blockIdx.x;
    const int64_t c = (int64_t) blockIdx.y;
    if (t >= args.n_tokens || c >= args.n_comp) {
        return;
    }
    const int lane = threadIdx.x;  // 0..HEAD_DIM-1

    // shared reduction buffer (one warp-sum slot per 32 lanes)
    __shared__ float s_red[HEAD_DIM / 32 > 0 ? HEAD_DIM / 32 : 1];

    // k[d, c] for this comp row (shared across all heads/tokens) — load once per lane.
    const float kv = dsv4_idx_load<kT>(k + (uint64_t) lane * args.nb_k0 + (uint64_t) c * args.nb_k1);

    float logit = 0.0f;
    for (int64_t h = 0; h < args.n_head; ++h) {
        const float qv = dsv4_idx_load<qT>(
            q + (uint64_t) lane * args.nb_q0 + (uint64_t) t * args.nb_q1 + (uint64_t) h * args.nb_q2);
        float dot = qv * kv;
        // warp reduce
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            dot += __shfl_down_sync(0xffffffff, dot, o);
        }
        // block reduce across warps
        if (HEAD_DIM > 32) {
            if ((lane & 31) == 0) {
                s_red[lane >> 5] = dot;
            }
            __syncthreads();
            if (lane == 0) {
                float full = 0.0f;
                #pragma unroll
                for (int w = 0; w < HEAD_DIM / 32; ++w) {
                    full += s_red[w];
                }
                dot = full;
            }
            __syncthreads();
        }
        if (lane == 0) {
            const float r = dot > 0.0f ? dot : 0.0f;   // relu
            const float wv = *((const float *) (weights + (uint64_t) h * args.nb_w0 + (uint64_t) t * args.nb_w1));
            logit += wv * r;
        }
    }

    if (lane == 0) {
        *((float *) (dst + (uint64_t) c * args.nb0 + (uint64_t) t * args.nb1)) = logit;
    }
}

// ---- WMMA (tensor-core) fused indexer logits ----------------------------------------
// Same math as kernel_dsv4_indexer_logits but the per-head Sigma_d q[d,t,h]*k[d,c] dot runs
// on tensor cores (bf16 inputs, F32 accumulate, m16n8k16/16x16x16 MMA) instead of CUDA-core
// block-reduce. One warp computes a 16(comp) x 16(token) output tile; for each of the 64
// heads it stages the 16x128 K-tile and 16x128 Q-tile into bf16 smem, runs 8 k-steps of MMA
// (128/16), ReLUs the F32 result, scales by weights[h,t] and adds into the running F32 tile
// accumulator. The epilogue (relu + weight + head-sum) stays F32 -> precision-tolerant
// (feeds the top-512 selection). head_dim is fixed to 128 (DSV4 indexer key_length).
#define DSV4_IDX_HD 128
#define DSV4_IDX_TILE 16

template <typename kT, typename qT>
static __global__ void kernel_dsv4_indexer_logits_wmma(
        const ggml_cuda_kargs_dsv4_indexer_logits args,
        const char * __restrict__ k,
        const char * __restrict__ q,
        const char * __restrict__ weights,
        char * __restrict__ dst) {
#ifdef DSV4_WMMA_AVAILABLE
    namespace wmma = nvcuda::wmma;

    const int64_t c0 = (int64_t) blockIdx.y * DSV4_IDX_TILE;   // first comp row of this tile
    const int64_t t0 = (int64_t) blockIdx.x * DSV4_IDX_TILE;   // first token of this tile
    const int      tid = threadIdx.x;                          // 0..31 (one warp)

    // bf16 staging tiles: K [16c x 128d], Q [16t x 128d]
    __shared__ __nv_bfloat16 sK[DSV4_IDX_TILE][DSV4_IDX_HD];
    __shared__ __nv_bfloat16 sQ[DSV4_IDX_TILE][DSV4_IDX_HD];
    __shared__ float         sW[DSV4_IDX_TILE];                  // weights[h, t]
    __shared__ float         sM[DSV4_IDX_TILE][DSV4_IDX_TILE];   // per-head M result [c][t]
    __shared__ float         sAcc[DSV4_IDX_TILE][DSV4_IDX_TILE]; // running F32 logits [c][t]

    for (int idx = tid; idx < DSV4_IDX_TILE * DSV4_IDX_TILE; idx += 32) {
        sAcc[idx / DSV4_IDX_TILE][idx % DSV4_IDX_TILE] = 0.0f;
    }

    const int64_t n_comp   = args.n_comp;
    const int64_t n_tokens = args.n_tokens;

    wmma::fragment<wmma::matrix_a,    16, 16, 16, __nv_bfloat16, wmma::row_major> fragK; // K[c,d]
    wmma::fragment<wmma::matrix_b,    16, 16, 16, __nv_bfloat16, wmma::col_major> fragQ; // -> Q[d,t]
    wmma::fragment<wmma::accumulator, 16, 16, 16, float>                          fragM;

    for (int64_t h = 0; h < args.n_head; ++h) {
        __syncthreads();
        // Stage K tile [16c x 128d] (bf16). 16*128/32 = 64 elements per lane.
        for (int idx = tid; idx < DSV4_IDX_TILE * DSV4_IDX_HD; idx += 32) {
            const int cc = idx / DSV4_IDX_HD;   // 0..15
            const int dd = idx % DSV4_IDX_HD;   // 0..127
            const int64_t cg = c0 + cc;
            float kval = 0.0f;
            if (cg < n_comp) {
                kval = (float) (*((const kT *) (k + (uint64_t) dd * args.nb_k0 + (uint64_t) cg * args.nb_k1)));
            }
            sK[cc][dd] = __float2bfloat16(kval);
        }
        // Stage Q tile [16t x 128d] (bf16) for head h.
        for (int idx = tid; idx < DSV4_IDX_TILE * DSV4_IDX_HD; idx += 32) {
            const int tt = idx / DSV4_IDX_HD;
            const int dd = idx % DSV4_IDX_HD;
            const int64_t tg = t0 + tt;
            float qval = 0.0f;
            if (tg < n_tokens) {
                qval = (float) (*((const qT *) (q + (uint64_t) dd * args.nb_q0 + (uint64_t) tg * args.nb_q1 + (uint64_t) h * args.nb_q2)));
            }
            sQ[tt][dd] = __float2bfloat16(qval);
        }
        if (tid < DSV4_IDX_TILE) {
            const int64_t tg = t0 + tid;
            sW[tid] = (tg < n_tokens) ? *((const float *) (weights + (uint64_t) h * args.nb_w0 + (uint64_t) tg * args.nb_w1)) : 0.0f;
        }
        __syncthreads();

        // M[c,t] = sum_d K[c,d] * Q[t,d] over the 128-dim contraction (8 k-steps). With matrix_b in
        // col_major the B operand reads sQ[t][d] as [d][t], so mma_sync yields D[c,t] = sum_d K[c,d]*Q[t,d].
        wmma::fill_fragment(fragM, 0.0f);
        #pragma unroll
        for (int kd = 0; kd < DSV4_IDX_HD; kd += 16) {
            wmma::load_matrix_sync(fragK, &sK[0][kd], DSV4_IDX_HD);
            wmma::load_matrix_sync(fragQ, &sQ[0][kd], DSV4_IDX_HD);
            wmma::mma_sync(fragM, fragK, fragQ, fragM);
        }
        wmma::store_matrix_sync(&sM[0][0], fragM, DSV4_IDX_TILE, wmma::mem_row_major);
        __syncthreads();

        // ReLU + weight[h,t] + head-accumulate (all F32).
        for (int idx = tid; idx < DSV4_IDX_TILE * DSV4_IDX_TILE; idx += 32) {
            const int cc = idx / DSV4_IDX_TILE;
            const int tt = idx % DSV4_IDX_TILE;
            const float m = sM[cc][tt];
            const float r = m > 0.0f ? m : 0.0f;
            sAcc[cc][tt] += sW[tt] * r;
        }
    }
    __syncthreads();

    // write the tile to dst[n_comp, n_tokens]
    for (int idx = tid; idx < DSV4_IDX_TILE * DSV4_IDX_TILE; idx += 32) {
        const int cc = idx / DSV4_IDX_TILE;
        const int tt = idx % DSV4_IDX_TILE;
        const int64_t cg = c0 + cc;
        const int64_t tg = t0 + tt;
        if (cg < n_comp && tg < n_tokens) {
            *((float *) (dst + (uint64_t) cg * args.nb0 + (uint64_t) tg * args.nb1)) = sAcc[cc][tt];
        }
    }
#else
    GGML_UNUSED_VARS(args, k, q, weights, dst);
    NO_DEVICE_CODE;
#endif
}

// Fused split_sinkhorn + weighted_sum (one block per token): thread 0 computes the
// 24-value sinkhorn split (also written to the split dst so the later post/comb views
// stay valid), then all threads do the pre-weighted sum over n_embd*n_hc.
static __global__ void kernel_dsv4_hc_split_sinkhorn_ws(
        const ggml_cuda_kargs_dsv4_hc_split_sinkhorn sargs,
        const float * mixes,
        const float * scale,
        const float * base,
        float * split_dst,
        const ggml_cuda_kargs_dsv4_hc_weighted_sum wargs,
        const char * x,
        char * y) {
    __shared__ float s_pre[DSV4_HC_MAX];

    const int64_t t = blockIdx.x;  // token row
    if (t >= sargs.n_rows) {
        return;
    }

    if (threadIdx.x == 0) {
        float * out = split_dst + t * sargs.mix_hc;
        dsv4_hc_split_sinkhorn_one(sargs, mixes + t * sargs.mix_hc, scale, base, out);
        for (int h = 0; h < sargs.n_hc; ++h) {
            s_pre[h] = out[h];
        }
    }
    __syncthreads();

    for (int64_t d = threadIdx.x; d < wargs.n_embd; d += blockDim.x) {
        float acc = 0.0f;
        for (int64_t h = 0; h < wargs.n_hc; ++h) {
            acc += *((const float *) (x + d * wargs.nb_x0 + h * wargs.nb_x1 + t * wargs.nb_x2)) * s_pre[h];
        }
        *((float *) (y + d * wargs.nb0 + t * wargs.nb1)) = acc;
    }
}

} // namespace

bool ggml_cuda_op_dsv4_hc_split_sinkhorn_ws_fused(ggml_backend_cuda_context & ctx, ggml_tensor * split, ggml_tensor * ws) {
    const ggml_tensor * mixes = split->src[0];
    const ggml_tensor * scale = split->src[1];
    const ggml_tensor * base  = split->src[2];
    const ggml_tensor * x     = ws->src[0];

    if (mixes->type != GGML_TYPE_F32 || split->type != GGML_TYPE_F32 ||
        x->type != GGML_TYPE_F32 || ws->type != GGML_TYPE_F32) {
        return false;
    }

    const int32_t n_hc           = ggml_get_op_params_i32(split, 0);
    const int32_t sinkhorn_iters = ggml_get_op_params_i32(split, 1);
    const float eps              = ggml_get_op_params_f32(split, 2);

    if (n_hc <= 0 || n_hc > DSV4_HC_MAX) {
        return false;
    }

    const int64_t n_rows   = mixes->ne[1];
    const int64_t n_embd   = ws->ne[0];
    const int64_t n_tokens = ws->ne[1];
    if (n_tokens != n_rows) {
        return false;
    }

    const ggml_cuda_kargs_dsv4_hc_split_sinkhorn sargs = {
        /*.n_hc            =*/ n_hc,
        /*.sinkhorn_iters  =*/ sinkhorn_iters,
        /*.n_rows          =*/ n_rows,
        /*.mix_hc          =*/ mixes->ne[0],
        /*.nb01            =*/ mixes->nb[1],
        /*.nb1             =*/ split->nb[1],
        /*.eps             =*/ eps,
    };
    const ggml_cuda_kargs_dsv4_hc_weighted_sum wargs = {
        /*.n_embd  =*/ n_embd,
        /*.n_hc    =*/ x->ne[1],
        /*.n_tokens =*/ n_tokens,
        /*.nb_x0   =*/ x->nb[0],
        /*.nb_x1   =*/ x->nb[1],
        /*.nb_x2   =*/ x->nb[2],
        /*.nb_w0   =*/ 0,
        /*.nb_w1   =*/ 0,
        /*.nb0     =*/ ws->nb[0],
        /*.nb1     =*/ ws->nb[1],
    };

    {
        static const bool probe = getenv("DSV4_GRAPH_PROBE") != nullptr;
        static int fired = 0;
        if (probe && fired++ < 3) {
            fprintf(stderr, "hc fused FIRED (n_rows=%lld n_embd=%lld)\n", (long long) n_rows, (long long) n_embd);
        }
    }

    const cudaStream_t stream = ctx.stream();
    kernel_dsv4_hc_split_sinkhorn_ws<<<n_rows, 256, 0, stream>>>(
        sargs,
        (const float *) mixes->data, (const float *) scale->data, (const float *) base->data,
        (float *) split->data,
        wargs,
        (const char *) x->data, (char *) ws->data);

    return true;
}

bool ggml_cuda_op_dsv4_hc_split_sinkhorn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * src2 = dst->src[2];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(src2->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(src0->ne[2] == 1);
    GGML_ASSERT(src0->ne[3] == 1);

    const int32_t n_hc           = ggml_get_op_params_i32(dst, 0);
    const int32_t sinkhorn_iters = ggml_get_op_params_i32(dst, 1);
    const float eps              = ggml_get_op_params_f32(dst, 2);

    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];
    const int64_t ne00 = src0->ne[0];

    const int64_t n_rows = ne01 * ne02 * ne03;

    const float * mixes_d = (const float *) src0->data;
    const float * scale_d = (const float *) src1->data;
    const float * base_d  = (const float *) src2->data;
    float * dst_d = (float *) dst->data;

    const int nth = std::min<int64_t>(256, std::max<int64_t>(1, n_rows));
    const int n_tg = (n_rows + nth - 1) / nth;

    ggml_cuda_kargs_dsv4_hc_split_sinkhorn args = {
        /*.n_hc            =*/ n_hc,
        /*.sinkhorn_iters  =*/ sinkhorn_iters,
        /*.n_rows          =*/ n_rows,
        /*.mix_hc          =*/ ne00,
        /*.nb01            =*/ src0->nb[1],
        /*.nb1             =*/ dst->nb[1],
        /*.eps             =*/ eps,
    };

    const cudaStream_t stream = ctx.stream();

    kernel_dsv4_hc_split_sinkhorn<<<n_tg, nth, 0, stream>>>(args, mixes_d, scale_d, base_d, dst_d);

    return true;
}

bool ggml_cuda_op_dsv4_hc_expand(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_tensor * block_out = dst->src[0];
    ggml_tensor * residual  = dst->src[1];
    ggml_tensor * post      = dst->src[2];
    ggml_tensor * comb      = dst->src[3];

    GGML_ASSERT(block_out->type == GGML_TYPE_F32);
    GGML_ASSERT(residual->type  == GGML_TYPE_F32);
    GGML_ASSERT(post->type      == GGML_TYPE_F32);
    GGML_ASSERT(comb->type      == GGML_TYPE_F32);
    GGML_ASSERT(dst->type       == GGML_TYPE_F32);

    const int64_t ne0 = dst->ne[0];
    const int64_t ne1 = dst->ne[1];
    const int64_t ne2 = dst->ne[2];

    const int64_t n_elem = ne0 * ne1 * ne2;

    const int nth = std::min<int64_t>(256, std::max<int64_t>(1, n_elem));
    const int n_tg = (n_elem + nth - 1) / nth;

    ggml_cuda_kargs_dsv4_hc_expand args = {
        /*.n_embd    =*/ ne0,
        /*.n_hc      =*/ ne1,
        /*.n_tokens  =*/ ne2,
        /*.nb_block0 =*/ block_out->nb[0],
        /*.nb_block1 =*/ block_out->nb[1],
        /*.nb_res0   =*/ residual->nb[0],
        /*.nb_res1   =*/ residual->nb[1],
        /*.nb_res2   =*/ residual->nb[2],
        /*.nb_post0  =*/ post->nb[0],
        /*.nb_post1  =*/ post->nb[1],
        /*.nb_comb0  =*/ comb->nb[0],
        /*.nb_comb1  =*/ comb->nb[1],
        /*.nb_comb2  =*/ comb->nb[2],
        /*.nb0       =*/ dst->nb[0],
        /*.nb1       =*/ dst->nb[1],
        /*.nb2       =*/ dst->nb[2],
    };

    const cudaStream_t stream = ctx.stream();

    kernel_dsv4_hc_expand<<<n_tg, nth, 0, stream>>>(
        args,
        (const char *) block_out->data,
        (const char *) residual->data,
        (const char *) post->data,
        (const char *) comb->data,
        (char *) dst->data);

    return true;
}

bool ggml_cuda_op_dsv4_rope_tail_supported(void) {
    // Supported: F32 input/output
    return true;
}

bool ggml_cuda_op_dsv4_hc_split_sinkhorn_supported(void) {
    return true;
}

bool ggml_cuda_op_dsv4_hc_expand_supported(void) {
    return true;
}

bool ggml_cuda_op_dsv4_fp8_kv_quantize_supported(void) {
    // Supported: F32 input/output
    return true;
}

bool ggml_cuda_op_dsv4_fp8_kv_quantize(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    const int32_t n_rot = ggml_get_op_params_i32(dst, 0);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];

    const int64_t n_rows = ne01 * ne02 * ne03;

    ggml_cuda_kargs_dsv4_fp8_kv_quantize args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.ne03 =*/ ne03,
        /*.nb00 =*/ src0->nb[0],
        /*.nb01 =*/ src0->nb[1],
        /*.nb02 =*/ src0->nb[2],
        /*.nb03 =*/ src0->nb[3],
        /*.nb0  =*/ dst->nb[0],
        /*.nb1  =*/ dst->nb[1],
        /*.nb2  =*/ dst->nb[2],
        /*.nb3  =*/ dst->nb[3],
        /*.n_rot =*/ n_rot,
    };

    const cudaStream_t stream = ctx.stream();

    kernel_dsv4_fp8_kv_quantize<<<n_rows, 64, 0, stream>>>(
        args,
        (const char *) src0->data,
        (char *) dst->data);

    return true;
}

bool ggml_cuda_op_dsv4_hc_weighted_sum_supported(void) {
    return true;
}

bool ggml_cuda_op_dsv4_hc_weighted_sum(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x       = dst->src[0];
    const ggml_tensor * weights = dst->src[1];

    GGML_ASSERT(x->type       == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type     == GGML_TYPE_F32);
    GGML_ASSERT(x->ne[3]       == 1);
    GGML_ASSERT(weights->ne[2] == 1);
    GGML_ASSERT(weights->ne[3] == 1);
    GGML_ASSERT(dst->ne[2]     == 1);
    GGML_ASSERT(dst->ne[3]     == 1);

    const int64_t n_embd   = dst->ne[0];
    const int64_t n_hc     = x->ne[1];
    const int64_t n_tokens = dst->ne[1];
    const int64_t n_elem   = n_embd * n_tokens;

    const int nth = std::min<int64_t>(256, std::max<int64_t>(1, n_elem));
    const int n_tg = (n_elem + nth - 1) / nth;

    ggml_cuda_kargs_dsv4_hc_weighted_sum args = {
        /*.n_embd  =*/ n_embd,
        /*.n_hc    =*/ n_hc,
        /*.n_tokens =*/ n_tokens,
        /*.nb_x0   =*/ x->nb[0],
        /*.nb_x1   =*/ x->nb[1],
        /*.nb_x2   =*/ x->nb[2],
        /*.nb_w0   =*/ weights->nb[0],
        /*.nb_w1   =*/ weights->nb[1],
        /*.nb0     =*/ dst->nb[0],
        /*.nb1     =*/ dst->nb[1],
    };

    const cudaStream_t stream = ctx.stream();

    kernel_dsv4_hc_weighted_sum<<<n_tg, nth, 0, stream>>>(
        args,
        (const char *) x->data,
        (const char *) weights->data,
        (char *) dst->data);

    return true;
}

bool ggml_cuda_op_dsv4_indexer_logits_supported(const ggml_tensor * dst) {
    const ggml_tensor * k = dst->src[0];
    const ggml_tensor * q = dst->src[1];
    const ggml_tensor * w = dst->src[2];
    if (!k || !q || !w) return false;
    if (w->type != GGML_TYPE_F32) return false;
    if (dst->type != GGML_TYPE_F32) return false;
    // only the head_dim==128 shape is instantiated (DSV4 indexer key_length)
    if (k->ne[0] != 128 || q->ne[0] != 128) return false;
    const bool kok = k->type == GGML_TYPE_F16 || k->type == GGML_TYPE_F32;
    const bool qok = q->type == GGML_TYPE_F16 || q->type == GGML_TYPE_F32;
    return kok && qok;
}

bool ggml_cuda_op_dsv4_indexer_logits(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * k = dst->src[0];   // [head_dim, n_comp, 1]
    const ggml_tensor * q = dst->src[1];   // [head_dim, n_tokens, n_head]
    const ggml_tensor * w = dst->src[2];   // [n_head, n_tokens]

    GGML_ASSERT(w->type   == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const int64_t head_dim = k->ne[0];
    const int64_t n_comp   = k->ne[1];
    const int64_t n_tokens = q->ne[1];
    const int64_t n_head   = q->ne[2];
    GGML_ASSERT(head_dim == 128);

    ggml_cuda_kargs_dsv4_indexer_logits args = {
        /*.head_dim =*/ head_dim,
        /*.n_comp   =*/ n_comp,
        /*.n_tokens =*/ n_tokens,
        /*.n_head   =*/ n_head,
        /*.nb_k0    =*/ k->nb[0],
        /*.nb_k1    =*/ k->nb[1],
        /*.nb_q0    =*/ q->nb[0],
        /*.nb_q1    =*/ q->nb[1],
        /*.nb_q2    =*/ q->nb[2],
        /*.nb_w0    =*/ w->nb[0],
        /*.nb_w1    =*/ w->nb[1],
        /*.nb0      =*/ dst->nb[0],
        /*.nb1      =*/ dst->nb[1],
    };

    const cudaStream_t stream = ctx.stream();

    const bool kf16 = k->type == GGML_TYPE_F16;
    const bool qf16 = q->type == GGML_TYPE_F16;

    // [DSV4_INDEXER_WMMA] tensor-core (bf16/F32-accumulate) dot for the indexer logits.
    //
    // The WMMA kernel tiles 16(comp) x 16(token) with ONE WARP per tile. That is right for prefill
    // and wrong for decode: at n_tokens == 1 it discards 15/16 of every tile and the grid collapses
    // to (1, n_comp/16) single-warp blocks. The CUDA-core block-reduce kernel launches
    // (n_tokens, n_comp) blocks of 128 lanes instead, which at decode is 16x the parallelism.
    // Measured end-to-end, 2x GB10, plain decode, 24.7k ctx: WMMA 8.26 t/s vs CUDA-core 10.11 t/s.
    // So pick by n_tokens. DSV4_INDEXER_NO_WMMA=1 forces the CUDA-core path everywhere (A/B).
    static const bool no_wmma = getenv("DSV4_INDEXER_NO_WMMA") != nullptr;

    if (!no_wmma && n_tokens >= DSV4_IDX_TILE) {
        // one warp per 16(comp) x 16(token) output tile
        const dim3 grid((unsigned) ((n_tokens + DSV4_IDX_TILE - 1) / DSV4_IDX_TILE),
                        (unsigned) ((n_comp   + DSV4_IDX_TILE - 1) / DSV4_IDX_TILE), 1);
        const dim3 block(32, 1, 1);
        auto launch_wmma = [&](auto kt_tag, auto qt_tag) {
            using kT = decltype(kt_tag);
            using qT = decltype(qt_tag);
            kernel_dsv4_indexer_logits_wmma<kT, qT><<<grid, block, 0, stream>>>(
                args, (const char *) k->data, (const char *) q->data,
                (const char *) w->data, (char *) dst->data);
        };
        if (kf16 && qf16)       launch_wmma(half{},  half{});
        else if (kf16 && !qf16) launch_wmma(half{},  float{});
        else if (!kf16 && qf16) launch_wmma(float{}, half{});
        else                    launch_wmma(float{}, float{});
        return true;
    }

    // CUDA-core fallback (v1): one 128-lane block per (comp,token).
    const dim3 grid((unsigned) n_tokens, (unsigned) n_comp, 1);
    const dim3 block(128, 1, 1);
    auto launch = [&](auto kt_tag, auto qt_tag) {
        using kT = decltype(kt_tag);
        using qT = decltype(qt_tag);
        kernel_dsv4_indexer_logits<kT, qT, 128><<<grid, block, 0, stream>>>(
            args, (const char *) k->data, (const char *) q->data,
            (const char *) w->data, (char *) dst->data);
    };
    if (kf16 && qf16)       launch(half{},  half{});
    else if (kf16 && !qf16) launch(half{},  float{});
    else if (!kf16 && qf16) launch(float{}, half{});
    else                    launch(float{}, float{});

    return true;
}

// [Fusion 3] see kernel_dsv4_norm_rope_f32
bool ggml_cuda_op_dsv4_norm_rope_supported(const ggml_tensor * dst) {
    const ggml_tensor * a   = dst->src[0];
    const ggml_tensor * pos = dst->src[1];
    const ggml_tensor * w   = dst->src[2];
    if (!a || !pos) return false;
    if (a->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) return false;
    if (pos->type != GGML_TYPE_I32) return false;
    if (w && w->type != GGML_TYPE_F32) return false;   // AUX_BF16 could retype a norm weight
    // the row lives in shared memory for the whole kernel
    if (a->ne[0] <= 0 || a->ne[0] > 8192) return false;
    return true;
}

bool ggml_cuda_op_dsv4_norm_rope(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * a   = dst->src[0];
    const ggml_tensor * pos = dst->src[1];
    const ggml_tensor * w   = dst->src[2];

    GGML_ASSERT(a->type   == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(pos->type == GGML_TYPE_I32);

    const int32_t * p = (const int32_t *) dst->op_params;

    ggml_cuda_kargs_dsv4_norm_rope args = {};
    args.ne00 = a->ne[0]; args.ne01 = a->ne[1]; args.ne02 = a->ne[2]; args.ne03 = a->ne[3];
    args.nb00 = a->nb[0]; args.nb01 = a->nb[1]; args.nb02 = a->nb[2]; args.nb03 = a->nb[3];
    args.nb0  = dst->nb[0]; args.nb1 = dst->nb[1]; args.nb2 = dst->nb[2]; args.nb3 = dst->nb[3];
    args.n_dims     = p[0];
    args.mode       = p[1];
    args.n_ctx_orig = p[2];
    args.fp8_kv     = p[3];
    memcpy(&args.eps,         p +  4, sizeof(float));
    memcpy(&args.freq_base,   p +  5, sizeof(float));
    memcpy(&args.freq_scale,  p +  6, sizeof(float));
    memcpy(&args.ext_factor,  p +  7, sizeof(float));
    memcpy(&args.attn_factor, p +  8, sizeof(float));
    memcpy(&args.beta_fast,   p +  9, sizeof(float));
    memcpy(&args.beta_slow,   p + 10, sizeof(float));
    args.has_w = (w != nullptr);

    const int64_t n_rows = args.ne01 * args.ne02 * args.ne03;
    if (n_rows == 0) {
        return true;
    }

    const size_t smem = (size_t) args.ne00 * sizeof(float);

    kernel_dsv4_norm_rope_f32<<<(unsigned) n_rows, DSV4_NR_NT, smem, ctx.stream()>>>(
        args,
        (const char *) a->data,
        (const char *) pos->data,
        w ? (const char *) w->data : nullptr,
        (char *) dst->data);

    return true;
}

bool ggml_cuda_op_dsv4_rope_tail(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * src2 = dst->src[2];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    const int32_t n_dims     = ggml_get_op_params_i32(dst, 0);
    const int32_t mode       = ggml_get_op_params_i32(dst, 1);
    const int32_t n_ctx_orig = ggml_get_op_params_i32(dst, 2);
    const int32_t inverse    = ggml_get_op_params_i32(dst, 3);

    float freq_base;
    float freq_scale;
    float ext_factor;
    float attn_factor;
    float beta_fast;
    float beta_slow;

    memcpy(&freq_base,   (const int32_t *) dst->op_params + 4, sizeof(float));
    memcpy(&freq_scale,  (const int32_t *) dst->op_params + 5, sizeof(float));
    memcpy(&ext_factor,  (const int32_t *) dst->op_params + 6, sizeof(float));
    memcpy(&attn_factor, (const int32_t *) dst->op_params + 7, sizeof(float));
    memcpy(&beta_fast,   (const int32_t *) dst->op_params + 8, sizeof(float));
    memcpy(&beta_slow,   (const int32_t *) dst->op_params + 9, sizeof(float));

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];

    const int nth = std::min<int64_t>(256, std::max<int64_t>(1, ne00));

    ggml_cuda_kargs_dsv4_rope_tail args = {
        /*.ne00        =*/ ne00,
        /*.ne01        =*/ ne01,
        /*.ne02        =*/ ne02,
        /*.ne03        =*/ ne03,
        /*.nb00        =*/ src0->nb[0],
        /*.nb01        =*/ src0->nb[1],
        /*.nb02        =*/ src0->nb[2],
        /*.nb03        =*/ src0->nb[3],
        /*.nb0         =*/ dst->nb[0],
        /*.nb1         =*/ dst->nb[1],
        /*.nb2         =*/ dst->nb[2],
        /*.nb3         =*/ dst->nb[3],
        /*.n_dims      =*/ n_dims,
        /*.mode        =*/ mode,
        /*.n_ctx_orig  =*/ n_ctx_orig,
        /*.inverse     =*/ inverse,
        /*.freq_base   =*/ freq_base,
        /*.freq_scale  =*/ freq_scale,
        /*.ext_factor  =*/ ext_factor,
        /*.attn_factor =*/ attn_factor,
        /*.beta_fast   =*/ beta_fast,
        /*.beta_slow   =*/ beta_slow,
        /*.src2        =*/ src2 != nullptr,
    };

    const cudaStream_t stream = ctx.stream();

    dim3 grid(ne03, ne02, ne01);

    kernel_dsv4_rope_tail_f32<<<grid, nth, 0, stream>>>(
        args,
        (const char *) src0->data,
        (const char *) src1->data,
        src2 ? (const char *) src2->data : (const char *) src0->data,
        (char *) dst->data);

    return true;
}