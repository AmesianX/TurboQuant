#include "convert.cuh"
#include "dequantize.cuh"

#include <cstdint>

#define CUDA_Q8_0_NE_ALIGN 2048

template <int qk, int qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
static __global__ void dequantize_block(const void * __restrict__ vx, dst_t * __restrict__ y,
        const int64_t ne00, const int64_t ne01,
        const int64_t ne0203, const uint3 ne02,
        const int64_t s01, const int64_t s02, const int64_t s03) {
    const int64_t i00 = 2 * (int64_t(blockDim.x)*blockIdx.x + threadIdx.x);

    if (i00 >= ne00) {
        return;
    }

    for (int64_t i01 = blockIdx.y; i01 < ne01; i01 += gridDim.y) {
        for (int64_t i0203 = blockIdx.z; i0203 < ne0203; i0203 += gridDim.z) {
            const uint2 dm = fast_div_modulo((uint32_t)i0203, ne02);
            const int64_t i02 = dm.y;
            const int64_t i03 = dm.x;

            const int64_t ibx0 = i03*s03 + i02*s02 + i01*s01;

            const int64_t ib = ibx0 + i00/qk; // block index
            const int64_t iqs = (i00%qk)/qr; // quant index
            const int64_t iybs = i00 - i00%qk; // y block start index
            const int64_t y_offset = qr == 1 ? 1 : qk/2;

            // dequantize
            float2 v;
            dequantize_kernel(vx, ib, iqs, v);

            const int64_t iy0 = (i0203*ne01 + i01)*ne00 + iybs + iqs;
            y[iy0 + 0]        = ggml_cuda_cast<dst_t>(v.x);
            y[iy0 + y_offset] = ggml_cuda_cast<dst_t>(v.y);
        }
    }
}

template <bool need_check>
static __global__ void dequantize_block_q8_0_f16(const void * __restrict__ vx, half * __restrict__ y, const int64_t k) {
#if __CUDA_ARCH__ >= GGML_CUDA_CC_PASCAL
    constexpr int nint = CUDA_Q8_0_NE_ALIGN/sizeof(int) + WARP_SIZE;

    const int64_t   i0 = CUDA_Q8_0_NE_ALIGN*blockIdx.x;
    const int * x0 = ((int *) vx) + blockIdx.x * nint;
    half2 * y2 = (half2 *) (y + i0);

    __shared__ int vals[nint];

#pragma unroll
    for (int ix0 = 0; ix0 < nint; ix0 += WARP_SIZE) {
        if (need_check && i0*sizeof(block_q8_0)/QK8_0 + sizeof(int)*(ix0 + threadIdx.x) >= k*sizeof(block_q8_0)/QK8_0) {
            break;
        }

        const int ix = ix0 + threadIdx.x;
        vals[ix] = x0[ix];
    }

    __syncthreads();

#pragma unroll
    for (int iy = 0; iy < CUDA_Q8_0_NE_ALIGN; iy += 2*WARP_SIZE) {
        if (need_check && i0 + iy + 2*threadIdx.x >= k) {
            return;
        }

        const half * b0 = ((const half  *) vals) + (sizeof(block_q8_0)/sizeof(half)) * ((iy + 2*threadIdx.x)/QK8_0);
        const half    d = *b0;
        const char2  qs = ((const char2 *) (b0 + 1))[threadIdx.x % (QK8_0/2)];

        y2[iy/2 + threadIdx.x] = __hmul2(make_half2(qs.x, qs.y), __half2half2(d));
    }
#else
    GGML_UNUSED_VARS(vx, y, k);
    NO_DEVICE_CODE;
#endif // __CUDA_ARCH__ >= GGML_CUDA_CC_PASCAL
}

template<typename dst_t>
static __global__ void dequantize_block_q4_0(const void * __restrict__ vx, dst_t * __restrict__ yy, int nb32) {

    const int64_t i = blockIdx.x;

    // assume 32 threads
    const int64_t tid = threadIdx.x;
    const int64_t il  = tid/8;
    const int64_t ir  = tid%8;
    const int64_t ib = 8*i + ir;
    if (ib >= nb32) {
        return;
    }

    dst_t * y = yy + 256*i + 32*ir + 4*il;

    const block_q4_0 * x = (const block_q4_0 *)vx + ib;
    const float d = __half2float(x->d);
    const float dm = -8*d;

    const uint8_t * q = x->qs + 4*il;

    for (int l = 0; l < 4; ++l) {
        y[l+ 0] = d * (q[l] & 0xF) + dm;
        y[l+16] = d * (q[l] >>  4) + dm;
    }
}

template<typename dst_t>
static __global__ void dequantize_block_q4_1(const void * __restrict__ vx, dst_t * __restrict__ yy, int nb32) {

    const int64_t i = blockIdx.x;

    // assume 32 threads
    const int64_t tid = threadIdx.x;
    const int64_t il  = tid/8;
    const int64_t ir  = tid%8;
    const int64_t ib = 8*i + ir;
    if (ib >= nb32) {
        return;
    }

    dst_t * y = yy + 256*i + 32*ir + 4*il;

    const block_q4_1 * x = (const block_q4_1 *)vx + ib;
    const float2 d = __half22float2(x->dm);

    const uint8_t * q = x->qs + 4*il;

    for (int l = 0; l < 4; ++l) {
        y[l+ 0] = d.x * (q[l] & 0xF) + d.y;
        y[l+16] = d.x * (q[l] >>  4) + d.y;
    }
}

//================================== k-quants

template<typename dst_t>
static __global__ void dequantize_block_q2_K(const void * __restrict__ vx, dst_t * __restrict__ yy) {

    const int64_t i   = blockIdx.x;
    const block_q2_K * x = (const block_q2_K *) vx;

    const int64_t tid = threadIdx.x;
    const int64_t n   = tid/32;
    const int64_t l   = tid - 32*n;
    const int64_t is  = 8*n + l/16;

    const uint8_t q = x[i].qs[32*n + l];
    dst_t * y = yy + i*QK_K + 128*n;

    float dall = __low2half(x[i].dm);
    float dmin = __high2half(x[i].dm);
    y[l+ 0] = dall * (x[i].scales[is+0] & 0xF) * ((q >> 0) & 3) - dmin * (x[i].scales[is+0] >> 4);
    y[l+32] = dall * (x[i].scales[is+2] & 0xF) * ((q >> 2) & 3) - dmin * (x[i].scales[is+2] >> 4);
    y[l+64] = dall * (x[i].scales[is+4] & 0xF) * ((q >> 4) & 3) - dmin * (x[i].scales[is+4] >> 4);
    y[l+96] = dall * (x[i].scales[is+6] & 0xF) * ((q >> 6) & 3) - dmin * (x[i].scales[is+6] >> 4);
}

template<typename dst_t>
static __global__ void dequantize_block_q3_K(const void * __restrict__ vx, dst_t * __restrict__ yy) {

    const int64_t i = blockIdx.x;
    const block_q3_K * x = (const block_q3_K *) vx;

    const int64_t r = threadIdx.x/4;
    const int64_t tid = r/2;
    const int64_t is0 = r%2;
    const int64_t l0 = 16*is0 + 4*(threadIdx.x%4);
    const int64_t n = tid / 4;
    const int64_t j = tid - 4*n;

    uint8_t m = 1 << (4*n + j);
    int64_t is = 8*n + 2*j + is0;
    int shift = 2*j;

    int8_t us = is <  4 ? (x[i].scales[is-0] & 0xF) | (((x[i].scales[is+8] >> 0) & 3) << 4) :
                is <  8 ? (x[i].scales[is-0] & 0xF) | (((x[i].scales[is+4] >> 2) & 3) << 4) :
                is < 12 ? (x[i].scales[is-8] >>  4) | (((x[i].scales[is+0] >> 4) & 3) << 4) :
                          (x[i].scales[is-8] >>  4) | (((x[i].scales[is-4] >> 6) & 3) << 4);
    float d_all = x[i].d;
    float dl = d_all * (us - 32);

    dst_t * y = yy + i*QK_K + 128*n + 32*j;
    const uint8_t * q = x[i].qs + 32*n;
    const uint8_t * hm = x[i].hmask;

    for (int l = l0; l < l0+4; ++l) y[l] = dl * ((int8_t)((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4));
}

static inline __device__ void get_scale_min_k4(int j, const uint8_t * q, uint8_t & d, uint8_t & m) {
    if (j < 4) {
        d = q[j] & 63; m = q[j + 4] & 63;
    } else {
        d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}

template<typename dst_t>
static __global__ void dequantize_block_q4_K(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const block_q4_K * x = (const block_q4_K *) vx;

    const int64_t i = blockIdx.x;

    // assume 32 threads
    const int64_t tid = threadIdx.x;
    const int64_t il  = tid/8;
    const int64_t ir  = tid%8;
    const int64_t is  = 2*il;
    const int64_t n   = 4;

    dst_t * y = yy + i*QK_K + 64*il + n*ir;

    const float dall = __low2half(x[i].dm);
    const float dmin = __high2half(x[i].dm);

    const uint8_t * q = x[i].qs + 32*il + n*ir;

    uint8_t sc, m;
    get_scale_min_k4(is + 0, x[i].scales, sc, m);
    const float d1 = dall * sc; const float m1 = dmin * m;
    get_scale_min_k4(is + 1, x[i].scales, sc, m);
    const float d2 = dall * sc; const float m2 = dmin * m;
    for (int l = 0; l < n; ++l) {
        y[l + 0] = d1 * (q[l] & 0xF) - m1;
        y[l +32] = d2 * (q[l] >>  4) - m2;
    }
}

template<typename dst_t>
static __global__ void dequantize_block_q5_K(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const block_q5_K * x = (const block_q5_K *) vx;

    const int64_t i = blockIdx.x;

    // assume 64 threads - this is very slightly better than the one below
    const int64_t tid = threadIdx.x;
    const int64_t il  = tid/16;   // il is in 0...3
    const int64_t ir  = tid%16;   // ir is in 0...15
    const int64_t is  = 2*il;     // is is in 0...6

    dst_t * y = yy + i*QK_K + 64*il + 2*ir;

    const float dall = __low2half(x[i].dm);
    const float dmin = __high2half(x[i].dm);

    const uint8_t * ql = x[i].qs + 32*il + 2*ir;
    const uint8_t * qh = x[i].qh + 2*ir;

    uint8_t sc, m;
    get_scale_min_k4(is + 0, x[i].scales, sc, m);
    const float d1 = dall * sc; const float m1 = dmin * m;
    get_scale_min_k4(is + 1, x[i].scales, sc, m);
    const float d2 = dall * sc; const float m2 = dmin * m;

    uint8_t   hm  = 1 << (2*il);
    y[ 0] = d1 * ((ql[ 0] & 0xF) + (qh[ 0] & hm ? 16 : 0)) - m1;
    y[ 1] = d1 * ((ql[ 1] & 0xF) + (qh[ 1] & hm ? 16 : 0)) - m1;
    hm <<= 1;
    y[32] = d2 * ((ql[ 0] >>  4) + (qh[ 0] & hm ? 16 : 0)) - m2;
    y[33] = d2 * ((ql[ 1] >>  4) + (qh[ 1] & hm ? 16 : 0)) - m2;
}

template<typename dst_t>
static __global__ void dequantize_block_q6_K(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const block_q6_K * x = (const block_q6_K *) vx;

    const int64_t i = blockIdx.x;

    // assume 64 threads - this is very slightly better than the one below
    const int64_t tid = threadIdx.x;
    const int64_t ip  = tid/32;   // ip is 0 or 1
    const int64_t il  = tid - 32*ip; // 0...32
    const int64_t is  = 8*ip + il/16;

    dst_t * y = yy + i*QK_K + 128*ip + il;

    const float d = x[i].d;

    const uint8_t * ql = x[i].ql + 64*ip + il;
    const uint8_t   qh = x[i].qh[32*ip + il];
    const int8_t  * sc = x[i].scales + is;

    y[ 0] = d * sc[0] * ((int8_t)((ql[ 0] & 0xF) | (((qh >> 0) & 3) << 4)) - 32);
    y[32] = d * sc[2] * ((int8_t)((ql[32] & 0xF) | (((qh >> 2) & 3) << 4)) - 32);
    y[64] = d * sc[4] * ((int8_t)((ql[ 0]  >> 4) | (((qh >> 4) & 3) << 4)) - 32);
    y[96] = d * sc[6] * ((int8_t)((ql[32]  >> 4) | (((qh >> 6) & 3) << 4)) - 32);
}

template<typename dst_t>
static __global__ void dequantize_block_iq2_xxs(const void * __restrict__ vx, dst_t * __restrict__ yy) {

    const int64_t i   = blockIdx.x;
    const block_iq2_xxs * x = (const block_iq2_xxs  *) vx;

    const int64_t tid = threadIdx.x;
    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    const uint16_t * q2 = x[i].qs + 4*ib;
    const uint8_t  * aux8 = (const uint8_t *)q2;
    const uint8_t  * grid = (const uint8_t *)(iq2xxs_grid + aux8[il]);
    const uint32_t aux32 = q2[2] | (q2[3] << 16);
    const float d = (float)x[i].d * (0.5f + (aux32 >> 28)) * 0.25f;
    const uint8_t signs = ksigns_iq2xs[(aux32 >> 7*il) & 127];
    for (int j = 0; j < 8; ++j) y[j] = d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
}

template<typename dst_t>
static __global__ void dequantize_block_iq2_xs(const void * __restrict__ vx, dst_t * __restrict__ yy) {

    const int64_t i   = blockIdx.x;
    const block_iq2_xs * x = (const block_iq2_xs *) vx;

    const int64_t tid = threadIdx.x;
    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    const uint16_t * q2 = x[i].qs + 4*ib;
    const uint8_t  * grid = (const uint8_t *)(iq2xs_grid + (q2[il] & 511));
    const float d = (float)x[i].d * (0.5f + ((x[i].scales[ib] >> 4*(il/2)) & 0xf)) * 0.25f;
    const uint8_t signs = ksigns_iq2xs[q2[il] >> 9];
    for (int j = 0; j < 8; ++j) y[j] = d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
}

template<typename dst_t>
static __global__ void dequantize_block_iq2_s(const void * __restrict__ vx, dst_t * __restrict__ yy) {

    const int64_t i   = blockIdx.x;
    const block_iq2_s * x = (const block_iq2_s *) vx;

    const int64_t tid = threadIdx.x;
    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    const uint8_t * grid = (const uint8_t *)(iq2s_grid + (x[i].qs[4*ib+il] | ((x[i].qh[ib] << (8-2*il)) & 0x300)));
    const float d = (float)x[i].d * (0.5f + ((x[i].scales[ib] >> 4*(il/2)) & 0xf)) * 0.25f;
    const uint8_t signs = x[i].qs[QK_K/8+4*ib+il];
    for (int j = 0; j < 8; ++j) y[j] = d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
}

template<typename dst_t>
static __global__ void dequantize_block_iq3_xxs(const void * __restrict__ vx, dst_t * __restrict__ yy) {

    const int64_t i   = blockIdx.x;
    const block_iq3_xxs * x = (const block_iq3_xxs  *) vx;

    const int64_t tid = threadIdx.x;
    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    const uint8_t  * q3 = x[i].qs + 8*ib;
    const uint16_t * gas = (const uint16_t *)(x[i].qs + QK_K/4) + 2*ib;
    const uint8_t  * grid1 = (const uint8_t *)(iq3xxs_grid + q3[2*il+0]);
    const uint8_t  * grid2 = (const uint8_t *)(iq3xxs_grid + q3[2*il+1]);
    const uint32_t aux32 = gas[0] | (gas[1] << 16);
    const float d = (float)x[i].d * (0.5f + (aux32 >> 28)) * 0.5f;
    const uint8_t signs = ksigns_iq2xs[(aux32 >> 7*il) & 127];
    for (int j = 0; j < 4; ++j) {
        y[j+0] = d * grid1[j] * (signs & kmask_iq2xs[j+0] ? -1.f : 1.f);
        y[j+4] = d * grid2[j] * (signs & kmask_iq2xs[j+4] ? -1.f : 1.f);
    }
}

template<typename dst_t>
static __global__ void dequantize_block_iq3_s(const void * __restrict__ vx, dst_t * __restrict__ yy) {

    const int64_t i   = blockIdx.x;
    const block_iq3_s * x = (const block_iq3_s *) vx;

    const int64_t tid = threadIdx.x;
    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    const uint8_t * qs = x[i].qs + 8*ib;
    const uint8_t * grid1 = (const uint8_t *)(iq3s_grid + (qs[2*il+0] | ((x[i].qh[ib] << (8-2*il)) & 256)));
    const uint8_t * grid2 = (const uint8_t *)(iq3s_grid + (qs[2*il+1] | ((x[i].qh[ib] << (7-2*il)) & 256)));
    const float d = (float)x[i].d * (1 + 2*((x[i].scales[ib/2] >> 4*(ib%2)) & 0xf));
    const uint8_t signs = x[i].signs[4*ib + il];
    for (int j = 0; j < 4; ++j) {
        y[j+0] = d * grid1[j] * (signs & kmask_iq2xs[j+0] ? -1.f : 1.f);
        y[j+4] = d * grid2[j] * (signs & kmask_iq2xs[j+4] ? -1.f : 1.f);
    }
}

template<typename dst_t>
static __global__ void dequantize_block_iq1_s(const void * __restrict__ vx, dst_t * __restrict__ yy) {

    const int64_t i   = blockIdx.x;
    const block_iq1_s * x = (const block_iq1_s  *) vx;

    const int64_t tid = threadIdx.x;
    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    const float delta = x[i].qh[ib] & 0x8000 ? -1 - IQ1S_DELTA : -1 + IQ1S_DELTA;
    const float d = (float)x[i].d * (2*((x[i].qh[ib] >> 12) & 7) + 1);
    uint32_t grid32[2]; const int8_t * q = (const int8_t *)grid32;
    grid32[0] = iq1s_grid_gpu[x[i].qs[4*ib+il] | (((x[i].qh[ib] >> 3*il) & 7) << 8)];
    grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0f;
    grid32[0] &= 0x0f0f0f0f;
    for (int j = 0; j < 8; ++j) {
        y[j] = d * (q[j] + delta);
    }
}

template<typename dst_t>
static __global__ void dequantize_block_iq1_m(const void * __restrict__ vx, dst_t * __restrict__ yy) {

    const int64_t i   = blockIdx.x;
    const block_iq1_m * x = (const block_iq1_m  *) vx;

    const int64_t tid = threadIdx.x;
    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    const uint16_t * sc = (const uint16_t *)x[i].scales;
    iq1m_scale_t scale;
    scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
    const int64_t ib16 = 2*ib + il/2; // sc[ib16/4] >> 3*(ib16%4) -> sc[ib/2] >> 3*((2*ib+il/2)%4);
    const float d = (float)scale.f16 * (2*((sc[ib16/4] >> 3*(ib16%4)) & 0x7) + 1);
    const float delta = x[i].qh[2*ib+il/2] & (0x08 << 4*(il%2)) ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA;
    uint32_t grid32[2]; const int8_t * q = (const int8_t *)grid32;
    grid32[0] = iq1s_grid_gpu[x[i].qs[4*ib+il] | (((x[i].qh[2*ib+il/2] >> 4*(il%2)) & 7) << 8)];
    grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0f;
    grid32[0] &= 0x0f0f0f0f;
    for (int j = 0; j < 8; ++j) {
        y[j] = d * (q[j] + delta);
    }
}

template<typename dst_t>
static __global__ void dequantize_block_iq4_nl(const void * __restrict__ vx, dst_t * __restrict__ yy) {

    const int64_t i   = blockIdx.x;
    const block_iq4_nl * x = (const block_iq4_nl *) vx + i*(QK_K/QK4_NL);

    const int64_t tid = threadIdx.x;
    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 4*il;
    const uint8_t  * q4 = x[ib].qs + 4*il;
    const float d = (float)x[ib].d;
    for (int j = 0; j < 4; ++j) {
        y[j+ 0] = d * kvalues_iq4nl[q4[j] & 0xf];
        y[j+16] = d * kvalues_iq4nl[q4[j] >>  4];
    }
}

template<typename dst_t>
static __global__ void dequantize_block_iq4_xs(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t i   = blockIdx.x;
    const block_iq4_xs * x = (const block_iq4_xs *)vx;

    const int64_t tid = threadIdx.x;
    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 4*il;
    const uint8_t  * q4 = x[i].qs + 16*ib + 4*il;
    const float d = (float)x[i].d * ((((x[i].scales_l[ib/2] >> 4*(ib%2)) & 0xf) | (((x[i].scales_h >> 2*ib) & 3) << 4)) - 32);
    for (int j = 0; j < 4; ++j) {
        y[j+ 0] = d * kvalues_iq4nl[q4[j] & 0xf];
        y[j+16] = d * kvalues_iq4nl[q4[j] >>  4];
    }
}

template<typename dst_t>
static __global__ void dequantize_block_mxfp4(const void * __restrict__ vx, dst_t * __restrict__ yy) {

    const int64_t i   = blockIdx.x;
    const block_mxfp4 * x = (const block_mxfp4 *) vx + i*(QK_K/QK_MXFP4);

    const int64_t tid = threadIdx.x;
    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 4*il;
    const uint8_t  * q4 = x[ib].qs + 4*il;
    const float d = ggml_cuda_e8m0_to_fp32(x[ib].e);
    for (int j = 0; j < 4; ++j) {
        y[j+ 0] = d * kvalues_mxfp4[q4[j] & 0xf]*0.5f;
        y[j+16] = d * kvalues_mxfp4[q4[j] >>  4]*0.5f;
    }
}

template <int qk, int qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
static void dequantize_block_cuda(const void * vx, dst_t * y,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t s01, const int64_t s02, const int64_t s03, cudaStream_t stream) {
    const int64_t ne0203 = ne02*ne03;
    const uint3 ne02_fdv = init_fastdiv_values(ne02);
    const dim3 num_blocks((ne00 + 2*CUDA_DEQUANTIZE_BLOCK_SIZE - 1) / (2*CUDA_DEQUANTIZE_BLOCK_SIZE), (int)std::min(ne01, (int64_t)65535), (int)std::min(ne0203, (int64_t)65535));
    dequantize_block<qk, qr, dequantize_kernel><<<num_blocks, CUDA_DEQUANTIZE_BLOCK_SIZE, 0, stream>>>
        (vx, y, ne00, ne01, ne0203, ne02_fdv, s01, s02, s03);
}

template <int qk, int qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
static void dequantize_block_cont_cuda(const void * __restrict__ vx, dst_t * __restrict__ y, const int64_t k, cudaStream_t stream) {
    dequantize_block_cuda<qk, qr, dequantize_kernel, dst_t>(vx, y, k, 1, 1, 1, k/qk, k/qk, k/qk, stream);
}

static void dequantize_block_q8_0_f16_cuda(const void * __restrict__ vx, half * __restrict__ y, const int64_t k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_Q8_0_NE_ALIGN - 1) / CUDA_Q8_0_NE_ALIGN;
    if (k % CUDA_Q8_0_NE_ALIGN == 0) {
        const bool need_check = false;
        dequantize_block_q8_0_f16<need_check><<<num_blocks, WARP_SIZE, 0, stream>>>(vx, y, k);
    } else {
        const bool need_check = true;
        dequantize_block_q8_0_f16<need_check><<<num_blocks, WARP_SIZE, 0, stream>>>(vx, y, k);
    }
}

template<typename dst_t>
static void dequantize_row_q2_K_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_q2_K<<<nb, 64, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_q3_K_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_q3_K<<<nb, 64, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_q4_0_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb32 = k / 32;
    const int nb = (k + 255) / 256;
    dequantize_block_q4_0<<<nb, 32, 0, stream>>>(vx, y, nb32);
}

template<typename dst_t>
static void dequantize_row_q4_1_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb32 = k / 32;
    const int nb = (k + 255) / 256;
    dequantize_block_q4_1<<<nb, 32, 0, stream>>>(vx, y, nb32);
}

template<typename dst_t>
static void dequantize_row_q4_K_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_q4_K<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_q5_K_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_q5_K<<<nb, 64, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_q6_K_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_q6_K<<<nb, 64, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq2_xxs_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_iq2_xxs<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq2_xs_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_iq2_xs<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq2_s_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_iq2_s<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq3_xxs_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_iq3_xxs<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq3_s_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_iq3_s<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq1_s_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_iq1_s<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq4_nl_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = (k + QK_K - 1) / QK_K;
    dequantize_block_iq4_nl<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq1_m_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_iq1_m<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq4_xs_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = (k + QK_K - 1) / QK_K;
    dequantize_block_iq4_xs<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_mxfp4_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = (k + QK_K - 1) / QK_K;
    dequantize_block_mxfp4<<<nb, 32, 0, stream>>>(vx, y);
}

template <typename dst_t>
static __global__ void dequantize_block_nvfp4(
        const void * __restrict__ vx,
        dst_t * __restrict__ yy,
        const int64_t ne) {
    const int64_t i = blockIdx.x;
    const int     tid = threadIdx.x;

    const int64_t base = i * QK_NVFP4;
    if (base >= ne) {
        return;
    }

    const block_nvfp4 * x = (const block_nvfp4 *) vx;
    const block_nvfp4 & xb = x[i];

    const int sub = tid / (QK_NVFP4_SUB / 2);
    const int j = tid % (QK_NVFP4_SUB / 2);

    const float d = ggml_cuda_ue4m3_to_fp32(xb.d[sub]);
    const uint8_t q = xb.qs[sub * (QK_NVFP4_SUB / 2) + j];

    const int64_t y0 = base + sub * QK_NVFP4_SUB + j;
    const int64_t y1 = y0 + QK_NVFP4_SUB / 2;

    yy[y0] = ggml_cuda_cast<dst_t>(d * kvalues_mxfp4[q & 0x0F]);
    yy[y1] = ggml_cuda_cast<dst_t>(d * kvalues_mxfp4[q >> 4]);
}

template <typename dst_t>
static void dequantize_row_nvfp4_cuda(
        const void * vx,
        dst_t * y,
        const int64_t k,
        cudaStream_t stream) {
    GGML_ASSERT(k % QK_NVFP4 == 0);
    const int nb = k / QK_NVFP4;
    dequantize_block_nvfp4<<<nb, 32, 0, stream>>>(vx, y, k);
}
template <typename src_t, typename dst_t>
static __global__ void convert_unary(
        const void * __restrict__ vx, dst_t * __restrict__ y, const int64_t ne00, const int64_t ne01,
        const int64_t ne0203, const uint3 ne02,
        const int64_t s01, const int64_t s02, const int64_t s03) {
    const int64_t i00 = (int64_t)blockDim.x*blockIdx.x + threadIdx.x;

    if (i00 >= ne00) {
        return;
    }

    const src_t * x = (const src_t *) vx;

    for (int64_t i01 = blockIdx.y; i01 < ne01; i01 += gridDim.y) {
        for (int64_t i0203 = blockIdx.z; i0203 < ne0203; i0203 += gridDim.z) {
            const uint2 dm = fast_div_modulo((uint32_t)i0203, ne02);
            const int64_t i02 = dm.y;
            const int64_t i03 = dm.x;

            const int64_t ix = i03*s03 + i02*s02 + i01*s01 + i00;
            const int64_t iy = (i0203*ne01 + i01)*ne00 + i00;
            y[iy] = ggml_cuda_cast<dst_t>(x[ix]);
        }
    }
}

template <typename src_t, typename dst_t>
static void convert_unary_cuda(const void * vx, dst_t * y,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t s01, const int64_t s02, const int64_t s03, cudaStream_t stream) {
    const int64_t ne0203 = ne02*ne03;
    const uint3 ne02_fdv = init_fastdiv_values(ne02);
    const dim3 num_blocks((ne00 + CUDA_DEQUANTIZE_BLOCK_SIZE - 1) / CUDA_DEQUANTIZE_BLOCK_SIZE, (int)std::min(ne01, (int64_t)65535), (int)std::min(ne0203, (int64_t)65535));
    convert_unary<src_t><<<num_blocks, CUDA_DEQUANTIZE_BLOCK_SIZE, 0, stream>>>
        (vx, y, ne00, ne01, ne0203, ne02_fdv, s01, s02, s03);
}

template <typename src_t, typename dst_t>
static void convert_unary_cont_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    convert_unary_cuda<src_t>(vx, y, k, 1, 1, 1, k, k, k, stream);
}

// ============================================================
// TBQ _4 → spatial f16: fused dequant + cooperative IWHT
// 1 CUDA block = 1 TBQ block. 256 threads cooperate on IWHT butterfly.
// ============================================================
template <typename block_type, int n_bits>
static __device__ __forceinline__ void tbq_dequant_sub(const block_type * blk, float * buf, int tid, int sub) {
    static constexpr float c3[8] = { -2.1520f,-1.3440f,-0.7560f,-0.2451f,0.2451f,0.7560f,1.3440f,2.1520f };
    static constexpr float c4[16] = { -2.7326f,-2.0690f,-1.6180f,-1.2562f,-0.9424f,-0.6568f,-0.3881f,-0.1284f,
                                       0.1284f,0.3881f,0.6568f,0.9424f,1.2562f,1.6180f,2.0690f,2.7326f };
    const float norm = __half2float(sub == 0 ? blk->d1 : blk->d2);
    const uint8_t * qs = sub == 0 ? blk->qs1 : blk->qs2;
    if constexpr (n_bits == 3) {
        const int bp = tid*3, by = bp/8, bo = bp%8;
        uint32_t r = (uint32_t)qs[by]>>bo; if (bo>5) r|=(uint32_t)qs[by+1]<<(8-bo);
        buf[tid] = c3[r&7] * norm;
    } else {
        buf[tid] = c4[(qs[tid/2]>>((tid%2)*4))&0xF] * norm;
    }
}

static __device__ __forceinline__ void tbq_iwht_256(float * buf, int tid) {
    static constexpr uint8_t signs[32] = {
        0xa7,0x3b,0x91,0xf4,0x6d,0xc2,0x58,0x0e,0xb3,0x7f,0x24,0xd6,0x89,0x45,0xea,0x1c,
        0x63,0xaf,0xd8,0x52,0x97,0x0b,0xe1,0x3d,0x76,0xc4,0x19,0xfe,0x4a,0x85,0x2c,0xdb,
    };
    __syncthreads();
    for (int len = 1; len < 256; len *= 2) {
        int grp = tid/(2*len), pos = tid%(2*len);
        if (pos < len) { float u=buf[grp*2*len+pos], v=buf[grp*2*len+pos+len]; buf[grp*2*len+pos]=u+v; buf[grp*2*len+pos+len]=u-v; }
        __syncthreads();
    }
    int s = ((signs[tid>>3]>>(tid&7))&1) ? -1 : 1;
    buf[tid] = buf[tid]*s/256.0f;
    __syncthreads();
}

template <typename block_type, int n_bits>
static __global__ void dequantize_tbq_4_spatial_f16_kernel(const void * __restrict__ vx, half * __restrict__ y, int64_t n_blocks) {
    int64_t bi = blockIdx.x;
    if (bi >= n_blocks) return;
    const block_type * blk = (const block_type *)vx + bi;
    half * out = y + bi * TBQ_K576;
    int tid = threadIdx.x;
    __shared__ float buf[256];
    // Sub-block 1: dequant → IWHT → f16
    tbq_dequant_sub<block_type, n_bits>(blk, buf, tid, 0);
    tbq_iwht_256(buf, tid);
    out[tid] = __float2half(buf[tid]);
    // Sub-block 2: dequant → IWHT → f16
    tbq_dequant_sub<block_type, n_bits>(blk, buf, tid, 1);
    tbq_iwht_256(buf, tid);
    out[256+tid] = __float2half(buf[tid]);
    // Sub-block 3 (rope): f16 passthrough
    if (tid < 64) out[512+tid] = blk->rope[tid];
}

// TBQP → WHT-domain f16 with QJL correction (NO IWHT).
// QJL is valid in WHT domain: value = centroid*norm + qjl_sign*d_qjl.
// MMA computes WHT(Q)·WHT(K)^T = D×Q·K^T (correct by WHT orthogonality).
// Q WHT and output IWHT are applied externally in launch_fattn.
// Simple per-element kernel — 288 threads (576/2 half2), no shared memory needed.
template <typename block_type, int n_bits>
static __global__ void dequantize_tbqp_4_wht_f16_kernel(const void * __restrict__ vx, half * __restrict__ y, int64_t n_blocks) {
    int64_t bi = blockIdx.x;
    if (bi >= n_blocks) return;
    const block_type * blk = (const block_type *)vx + bi;
    half2 * out = (half2 *)(y + bi * TBQ_K576);
    const int tid = threadIdx.x;
    const int elem = tid * 2;

    static constexpr float c2[4] = { -1.5104f,-0.4528f,0.4528f,1.5104f };
    static constexpr float c3[8] = { -2.1520f,-1.3440f,-0.7560f,-0.2451f,0.2451f,0.7560f,1.3440f,2.1520f };

    float v0, v1;
    if (elem < 512) {
        const int sub = elem < 256 ? 0 : 1;
        const int e = elem - sub * 256;
        const float norm = __half2float(sub == 0 ? blk->d1 : blk->d2);
        const float dq   = __half2float(sub == 0 ? blk->d1_qjl : blk->d2_qjl);
        const uint8_t * qs  = sub == 0 ? blk->qs1  : blk->qs2;
        const uint8_t * qjl = sub == 0 ? blk->qjl1 : blk->qjl2;
        const int s0 = ((qjl[e/8]>>(e%8))&1) ? 1 : -1;
        const int s1 = ((qjl[(e+1)/8]>>((e+1)%8))&1) ? 1 : -1;
        if constexpr (n_bits == 3) {
            // TBQP3: 2-bit MSE + 1-bit QJL
            v0 = c2[(qs[e/4]>>((e%4)*2))&0x3]*norm + s0*dq;
            v1 = c2[(qs[(e+1)/4]>>(((e+1)%4)*2))&0x3]*norm + s1*dq;
        } else {
            // TBQP4: 3-bit MSE + 1-bit QJL
            int bp0=e*3, by0=bp0/8, bo0=bp0%8;
            int bp1=(e+1)*3, by1=bp1/8, bo1=bp1%8;
            uint32_t r0=(uint32_t)qs[by0]>>bo0; if(bo0>5) r0|=(uint32_t)qs[by0+1]<<(8-bo0);
            uint32_t r1=(uint32_t)qs[by1]>>bo1; if(bo1>5) r1|=(uint32_t)qs[by1+1]<<(8-bo1);
            v0 = c3[r0&7]*norm + s0*dq;
            v1 = c3[r1&7]*norm + s1*dq;
        }
    } else {
        const int e = elem - 512;
        v0 = __half2float(blk->rope[e]);
        v1 = __half2float(blk->rope[e+1]);
    }
    out[tid] = make_half2(__float2half(v0), __float2half(v1));
}

// Q WHT: transform Q from spatial to WHT domain (for TBQP MMA pre-processing).
// 1 CUDA block = 1 sub-block (256 elements). 256 threads cooperate on butterfly.
static __global__ void tbq_q_wht_kernel(float * __restrict__ Q, int64_t n_rows, int64_t row_stride) {
    int64_t row = blockIdx.x;
    if (row >= n_rows) return;
    const int sub = blockIdx.y;  // 0 or 1
    const int tid = threadIdx.x;

    static constexpr uint8_t signs[32] = {
        0xa7,0x3b,0x91,0xf4,0x6d,0xc2,0x58,0x0e,0xb3,0x7f,0x24,0xd6,0x89,0x45,0xea,0x1c,
        0x63,0xaf,0xd8,0x52,0x97,0x0b,0xe1,0x3d,0x76,0xc4,0x19,0xfe,0x4a,0x85,0x2c,0xdb,
    };

    __shared__ float buf[256];
    float * q = Q + row * row_stride + sub * 256;

    int s = ((signs[tid>>3]>>(tid&7))&1) ? -1 : 1;
    buf[tid] = q[tid] * s;
    __syncthreads();

    for (int len = 1; len < 256; len *= 2) {
        int grp = tid/(2*len), pos = tid%(2*len);
        if (pos < len) { float u=buf[grp*2*len+pos], v=buf[grp*2*len+pos+len]; buf[grp*2*len+pos]=u+v; buf[grp*2*len+pos+len]=u-v; }
        __syncthreads();
    }

    q[tid] = buf[tid] / 256.0f;
}

// Fused Q_wht1 + Q_wht2: signs1 WHT → store Q_wht1, then signs2 WHT → store Q_wht2.
// Single kernel, single block read, shared memory reused between the two WHTs.
// Output: Q_wht1 at q1[sub*256+tid], Q_wht2 at q2[sub*256+tid]
static __global__ void tbq_q_wht12_kernel(const float * __restrict__ Q_src,
                                           float * __restrict__ Q_wht1, float * __restrict__ Q_wht2,
                                           int64_t n_rows, int64_t row_stride) {
    int64_t row = blockIdx.x;
    if (row >= n_rows) return;
    const int sub = blockIdx.y;
    const int tid = threadIdx.x;

    static constexpr uint8_t signs1[32] = {
        0xa7,0x3b,0x91,0xf4,0x6d,0xc2,0x58,0x0e,0xb3,0x7f,0x24,0xd6,0x89,0x45,0xea,0x1c,
        0x63,0xaf,0xd8,0x52,0x97,0x0b,0xe1,0x3d,0x76,0xc4,0x19,0xfe,0x4a,0x85,0x2c,0xdb,
    };
    static constexpr uint8_t signs2[32] = {
        0xd3,0x4e,0xa8,0x17,0x9c,0x5b,0xe6,0x31,0x72,0xb9,0x0d,0xf5,0x43,0x8a,0x6e,0xc7,
        0x58,0x2f,0x94,0xe1,0xb6,0x3d,0x0a,0x7c,0xc5,0x61,0xd8,0x4f,0xa3,0x97,0x1e,0x85,
    };

    __shared__ float buf[256];
    const float * q_src = Q_src + row * row_stride + sub * 256;
    float * q1 = Q_wht1 + row * row_stride + sub * 256;
    float * q2 = Q_wht2 + row * row_stride + sub * 256;

    // === WHT 1: signs1 → butterfly → /256 ===
    int s1 = ((signs1[tid>>3]>>(tid&7))&1) ? -1 : 1;
    buf[tid] = q_src[tid] * s1;  // Read from Q_src directly
    __syncthreads();

    for (int len = 1; len < 256; len *= 2) {
        int grp = tid/(2*len), pos = tid%(2*len);
        if (pos < len) { float u=buf[grp*2*len+pos], v=buf[grp*2*len+pos+len]; buf[grp*2*len+pos]=u+v; buf[grp*2*len+pos+len]=u-v; }
        __syncthreads();
    }

    q1[tid] = buf[tid] / 256.0f;  // Store Q_wht1
    // buf still has WHT(signs1*Q) (un-scaled) — use for Q_wht2

    // === WHT 2: signs2 applied to WHT result, then butterfly, then qjl_factor/(256*256) ===
    int s2 = ((signs2[tid>>3]>>(tid&7))&1) ? -1 : 1;
    buf[tid] = buf[tid] * s2;  // signs2 * WHT(signs1*Q)
    __syncthreads();

    for (int len = 1; len < 256; len *= 2) {
        int grp = tid/(2*len), pos = tid%(2*len);
        if (pos < len) { float u=buf[grp*2*len+pos], v=buf[grp*2*len+pos+len]; buf[grp*2*len+pos]=u+v; buf[grp*2*len+pos+len]=u-v; }
        __syncthreads();
    }

    constexpr float qjl_factor = 1.2533f;
    q2[tid] = buf[tid] * qjl_factor / (256.0f * 256.0f);  // Store Q_wht2
}

// Output IWHT: transform attention output from WHT to spatial domain (for TBQP MMA post-processing).
static __global__ void tbq_output_iwht_kernel(float * __restrict__ dst, int64_t n_rows, int64_t row_stride) {
    int64_t row = blockIdx.x;
    if (row >= n_rows) return;
    const int sub = blockIdx.y;
    const int tid = threadIdx.x;

    static constexpr uint8_t signs[32] = {
        0xa7,0x3b,0x91,0xf4,0x6d,0xc2,0x58,0x0e,0xb3,0x7f,0x24,0xd6,0x89,0x45,0xea,0x1c,
        0x63,0xaf,0xd8,0x52,0x97,0x0b,0xe1,0x3d,0x76,0xc4,0x19,0xfe,0x4a,0x85,0x2c,0xdb,
    };

    __shared__ float buf[256];
    float * out = dst + row * row_stride + sub * 256;

    buf[tid] = out[tid];
    __syncthreads();

    for (int len = 1; len < 256; len *= 2) {
        int grp = tid/(2*len), pos = tid%(2*len);
        if (pos < len) { float u=buf[grp*2*len+pos], v=buf[grp*2*len+pos+len]; buf[grp*2*len+pos]=u+v; buf[grp*2*len+pos+len]=u-v; }
        __syncthreads();
    }

    int s = ((signs[tid>>3]>>(tid&7))&1) ? -1 : 1;
    out[tid] = buf[tid] * s / 256.0f;
}

// TBQP K_qjl dequant: QJL-only component → f16.
// Output: s_qjl * dq per element (latent sub-blocks), 0 for rope.
// Used for separate QJL MMA pass: KQ_qjl = Q_wht2 @ K_qjl^T.
template <typename block_type, int n_bits>
static __global__ void dequantize_tbqp_4_k_qjl_f16_kernel(const void * __restrict__ vx, half * __restrict__ y, int64_t n_blocks) {
    int64_t bi = blockIdx.x;
    if (bi >= n_blocks) return;
    const block_type * blk = (const block_type *)vx + bi;
    half2 * out = (half2 *)(y + bi * TBQ_K576);
    const int tid = threadIdx.x;
    const int elem = tid * 2;

    float v0, v1;
    if (elem < 512) {
        const int sub = elem < 256 ? 0 : 1;
        const int e = elem - sub * 256;
        const float dq = __half2float(sub == 0 ? blk->d1_qjl : blk->d2_qjl);
        const uint8_t * qjl = sub == 0 ? blk->qjl1 : blk->qjl2;
        const int s0 = ((qjl[e/8]>>(e%8))&1) ? 1 : -1;
        const int s1 = ((qjl[(e+1)/8]>>((e+1)%8))&1) ? 1 : -1;
        v0 = s0 * dq;
        v1 = s1 * dq;
    } else {
        v0 = 0.0f;  // rope has no QJL
        v1 = 0.0f;
    }
    out[tid] = make_half2(__float2half(v0), __float2half(v1));
}

// TBQP fused K_mse + K_qjl dequant: reads each block ONCE, writes two outputs.
// out_mse: centroid*norm (WHT domain, no QJL) — for K and V in MMA
// out_qjl: s_qjl * dq (QJL-only) — for separate QJL MMA pass
// n_blocks_total = total blocks. out_qjl = out_mse + n_blocks_total * TBQ_K576 (concatenated).
template <typename block_type, int n_bits>
static __global__ void dequantize_tbqp_4_fused_mse_qjl_f16_kernel(
        const void * __restrict__ vx, half * __restrict__ out_mse, int64_t n_blocks) {
    int64_t bi = blockIdx.x;
    if (bi >= n_blocks) return;
    const block_type * blk = (const block_type *)vx + bi;
    half2 * mse = (half2 *)(out_mse + bi * TBQ_K576);
    half2 * qjl = (half2 *)(out_mse + (n_blocks + bi) * TBQ_K576);  // second half of buffer
    const int tid = threadIdx.x;
    const int elem = tid * 2;

    static constexpr float c2[4] = { -1.5104f,-0.4528f,0.4528f,1.5104f };
    static constexpr float c3[8] = { -2.1520f,-1.3440f,-0.7560f,-0.2451f,0.2451f,0.7560f,1.3440f,2.1520f };

    float mse0, mse1, qjl0, qjl1;
    if (elem < 512) {
        const int sub = elem < 256 ? 0 : 1;
        const int e = elem - sub * 256;
        const float norm = __half2float(sub == 0 ? blk->d1 : blk->d2);
        const float dq   = __half2float(sub == 0 ? blk->d1_qjl : blk->d2_qjl);
        const uint8_t * qs  = sub == 0 ? blk->qs1  : blk->qs2;
        const uint8_t * qjl_bits = sub == 0 ? blk->qjl1 : blk->qjl2;
        const int s0 = ((qjl_bits[e/8]>>(e%8))&1) ? 1 : -1;
        const int s1 = ((qjl_bits[(e+1)/8]>>((e+1)%8))&1) ? 1 : -1;
        if constexpr (n_bits == 3) {
            mse0 = c2[(qs[e/4]>>((e%4)*2))&0x3]*norm;
            mse1 = c2[(qs[(e+1)/4]>>(((e+1)%4)*2))&0x3]*norm;
        } else {
            int bp0=e*3, by0=bp0/8, bo0=bp0%8;
            int bp1=(e+1)*3, by1=bp1/8, bo1=bp1%8;
            uint32_t r0=(uint32_t)qs[by0]>>bo0; if(bo0>5) r0|=(uint32_t)qs[by0+1]<<(8-bo0);
            uint32_t r1=(uint32_t)qs[by1]>>bo1; if(bo1>5) r1|=(uint32_t)qs[by1+1]<<(8-bo1);
            mse0 = c3[r0&7]*norm;
            mse1 = c3[r1&7]*norm;
        }
        qjl0 = s0 * dq;
        qjl1 = s1 * dq;
    } else {
        const int e = elem - 512;
        mse0 = __half2float(blk->rope[e]);
        mse1 = __half2float(blk->rope[e+1]);
        qjl0 = 0.0f;
        qjl1 = 0.0f;
    }
    mse[tid] = make_half2(__float2half(mse0), __float2half(mse1));
    qjl[tid] = make_half2(__float2half(qjl0), __float2half(qjl1));
}

// TBQP Q_wht2: second WHT with QJL sign pattern for QJL dot product correction.
// Input: Q_wht1 (already WHT'd with signs1, stored as float, /256 applied).
// Output: Q_wht2 = WHT(qjl_signs * Q_wht1 * 256) * qjl_factor / (256*256)
// The *256 undoes Q_wht1's /256, then /256² for two WHTs, * qjl_factor for QJL scaling.
// Net: Q_wht2 = WHT(qjl_signs * WHT(signs1*Q)) * qjl_factor / (256*256)
static __global__ void tbq_q_wht2_kernel(const float * __restrict__ Q_wht1, float * __restrict__ Q_wht2,
                                          int64_t n_rows, int64_t row_stride) {
    int64_t row = blockIdx.x;
    if (row >= n_rows) return;
    const int sub = blockIdx.y;  // 0 or 1
    const int tid = threadIdx.x;

    // QJL sign pattern (same as qjl_signs_256 in fattn-vec.cuh)
    static constexpr uint8_t qjl_signs[32] = {
        0xd3,0x4e,0xa8,0x17,0x9c,0x5b,0xe6,0x31,0x72,0xb9,0x0d,0xf5,0x43,0x8a,0x6e,0xc7,
        0x58,0x2f,0x94,0xe1,0xb6,0x3d,0x0a,0x7c,0xc5,0x61,0xd8,0x4f,0xa3,0x97,0x1e,0x85,
    };

    __shared__ float buf[256];
    const float * src = Q_wht1 + row * row_stride + sub * 256;

    // Load Q_wht1 * 256 (undo /256) and apply QJL sign
    int s = ((qjl_signs[tid>>3]>>(tid&7))&1) ? -1 : 1;
    buf[tid] = src[tid] * 256.0f * s;
    __syncthreads();

    // WHT butterfly
    for (int len = 1; len < 256; len *= 2) {
        int grp = tid/(2*len), pos = tid%(2*len);
        if (pos < len) { float u=buf[grp*2*len+pos], v=buf[grp*2*len+pos+len]; buf[grp*2*len+pos]=u+v; buf[grp*2*len+pos+len]=u-v; }
        __syncthreads();
    }

    // Scale: qjl_factor / (256*256)
    constexpr float qjl_factor = 1.2533f;
    float * dst = Q_wht2 + row * row_stride + sub * 256;
    dst[tid] = buf[tid] * qjl_factor / (256.0f * 256.0f);
}

// Host wrappers
void tbq_q_wht_cuda(float * Q, int64_t DKQ, int64_t n_rows, int64_t row_stride, cudaStream_t stream) {
    tbq_q_wht_kernel<<<dim3(n_rows, 2), 256, 0, stream>>>(Q, n_rows, row_stride);
}

void tbq_q_wht2_cuda(const float * Q_wht1, float * Q_wht2, int64_t DKQ, int64_t n_rows, int64_t row_stride, cudaStream_t stream) {
    tbq_q_wht2_kernel<<<dim3(n_rows, 2), 256, 0, stream>>>(Q_wht1, Q_wht2, n_rows, row_stride);
}

// Fused: Q_wht1 + Q_wht2 in one kernel. Q_wht1_buf initially contains original Q data.
void tbq_q_wht12_cuda(const float * Q_src, float * Q_wht1, float * Q_wht2, int64_t DKQ, int64_t n_rows, int64_t row_stride, cudaStream_t stream) {
    tbq_q_wht12_kernel<<<dim3(n_rows, 2), 256, 0, stream>>>(Q_src, Q_wht1, Q_wht2, n_rows, row_stride);
}

void tbq_output_iwht_cuda(float * dst, int64_t DV, int64_t n_rows, cudaStream_t stream) {
    tbq_output_iwht_kernel<<<dim3(n_rows, 2), 256, 0, stream>>>(dst, n_rows, DV);
}

void tbqp3_k_qjl_f16_cuda(const void * vx, half * y, int64_t k, cudaStream_t stream) {
    dequantize_tbqp_4_k_qjl_f16_kernel<block_tbqp3_4, 3><<<k/TBQ_K576, 288, 0, stream>>>(vx, y, k/TBQ_K576);
}
void tbqp4_k_qjl_f16_cuda(const void * vx, half * y, int64_t k, cudaStream_t stream) {
    dequantize_tbqp_4_k_qjl_f16_kernel<block_tbqp4_4, 4><<<k/TBQ_K576, 288, 0, stream>>>(vx, y, k/TBQ_K576);
}

// Fused: reads each TBQ block ONCE, writes K_mse and K_qjl to concatenated buffer.
// out points to a 2*k buffer: [K_mse (k halves) | K_qjl (k halves)]
void tbqp3_fused_mse_qjl_f16_cuda(const void * vx, half * out, int64_t k, cudaStream_t stream) {
    dequantize_tbqp_4_fused_mse_qjl_f16_kernel<block_tbqp3_4, 3><<<k/TBQ_K576, 288, 0, stream>>>(vx, out, k/TBQ_K576);
}
void tbqp4_fused_mse_qjl_f16_cuda(const void * vx, half * out, int64_t k, cudaStream_t stream) {
    dequantize_tbqp_4_fused_mse_qjl_f16_kernel<block_tbqp4_4, 4><<<k/TBQ_K576, 288, 0, stream>>>(vx, out, k/TBQ_K576);
}

static void dequantize_row_tbq3_4_to_f16_cuda(const void * vx, half * y, int64_t k, cudaStream_t stream) {
    dequantize_tbq_4_spatial_f16_kernel<block_tbq3_4, 3><<<k/TBQ_K576, 256, 0, stream>>>(vx, y, k/TBQ_K576);
}
static void dequantize_row_tbq4_4_to_f16_cuda(const void * vx, half * y, int64_t k, cudaStream_t stream) {
    dequantize_tbq_4_spatial_f16_kernel<block_tbq4_4, 4><<<k/TBQ_K576, 256, 0, stream>>>(vx, y, k/TBQ_K576);
}
static void dequantize_row_tbqp3_4_to_f16_cuda(const void * vx, half * y, int64_t k, cudaStream_t stream) {
    // WHT-domain dequant with QJL. Q WHT + output IWHT applied externally.
    dequantize_tbqp_4_wht_f16_kernel<block_tbqp3_4, 3><<<k/TBQ_K576, 288, 0, stream>>>(vx, y, k/TBQ_K576);
}
static void dequantize_row_tbqp4_4_to_f16_cuda(const void * vx, half * y, int64_t k, cudaStream_t stream) {
    dequantize_tbqp_4_wht_f16_kernel<block_tbqp4_4, 4><<<k/TBQ_K576, 288, 0, stream>>>(vx, y, k/TBQ_K576);
}

// TBQP V WHT dequant: MSE only (no QJL) in WHT domain → f16.
// K gets QJL (for dot product accuracy), V must NOT have QJL (pollutes V accumulation).
// Same structure as K WHT dequant but without + s*dq terms.
template <typename block_type, int n_bits>
static __global__ void dequantize_tbqp_4_v_wht_f16_kernel(const void * __restrict__ vx, half * __restrict__ y, int64_t n_blocks) {
    int64_t bi = blockIdx.x;
    if (bi >= n_blocks) return;
    const block_type * blk = (const block_type *)vx + bi;
    half2 * out = (half2 *)(y + bi * TBQ_K576);
    const int tid = threadIdx.x;
    const int elem = tid * 2;

    static constexpr float c2[4] = { -1.5104f,-0.4528f,0.4528f,1.5104f };
    static constexpr float c3[8] = { -2.1520f,-1.3440f,-0.7560f,-0.2451f,0.2451f,0.7560f,1.3440f,2.1520f };

    float v0, v1;
    if (elem < 512) {
        const int sub = elem < 256 ? 0 : 1;
        const int e = elem - sub * 256;
        const float norm = __half2float(sub == 0 ? blk->d1 : blk->d2);
        const uint8_t * qs = sub == 0 ? blk->qs1 : blk->qs2;
        if constexpr (n_bits == 3) {
            v0 = c2[(qs[e/4]>>((e%4)*2))&0x3]*norm;
            v1 = c2[(qs[(e+1)/4]>>(((e+1)%4)*2))&0x3]*norm;
        } else {
            int bp0=e*3, by0=bp0/8, bo0=bp0%8;
            int bp1=(e+1)*3, by1=bp1/8, bo1=bp1%8;
            uint32_t r0=(uint32_t)qs[by0]>>bo0; if(bo0>5) r0|=(uint32_t)qs[by0+1]<<(8-bo0);
            uint32_t r1=(uint32_t)qs[by1]>>bo1; if(bo1>5) r1|=(uint32_t)qs[by1+1]<<(8-bo1);
            v0 = c3[r0&7]*norm;
            v1 = c3[r1&7]*norm;
        }
    } else {
        const int e = elem - 512;
        v0 = __half2float(blk->rope[e]);
        v1 = __half2float(blk->rope[e+1]);
    }
    out[tid] = make_half2(__float2half(v0), __float2half(v1));
}

void tbqp3_v_wht_noqjl_f16_cuda(const void * vx, half * y, int64_t k, cudaStream_t stream) {
    dequantize_tbqp_4_v_wht_f16_kernel<block_tbqp3_4, 3><<<k/TBQ_K576, 288, 0, stream>>>(vx, y, k/TBQ_K576);
}
void tbqp4_v_wht_noqjl_f16_cuda(const void * vx, half * y, int64_t k, cudaStream_t stream) {
    dequantize_tbqp_4_v_wht_f16_kernel<block_tbqp4_4, 4><<<k/TBQ_K576, 288, 0, stream>>>(vx, y, k/TBQ_K576);
}

// TBQP V spatial dequant: MSE only (no QJL) + cooperative IWHT → spatial f16.
// For TBQP MMA: K uses WHT+QJL, V needs MSE-only spatial f16 (QJL breaks IWHT reconstruction).
template <typename block_type, int n_bits>
static __global__ void dequantize_tbqp_4_v_spatial_f16_kernel(const void * __restrict__ vx, half * __restrict__ y, int64_t n_blocks) {
    int64_t bi = blockIdx.x;
    if (bi >= n_blocks) return;
    const block_type * blk = (const block_type *)vx + bi;
    half * out = y + bi * TBQ_K576;
    int tid = threadIdx.x;
    __shared__ float buf[256];
    static constexpr float c2[4] = { -1.5104f,-0.4528f,0.4528f,1.5104f };
    static constexpr float c3[8] = { -2.1520f,-1.3440f,-0.7560f,-0.2451f,0.2451f,0.7560f,1.3440f,2.1520f };
    static constexpr uint8_t signs[32] = {
        0xa7,0x3b,0x91,0xf4,0x6d,0xc2,0x58,0x0e,0xb3,0x7f,0x24,0xd6,0x89,0x45,0xea,0x1c,
        0x63,0xaf,0xd8,0x52,0x97,0x0b,0xe1,0x3d,0x76,0xc4,0x19,0xfe,0x4a,0x85,0x2c,0xdb,
    };
    for (int sub = 0; sub < 2; ++sub) {
        const float norm = __half2float(sub == 0 ? blk->d1 : blk->d2);
        const uint8_t * qs = sub == 0 ? blk->qs1 : blk->qs2;
        if constexpr (n_bits == 3) {
            buf[tid] = c2[(qs[tid/4]>>((tid%4)*2))&0x3] * norm;
        } else {
            int bp = tid*3, by = bp/8, bo = bp%8;
            uint32_t r = (uint32_t)qs[by]>>bo; if (bo>5) r|=(uint32_t)qs[by+1]<<(8-bo);
            buf[tid] = c3[r&7] * norm;
        }
        __syncthreads();
        for (int len = 1; len < 256; len *= 2) {
            int grp = tid/(2*len), pos = tid%(2*len);
            if (pos < len) { float u=buf[grp*2*len+pos], v=buf[grp*2*len+pos+len]; buf[grp*2*len+pos]=u+v; buf[grp*2*len+pos+len]=u-v; }
            __syncthreads();
        }
        int s = ((signs[tid>>3]>>(tid&7))&1) ? -1 : 1;
        out[sub*256+tid] = __float2half(buf[tid]*s/256.0f);
        __syncthreads();
    }
    if (tid < 64) out[512+tid] = blk->rope[tid];
}

void tbqp3_v_spatial_f16_cuda(const void * vx, half * y, int64_t k, cudaStream_t stream) {
    dequantize_tbqp_4_v_spatial_f16_kernel<block_tbqp3_4, 3><<<k/TBQ_K576, 256, 0, stream>>>(vx, y, k/TBQ_K576);
}
void tbqp4_v_spatial_f16_cuda(const void * vx, half * y, int64_t k, cudaStream_t stream) {
    dequantize_tbqp_4_v_spatial_f16_kernel<block_tbqp4_4, 4><<<k/TBQ_K576, 256, 0, stream>>>(vx, y, k/TBQ_K576);
}

// (Q WHT and output IWHT kernels defined above — no duplicates)

to_bf16_cuda_t ggml_get_to_bf16_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
            return convert_unary_cont_cuda<float>;
        case GGML_TYPE_F16:
            return convert_unary_cont_cuda<half>;
        default:
            return nullptr;
    }
}

to_fp16_cuda_t ggml_get_to_fp16_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:
            return dequantize_row_q4_0_cuda;
        case GGML_TYPE_Q4_1:
            return dequantize_row_q4_1_cuda;
        case GGML_TYPE_Q5_0:
            return dequantize_block_cont_cuda<QK5_0, QR5_0, dequantize_q5_0>;
        case GGML_TYPE_Q5_1:
            return dequantize_block_cont_cuda<QK5_1, QR5_1, dequantize_q5_1>;
        case GGML_TYPE_Q8_0:
            if (fp16_available(ggml_cuda_info().devices[ggml_cuda_get_device()].cc)) {
                return dequantize_block_q8_0_f16_cuda;
            }
            return dequantize_block_cont_cuda<QK8_0, QR8_0, dequantize_q8_0>;
        case GGML_TYPE_Q2_K:
            return dequantize_row_q2_K_cuda;
        case GGML_TYPE_Q3_K:
            return dequantize_row_q3_K_cuda;
        case GGML_TYPE_Q4_K:
            return dequantize_row_q4_K_cuda;
        case GGML_TYPE_Q5_K:
            return dequantize_row_q5_K_cuda;
        case GGML_TYPE_Q6_K:
            return dequantize_row_q6_K_cuda;
        case GGML_TYPE_IQ2_XXS:
            return dequantize_row_iq2_xxs_cuda;
        case GGML_TYPE_IQ2_XS:
            return dequantize_row_iq2_xs_cuda;
        case GGML_TYPE_IQ2_S:
            return dequantize_row_iq2_s_cuda;
        case GGML_TYPE_IQ3_XXS:
            return dequantize_row_iq3_xxs_cuda;
        case GGML_TYPE_IQ1_S:
            return dequantize_row_iq1_s_cuda;
        case GGML_TYPE_IQ1_M:
            return dequantize_row_iq1_m_cuda;
        case GGML_TYPE_IQ4_NL:
            return dequantize_row_iq4_nl_cuda;
        case GGML_TYPE_IQ4_XS:
            return dequantize_row_iq4_xs_cuda;
        case GGML_TYPE_IQ3_S:
            return dequantize_row_iq3_s_cuda;
        case GGML_TYPE_MXFP4:
            return dequantize_row_mxfp4_cuda;
        case GGML_TYPE_NVFP4:
            return dequantize_row_nvfp4_cuda;
        case GGML_TYPE_F32:
            return convert_unary_cont_cuda<float>;
        case GGML_TYPE_BF16:
            return convert_unary_cont_cuda<nv_bfloat16>;
        case GGML_TYPE_TBQ3_4:
            return dequantize_row_tbq3_4_to_f16_cuda;
        case GGML_TYPE_TBQ4_4:
            return dequantize_row_tbq4_4_to_f16_cuda;
        case GGML_TYPE_TBQP3_4:
            return (to_fp16_cuda_t)tbqp3_v_spatial_f16_cuda;  // Spatial (MSE + IWHT). V=K view=spatial, no output IWHT.
        case GGML_TYPE_TBQP4_4:
            return (to_fp16_cuda_t)tbqp4_v_spatial_f16_cuda;
        default:
            return nullptr;
    }
}

// ============================================================
// TBQ _4 → spatial f16: fused dequant + cooperative IWHT
// 1 CUDA block = 1 TBQ block (576 elements). 256 threads cooperate on IWHT butterfly.
// (duplicate removed — definitions above)

to_fp32_cuda_t ggml_get_to_fp32_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:
            return dequantize_row_q4_0_cuda;
        case GGML_TYPE_Q4_1:
            return dequantize_row_q4_1_cuda;
        case GGML_TYPE_Q5_0:
            return dequantize_block_cont_cuda<QK5_0, QR5_0, dequantize_q5_0>;
        case GGML_TYPE_Q5_1:
            return dequantize_block_cont_cuda<QK5_1, QR5_1, dequantize_q5_1>;
        case GGML_TYPE_Q8_0:
            return dequantize_block_cont_cuda<QK8_0, QR8_0, dequantize_q8_0>;
        case GGML_TYPE_Q2_K:
            return dequantize_row_q2_K_cuda;
        case GGML_TYPE_Q3_K:
            return dequantize_row_q3_K_cuda;
        case GGML_TYPE_Q4_K:
            return dequantize_row_q4_K_cuda;
        case GGML_TYPE_Q5_K:
            return dequantize_row_q5_K_cuda;
        case GGML_TYPE_Q6_K:
            return dequantize_row_q6_K_cuda;
        case GGML_TYPE_IQ2_XXS:
            return dequantize_row_iq2_xxs_cuda;
        case GGML_TYPE_IQ2_XS:
            return dequantize_row_iq2_xs_cuda;
        case GGML_TYPE_IQ2_S:
            return dequantize_row_iq2_s_cuda;
        case GGML_TYPE_IQ3_XXS:
            return dequantize_row_iq3_xxs_cuda;
        case GGML_TYPE_IQ1_S:
            return dequantize_row_iq1_s_cuda;
        case GGML_TYPE_IQ1_M:
            return dequantize_row_iq1_m_cuda;
        case GGML_TYPE_IQ4_NL:
            return dequantize_row_iq4_nl_cuda;
        case GGML_TYPE_IQ4_XS:
            return dequantize_row_iq4_xs_cuda;
        case GGML_TYPE_IQ3_S:
            return dequantize_row_iq3_s_cuda;
        case GGML_TYPE_MXFP4:
            return dequantize_row_mxfp4_cuda;
        case GGML_TYPE_NVFP4:
            return dequantize_row_nvfp4_cuda;
        case GGML_TYPE_F16:
            return convert_unary_cont_cuda<half>;
        case GGML_TYPE_BF16:
            return convert_unary_cont_cuda<nv_bfloat16>;
        default:
            return nullptr;
    }
}

to_fp16_nc_cuda_t ggml_get_to_fp16_nc_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
            return convert_unary_cuda<float>;
        case GGML_TYPE_Q4_0:
            return dequantize_block_cuda<QK4_0, QR4_0, dequantize_q4_0>;
        case GGML_TYPE_Q4_1:
            return dequantize_block_cuda<QK4_1, QR4_1, dequantize_q4_1>;
        case GGML_TYPE_Q5_0:
            return dequantize_block_cuda<QK5_0, QR5_0, dequantize_q5_0>;
        case GGML_TYPE_Q5_1:
            return dequantize_block_cuda<QK5_1, QR5_1, dequantize_q5_1>;
        case GGML_TYPE_Q8_0:
            return dequantize_block_cuda<QK8_0, QR8_0, dequantize_q8_0>;
        case GGML_TYPE_BF16:
            return convert_unary_cuda<nv_bfloat16>;
        default:
            return nullptr;
    }
}

to_bf16_nc_cuda_t ggml_get_to_bf16_nc_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
            return convert_unary_cuda<float, nv_bfloat16>;
        case GGML_TYPE_Q4_0:
            return dequantize_block_cuda<QK4_0, QR4_0, dequantize_q4_0>;
        case GGML_TYPE_Q4_1:
            return dequantize_block_cuda<QK4_1, QR4_1, dequantize_q4_1>;
        case GGML_TYPE_Q5_0:
            return dequantize_block_cuda<QK5_0, QR5_0, dequantize_q5_0>;
        case GGML_TYPE_Q5_1:
            return dequantize_block_cuda<QK5_1, QR5_1, dequantize_q5_1>;
        case GGML_TYPE_Q8_0:
            return dequantize_block_cuda<QK8_0, QR8_0, dequantize_q8_0>;
        case GGML_TYPE_F16:
            return convert_unary_cuda<half, nv_bfloat16>;
        default:
            return nullptr;
    }
}

to_fp32_nc_cuda_t ggml_get_to_fp32_nc_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F16:
            return convert_unary_cuda<half, float>;
        case GGML_TYPE_Q4_0:
            return dequantize_block_cuda<QK4_0, QR4_0, dequantize_q4_0>;
        case GGML_TYPE_Q4_1:
            return dequantize_block_cuda<QK4_1, QR4_1, dequantize_q4_1>;
        case GGML_TYPE_Q5_0:
            return dequantize_block_cuda<QK5_0, QR5_0, dequantize_q5_0>;
        case GGML_TYPE_Q5_1:
            return dequantize_block_cuda<QK5_1, QR5_1, dequantize_q5_1>;
        case GGML_TYPE_Q8_0:
            return dequantize_block_cuda<QK8_0, QR8_0, dequantize_q8_0>;
        case GGML_TYPE_BF16:
            return convert_unary_cuda<nv_bfloat16, float>;
        default:
            return nullptr;
    }
}
