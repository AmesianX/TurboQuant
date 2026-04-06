#include "set-rows.cuh"
#include "cpy-utils.cuh"

typedef void (*set_rows_kernel_t)(const char * src, char * dst);

// Generic quantized set_rows kernel template
template <typename idx_t, typename block_type, int qk, void (*quantize_func)(const float *, block_type *)>
static __global__ void k_set_rows_quant(const float * __restrict__ src0,
                                        const idx_t * __restrict__ src1,
                                        block_type * __restrict__ dst,
                                        const int64_t ne_total,
                                        const int64_t ne10,
                                        const int64_t ne11,
                                        const int64_t ne12,
                                        const int64_t ne13,
                                        const int64_t s01,
                                        const int64_t s02,
                                        const int64_t s03,
                                        const int64_t s10,
                                        const int64_t s11,
                                        const int64_t s12,
                                        const int64_t s1,
                                        const int64_t s2,
                                        const int64_t s3,
                                        const uint3   ne00,
                                        const uint3   ne01,
                                        const uint3   ne02,
                                        const uint3   ne11_fd,
                                        const uint3   ne12_fd) {
    const int64_t i = int64_t(blockDim.x) * blockIdx.x + threadIdx.x;

    if (i >= ne_total) {
        return;
    }

    const int64_t i_base = i * qk;
    uint32_t      tmp    = (uint32_t) i_base;
    uint2         div_mod;

    div_mod           = fast_div_modulo(tmp, ne00);
    const int64_t i00 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne01);
    const int64_t i01 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne02);
    const int64_t i02 = div_mod.y;
    const int64_t i03 = div_mod.x;

    const int64_t i12 = fastmodulo((uint32_t) i03, ne12_fd);
    const int64_t i11 = fastmodulo((uint32_t) i02, ne11_fd);
    const int64_t i10 = i01;

    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);

    const float * src0_row = src0 + i01*s01 + i02*s02 + i03*s03;
    block_type * dst_row_ptr = dst + (dst_row*s1 + i02*s2 + i03*s3) / sizeof(block_type);

    const float * src_block = src0_row + i00;
    block_type * dst_block = dst_row_ptr + i00 / qk;

    quantize_func(src_block, dst_block);

    GGML_UNUSED(ne10);
    GGML_UNUSED(ne11);
    GGML_UNUSED(ne12);
    GGML_UNUSED(ne13);
}

// 512-point WHT quantization kernel: processes 2 consecutive blocks at once
// Each thread handles 512 elements (2 × qk), called only for even-indexed block pairs
template<typename idx_t, typename block_type, int qk, void (*quantize_func_512)(const float*, block_type*)>
static __global__ void k_set_rows_quant_512(const float * __restrict__ src0,
                                        const idx_t * __restrict__ src1,
                                        block_type * __restrict__ dst,
                                        const int64_t ne_total_pairs,
                                        const int64_t ne10,
                                        const int64_t ne11,
                                        const int64_t ne12,
                                        const int64_t ne13,
                                        const int64_t s01,
                                        const int64_t s02,
                                        const int64_t s03,
                                        const int64_t s10,
                                        const int64_t s11,
                                        const int64_t s12,
                                        const int64_t s1,
                                        const int64_t s2,
                                        const int64_t s3,
                                        const uint3   ne00,
                                        const uint3   ne01,
                                        const uint3   ne02,
                                        const uint3   ne11_fd,
                                        const uint3   ne12_fd) {
    const int64_t i = int64_t(blockDim.x) * blockIdx.x + threadIdx.x;

    if (i >= ne_total_pairs) {
        return;
    }

    // Each thread handles a pair of blocks (512 elements)
    const int64_t i_base = i * (2 * qk);
    uint32_t      tmp    = (uint32_t) i_base;
    uint2         div_mod;

    div_mod           = fast_div_modulo(tmp, ne00);
    const int64_t i00 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne01);
    const int64_t i01 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne02);
    const int64_t i02 = div_mod.y;
    const int64_t i03 = div_mod.x;

    const int64_t i12 = fastmodulo((uint32_t) i03, ne12_fd);
    const int64_t i11 = fastmodulo((uint32_t) i02, ne11_fd);
    const int64_t i10 = i01;

    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);

    const float * src0_row = src0 + i01*s01 + i02*s02 + i03*s03;
    block_type * dst_row_ptr = dst + (dst_row*s1 + i02*s2 + i03*s3) / sizeof(block_type);

    const float * src_block = src0_row + i00;
    block_type * dst_block = dst_row_ptr + i00 / qk;

    quantize_func_512(src_block, dst_block);

    GGML_UNUSED(ne10);
    GGML_UNUSED(ne11);
    GGML_UNUSED(ne12);
    GGML_UNUSED(ne13);
}

// Template dispatch for 512-point WHT quantization
template<typename idx_t, typename block_type, int qk, void (*quantize_func_512)(const float*, block_type*)>
static void set_rows_cuda_quant_512(
        const float * src0_d, const idx_t * src1_d, block_type * dst_d,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {

    GGML_ASSERT(ne00 % (2 * qk) == 0); // must be multiple of 512
    const int64_t ne_total_pairs = (ne00 * ne01 * ne02 * ne03) / (2 * qk);
    const int num_blocks = (ne_total_pairs + CUDA_SET_ROWS_BLOCK_SIZE - 1) / CUDA_SET_ROWS_BLOCK_SIZE;
    const dim3 block_size(CUDA_SET_ROWS_BLOCK_SIZE);
    const dim3 grid_size(num_blocks);

    const int64_t s01 = nb01/sizeof(float);
    const int64_t s02 = nb02/sizeof(float);
    const int64_t s03 = nb03/sizeof(float);
    const int64_t s10 = nb10/sizeof(idx_t);
    const int64_t s11 = nb11/sizeof(idx_t);
    const int64_t s12 = nb12/sizeof(idx_t);
    const int64_t s1  = nb1;
    const int64_t s2  = nb2;
    const int64_t s3  = nb3;

    if (ne_total_pairs > 0 && ne00 > 0 && ne01 > 0 && ne02 > 0 && ne11 > 0 && ne12 > 0) {
        const uint3 ne00_fd = init_fastdiv_values((uint32_t) ne00);
        const uint3 ne01_fd = init_fastdiv_values((uint32_t) ne01);
        const uint3 ne02_fd = init_fastdiv_values((uint32_t) ne02);
        const uint3 ne11_fd = init_fastdiv_values((uint32_t) ne11);
        const uint3 ne12_fd = init_fastdiv_values((uint32_t) ne12);
        k_set_rows_quant_512<idx_t, block_type, qk, quantize_func_512>
            <<<grid_size, block_size, 0, stream>>>(
                src0_d, src1_d, dst_d,
                ne_total_pairs,
                ne10, ne11, ne12, ne13,
                s01, s02, s03,
                s10, s11, s12,
                s1, s2, s3,
                ne00_fd, ne01_fd, ne02_fd,
                ne11_fd, ne12_fd);
    }
}

// Template dispatch function for quantized set_rows
template<typename idx_t, typename block_type, int qk, void (*quantize_func)(const float*, block_type*)>
static void set_rows_cuda_quant(
        const float * src0_d, const idx_t * src1_d, block_type * dst_d,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {

    GGML_ASSERT(ne00 % qk == 0);
    const int64_t ne_total = (ne00 * ne01 * ne02 * ne03) / qk;
    const int num_blocks = (ne_total + CUDA_SET_ROWS_BLOCK_SIZE - 1) / CUDA_SET_ROWS_BLOCK_SIZE;
    const dim3 block_size(CUDA_SET_ROWS_BLOCK_SIZE);
    const dim3 grid_size(num_blocks);

    const int64_t s01 = nb01/sizeof(float);
    const int64_t s02 = nb02/sizeof(float);
    const int64_t s03 = nb03/sizeof(float);
    const int64_t s10 = nb10/sizeof(idx_t);
    const int64_t s11 = nb11/sizeof(idx_t);
    const int64_t s12 = nb12/sizeof(idx_t);
    const int64_t s1  = nb1;
    const int64_t s2  = nb2;
    const int64_t s3  = nb3;

    if (ne_total > 0 && ne00 > 0 && ne01 > 0 && ne02 > 0 && ne11 > 0 && ne12 > 0) {
        const uint3 ne00_fd = init_fastdiv_values((uint32_t) ne00);
        const uint3 ne01_fd = init_fastdiv_values((uint32_t) ne01);
        const uint3 ne02_fd = init_fastdiv_values((uint32_t) ne02);
        const uint3 ne11_fd = init_fastdiv_values((uint32_t) ne11);
        const uint3 ne12_fd = init_fastdiv_values((uint32_t) ne12);

        k_set_rows_quant<idx_t, block_type, qk, quantize_func><<<grid_size, block_size, 0, stream>>>(
            src0_d, src1_d, dst_d, ne_total, ne10, ne11, ne12, ne13, s01, s02, s03, s10, s11, s12, s1, s2, s3, ne00_fd,
            ne01_fd, ne02_fd, ne11_fd, ne12_fd);
    }
}

// ============================================================
// Cross-head WHT kernel v1: simple 1-thread-per-group (proven correct, PPL=1340)
// ============================================================
template <typename idx_t, typename block_type,
    void (*quantize_xhead)(const float*, const float*, const float*, const float*,
                           const float*, const float*, const float*, const float*,
                           block_type*, block_type*, block_type*, block_type*,
                           block_type*, block_type*, block_type*, block_type*)>
static __global__ void k_set_rows_xhead_v1(
        const float * __restrict__ src0, const idx_t * __restrict__ src1, block_type * __restrict__ dst,
        const int64_t ne_total_groups,
        const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t s10, const int64_t s11, const int64_t s12,
        const int64_t ne11, const int64_t ne12,
        const int64_t s1, const int64_t s2, const int64_t s3,
        const int64_t groups_per_ne00,
        const uint3 ne01_fd, const uint3 ne02_fd, const uint3 ne11_fd, const uint3 ne12_fd) {
    const int64_t ig = int64_t(blockDim.x) * blockIdx.x + threadIdx.x;
    if (ig >= ne_total_groups) return;
    const int64_t i_g00 = ig % groups_per_ne00;
    const int64_t r1 = ig / groups_per_ne00;
    uint2 div_mod = fast_div_modulo((uint32_t)r1, ne01_fd);
    const int64_t i01 = div_mod.y; uint32_t tmp2 = div_mod.x;
    div_mod = fast_div_modulo(tmp2, ne02_fd);
    const int64_t i02 = div_mod.y, i03 = div_mod.x;
    const int64_t i12 = fastmodulo((uint32_t)i03, ne12_fd);
    const int64_t i11v = fastmodulo((uint32_t)i02, ne11_fd);
    const idx_t dst_row = *(src1 + i01*s10 + i11v*s11 + i12*s12);
    const int64_t blk_base = i_g00 * 8;
    const float * sb = src0 + i01*s01 + i02*s02 + i03*s03;
    block_type * db = (block_type *)((char *)dst + dst_row*s1 + i02*s2 + i03*s3);
    quantize_xhead(
        sb+(blk_base+0)*TBQ_K64, sb+(blk_base+1)*TBQ_K64, sb+(blk_base+2)*TBQ_K64, sb+(blk_base+3)*TBQ_K64,
        sb+(blk_base+4)*TBQ_K64, sb+(blk_base+5)*TBQ_K64, sb+(blk_base+6)*TBQ_K64, sb+(blk_base+7)*TBQ_K64,
        db+blk_base+0, db+blk_base+1, db+blk_base+2, db+blk_base+3,
        db+blk_base+4, db+blk_base+5, db+blk_base+6, db+blk_base+7);
    GGML_UNUSED(ne11); GGML_UNUSED(ne12);
}

template<typename idx_t, typename block_type,
    void (*quantize_xhead)(const float*, const float*, const float*, const float*,
                           const float*, const float*, const float*, const float*,
                           block_type*, block_type*, block_type*, block_type*,
                           block_type*, block_type*, block_type*, block_type*)>
static void set_rows_cuda_xhead_v1(
        const float * src0_d, const idx_t * src1_d, block_type * dst_d,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {
    GGML_ASSERT(ne00 % TBQ_K64 == 0);
    const int64_t blks_per_ne00 = ne00 / TBQ_K64;
    GGML_ASSERT(blks_per_ne00 % 8 == 0);
    const int64_t groups_per_ne00 = blks_per_ne00 / 8;
    const int64_t n_total = groups_per_ne00 * ne01 * ne02 * ne03;
    if (n_total == 0) return;
    constexpr int BS = 1; // 1 thread/block: heavy per-thread work (2KB+ stack)
    const int nb = (n_total + BS - 1) / BS;
    const uint3 ne01_fd = init_fastdiv_values((uint32_t)ne01);
    const uint3 ne02_fd = init_fastdiv_values((uint32_t)ne02);
    const uint3 ne11_fd = init_fastdiv_values((uint32_t)ne11);
    const uint3 ne12_fd = init_fastdiv_values((uint32_t)ne12);
    k_set_rows_xhead_v1<idx_t, block_type, quantize_xhead><<<nb, BS, 0, stream>>>(
        src0_d, src1_d, dst_d, n_total,
        nb01/sizeof(float), nb02/sizeof(float), nb03/sizeof(float),
        nb10/sizeof(idx_t), nb11/sizeof(idx_t), nb12/sizeof(idx_t),
        ne11, ne12, (int64_t)nb1, (int64_t)nb2, (int64_t)nb3,
        groups_per_ne00, ne01_fd, ne02_fd, ne11_fd, ne12_fd);
    GGML_UNUSED(ne00); GGML_UNUSED(ne10); GGML_UNUSED(ne13);
}

// ============================================================
// Cross-head WHT kernel v2: Kronecker decomposition + shared memory
// H_512 = H_8 ⊗ H_64 → 8 parallel per-head WHTs + 3-stage cross-head WHT
// 8 threads per group, each handles one head's 64-element WHT
// Shared memory: 8×64 floats = 2KB per group for cross-head exchange
// ============================================================

// Per-head quantize helpers (called by each thread for its 64 elements)
// These operate on tmp[64] in registers, writing to the output block
template<typename block_type>
static __device__ void xhead_quantize_tbq3(float * tmp, block_type * out, float norm) {
    static constexpr float b3[7] = { -1.7480f,-1.0500f,-0.5006f,0.0f,0.5006f,1.0500f,1.7480f };
    out->d = __float2half(norm);
    int bit_pos = 0;
    for (int j = 0; j < TBQ_K64*3/8; j++) out->qs[j] = 0;
    for (int j = 0; j < TBQ_K64; j++) {
        uint8_t idx = 7;
        for (int b = 0; b < 7; b++) { if (tmp[j] < b3[b]) { idx = b; break; } }
        int byte_idx = bit_pos/8, bit_off = bit_pos%8;
        out->qs[byte_idx] |= (uint8_t)((idx & 0x7) << bit_off);
        if (bit_off > 5) out->qs[byte_idx+1] |= (uint8_t)((idx & 0x7) >> (8-bit_off));
        bit_pos += 3;
    }
}

template<typename block_type>
static __device__ void xhead_quantize_tbq4(float * tmp, block_type * out, float norm) {
    static constexpr float b4[15] = { -2.4008f,-1.8435f,-1.4371f,-1.0993f,-0.7996f,-0.5225f,-0.2583f,0.0f,0.2583f,0.5225f,0.7996f,1.0993f,1.4371f,1.8435f,2.4008f };
    out->d = __float2half(norm);
    for (int j = 0; j < TBQ_K64/2; j++) {
        uint8_t idx0 = 15, idx1 = 15;
        for (int b = 0; b < 15; b++) { if (tmp[2*j]   < b4[b]) { idx0 = b; break; } }
        for (int b = 0; b < 15; b++) { if (tmp[2*j+1] < b4[b]) { idx1 = b; break; } }
        out->qs[j] = idx0 | (idx1 << 4);
    }
}

template<typename block_type>
static __device__ void xhead_quantize_tbqp3(float * tmp, block_type * out, float norm) {
    static constexpr float c2[4] = { -1.5104f,-0.4528f,0.4528f,1.5104f };
    static constexpr float b2[3] = { -0.9816f,0.0f,0.9816f };
    out->d = __float2half(norm);
    float recon[TBQ_K64];
    for (int j = 0; j < TBQ_K64/4; j++) {
        uint8_t packed = 0;
        for (int k = 0; k < 4; k++) {
            uint8_t idx = 3;
            for (int b = 0; b < 3; b++) { if (tmp[j*4+k] < b2[b]) { idx = b; break; } }
            packed |= (idx & 0x3) << (k*2);
            recon[j*4+k] = c2[idx];
        }
        out->qs[j] = packed;
    }
    float res_abs = 0.0f;
    for (int j = 0; j < TBQ_K64; j++) res_abs += fabsf(tmp[j] - recon[j]);
    out->d_qjl = __float2half((res_abs/TBQ_K64)*norm);
    for (int j = 0; j < TBQ_K64/8; j++) out->qjl[j] = 0;
    for (int j = 0; j < TBQ_K64; j++) { if (tmp[j]-recon[j] >= 0.0f) out->qjl[j/8] |= (1<<(j%8)); }
}

template<typename block_type>
static __device__ void xhead_quantize_tbqp4(float * tmp, block_type * out, float norm) {
    static constexpr float c3[8] = { -2.1520f,-1.3440f,-0.7560f,-0.2451f,0.2451f,0.7560f,1.3440f,2.1520f };
    static constexpr float b3[7] = { -1.7480f,-1.0500f,-0.5006f,0.0f,0.5006f,1.0500f,1.7480f };
    out->d = __float2half(norm);
    float recon[TBQ_K64];
    int bit_pos = 0;
    for (int j = 0; j < TBQ_K64*3/8; j++) out->qs[j] = 0;
    for (int j = 0; j < TBQ_K64; j++) {
        uint8_t idx = 7;
        for (int b = 0; b < 7; b++) { if (tmp[j] < b3[b]) { idx = b; break; } }
        int byte_idx = bit_pos/8, bit_off = bit_pos%8;
        out->qs[byte_idx] |= (uint8_t)((idx&0x7)<<bit_off);
        if (bit_off > 5) out->qs[byte_idx+1] |= (uint8_t)((idx&0x7)>>(8-bit_off));
        recon[j] = c3[idx]; bit_pos += 3;
    }
    float res_abs = 0.0f;
    for (int j = 0; j < TBQ_K64; j++) res_abs += fabsf(tmp[j] - recon[j]);
    out->d_qjl = __float2half((res_abs/TBQ_K64)*norm);
    for (int j = 0; j < TBQ_K64/8; j++) out->qjl[j] = 0;
    for (int j = 0; j < TBQ_K64; j++) { if (tmp[j]-recon[j] >= 0.0f) out->qjl[j/8] |= (1<<(j%8)); }
}

// Quantize type tag for template dispatch
enum xhead_qtype { XHEAD_TBQ3, XHEAD_TBQ4, XHEAD_TBQP3, XHEAD_TBQP4 };

template <typename idx_t, typename block_type, xhead_qtype QT>
static __global__ void k_set_rows_xhead_v2(
        const float * __restrict__ src0,
        const idx_t * __restrict__ src1,
        block_type  * __restrict__ dst,
        const int64_t ne_total_groups,
        const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t s10, const int64_t s11, const int64_t s12,
        const int64_t ne11, const int64_t ne12,
        const int64_t s1, const int64_t s2, const int64_t s3,
        const int64_t groups_per_ne00,
        const uint3 ne01_fd, const uint3 ne02_fd,
        const uint3 ne11_fd, const uint3 ne12_fd) {

    // 8 threads per group. threadIdx.x % 8 = head index within group.
    const int h = threadIdx.x % 8;          // head index within 8-head group
    const int local_group = threadIdx.x / 8; // which group within this block
    const int64_t ig = int64_t(blockIdx.x) * (blockDim.x / 8) + local_group;
    const bool active = (ig < ne_total_groups);

    // Shared memory: 8×64 floats per group for cross-head WHT exchange
    extern __shared__ float smem[];
    float * xhead_smem = smem + local_group * 8 * TBQ_K64; // this group's 512-float region

    // Decompose group index (only meaningful if active, but compute anyway to avoid branching)
    int64_t i_g00 = 0, i01 = 0, i02 = 0, i03 = 0;
    idx_t dst_row = 0;
    int64_t blk_base = 0;
    float tmp[TBQ_K64];
    float sum_sq_local = 0.0f;

    if (active) {
        i_g00 = ig % groups_per_ne00;
        const int64_t r1 = ig / groups_per_ne00;
        uint2 div_mod = fast_div_modulo((uint32_t)r1, ne01_fd);
        i01 = div_mod.y;
        uint32_t tmp2 = div_mod.x;
        div_mod = fast_div_modulo(tmp2, ne02_fd);
        i02 = div_mod.y;
        i03 = div_mod.x;

        const int64_t i12 = fastmodulo((uint32_t)i03, ne12_fd);
        const int64_t i11_val = fastmodulo((uint32_t)i02, ne11_fd);
        dst_row = *(src1 + i01*s10 + i11_val*s11 + i12*s12);
        blk_base = i_g00 * 8;

        // Each thread loads its head's 64 floats
        const float * src_head = src0 + i01*s01 + i02*s02 + i03*s03 + (blk_base + h)*TBQ_K64;
        for (int j = 0; j < TBQ_K64; j++) {
            float v = src_head[j];
            tmp[j] = v;
            sum_sq_local += v * v;
        }
    } else {
        for (int j = 0; j < TBQ_K64; j++) tmp[j] = 0.0f;
    }

    // Warp-level reduction for shared norm (sum across 8 heads)
    float sum_sq = sum_sq_local;
    #pragma unroll
    for (int offset = 4; offset >= 1; offset >>= 1) {
        sum_sq += __shfl_xor_sync(0xff << (local_group*8), sum_sq, offset, WARP_SIZE);
    }
    // Now all 8 threads have the same sum_sq (total across all heads)
    float norm = sqrtf(sum_sq);

    const bool zero_norm = (norm < 1e-10f);
    if (active && zero_norm) {
        block_type * out = (block_type *)((char *)dst + dst_row*s1 + i02*s2 + i03*s3) + blk_base + h;
        out->d = __float2half(0.0f);
        for (int j = 0; j < (int)sizeof(out->qs); j++) out->qs[j] = 0;
    }
    // Don't return early — must participate in __syncthreads below
    if (!active || zero_norm) {
        // Still need to participate in all syncthreads
        for (int j = 0; j < TBQ_K64; j++) tmp[j] = 0;
    } else {

    // Apply per-head signs + normalize
    static __device__ constexpr uint8_t tbq_signs_512_k[64] = {
        0xa7,0x3b,0x91,0xf4,0x6d,0xc2,0x58,0x0e,
        0xb3,0x7f,0x24,0xd6,0x89,0x45,0xea,0x1c,
        0x63,0xaf,0xd8,0x52,0x97,0x0b,0xe1,0x3d,
        0x76,0xc4,0x19,0xfe,0x4a,0x85,0x2c,0xdb,
        0xd3,0x4e,0xa8,0x17,0x9c,0x5b,0xe6,0x31,
        0x72,0xb9,0x0d,0xf5,0x43,0x8a,0x6e,0xc7,
        0x58,0x2f,0x94,0xe1,0xb6,0x3d,0x0a,0x7c,
        0xc5,0x61,0xd8,0x4f,0xa3,0x97,0x1e,0x85,
    };
    const float inv_norm = 1.0f / norm;
    const int sign_offset = h * 8; // byte offset for this head's 64-bit signs
    for (int j = 0; j < TBQ_K64; j++) {
        const int global_j = h * TBQ_K64 + j;
        int sign = ((tbq_signs_512_k[global_j >> 3] >> (global_j & 7)) & 1) ? -1 : 1;
        tmp[j] *= inv_norm * sign;
    }

    // Phase 1: Per-head 64-element serial WHT (6 stages) — each thread independent
    for (int len = 1; len < TBQ_K64; len *= 2)
        for (int i = 0; i < TBQ_K64; i += 2*len)
            for (int j = 0; j < len; j++) {
                float u = tmp[i+j], v = tmp[i+j+len];
                tmp[i+j] = u+v; tmp[i+j+len] = u-v;
            }
    } // end else (active && !zero_norm)

    // Phase 2: Cross-head 8-element WHT (3 stages) via shared memory
    // All threads participate in syncthreads (including inactive ones)
    // Store per-head WHT results to shared memory
    for (int j = 0; j < TBQ_K64; j++) {
        xhead_smem[h * TBQ_K64 + j] = tmp[j];
    }
    __syncthreads();

    // 3-stage butterfly: each thread reads partner's values, computes in registers
    #pragma unroll
    for (int stage = 0; stage < 3; stage++) {
        const int partner = h ^ (1 << stage);
        for (int j = 0; j < TBQ_K64; j++) {
            float my_val = xhead_smem[h * TBQ_K64 + j];
            float partner_val = xhead_smem[partner * TBQ_K64 + j];
            tmp[j] = (h & (1 << stage)) ? (partner_val - my_val) : (my_val + partner_val);
        }
        __syncthreads();
        for (int j = 0; j < TBQ_K64; j++) {
            xhead_smem[h * TBQ_K64 + j] = tmp[j];
        }
        __syncthreads();
    }

    // Quantize this thread's 64 elements to output block
    if (active && !zero_norm) {
        block_type * out = (block_type *)((char *)dst + dst_row*s1 + i02*s2 + i03*s3) + blk_base + h;
        if constexpr (QT == XHEAD_TBQ3)  { xhead_quantize_tbq3(tmp, out, norm); }
        if constexpr (QT == XHEAD_TBQ4)  { xhead_quantize_tbq4(tmp, out, norm); }
        if constexpr (QT == XHEAD_TBQP3) { xhead_quantize_tbqp3(tmp, out, norm); }
        if constexpr (QT == XHEAD_TBQP4) { xhead_quantize_tbqp4(tmp, out, norm); }
    }

    GGML_UNUSED(ne11); GGML_UNUSED(ne12);
}

template<typename idx_t, typename block_type, xhead_qtype QT>
static void set_rows_cuda_xhead(
        const float * src0_d, const idx_t * src1_d, block_type * dst_d,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {

    GGML_ASSERT(ne00 % TBQ_K64 == 0);
    const int64_t blks_per_ne00 = ne00 / TBQ_K64;
    GGML_ASSERT(blks_per_ne00 % 8 == 0 && "cross-head WHT requires n_kv_head divisible by 8");

    const int64_t groups_per_ne00 = blks_per_ne00 / 8;
    const int64_t ne_total_groups = groups_per_ne00 * ne01 * ne02 * ne03;
    if (ne_total_groups == 0) return;

    // 8 threads per group, multiple groups per block
    constexpr int GROUPS_PER_BLOCK = 4;  // 4 groups × 8 threads = 32 threads/block
    constexpr int THREADS_PER_BLOCK = GROUPS_PER_BLOCK * 8;
    const int num_blocks = (ne_total_groups + GROUPS_PER_BLOCK - 1) / GROUPS_PER_BLOCK;
    const size_t smem_size = GROUPS_PER_BLOCK * 8 * TBQ_K64 * sizeof(float); // 4 × 2KB = 8KB

    const uint3 ne01_fd = init_fastdiv_values((uint32_t)ne01);
    const uint3 ne02_fd = init_fastdiv_values((uint32_t)ne02);
    const uint3 ne11_fd = init_fastdiv_values((uint32_t)ne11);
    const uint3 ne12_fd = init_fastdiv_values((uint32_t)ne12);

    k_set_rows_xhead_v2<idx_t, block_type, QT><<<num_blocks, THREADS_PER_BLOCK, smem_size, stream>>>(
        src0_d, src1_d, dst_d,
        ne_total_groups,
        nb01/sizeof(float), nb02/sizeof(float), nb03/sizeof(float),
        nb10/sizeof(idx_t), nb11/sizeof(idx_t), nb12/sizeof(idx_t),
        ne11, ne12,
        (int64_t)nb1, (int64_t)nb2, (int64_t)nb3,
        groups_per_ne00,
        ne01_fd, ne02_fd, ne11_fd, ne12_fd);

    GGML_UNUSED(ne00); GGML_UNUSED(ne10); GGML_UNUSED(ne13);
}

template <typename src_t, typename idx_t, typename dst_t>
static __global__ void k_set_rows(const src_t * __restrict__ src0,
                                  const idx_t * __restrict__ src1,
                                  dst_t * __restrict__ dst,
                                  const int64_t ne_total,
                                  const int64_t ne10,
                                  const int64_t ne11,
                                  const int64_t ne12,
                                  const int64_t ne13,
                                  const int64_t s01,
                                  const int64_t s02,
                                  const int64_t s03,
                                  const int64_t s10,
                                  const int64_t s11,
                                  const int64_t s12,
                                  const int64_t s1,
                                  const int64_t s2,
                                  const int64_t s3,
                                  const uint3   ne00,
                                  const uint3   ne01,
                                  const uint3   ne02,
                                  const uint3   ne11_fd,
                                  const uint3   ne12_fd) {
    const int64_t i = int64_t(blockDim.x) * blockIdx.x + threadIdx.x;

    if (i >= ne_total) {
        return;
    }

    uint32_t tmp = (uint32_t) i;
    uint2    div_mod;

    div_mod           = fast_div_modulo(tmp, ne00);
    const int64_t i00 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne01);
    const int64_t i01 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne02);
    const int64_t i02 = div_mod.y;
    const int64_t i03 = div_mod.x;

    const int64_t i12 = fastmodulo((uint32_t) i03, ne12_fd);
    const int64_t i11 = fastmodulo((uint32_t) i02, ne11_fd);
    const int64_t i10 = i01;

    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);

    const src_t * src0_row = src0 + i01*s01 + i02*s02 + i03*s03;
    dst_t * dst_row_ptr    = dst + dst_row*s1 + i02*s2 + i03*s3;

    dst_row_ptr[i00] = ggml_cuda_cast<dst_t>(src0_row[i00]);

    GGML_UNUSED(ne10);
    GGML_UNUSED(ne11);
    GGML_UNUSED(ne12);
    GGML_UNUSED(ne13);
}

template<typename src_t, typename idx_t, typename dst_t>
static void set_rows_cuda(
        const src_t * src0_d, const idx_t * src1_d, dst_t * dst_d,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {

    const int64_t ne_total = ne00 * ne01 * ne02 * ne03;
    const int num_blocks = (ne_total + CUDA_SET_ROWS_BLOCK_SIZE - 1) / CUDA_SET_ROWS_BLOCK_SIZE;
    const dim3 block_size(CUDA_SET_ROWS_BLOCK_SIZE);
    const dim3 grid_size(num_blocks);


    const int64_t s01 = nb01/sizeof(src_t);
    const int64_t s02 = nb02/sizeof(src_t);
    const int64_t s03 = nb03/sizeof(src_t);
    const int64_t s10 = nb10/sizeof(idx_t);
    const int64_t s11 = nb11/sizeof(idx_t);
    const int64_t s12 = nb12/sizeof(idx_t);
    const int64_t s1  = nb1/sizeof(dst_t);
    const int64_t s2  = nb2/sizeof(dst_t);
    const int64_t s3  = nb3/sizeof(dst_t);

    if (ne_total > 0 && ne00 > 0 && ne01 > 0 && ne02 > 0 && ne11 > 0 && ne12 > 0) {
        const uint3 ne00_fd = init_fastdiv_values((uint32_t) ne00);
        const uint3 ne01_fd = init_fastdiv_values((uint32_t) ne01);
        const uint3 ne02_fd = init_fastdiv_values((uint32_t) ne02);
        const uint3 ne11_fd = init_fastdiv_values((uint32_t) ne11);
        const uint3 ne12_fd = init_fastdiv_values((uint32_t) ne12);

        k_set_rows<<<grid_size, block_size, 0, stream>>>(src0_d, src1_d, dst_d, ne_total, ne10, ne11, ne12, ne13, s01,
                                                         s02, s03, s10, s11, s12, s1, s2, s3, ne00_fd, ne01_fd, ne02_fd,
                                                         ne11_fd, ne12_fd);
    }
}

template<typename src_t, typename idx_t>
static void set_rows_cuda(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const src_t * src0_d = (const src_t *)src0->data;
    const idx_t * src1_d = (const idx_t *)src1->data;

    GGML_TENSOR_BINARY_OP_LOCALS

    cudaStream_t stream = ctx.stream();


    if (dst->type == GGML_TYPE_F32) {
        set_rows_cuda(
            src0_d, src1_d, (float*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_F16) {
        set_rows_cuda(
            src0_d, src1_d, (half*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_BF16) {
        set_rows_cuda(
            src0_d, src1_d, (nv_bfloat16*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q4_0) {
        set_rows_cuda_quant<idx_t, block_q4_0, QK4_0, quantize_f32_q4_0_block>(
            src0_d, src1_d, (block_q4_0*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q4_1) {
        set_rows_cuda_quant<idx_t, block_q4_1, QK4_1, quantize_f32_q4_1_block>(
            src0_d, src1_d, (block_q4_1*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q5_0) {
        set_rows_cuda_quant<idx_t, block_q5_0, QK5_0, quantize_f32_q5_0_block>(
            src0_d, src1_d, (block_q5_0*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q5_1) {
        set_rows_cuda_quant<idx_t, block_q5_1, QK5_1, quantize_f32_q5_1_block>(
            src0_d, src1_d, (block_q5_1*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q8_0) {
        set_rows_cuda_quant<idx_t, block_q8_0, QK8_0, quantize_f32_q8_0_block>(
            src0_d, src1_d, (block_q8_0*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_IQ4_NL) {
        set_rows_cuda_quant<idx_t, block_iq4_nl, QK4_NL, quantize_f32_iq4_nl_block>(
            src0_d, src1_d, (block_iq4_nl*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_TBQ4_0) {
        set_rows_cuda_quant<idx_t, block_tbq4_0, QK_K, quantize_f32_tbq4_0_block>(
            src0_d, src1_d, (block_tbq4_0*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_TBQP3_0) {
        // Note: D=512 TBQP3 auto-downgraded to TBQ3 in common.cpp (QJL ineffective at D=512)
        set_rows_cuda_quant<idx_t, block_tbqp3_0, QK_K, quantize_f32_tbqp3_0_block>(
            src0_d, src1_d, (block_tbqp3_0*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_TBQP4_0) {
        set_rows_cuda_quant<idx_t, block_tbqp4_0, QK_K, quantize_f32_tbqp4_0_block>(
            src0_d, src1_d, (block_tbqp4_0*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_TBQ3_0) {
        const int32_t head_dim = dst->op_params[0];
        const int32_t no_wht   = dst->op_params[1]; // 1 = skip WHT (V cache experiment)
        if (no_wht == 1) {
            // no_wht is no longer set for V — this path is now unused for V
            set_rows_cuda_quant<idx_t, block_tbq3_0, QK_K, quantize_f32_tbq3_0_block_nowht>(
                src0_d, src1_d, (block_tbq3_0*)dst->data,
                ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream
            );
        } else if (head_dim >= 512) {
            set_rows_cuda_quant_512<idx_t, block_tbq3_0, QK_K, quantize_f32_tbq3_0_block_512>(
                src0_d, src1_d, (block_tbq3_0*)dst->data,
                ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream
            );
        } else {
            set_rows_cuda_quant<idx_t, block_tbq3_0, QK_K, quantize_f32_tbq3_0_block>(
                src0_d, src1_d, (block_tbq3_0*)dst->data,
                ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream
            );
        }
    } else if (dst->type == GGML_TYPE_TBQ3_1) {
        set_rows_cuda_quant<idx_t, block_tbq3_1, TBQ_K128, quantize_f32_tbq3_1_block>(
            src0_d, src1_d, (block_tbq3_1*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_TBQ4_1) {
        set_rows_cuda_quant<idx_t, block_tbq4_1, TBQ_K128, quantize_f32_tbq4_1_block>(
            src0_d, src1_d, (block_tbq4_1*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_TBQP3_1) {
        set_rows_cuda_quant<idx_t, block_tbqp3_1, TBQ_K128, quantize_f32_tbqp3_1_block>(
            src0_d, src1_d, (block_tbqp3_1*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_TBQP4_1) {
        set_rows_cuda_quant<idx_t, block_tbqp4_1, TBQ_K128, quantize_f32_tbqp4_1_block>(
            src0_d, src1_d, (block_tbqp4_1*)dst->data,
            ne00, ne01, ne02, ne03, ne10, ne11, ne12, ne13, nb01, nb02, nb03, nb10, nb11, nb12, nb1, nb2, nb3, stream);
    } else if (dst->type == GGML_TYPE_TBQ3_2) {
        set_rows_cuda_quant<idx_t, block_tbq3_2, TBQ_K64, quantize_f32_tbq3_2_block>(
            src0_d, src1_d, (block_tbq3_2*)dst->data,
            ne00, ne01, ne02, ne03, ne10, ne11, ne12, ne13, nb01, nb02, nb03, nb10, nb11, nb12, nb1, nb2, nb3, stream);
    } else if (dst->type == GGML_TYPE_TBQ4_2) {
        set_rows_cuda_quant<idx_t, block_tbq4_2, TBQ_K64, quantize_f32_tbq4_2_block>(
            src0_d, src1_d, (block_tbq4_2*)dst->data,
            ne00, ne01, ne02, ne03, ne10, ne11, ne12, ne13, nb01, nb02, nb03, nb10, nb11, nb12, nb1, nb2, nb3, stream);
    } else if (dst->type == GGML_TYPE_TBQP3_2) {
        set_rows_cuda_quant<idx_t, block_tbqp3_2, TBQ_K64, quantize_f32_tbqp3_2_block>(
            src0_d, src1_d, (block_tbqp3_2*)dst->data,
            ne00, ne01, ne02, ne03, ne10, ne11, ne12, ne13, nb01, nb02, nb03, nb10, nb11, nb12, nb1, nb2, nb3, stream);
    } else if (dst->type == GGML_TYPE_TBQP4_2) {
        set_rows_cuda_quant<idx_t, block_tbqp4_2, TBQ_K64, quantize_f32_tbqp4_2_block>(
            src0_d, src1_d, (block_tbqp4_2*)dst->data,
            ne00, ne01, ne02, ne03, ne10, ne11, ne12, ne13, nb01, nb02, nb03, nb10, nb11, nb12, nb1, nb2, nb3, stream);
    } else if (dst->type == GGML_TYPE_TBQ3_4) {
        set_rows_cuda_quant<idx_t, block_tbq3_4, TBQ_K576, quantize_f32_tbq3_4_block>(
            src0_d, src1_d, (block_tbq3_4*)dst->data,
            ne00, ne01, ne02, ne03, ne10, ne11, ne12, ne13, nb01, nb02, nb03, nb10, nb11, nb12, nb1, nb2, nb3, stream);
    } else if (dst->type == GGML_TYPE_TBQ4_4) {
        set_rows_cuda_quant<idx_t, block_tbq4_4, TBQ_K576, quantize_f32_tbq4_4_block>(
            src0_d, src1_d, (block_tbq4_4*)dst->data,
            ne00, ne01, ne02, ne03, ne10, ne11, ne12, ne13, nb01, nb02, nb03, nb10, nb11, nb12, nb1, nb2, nb3, stream);
    } else if (dst->type == GGML_TYPE_TBQP3_4) {
        set_rows_cuda_quant<idx_t, block_tbqp3_4, TBQ_K576, quantize_f32_tbqp3_4_block>(
            src0_d, src1_d, (block_tbqp3_4*)dst->data,
            ne00, ne01, ne02, ne03, ne10, ne11, ne12, ne13, nb01, nb02, nb03, nb10, nb11, nb12, nb1, nb2, nb3, stream);
    } else if (dst->type == GGML_TYPE_TBQP4_4) {
        set_rows_cuda_quant<idx_t, block_tbqp4_4, TBQ_K576, quantize_f32_tbqp4_4_block>(
            src0_d, src1_d, (block_tbqp4_4*)dst->data,
            ne00, ne01, ne02, ne03, ne10, ne11, ne12, ne13, nb01, nb02, nb03, nb10, nb11, nb12, nb1, nb2, nb3, stream);
    } else if (dst->type == GGML_TYPE_TBQ3_3) {
        set_rows_cuda_xhead_v1<idx_t, block_tbq3_3, quantize_f32_tbq3_3_xhead>(
            src0_d, src1_d, (block_tbq3_3*)dst->data,
            ne00, ne01, ne02, ne03, ne10, ne11, ne12, ne13, nb01, nb02, nb03, nb10, nb11, nb12, nb1, nb2, nb3, stream);
    } else if (dst->type == GGML_TYPE_TBQ4_3) {
        set_rows_cuda_xhead_v1<idx_t, block_tbq4_3, quantize_f32_tbq4_3_xhead>(
            src0_d, src1_d, (block_tbq4_3*)dst->data,
            ne00, ne01, ne02, ne03, ne10, ne11, ne12, ne13, nb01, nb02, nb03, nb10, nb11, nb12, nb1, nb2, nb3, stream);
    } else if (dst->type == GGML_TYPE_TBQP3_3) {
        set_rows_cuda_xhead_v1<idx_t, block_tbqp3_3, quantize_f32_tbqp3_3_xhead>(
            src0_d, src1_d, (block_tbqp3_3*)dst->data,
            ne00, ne01, ne02, ne03, ne10, ne11, ne12, ne13, nb01, nb02, nb03, nb10, nb11, nb12, nb1, nb2, nb3, stream);
    } else if (dst->type == GGML_TYPE_TBQP4_3) {
        set_rows_cuda_xhead_v1<idx_t, block_tbqp4_3, quantize_f32_tbqp4_3_xhead>(
            src0_d, src1_d, (block_tbqp4_3*)dst->data,
            ne00, ne01, ne02, ne03, ne10, ne11, ne12, ne13, nb01, nb02, nb03, nb10, nb11, nb12, nb1, nb2, nb3, stream);
    } else {
        GGML_ABORT("unsupported type %s", ggml_type_name(dst->type));
    }
}


void ggml_cuda_op_set_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_I64 || src1->type == GGML_TYPE_I32);

    if (src1->type == GGML_TYPE_I64) {
        set_rows_cuda<float, int64_t>(ctx, src0, src1, dst);
    } else {
        set_rows_cuda<float, int32_t>(ctx, src0, src1, dst);
    }
}
