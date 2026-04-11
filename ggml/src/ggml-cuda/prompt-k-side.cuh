// TurboQuant: f16 side buffer for K (indexed write, matches set-rows KV positions)
#pragma once

#include <cuda_fp16.h>
#include <cstdint>
#include <mutex>
#include <unordered_map>

namespace turboquant {

struct prompt_k_side_t {
    half * buf      = nullptr;
    int    n_tokens = 0;
    int    capacity = 0;
    int    ne00     = 0;
};

inline std::mutex & get_side_mutex() {
    static std::mutex m;
    return m;
}

inline std::unordered_map<const void*, prompt_k_side_t> & get_side_map() {
    static std::unordered_map<const void*, prompt_k_side_t> m;
    return m;
}

inline void reset_all_sides() {
    std::lock_guard<std::mutex> lock(get_side_mutex());
    for (auto & [k, v] : get_side_map()) {
        v.n_tokens = 0;
    }
}

// f32 → f16 kernel with src1 index mapping (writes to KV cache positions)
template <typename idx_t>
static __global__ void k_f32_to_f16_indexed(
    const float * __restrict__ src,
    const idx_t * __restrict__ indices,
    half * __restrict__ dst,
    int ne00, int ne01, int s01) {

    const int i_token = blockIdx.x;
    const int i_elem = threadIdx.x;
    if (i_token >= ne01 || i_elem >= ne00) return;

    const int dst_row = (int)indices[i_token];
    dst[dst_row * ne00 + i_elem] = __float2half(src[i_token * s01 + i_elem]);
}

template <typename idx_t>
inline void write_side(const void * dst_data, const float * src0_d, const idx_t * src1_d,
                       int ne00, int ne01, int nb01, cudaStream_t stream,
                       bool force_reset = false) {
    std::lock_guard<std::mutex> lock(get_side_mutex());
    auto & m = get_side_map();
    auto & side = m[dst_data];
    side.ne00 = ne00;

    if (force_reset) {
        side.n_tokens = 0;
        // Shrink oversized buffer to reclaim memory
        if (side.capacity > 8192) {
            if (side.buf) cudaFree(side.buf);
            side.buf = nullptr;
            side.capacity = 0;
        }
    }

    int needed = side.n_tokens + ne01;
    if (needed > side.capacity) {
        int new_cap = needed + 4096;
        half * new_buf = nullptr;
        size_t alloc_bytes = (size_t)new_cap * ne00 * sizeof(half);
        cudaMalloc(&new_buf, alloc_bytes);
        cudaMemsetAsync(new_buf, 0, alloc_bytes, stream);
        if (side.buf && side.capacity > 0) {
            cudaMemcpyAsync(new_buf, side.buf,
                (size_t)side.capacity * ne00 * sizeof(half),
                cudaMemcpyDeviceToDevice, stream);
        }
        if (side.buf) cudaFree(side.buf);
        side.buf = new_buf;
        side.capacity = new_cap;
    }

    k_f32_to_f16_indexed<<<ne01, ne00, 0, stream>>>(
        src0_d, src1_d, side.buf,
        ne00, ne01, nb01 / (int)sizeof(float));
    side.n_tokens += ne01;
}

inline const char * get_prompt_k_ptr(const void * k_data) {
    std::lock_guard<std::mutex> lock(get_side_mutex());
    auto & m = get_side_map();
    auto it = m.find(k_data);
    if (it != m.end() && it->second.buf && it->second.n_tokens > 0) {
        return (const char *)it->second.buf;
    }
    return nullptr;
}

inline int32_t get_prompt_k_stride(const void * k_data) {
    std::lock_guard<std::mutex> lock(get_side_mutex());
    auto & m = get_side_map();
    auto it = m.find(k_data);
    if (it != m.end() && it->second.buf && it->second.n_tokens > 0) {
        return it->second.n_tokens;
    }
    return 0;
}

} // namespace turboquant
