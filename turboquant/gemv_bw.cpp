// Decode weight-streaming bandwidth: what GB/s do our decode GEMV kernels actually reach?
//
// A DSV4 decode step reads ~4 GB of weights per rank. We take 61.7 ms => 65 GB/s.
// vLLM/b12x takes ~25 ms => 160 GB/s. GB10 DRAM tops out near 230-273 GB/s.
// So the question is not FLOPs, it is: which of our kernels fails to stream at DRAM speed.
//
// This measures each decode-shaped mul_mat (n=1) in isolation and reports achieved GB/s.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cstdio>
#include <cstring>
#include <vector>
#include <string>
#include <chrono>

struct Case { const char * name; ggml_type type; int64_t k; int64_t n; };

int main() {
    ggml_backend_dev_t dev = nullptr;
    printf("devices: %zu\n", ggml_backend_dev_count());
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        printf("  [%zu] %s  type=%d\n", i, ggml_backend_dev_name(d), (int) ggml_backend_dev_type(d));
        if ((ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_GPU || ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_IGPU) && !dev) dev = d;
    }
    if (!dev) { printf("no GPU device\n"); return 1; }
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    if (!backend) { printf("no GPU backend\n"); return 1; }
    printf("backend: %s\n\n", ggml_backend_name(backend));

    // The weights a DSV4 rank streams every decode token (ATTN_SPLIT on => attn halves).
    const std::vector<Case> cases = {
        { "lm_head        f8", GGML_TYPE_F8_E4M3_B128, 4096, 129280 },
        { "lm_head      bf16", GGML_TYPE_BF16,         4096, 129280 },
        { "attn_q_b (split)f8", GGML_TYPE_F8_E4M3_B128, 1024, 16384 },
        { "attn_wo_b(split)f8", GGML_TYPE_F8_E4M3_B128, 4096,  4096 },
        { "attn_q_a       f8", GGML_TYPE_F8_E4M3_B128, 4096,  1024 },
        { "shexp_up       f8", GGML_TYPE_F8_E4M3_B128, 4096,  2048 },
        { "shexp_down     f8", GGML_TYPE_F8_E4M3_B128, 2048,  4096 },
        // reference points: kernels we know are well tuned
        { "big square   bf16", GGML_TYPE_BF16,         4096, 32768 },
        { "big square     f16", GGML_TYPE_F16,         4096, 32768 },
        { "big square    q4_0", GGML_TYPE_Q4_0,        4096, 32768 },
    };

    printf("%-20s %10s %10s %10s   %s\n", "weight", "MiB", "ms/call", "GB/s", "% of 230 GB/s peak");
    printf("---------------------------------------------------------------------------\n");

    for (const auto & c : cases) {
        ggml_init_params ip = { (size_t) 8*1024*1024, nullptr, true };
        ggml_context * ctx = ggml_init(ip);

        ggml_tensor * w = ggml_new_tensor_2d(ctx, c.type,        c.k, c.n);
        ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.k, 1);
        ggml_tensor * y = ggml_mul_mat(ctx, w, x);

        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (!buf) { printf("%-20s  alloc failed (unsupported type?)\n", c.name); ggml_free(ctx); continue; }

        // contents are irrelevant for a bandwidth measurement; just make them defined
        std::vector<uint8_t> zw(ggml_nbytes(w), 0x3c);
        std::vector<float>   zx(c.k, 1.0f);
        ggml_backend_tensor_set(w, zw.data(), 0, zw.size());
        ggml_backend_tensor_set(x, zx.data(), 0, zx.size() * sizeof(float));

        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, y);

        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
            printf("%-20s  compute failed (kernel missing)\n", c.name);
            ggml_backend_buffer_free(buf); ggml_free(ctx); continue;
        }
        ggml_backend_synchronize(backend);

        const int reps = 50;
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < reps; i++) ggml_backend_graph_compute(backend, gf);
        ggml_backend_synchronize(backend);
        auto t1 = std::chrono::high_resolution_clock::now();

        const double ms   = std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;
        const double mib  = ggml_nbytes(w) / (1024.0*1024.0);
        const double gbs  = (ggml_nbytes(w) / 1e9) / (ms / 1e3);
        printf("%-20s %10.1f %10.3f %10.1f   %.0f%%\n", c.name, mib, ms, gbs, 100.0*gbs/230.0);

        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }

    ggml_backend_free(backend);
    return 0;
}
