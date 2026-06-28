// DSV4 NVFP4 (W4A4) FUSED MoE op -- flashinfer CUTLASS CutlassMoeFCRunner.
//
// ROUND 1 SKELETON: op registration + registry plumbing + input validation.
// The heavy CUTLASS/trtllm fused-runner instantiation (CutlassMoeFCRunner<
// __nv_fp4_e2m1,__nv_fp4_e2m1,bf16,bf16>) lives in a SEPARATE TU
// (dsv4-moe-fused-cutlass.cu, Round 2) so only it pays the long compile and this
// op TU stays light -- mirroring how dsv4-moe-gemm.cu isolates the grouped-GEMM
// instantiation. This TU calls that TU through an extern-"C" shim guarded by
// DSV4_MOE_FUSED_CUTLASS (defined once the instantiation TU compiles+links).
//
// Until the shim is built, ggml_cuda_op_dsv4_moe_fused returns false so the graph
// falls back to the proven grouped path (the model graph only routes here when
// DSV4_MOE_FUSED=1 AND the op reports supported). See the port doc.

#include "dsv4-moe-fused.cuh"

#include <cstdint>
#include <cstdio>
#include <cstdlib>

// Registry accessor implemented in dsv4-moe-grouped.cu. Hands us this rank's raw
// NVFP4 device pointers + dims for layer `il` (false if not registered).
extern "C" bool dsv4_moe_grouped_get_layer_nvfp4(
        int il,
        int* n_expert, int* n_embd, int* n_ff_exp,
        const void** dq_gate,  const void** dq_up,  const void** dq_down,
        const void** sf_gate_simple, const void** sf_up_simple, const void** sf_down_simple,
        const float** global_gate, const float** global_up, const float** global_down);

#ifdef DSV4_MOE_FUSED_CUTLASS
// Round-2 shim: runs the flashinfer CUTLASS fused MoE for one layer. Implemented
// in dsv4-moe-fused-cutlass.cu. Takes this rank's NVFP4 expert registry pointers,
// the per-token routing (selected experts + final scales), the F32 hidden, and
// writes the F32 moe_out. Returns cudaSuccess on success.
extern "C" cudaError_t dsv4_moe_fused_run(
        int il,
        const float* hidden,            // [n_embd, n_tokens] device F32
        const int*   sel,               // [n_expert_used, n_tokens] device I32
        const float* weights,           // [n_expert_used, n_tokens] device F32
        float*       moe_out,           // [n_embd, n_tokens] device F32
        int n_tokens, int n_embd, int n_ff_exp, int n_expert, int n_expert_used,
        float swiglu_limit,
        const void* dq_gate, const void* dq_up, const void* dq_down,
        const void* sf_gate, const void* sf_up, const void* sf_down,
        const float* g_gate, const float* g_up, const float* g_down,
        cudaStream_t stream);
#endif

bool ggml_cuda_op_dsv4_moe_fused_supported(void) {
#ifdef DSV4_MOE_FUSED_CUTLASS
    // Built with the CUTLASS FP4 MoE TU. Runtime arch (sm_121a) is implied by the
    // ggml-cuda build target; the per-layer weight check happens in the op itself.
    return true;
#else
    return false;
#endif
}

bool ggml_cuda_op_dsv4_moe_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * hidden  = dst->src[0]; // [n_embd, n_tokens] F32
    const ggml_tensor * sel     = dst->src[1]; // [n_expert_used, n_tokens] I32
    const ggml_tensor * weights = dst->src[2]; // [n_expert_used, n_tokens] F32

    GGML_ASSERT(hidden  && hidden->type  == GGML_TYPE_F32);
    GGML_ASSERT(sel     && sel->type     == GGML_TYPE_I32);
    GGML_ASSERT(weights && weights->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const int   il           = ggml_get_op_params_i32(dst, 0);
    const float swiglu_limit = ggml_get_op_params_f32(dst, 1);

    const int n_embd        = (int) hidden->ne[0];
    const int n_tokens      = (int) hidden->ne[1];
    const int n_expert_used = (int) sel->ne[0];
    GGML_ASSERT((int) sel->ne[1] == n_tokens);

    // Look up this rank's NVFP4 expert registry for the layer.
    int n_expert = 0, reg_n_embd = 0, n_ff_exp = 0;
    const void *dq_gate = nullptr, *dq_up = nullptr, *dq_down = nullptr;
    const void *sf_gate = nullptr, *sf_up = nullptr, *sf_down = nullptr;
    const float *g_gate = nullptr, *g_up = nullptr, *g_down = nullptr;
    const bool have = dsv4_moe_grouped_get_layer_nvfp4(
        il, &n_expert, &reg_n_embd, &n_ff_exp,
        &dq_gate, &dq_up, &dq_down, &sf_gate, &sf_up, &sf_down,
        &g_gate, &g_up, &g_down);

    if (!have) {
        // Layer not registered -> cannot run fused. Caller treats false as "fall
        // back to the grouped path" (the graph builder only routes here when the
        // op is supported + the layer has NVFP4 weights, so this is defensive).
        return false;
    }
    GGML_ASSERT(reg_n_embd == n_embd);

#ifdef DSV4_MOE_FUSED_CUTLASS
    cudaStream_t stream = ctx.stream();
    const cudaError_t st = dsv4_moe_fused_run(
        il,
        (const float*) hidden->data, (const int*) sel->data, (const float*) weights->data,
        (float*) dst->data,
        n_tokens, n_embd, n_ff_exp, n_expert, n_expert_used, swiglu_limit,
        dq_gate, dq_up, dq_down, sf_gate, sf_up, sf_down, g_gate, g_up, g_down,
        stream);
    if (st != cudaSuccess) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            fprintf(stderr, "[DSV4_MOE_FUSED] runner failed at layer %d: %s -- falling back\n",
                    il, cudaGetErrorString(st));
        }
        return false;
    }
    return true;
#else
    // CUTLASS fused runner not compiled in yet (Round 2). Fall back to grouped.
    (void) ctx; (void) swiglu_limit; (void) n_expert; (void) n_ff_exp;
    (void) dq_gate; (void) dq_up; (void) dq_down;
    (void) sf_gate; (void) sf_up; (void) sf_down;
    (void) g_gate; (void) g_up; (void) g_down; (void) n_expert_used;
    static bool warned = false;
    if (!warned) {
        warned = true;
        fprintf(stderr, "[DSV4_MOE_FUSED] CUTLASS fused runner not built "
                        "(DSV4_MOE_FUSED_CUTLASS undefined); using grouped path.\n");
    }
    return false;
#endif
}
