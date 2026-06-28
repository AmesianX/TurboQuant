// Minimal NVFP4 W4A4 -> BF16 fused-MoE runner instantiation.
//
// flashinfer's own cutlass_fused_moe_instantiation.cu instantiates the FULL set
// of runner variants (half/bf16/fp8/int4/...), each dragging in its own
// MoeGemmRunner<...> that we do NOT compile. We only need the single NVFP4
// act + NVFP4 weight -> bf16 runner (the isNvfp4Quant NeedQuant=true path:
// <ActType, WeightType, OutputType, ScaleBiasType>). Instantiating just that one
// keeps the link closure to moe_gemm_kernels_fp4_fp4.cu + the SM120 launchers.
//
// Builds against flash-attention's vendored CUTLASS 4.3.0 for sm_121a. Heavy
// compile is isolated to this scoped static lib (dsv4-moe-fused-cutlass).

#include "cutlass_fused_moe_kernels.cuh"
#include "moe_kernels.h"

namespace tensorrt_llm::kernels::cutlass_kernels {
template class CutlassMoeFCRunner<__nv_fp4_e2m1, __nv_fp4_e2m1, __nv_bfloat16, __nv_bfloat16>;
}  // namespace tensorrt_llm::kernels::cutlass_kernels
