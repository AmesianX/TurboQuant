// Compile-only probe: which MmaTileShape M values instantiate the sm120 NVFP4
// blockscaled GROUPED GEMM type tree? Mirrors dsv4-moe-grouped.cu's tree exactly,
// parameterized on TBS_M via -DTBS_M=NN. If get_workspace_size / can_implement
// instantiate, the tile is structurally feasible. We DON'T run; we just force the
// full template instantiation by referencing the kernel + adapter types.
#include <cstdio>
#include <cutlass/cutlass.h>
#include <cute/tensor.hpp>
#include <cutlass/gemm/group_array_problem_shape.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/util/packed_stride.hpp>
#include <cutlass/float8.h>
#include <cutlass/float_subbyte.h>
#include <cutlass/bfloat16.h>

using namespace cute;

#ifndef TBS_M
#define TBS_M 128
#endif
#ifndef TBS_N
#define TBS_N 128
#endif

using ProblemShape  = cutlass::gemm::GroupProblemShape<Shape<int,int,int>>;
using ElementInput  = cutlass::float_e2m1_t;
using ElementA      = cutlass::nv_float4_t<ElementInput>;
using ElementB      = cutlass::nv_float4_t<ElementInput>;
using LayoutATag    = cutlass::layout::RowMajor;
using LayoutBTag    = cutlass::layout::ColumnMajor;
constexpr int AlignmentA = 32, AlignmentB = 32;
using ElementD      = cutlass::bfloat16_t;
using ElementC      = cutlass::bfloat16_t;
using LayoutCTag    = cutlass::layout::RowMajor;
using LayoutDTag    = cutlass::layout::RowMajor;
constexpr int AlignmentC = 128/cutlass::sizeof_bits<ElementC>::value;
constexpr int AlignmentD = 128/cutlass::sizeof_bits<ElementD>::value;
using ElementAccumulator = float;
using ElementCompute     = float;
using ArchTag       = cutlass::arch::Sm120;
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;
using ThreadBlockShape = Shape<Int<TBS_M>,Int<TBS_N>,_128>;
using ClusterShape     = Shape<_1,_1,_1>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass, ThreadBlockShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator, ElementCompute,
    ElementC, LayoutCTag*, AlignmentC, ElementD, LayoutDTag*, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass, ElementA, LayoutATag*, AlignmentA, ElementB, LayoutBTag*, AlignmentB,
    ElementAccumulator, ThreadBlockShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloop, CollectiveEpilogue>;
using Gemm       = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
using StrideA    = typename Gemm::GemmKernel::InternalStrideA;
using LayoutSFA  = typename Gemm::GemmKernel::CollectiveMainloop::InternalLayoutSFA;
using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

// force instantiation
__attribute__((used)) size_t probe(){
  Gemm g;
  typename Gemm::Arguments args{};
  size_t w = Gemm::get_workspace_size(args);
  (void)g;
  return w + sizeof(StrideA) + sizeof(LayoutSFA) + sizeof(Sm1xxBlkScaledConfig);
}
int main(){ printf("TBS_M=%d TBS_N=%d workspace_probe=%zu\n", TBS_M, TBS_N, probe()); return 0; }
