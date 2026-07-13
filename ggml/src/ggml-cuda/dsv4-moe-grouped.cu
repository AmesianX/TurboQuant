// DSV4 NVFP4 (W4A4) grouped-GEMM MoE op (STEP 2b) -- CUTLASS-heavy translation unit.
//
// Ports the validated standalone pipeline (dsv4_moe_nvfp4_test.cu, mean_rel_err
// 0.024 vs W4A4 ref) into a ggml-cuda custom op + a per-layer device weight
// registry filled by the MXFP4->NVFP4 load adapter (mxfp4_to_nvfp4.cu math).
//
// This file pulls in the full CUTLASS grouped-GEMM template tree -- long compile
// is expected and isolated here. Include paths (cutlass + flashinfer + util) are
// already wired into the ggml-cuda target in CMakeLists.txt.

#include "dsv4-moe-grouped.cuh"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <algorithm>
#include <cmath>

#include <cuda_runtime.h>

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

#define DSV4_CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ fprintf(stderr,"[dsv4-moe-grouped] CUDA ERR %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); abort(); } }while(0)
#define DSV4_CC(x) do{ cutlass::Status cs_=(x); if(cs_!=cutlass::Status::kSuccess){ fprintf(stderr,"[dsv4-moe-grouped] CUTLASS ERR %s:%d status=%d\n",__FILE__,__LINE__,(int)cs_); abort(); } }while(0)

// Grouped-decode HP buffer warm-up counter (fused-prefill -> grouped-decode capture crash fix).
// GLOBAL scope so both the in-namespace run() increment and the post-namespace decode_warmed()
// query (+ the engine gate via extern) share it.
static std::atomic<int> g_grouped_decode_warm_count{0};

namespace dsv4_moe_grouped_detail {
using namespace cute;

// ===== NVFP4 grouped-GEMM type tree (validated in dsv4_moe_nvfp4_test.cu) =====
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
using ThreadBlockShape = Shape<_128,_128,_128>;
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
using StrideB    = typename Gemm::GemmKernel::InternalStrideB;
using StrideD    = typename Gemm::GemmKernel::InternalStrideD;
using LayoutSFA  = typename Gemm::GemmKernel::CollectiveMainloop::InternalLayoutSFA;
using LayoutSFB  = typename Gemm::GemmKernel::CollectiveMainloop::InternalLayoutSFB;
using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
using ElementSF  = typename Gemm::GemmKernel::CollectiveMainloop::ElementSF; // float_ue4m3_t

static constexpr int   SFVec    = 16;
static constexpr float E2M1_MAX = 6.0f, E4M3_MAX = 448.0f;

// ---- per-group prep kernel (validated) --------------------------------------
template<typename PS>
__global__ void prep(ElementInput*A,ElementInput*B,ElementSF*SFA,ElementSF*SFB,ElementD*D,
   int*m_indptr,const int* sfa_off,int n,int k,int ng,PS*ps,
   ElementInput const**Ap,ElementInput const**Bp,ElementSF const**SFAp,ElementSF const**SFBp,
   ElementD**Dp,StrideA*sa,StrideB*sb,StrideD*sd,LayoutSFA*lsa,LayoutSFB*lsb,
   const float* gscaleB,float* alpha_grp,const float** alpha_ptr_grp,const float* gscaleA_ptr,int sfb_words_per_grp,int64_t b_estride_bytes){
  float gscaleA=gscaleA_ptr[0];
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=ng)return;
  int mo=m_indptr[i], m=m_indptr[i+1]-mo;
  int mm = m>0? m : 1;
  ps[i]=PS(m,n,k);
  sa[i]=cutlass::make_cute_packed_stride(StrideA{},{m,k,1});
  sb[i]=cutlass::make_cute_packed_stride(StrideB{},{n,k,1});
  sd[i]=cutlass::make_cute_packed_stride(StrideD{},{m,n,1});
  Ap[i]=reinterpret_cast<ElementInput const*>(reinterpret_cast<uint8_t const*>(A)+(int64_t(mo)*k>>1));
  Bp[i]=reinterpret_cast<ElementInput const*>(reinterpret_cast<uint8_t const*>(B)+int64_t(i)*b_estride_bytes);
  Dp[i]=D+int64_t(mo)*n;
  lsa[i]=Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(mm,n,k,1));
  SFAp[i]=SFA + sfa_off[i];
  lsb[i]=Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(m,n,k,1));
  SFBp[i]=SFB + int64_t(i)*sfb_words_per_grp;
  alpha_grp[i]=gscaleA*gscaleB[i];
  alpha_ptr_grp[i]=alpha_grp+i;
}

// ---- grouped GEMM runner (validated) ----------------------------------------
struct GroupedGemm {
  // [fc1-fused-layout] per-expert byte stride of the packed B (weight) buffer;
  // <=0 => legacy tight (n*k)/2. Set by the loader when fc1 is stored fused.
  int64_t b_estride_bytes = 0;
  int ng=0,n=0,k=0,sfb_words=0;
  typename ProblemShape::UnderlyingProblemShape* ps=nullptr;
  const ElementInput **Ap=nullptr,**Bp=nullptr; const ElementSF **SFAp=nullptr,**SFBp=nullptr; ElementD** Dp=nullptr;
  StrideA* sa=nullptr; StrideB* sb=nullptr; StrideD* sd=nullptr; LayoutSFA* lsa=nullptr; LayoutSFB* lsb=nullptr;
  float* alpha_grp=nullptr; const float** alpha_ptr_grp=nullptr;
  Gemm gemm; void* wsp=nullptr; size_t wsp_bytes=0; bool inited=false; int sm_count=0;

  void alloc(int ng_,int n_,int k_,int sfb_words_){
    ng=ng_; n=n_; k=k_; sfb_words=sfb_words_;
    DSV4_CK(cudaMalloc(&ps, ng*sizeof(*ps)));
    DSV4_CK(cudaMalloc(&Ap, ng*sizeof(*Ap)));   DSV4_CK(cudaMalloc(&Bp, ng*sizeof(*Bp)));
    DSV4_CK(cudaMalloc(&Dp, ng*sizeof(*Dp)));
    DSV4_CK(cudaMalloc(&SFAp,ng*sizeof(*SFAp))); DSV4_CK(cudaMalloc(&SFBp,ng*sizeof(*SFBp)));
    DSV4_CK(cudaMalloc(&sa, ng*sizeof(*sa)));    DSV4_CK(cudaMalloc(&sb, ng*sizeof(*sb)));
    DSV4_CK(cudaMalloc(&sd, ng*sizeof(*sd)));
    DSV4_CK(cudaMalloc(&lsa,ng*sizeof(*lsa)));   DSV4_CK(cudaMalloc(&lsb,ng*sizeof(*lsb)));
    DSV4_CK(cudaMalloc(&alpha_grp, ng*sizeof(float)));
    DSV4_CK(cudaMalloc(&alpha_ptr_grp, ng*sizeof(const float*)));
    sm_count=cutlass::KernelHardwareInfo::query_device_multiprocessor_count();
  }
  void free_all(){
    auto F=[](void*p){ if(p) cudaFree(p); };
    F(ps);F(Ap);F(Bp);F(Dp);F(SFAp);F(SFBp);F(sa);F(sb);F(sd);F(lsa);F(lsb);F(alpha_grp);F(alpha_ptr_grp);F(wsp);
    ps=nullptr; wsp=nullptr; inited=false;
  }
  // m_indptr: device int[ng+1]; d_sfa_off: device int[ng+1]; gscaleA_ptr: device float.
  // CAPTURE-SAFE: host_problem_shapes is passed as NULLPTR so CUTLASS sizes the grid
  // to hw.sm_count (M-independent) and the persistent tile scheduler reads the DEVICE
  // problem_shapes (ps, filled by prep) at runtime -> the launch is identical every
  // call regardless of M, and nothing on this path allocates or syncs. Workspace is
  // allocated ONCE (first call, outside any capture) and reused.
  void run(const ElementInput*A,const ElementInput*B,const ElementSF*SFA,const ElementSF*SFB,ElementD*D,
           const int* m_indptr,const int* d_sfa_off,const float* gscaleB,const float* gscaleA_ptr,
           cudaStream_t s){
    int nt=std::min(ng,1024), nb=(ng+nt-1)/nt;
    prep<typename ProblemShape::UnderlyingProblemShape><<<nb,nt,0,s>>>(
       const_cast<ElementInput*>(A),const_cast<ElementInput*>(B),
       const_cast<ElementSF*>(SFA),const_cast<ElementSF*>(SFB),D,
       const_cast<int*>(m_indptr),d_sfa_off,n,k,ng,ps,Ap,Bp,SFAp,SFBp,Dp,sa,sb,sd,lsa,lsb,
       const_cast<float*>(gscaleB),alpha_grp,alpha_ptr_grp,gscaleA_ptr,sfb_words,
       b_estride_bytes>0 ? b_estride_bytes : ((int64_t)n*k>>1));
    cutlass::KernelHardwareInfo hw; hw.device_id=0; hw.sm_count=sm_count;
    decltype(std::declval<typename Gemm::Arguments>().epilogue.thread) fa;
    fa.alpha=0.f; fa.beta=0.f; fa.alpha_ptr=nullptr; fa.beta_ptr=nullptr;
    fa.alpha_ptr_array=alpha_ptr_grp; fa.beta_ptr_array=nullptr;
    fa.dAlpha={_0{},_0{},1}; fa.dBeta={_0{},_0{},0};
    typename Gemm::Arguments args{cutlass::gemm::GemmUniversalMode::kGrouped,
      {ng,ps,/*host_problem_shapes=*/nullptr},{Ap,sa,Bp,sb,SFAp,lsa,SFBp,lsb},{fa,nullptr,nullptr,Dp,sd},hw};
    if(!inited){ wsp_bytes=Gemm::get_workspace_size(args); if(wsp_bytes) DSV4_CK(cudaMalloc(&wsp,wsp_bytes));
      DSV4_CC(gemm.can_implement(args)); inited=true; }
    DSV4_CC(gemm.initialize(args,wsp,s)); DSV4_CC(gemm.run(s,nullptr,false));
  }
};

// ===== CAPTURE-SAFE on-device routing (replaces thrust::sort_by_key + host indptr) =====
// 1) histogram: count rows per expert via atomicAdd into E bins.
__global__ void hist_expert(const int* sel,int* hist,int rows){
  int p=blockIdx.x*blockDim.x+threadIdx.x; if(p>=rows)return;
  atomicAdd(&hist[sel[p]],1);
}
// 2) exclusive scan of the E-bin histogram -> indptr[E+1] (single block, E<=1024).
//    Also clears a per-expert cursor[E] (running write offset) for the scatter step.
__global__ void excl_scan_indptr(const int* hist,int* indptr,int* cursor,int E){
  // simple serial exclusive scan in ONE thread (E=256 -> trivial cost, fully
  // capture-safe, deterministic). indptr[0]=0; indptr[e+1]=indptr[e]+hist[e].
  if(threadIdx.x!=0||blockIdx.x!=0) return;
  int acc=0;
  for(int e=0;e<E;e++){ indptr[e]=acc; cursor[e]=acc; acc+=hist[e]; }
  indptr[E]=acc;
}
// 3) stable-enough scatter: assign each (token,slot) pair its sorted position by
//    atomically claiming a slot in its expert's contiguous range. Produces the
//    permutation `vals` (= original pair index) and `grp_of_row` (= expert id),
//    grouping all rows of one expert contiguously. Intra-group ORDER is irrelevant
//    to the result (per-row math is identical; the GEMM is dense per group), so any
//    grouping is bit-identical to the old thrust sort by expert.
__global__ void counting_scatter(const int* sel,const int* indptr_unused,int* cursor,
                                 int* vals,int* grp_of_row,int rows,int E){
  int p=blockIdx.x*blockDim.x+threadIdx.x; if(p>=rows)return;
  int e=sel[p];
  int pos=atomicAdd(&cursor[e],1);   // contiguous slot within expert e's range
  vals[pos]=p;
  grp_of_row[pos]=e;
}
// 4) per-group SFA atom-aligned offsets (ceil(m/128)*128 * kb), exclusive scan.
//    On-device replacement for the host build_sfa_off loop. Single thread (E small).
__global__ void compute_sfa_off(const int* indptr,int* sfa_off,int E,int kb){
  if(threadIdx.x!=0||blockIdx.x!=0) return;
  int acc=0;
  for(int e=0;e<E;e++){ sfa_off[e]=acc; int m=indptr[e+1]-indptr[e]; int mpad=((m+127)/128)*128; acc+=mpad*kb; }
  sfa_off[E]=acc;
}
// 5) on-device per-tensor global activation scale (absmax over [n] floats).
//    Block-grid reduction into a scratch[gridDim.x], then a single-block final
//    reduction folds scratch + applies gscale = (absmax/E2M1_MAX)/E4M3_MAX.
//    Writes ONE float to d_gscale. Capture-safe (pure kernels, no host).
__global__ void absmax_partial(const float* x,float* part,int64_t n){
  extern __shared__ float sm[];
  float mx=0.f;
  for(int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=(int64_t)gridDim.x*blockDim.x)
    mx=fmaxf(mx,fabsf(x[i]));
  sm[threadIdx.x]=mx; __syncthreads();
  for(int off=blockDim.x/2;off>0;off>>=1){ if(threadIdx.x<off) sm[threadIdx.x]=fmaxf(sm[threadIdx.x],sm[threadIdx.x+off]); __syncthreads(); }
  if(threadIdx.x==0) part[blockIdx.x]=sm[0];
}
// final reduce of `part[nparts]` -> gscale float. Single block.
__global__ void absmax_finalize(const float* part,int nparts,float* d_gscale){
  extern __shared__ float sm[];
  float mx=0.f;
  for(int i=threadIdx.x;i<nparts;i+=blockDim.x) mx=fmaxf(mx,part[i]);
  sm[threadIdx.x]=mx; __syncthreads();
  for(int off=blockDim.x/2;off>0;off>>=1){ if(threadIdx.x<off) sm[threadIdx.x]=fmaxf(sm[threadIdx.x],sm[threadIdx.x+off]); __syncthreads(); }
  if(threadIdx.x==0){ float am=sm[0]; float g=(am/E2M1_MAX)/E4M3_MAX; d_gscale[0]= g>0.f? g : 1.f; }
}

// gather_quant: gscaleA is now read from a DEVICE pointer (computed on-device).
__global__ void gather_quant(const float* hidden,const int* sorted_vals,ElementInput* Xs,
                             ElementSF* SF_tight,int rows,int D,int U,const float* gscaleA_ptr){
  float gscaleA=gscaleA_ptr[0];
  int i=blockIdx.x; if(i>=rows)return;
  int pair=sorted_vals[i]; int token=pair/U;
  const float* src=hidden+(int64_t)token*D;
  extern __shared__ float sm[];
  int nblk=D/SFVec;
  for(int b=0;b<nblk;b++){
    int base=b*SFVec; float mx=0.f;
    for(int t=threadIdx.x;t<SFVec;t+=blockDim.x) mx=fmaxf(mx,fabsf(src[base+t]));
    sm[threadIdx.x]=mx; __syncthreads();
    for(int off=blockDim.x/2;off>0;off>>=1){ if(threadIdx.x<off) sm[threadIdx.x]=fmaxf(sm[threadIdx.x],sm[threadIdx.x+off]); __syncthreads(); }
    float amax=sm[0]; __syncthreads();
    float bs_raw=amax>0.f? amax/E2M1_MAX:0.f;
    ElementSF e4=ElementSF(bs_raw/gscaleA); float eff=float(e4)*gscaleA; float inv=eff>0.f?1.f/eff:0.f;
    for(int t=threadIdx.x;t<SFVec;t+=blockDim.x) Xs[(int64_t)i*D+base+t]=ElementInput(src[base+t]*inv);
    if(threadIdx.x==0) SF_tight[(int64_t)i*nblk+b]=e4;
  }
}
__global__ void pack_e2m1(const ElementInput* full,uint8_t* packed,int64_t nelem){
  int64_t j=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; int64_t nb=(nelem+1)/2; if(j>=nb)return;
  uint8_t lo=full[2*j].storage&0xF; uint8_t hi=(2*j+1<nelem)?(full[2*j+1].storage&0xF):0;
  packed[j]=lo|(hi<<4);
}
__global__ void scatter_sfa(const ElementSF* tight,ElementSF* SFA,int rows,int kb,LayoutSFA layout,
                            const int* grp_of_row,const int* row_base,const int* sfa_off){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=rows*kb)return;
  int row=i/kb, b=i%kb; int g=grp_of_row[row]; int lr=row-row_base[g];
  int64_t off=(int64_t)sfa_off[g] + layout(lr,b*SFVec,0);
  SFA[off]=tight[(int64_t)row*kb+b];
}
// on-device act absmax over the CLAMPED swiglu activation silu(clamp(g))*clamp(u).
// Matches the host reduction (and swiglu_quant) EXACTLY so the down global scale is
// bit-identical. Grid-stride partial reduction -> part[gridDim.x]; finalize with
// absmax_finalize (shared E2M1/E4M3 folding) -> gscaleAct device float.
__global__ void swiglu_absmax_partial(const ElementD* gate,const ElementD* up,
                                      float* part,int64_t n,float limit){
  extern __shared__ float sm[];
  float mx=0.f;
  for(int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=(int64_t)gridDim.x*blockDim.x){
    float gv=float(gate[i]); float uv=float(up[i]);
    if(limit>0.f){ gv=fminf(gv,limit); uv=fmaxf(-limit,fminf(uv,limit)); }
    float a=(gv/(1.f+expf(-gv)))*uv;
    mx=fmaxf(mx,fabsf(a));
  }
  sm[threadIdx.x]=mx; __syncthreads();
  for(int off=blockDim.x/2;off>0;off>>=1){ if(threadIdx.x<off) sm[threadIdx.x]=fmaxf(sm[threadIdx.x],sm[threadIdx.x+off]); __syncthreads(); }
  if(threadIdx.x==0) part[blockIdx.x]=sm[0];
}

__global__ void swiglu_quant(const ElementD* gate,const ElementD* up,ElementInput* act,
                             ElementSF* SF_tight,int rows,int F,const float* gscaleAct_ptr,float limit){
  float gscaleAct=gscaleAct_ptr[0];
  int i=blockIdx.x; if(i>=rows)return;
  extern __shared__ float sm[];
  int nblk=F/SFVec; const ElementD* g=gate+(int64_t)i*F; const ElementD* u=up+(int64_t)i*F;
  ElementInput* o=act+(int64_t)i*F;
  // DSV4 per-layer SwiGLU clamp (matches reference): gate -> (-INF, limit], up -> [-limit, limit].
  // Applied in BOTH passes so the per-16 SF block scale is computed from the CLAMPED activation.
  for(int b=0;b<nblk;b++){
    int base=b*SFVec; float mx=0.f;
    for(int t=threadIdx.x;t<SFVec;t+=blockDim.x){
      float gv=float(g[base+t]); if(limit>0.f) gv=fminf(gv,limit);
      float uv=float(u[base+t]); if(limit>0.f) uv=fmaxf(-limit,fminf(uv,limit));
      float sv=gv/(1.f+expf(-gv)); mx=fmaxf(mx,fabsf(sv*uv)); }
    sm[threadIdx.x]=mx; __syncthreads();
    for(int off=blockDim.x/2;off>0;off>>=1){ if(threadIdx.x<off) sm[threadIdx.x]=fmaxf(sm[threadIdx.x],sm[threadIdx.x+off]); __syncthreads(); }
    float amax=sm[0]; __syncthreads();
    float bs_raw=amax>0.f? amax/E2M1_MAX:0.f;
    ElementSF e4=ElementSF(bs_raw/gscaleAct); float eff=float(e4)*gscaleAct; float inv=eff>0.f?1.f/eff:0.f;
    for(int t=threadIdx.x;t<SFVec;t+=blockDim.x){
      float gv=float(g[base+t]); if(limit>0.f) gv=fminf(gv,limit);
      float uv=float(u[base+t]); if(limit>0.f) uv=fmaxf(-limit,fminf(uv,limit));
      float sv=gv/(1.f+expf(-gv)); o[base+t]=ElementInput(sv*uv*inv); }
    if(threadIdx.x==0) SF_tight[(int64_t)i*nblk+b]=e4;
  }
}
// scatter-add into a FLOAT output [n_embd, n_tokens] (ggml column-major: token-major rows of D)
__global__ void scatter_add(const ElementD* down,const int* sorted_vals,const float* router_w,
                            float* out,int rows,int D,int U){
  int i=blockIdx.x; if(i>=rows)return;
  int pair=sorted_vals[i]; int token=pair/U; float w=router_w[pair];
  const ElementD* d=down+(int64_t)i*D; float* o=out+(int64_t)token*D;
  for(int t=threadIdx.x;t<D;t+=blockDim.x) atomicAdd(&o[t], w*float(d[t]));
}

// ===== load-adapter math: MXFP4 -> NVFP4 (mxfp4_to_nvfp4.cu) ==================
// ggml block_mxfp4: { uint8_t e; uint8_t qs[16]; } 32 elems/block, e=E8M0 (bias 127).
struct host_block_mxfp4 { uint8_t e; uint8_t qs[16]; };
static inline float e8m0_scale(uint8_t e){ return (e==0)?0.f:ldexpf(1.0f,(int)e-127); }

// Convert one expert weight [n rows, k cols] (k%32==0) MXFP4 -> NVFP4:
//   q_packed : e2m1 nibbles, 2/byte, n*k/2 bytes (nibbles copied verbatim from MXFP4)
//   sf_tight : ue4m3 per-16 block scale, n*(k/16) cells (each MXFP4 32-block -> 2 NVFP4 cells)
//   returns per-expert fp32 global = max(S_mx)/448
static float convert_expert_mxfp4_to_nvfp4(const host_block_mxfp4* blocks,int n,int k,
                                           std::vector<uint8_t>& q_packed,
                                           std::vector<ElementSF>& sf_tight){
  int kb32=k/32, kb16=k/16;
  q_packed.assign((size_t)n*k/2,0);
  sf_tight.assign((size_t)n*kb16,ElementSF(0.f));
  // pass1: global
  float smax=0.f;
  for(size_t r=0;r<(size_t)n;r++) for(int b=0;b<kb32;b++)
    smax=std::max(smax,e8m0_scale(blocks[r*kb32+b].e));
  float global = smax>0.f? smax/E4M3_MAX : 1.f;
  // pass2: nibbles verbatim + per-16 e4m3 scale = quant(S_mx/global)
  for(size_t r=0;r<(size_t)n;r++){
    for(int b=0;b<kb32;b++){
      const host_block_mxfp4& bl=blocks[r*kb32+b];
      float S_mx=e8m0_scale(bl.e);
      ElementSF s16=ElementSF(S_mx/global);
      sf_tight[r*kb16 + b*2 + 0]=s16;
      sf_tight[r*kb16 + b*2 + 1]=s16;
      // MXFP4 packs nibbles DE-INTERLEAVED within each 32-block (ggml canonical):
      // byte m holds element m (low nibble) and element m+16 (high nibble).
      // CUTLASS float_e2m1_t (and our activation pack_e2m1) expect CONSECUTIVE
      // packing: byte p holds elements 2p, 2p+1. Copying verbatim scrambles the
      // weights along K vs the activations -> garbage. Re-pack de-interleaved->consecutive.
      uint8_t code[32];
      for(int m=0;m<16;m++){ uint8_t byte=bl.qs[m]; code[m]=byte&0xF; code[m+16]=byte>>4; }
      for(int p=0;p<16;p++)
        q_packed[(r*k + b*32)/2 + p] = code[2*p] | (code[2*p+1] << 4);
    }
  }
  return global;
}

// Build the swizzled per-group tile-atom SFB layout for ALL experts into a host buffer.
// returns words-per-group (cosize of the SFB atom layout for a 512-padded m proxy); fills `out`.
static int build_sfb_swizzled_host(const std::vector<std::vector<ElementSF>>& sft_per_expert,
                                   int n,int k,std::vector<ElementSF>& out){
  int E=(int)sft_per_expert.size(); int kb=k/SFVec;
  auto lsb=Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_shape(512,n,k,1));
  size_t per=cosize(lsb); int words=(int)per;
  out.assign((size_t)E*per,ElementSF(0));
  for(int g=0;g<E;g++){
    auto t=make_tensor(out.data()+(size_t)g*per,lsb);
    const ElementSF* src=sft_per_expert[g].data();
    for(int ni=0;ni<n;ni++) for(int b=0;b<kb;b++) t(ni,b*SFVec,0)=src[(size_t)ni*kb+b];
  }
  return words;
}

// Inverse of build_sfb_swizzled_host: read a swizzled per-group SFB host buffer
// (E groups, `per` words each, built with the 512-padded m proxy atom layout) and
// extract the SIMPLE tight per-expert per-16-block ue4m3 scale [E][n][k/16] for the
// HIGH-PRECISION DECODE path. Robust: uses the exact same CUTLASS layout to index.
static void unswizzle_sfb_to_simple(const ElementSF* swiz, int E, int n, int k,
                                    int per_words, std::vector<uint8_t>& out){
  int kb=k/SFVec;
  auto lsb=Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_shape(512,n,k,1));
  out.assign((size_t)E*n*kb, 0);
  for(int g=0;g<E;g++){
    auto t=make_tensor(const_cast<ElementSF*>(swiz)+(size_t)g*per_words,lsb);
    uint8_t* dst=out.data()+(size_t)g*n*kb;
    for(int ni=0;ni<n;ni++) for(int b=0;b<kb;b++)
      dst[(size_t)ni*kb+b]=t(ni,b*SFVec,0).storage; // raw ue4m3 byte
  }
}

// upload tight ue4m3 SFB into the swizzled per-group tile-atom layout for ALL experts.
// returns words-per-group (cosize of the SFB atom layout for a 512-padded m proxy).
[[maybe_unused]] static int upload_sfb_swizzled(const std::vector<std::vector<ElementSF>>& sft_per_expert,
                               int n,int k,ElementSF** dptr){
  std::vector<ElementSF> all;
  int words=build_sfb_swizzled_host(sft_per_expert,n,k,all);
  DSV4_CK(cudaMalloc(dptr, all.size()*sizeof(ElementSF)));
  DSV4_CK(cudaMemcpy(*dptr, all.data(), all.size()*sizeof(ElementSF), cudaMemcpyHostToDevice));
  return words;
}

// ===== per-layer registry ====================================================
struct LayerWeights {
  int n_expert=0, n_embd=0, n_ff_exp=0;
  // packed e2m1 (device), per expert concatenated [E*n*k/2 bytes]
  ElementInput *dq_gate=nullptr, *dq_up=nullptr, *dq_down=nullptr;
  // swizzled ue4m3 SFB (device)
  ElementSF *dsf_gate=nullptr, *dsf_up=nullptr, *dsf_down=nullptr;
  int sfb_words_gu=0, sfb_words_d=0;
  // per-expert fp32 global scales (device)
  float *dglobal_gate=nullptr, *dglobal_up=nullptr, *dglobal_down=nullptr;
  // reusable grouped GEMMs
  GroupedGemm gg_gate, gg_up, gg_down;
  // ---- HIGH-PRECISION DECODE path: simple (un-swizzled) per-expert per-16-block
  // ue4m3 SFB copies, tight layout [E][n][k/16] as uint8 raw e4m3 storage. Built by
  // un-swizzling the swizzled SFB at registration; used by the GEMV decode kernels.
  uint8_t *dsf_gate_simple=nullptr, *dsf_up_simple=nullptr, *dsf_down_simple=nullptr;
  // ---- HP DECODE: persistent scratch for swiglu activations [rows][n_ff_half].
  // Allocated ONCE (first HP call) and reused so the HP path performs NO per-call
  // cudaMalloc/cudaFree -> CUDA-graph capture safe. Sized to the largest `rows`
  // (= M*U) seen on the HP path; grown if a bigger small-batch arrives.
  float  *d_act_decode=nullptr; size_t act_decode_elems=0;
  // [fc1-fused-layout] bytes between consecutive experts in dq_gate/dq_up. 0 = legacy
  // (separate gate/up tensors, stride (n_ff*n_embd)/2). Non-zero = the CUTLASS fused-fc1
  // interleave (per expert: up rows then gate rows -> stride n_ff*n_embd), where dq_up is
  // the buffer base and dq_gate = base + (n_ff*n_embd)/2. Shared byte-identically with the
  // fused runner's fc1 (zero-copy alias -> no repack malloc/free, no decode UAF).
  int64_t gu_estride = 0;

  // ---- PREFILL (CUTLASS grouped-GEMM) persistent arena -----------------------
  // Pre-allocated ONCE (grow-once, never freed mid-capture) so the whole prefill
  // path is CUDA-graph capture-safe: NO per-call cudaMalloc/cudaFree/sync/thrust.
  // Sized to a max prefill row budget rows_cap = Mcap*U. All buffers below are
  // re-used every prefill call; only the routing kernels rewrite their contents.
  bool   pf_inited=false;
  int    pf_rows_cap=0;                       // rows_cap = Mcap*U we allocated for
  size_t pf_sfx_words_cap=0, pf_sfa_words_cap=0;
  int    *pf_hist=nullptr, *pf_cursor=nullptr;       // [E], [E]
  int    *pf_indptr=nullptr;                          // [E+1]
  int    *pf_sfa_off_D=nullptr, *pf_sfa_off_F=nullptr;// [E+1]
  int    *pf_vals=nullptr, *pf_grp_of_row=nullptr;    // [rows_cap]
  float  *pf_gscaleX=nullptr, *pf_gscaleAct=nullptr;  // [1], [1] (device scalars)
  float  *pf_absmax_part=nullptr; int pf_absmax_nparts=0; // [nparts] reduction scratch
  ElementInput *pf_Xs_full=nullptr, *pf_Xs=nullptr;   // [rows_cap*D], [rows_cap*D/2]
  ElementSF    *pf_SFx_tight=nullptr;                  // [rows_cap*kbD]
  ElementD     *pf_gate=nullptr, *pf_up=nullptr;       // [rows_cap*F]
  ElementInput *pf_act_full=nullptr, *pf_act=nullptr;  // [rows_cap*F], [rows_cap*F/2]
  ElementSF    *pf_SFa_tight=nullptr;                  // [rows_cap*kbF]
  ElementD     *pf_down=nullptr;                       // [rows_cap*D]
  ElementSF    *pf_SFx=nullptr, *pf_SFa=nullptr;       // [sfx_words_cap], [sfa_words_cap]
};

static std::mutex                              g_reg_mu;
static std::unordered_map<int,LayerWeights*>   g_registry;

// ---- EP (expert-parallel) config (process-global; set once at load) ---------
// Records this rank's expert shard so the FUSED op can pass flashinfer's runMoe the
// right MOEParallelismConfig (ep_size, ep_rank) + the GLOBAL num_experts. ep==0 =>
// no EP (full local expert set, ep_size=1) => byte-identical to today.
static int g_ep            = 0;
static int g_ep_base       = 0;   // GLOBAL id of local expert 0
static int g_ep_n_global   = 0;   // total experts across ranks (e.g. 256)
static int g_ep_n_local    = 0;   // experts on THIS rank (e.g. 128)

extern "C" void dsv4_moe_set_ep_config(int ep, int expert_base, int n_expert_global, int n_expert_local){
    g_ep = ep; g_ep_base = expert_base; g_ep_n_global = n_expert_global; g_ep_n_local = n_expert_local;
    fprintf(stderr,"[dsv4-moe-grouped] EP config: ep=%d expert_base=%d n_expert_global=%d n_expert_local=%d "
                   "(ep_size=%d ep_rank=%d)\n", ep, expert_base, n_expert_global, n_expert_local,
            (n_expert_local>0?n_expert_global/n_expert_local:1), (n_expert_local>0?expert_base/n_expert_local:0));
}
extern "C" int dsv4_moe_get_ep_config(int* expert_base, int* n_expert_global, int* n_expert_local){
    if (expert_base)     *expert_base     = g_ep_base;
    if (n_expert_global) *n_expert_global = g_ep_n_global;
    if (n_expert_local)  *n_expert_local  = g_ep_n_local;
    return g_ep;
}

// ---- DEFER-FREE retire list (CUDA-graph UAF guard) -------------------------
// The prefill arena pointers live in LayerWeights (g_registry), NOT in any
// ggml_tensor, so the CUDA-graph update check (ggml-cuda.cu, which only memcmp's
// tensor data/ne/nb) is BLIND to them. If a GROW did cudaFree()+cudaMalloc() to
// resize the arena, an already-captured prefill graph would replay using the
// FREED old addresses -> use-after-free crash (fires under --parallel 2 when a
// contended slot forces a clear & variable-sized re-prefill after capture).
// FIX: never cudaFree the live arena inline. On grow we ALLOCATE-NEW, hand the
// OLD pointers to this retire list, and free them only at the next device
// synchronize (drained from ggml_backend_cuda_synchronize). Any in-flight
// captured graph still referencing the old addresses stays valid until the
// stream has drained -> the UAF class is removed while growth is still allowed.
static std::mutex                              g_retire_mu;
static std::vector<void*>                      g_retire_ptrs;
static void dsv4_retire(void* p){ if(!p) return; std::lock_guard<std::mutex> lk(g_retire_mu); g_retire_ptrs.push_back(p); }

extern "C" void dsv4_moe_grouped_drain_retired(void){
  std::vector<void*> tmp;
  { std::lock_guard<std::mutex> lk(g_retire_mu); tmp.swap(g_retire_ptrs); }
  for(void* p : tmp) cudaFree(p);
}

// Public defer-free entry for the FUSED op: retire a device pointer to be freed at
// the next backend synchronize (drained by dsv4_moe_grouped_drain_retired above),
// so grow-reallocs in the fused op are CUDA-graph capture-safe (no immediate free
// of a pointer a pending captured graph may still reference).
extern "C" void dsv4_moe_grouped_retire_ptr(void* p){ dsv4_retire(p); }

// Grouped-decode HP buffer warm-up gate (fixes the fused-prefill -> grouped-decode capture crash).
// Under DSV4_MOE_FUSED, prefill never touches the grouped op, so the grouped decode buffers
// (d_act_decode) are first allocated on the FIRST decode — which, if CUDA-graph-captured, malloc's
// mid-capture -> "internal operation failed". The engine's grouped graph-gate disables graphs for
// grouped DECODE steps until this flag reports warmed; the op sets it once a decode-shaped call has
// allocated its buffer outside capture. After warm-up, decode graphs re-enable (steady-state fast).
// g_grouped_decode_warm_count + dsv4_moe_grouped_decode_warmed() are defined at GLOBAL scope after
// the namespace closes (below), alongside dsv4_grouped_layer_count(). The increment site uses the
// global via the ::-qualified name.

[[maybe_unused]] static void pack_upload_experts(const std::vector<std::vector<uint8_t>>& q_per_expert,ElementInput** dptr){
  size_t total=0; for(auto& q:q_per_expert) total+=q.size();
  std::vector<uint8_t> all(total); size_t off=0;
  for(auto& q:q_per_expert){ memcpy(all.data()+off,q.data(),q.size()); off+=q.size(); }
  uint8_t* dp; DSV4_CK(cudaMalloc(&dp, all.size()));
  DSV4_CK(cudaMemcpy(dp, all.data(), all.size(), cudaMemcpyHostToDevice));
  *dptr=reinterpret_cast<ElementInput*>(dp);
}

// ============================================================================
// HIGH-PRECISION DECODE path (small M): FP32 activations x dequantized NVFP4
// weights via custom GEMV kernels. NO 4-bit activation quantization.
// ============================================================================
// E2M1 nibble decode: bit3 = sign, bits[2:0] = magnitude code.
// magnitude codes 0..7 -> {0, .5, 1, 1.5, 2, 3, 4, 6}.
// e2m1 (FP4): [s | e1 e0 | m0] -> {0, .5, 1, 1.5, 2, 3, 4, 6} and their negatives.
//
// This used to be `static const float MAG[8]` indexed by the nibble. In device code that array is
// NOT a register file -- it is a memory lookup, and the index differs per lane so it cannot even
// broadcast. The MoE GEVM decodes ~4.3e9 FP4 values per token, so the kernel was bound by that LUT,
// not by DRAM: which is exactly why widening the weight loads to 16 B (DSV4_MOE_VEC16) and why
// swapping in b12x's own fp4_dot8 inner product BOTH moved the MoE time by less than 1%.
//
// Build the float bits directly instead. Zero memory traffic, ~4 ALU ops.
//   e == 0 : value = 0.5 * m                      (0 or 0.5)
//   e >= 1 : value = (1 + 0.5*m) * 2^(e-1)        (1, 1.5, 2, 3, 4, 6)
__device__ __forceinline__ float e2m1_decode(uint8_t nib){
  const uint32_t e = (nib >> 1) & 0x3;
  const uint32_t m =  nib       & 0x1;
  const uint32_t norm = (((e - 1u) + 127u) << 23) | (m << 22);   // (1 + m/2) * 2^(e-1)
  const uint32_t sub  = m ? 0x3F000000u : 0u;                    // 0.5 or 0
  uint32_t bits = (e != 0u) ? norm : sub;
  bits |= ((uint32_t) (nib & 0x8)) << 28;                        // sign
  return __int_as_float((int) bits);
}
// raw ue4m3 (float_ue4m3_t) storage byte -> float. E4M3 unsigned: 4 exp bits, 3 mant,
// bias 7. Build via cutlass ElementSF bitcast for exactness.
__device__ __forceinline__ float ue4m3_decode(uint8_t s){
  return float(ElementSF::bitcast(s));
}

// ---- warp-cooperative dot helpers (coalesced 4-bit weight read along K) ------
// One WARP computes a single output element. The 32 lanes stride over the K
// dimension; each lane consumes one packed byte (= 2 nibbles = 2 input elems) per
// step, so reads of Wrow are perfectly coalesced (lane t -> byte t). Block scales
// are applied per 16-element (SFVec) block: byte index `c/2`, K index `c=2*p`, the
// SF block is (2*p)/SFVec. A warp shuffle reduces the partial sums.
__device__ __forceinline__ float warp_reduce_sum(float v){
  #pragma unroll
  for(int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
  return v;
}

// Stage 1: gate/up + swiglu, WARP-PER-OUTPUT.  grid = (F/warps_per_block, M*U),
// block = warps_per_block*32 threads.  Computes act[row][j] = silu(gate)*up.
__global__ void dec_gate_up_swiglu(
    const float* __restrict__ hidden, const int* __restrict__ sel,
    const uint8_t* __restrict__ Wg, const uint8_t* __restrict__ Wu,
    const uint8_t* __restrict__ SFg, const uint8_t* __restrict__ SFu,
    const float* __restrict__ glg, const float* __restrict__ glu,
    float* __restrict__ act, int M, int U, int D, int F, float limit,
    int expert_base, int E_local, int64_t gu_estride){
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int warps_per_block = blockDim.x >> 5;
  const int j = blockIdx.x * warps_per_block + warp;   // output row index in [0,F)
  const int row = blockIdx.y;                          // token*U + slot
  if(j >= F || row >= M*U) return;
  const int token = row / U;
  // [ep2-dp] sel[] = GLOBAL ids; this rank owns local experts [expert_base, expert_base+E_local).
  // Remap to local + SKIP remote (other rank covers them; the PARTIAL AllReduce sums both partials).
  // Non-EP: expert_base=0, E_local=E => range check always passes (byte-identical).
  const int e = sel[row] - expert_base;
  if(e < 0 || e >= E_local){ act[(int64_t)row * F + j] = 0.f; return; }
  const float* __restrict__ x = hidden + (int64_t)token * D;
  const int kb = D / SFVec;
  const int nbytes = D >> 1;                            // packed bytes per weight row
  const int64_t wbase = (int64_t)e * gu_estride;
  const int64_t sbase = (int64_t)e * F * kb;
  const float ge = glg[e], ue = glu[e];
  const uint8_t* __restrict__ wgr = Wg + wbase + (int64_t)j * nbytes;
  const uint8_t* __restrict__ wur = Wu + wbase + (int64_t)j * nbytes;
  const uint8_t* __restrict__ sgr = SFg + sbase + (int64_t)j * kb;
  const uint8_t* __restrict__ sur = SFu + sbase + (int64_t)j * kb;
  float accg = 0.f, accu = 0.f;
  // lane p handles packed byte p, p+32, p+64, ...  -> input elems (2p, 2p+1).
  for(int p = lane; p < nbytes; p += 32){
    int c = 2 * p;
    int blk = c / SFVec;                               // SF block for this pair
    float sg = ue4m3_decode(sgr[blk]) * ge;
    float su = ue4m3_decode(sur[blk]) * ue;
    uint8_t bg = wgr[p], bu = wur[p];
    float x0 = x[c], x1 = x[c+1];
    accg += e2m1_decode(bg & 0xF) * sg * x0 + e2m1_decode(bg >> 4) * sg * x1;
    accu += e2m1_decode(bu & 0xF) * su * x0 + e2m1_decode(bu >> 4) * su * x1;
  }
  accg = warp_reduce_sum(accg);
  accu = warp_reduce_sum(accu);
  if(lane == 0){
    // DSV4 per-layer SwiGLU clamp (matches reference): gate -> (-INF, limit], up -> [-limit, limit].
    float g = accg; if(limit > 0.f) g = fminf(g, limit);
    float u = accu; if(limit > 0.f) u = fmaxf(-limit, fminf(u, limit));
    float sv = g / (1.f + expf(-g));
    act[(int64_t)row * F + j] = sv * u;
  }
}

// Stage 2: down + router-weighted scatter-add into FP32 out[n_embd, n_tokens],
// WARP-PER-OUTPUT.  grid = (D/warps_per_block, M*U), block = warps_per_block*32.
// out[token][i] += rw[row] * sum_j dequant(W_down[i,j]) * act[row][j].
__global__ void dec_down_scatter(
    const float* __restrict__ act, const int* __restrict__ sel,
    const float* __restrict__ rw, const uint8_t* __restrict__ Wd,
    const uint8_t* __restrict__ SFd, const float* __restrict__ gld,
    float* __restrict__ out, int M, int U, int D, int F,
    int expert_base, int E_local){
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int warps_per_block = blockDim.x >> 5;
  const int i = blockIdx.x * warps_per_block + warp;   // output row index in [0,D)
  const int row = blockIdx.y;                          // token*U + slot
  if(i >= D || row >= M*U) return;
  const int token = row / U;
  // [ep2-dp] GLOBAL->LOCAL remap + skip remote experts (act[] was zeroed for them; don't index local
  // weights OOB). Non-EP: expert_base=0, E_local=E => no-op.
  const int e = sel[row] - expert_base;
  if(e < 0 || e >= E_local) return;
  const float w = rw[row];
  const float* __restrict__ a = act + (int64_t)row * F;
  const int kb = F / SFVec;
  const int nbytes = F >> 1;
  const int64_t wbase = (int64_t)e * D * F / 2;
  const int64_t sbase = (int64_t)e * D * kb;
  const float de = gld[e];
  const uint8_t* __restrict__ wdr = Wd + wbase + (int64_t)i * nbytes;
  const uint8_t* __restrict__ sdr = SFd + sbase + (int64_t)i * kb;
  float acc = 0.f;
  for(int p = lane; p < nbytes; p += 32){
    int c = 2 * p;
    int blk = c / SFVec;
    float sd = ue4m3_decode(sdr[blk]) * de;
    uint8_t bd = wdr[p];
    acc += e2m1_decode(bd & 0xF) * sd * a[c] + e2m1_decode(bd >> 4) * sd * a[c+1];
  }
  acc = warp_reduce_sum(acc);
  if(lane == 0){
    atomicAdd(&out[(int64_t)token * D + i], w * acc);
  }
}

// ============================================================================
// FAITHFUL b12x decode variant (DSV4_MOE_W4A16_DECODE): same warp-per-output
// structure/args/EP/scale as dec_* above, but the inner contraction uses b12x's
// fp4_dot8 (f16x2-packed 16-elem block dot; bit-trick e2m1 decode since ptxas
// rejects cvt.e2m1x2 on sm_121 — b12x's own GEMM technique) instead of scalar-f32.
// Lane owns 16-K blocks strided by 32; per-block scale ue4m3(sf)*global, the dot's
// 2^-14 prescale folded via ldexpf(...,14). A/B target vs dec_*.
// ============================================================================
#include "dsv4-w4a16/decode/faithful_micro_dots.cuh"
namespace w4dec = dsv4::w4a16::decode;
__device__ __forceinline__ uint32_t w4dec_packf16x2(float a, float b){
  __half2 h = __floats2half2_rn(a, b); uint32_t r; memcpy(&r,&h,4); return r;
}
__global__ void w4a16_fc1_swiglu(
    const float* __restrict__ hidden, const int* __restrict__ sel,
    const uint8_t* __restrict__ Wg, const uint8_t* __restrict__ Wu,
    const uint8_t* __restrict__ SFg, const uint8_t* __restrict__ SFu,
    const float* __restrict__ glg, const float* __restrict__ glu,
    float* __restrict__ act, int M, int U, int D, int F, float limit,
    int expert_base, int E_local, int64_t gu_estride){
  const int lane = threadIdx.x & 31;
  const int j = blockIdx.x * (blockDim.x>>5) + (threadIdx.x>>5);
  const int row = blockIdx.y;
  if(j >= F || row >= M*U) return;
  const int token = row / U;
  const int e = sel[row] - expert_base;
  if(e < 0 || e >= E_local){ act[(int64_t)row*F + j] = 0.f; return; }
  const float* __restrict__ x = hidden + (int64_t)token * D;
  const int kb = D / SFVec, nbytes = D >> 1;
  const int64_t wbase = (int64_t)e*gu_estride, sbase = (int64_t)e*F*kb;
  const float ge = glg[e], ue = glu[e];
  const uint8_t* wgr = Wg + wbase + (int64_t)j*nbytes;
  const uint8_t* wur = Wu + wbase + (int64_t)j*nbytes;
  const uint8_t* sgr = SFg + sbase + (int64_t)j*kb;
  const uint8_t* sur = SFu + sbase + (int64_t)j*kb;
  float accg = 0.f, accu = 0.f;
  const int nblk = D / 16;
  for(int b = lane; b < nblk; b += 32){
    const int byte0 = b*8, k0 = b*16;
    uint32_t g0,g1,u0,u1; memcpy(&g0,wgr+byte0,4); memcpy(&g1,wgr+byte0+4,4);
    memcpy(&u0,wur+byte0,4); memcpy(&u1,wur+byte0+4,4);
    uint32_t xh[8];
    #pragma unroll
    for(int t=0;t<8;t++) xh[t]=w4dec_packf16x2(x[k0+t*2], x[k0+t*2+1]);
    float dg = w4dec::fp4_dot8_sum_prescale(g0,g1, xh[0],xh[1],xh[2],xh[3],xh[4],xh[5],xh[6],xh[7]);
    float du = w4dec::fp4_dot8_sum_prescale(u0,u1, xh[0],xh[1],xh[2],xh[3],xh[4],xh[5],xh[6],xh[7]);
    accg += ldexpf(dg,14) * (ue4m3_decode(sgr[b]) * ge);
    accu += ldexpf(du,14) * (ue4m3_decode(sur[b]) * ue);
  }
  accg = warp_reduce_sum(accg); accu = warp_reduce_sum(accu);
  if(lane == 0){
    float g = accg; if(limit>0.f) g = fminf(g, limit);
    float u = accu; if(limit>0.f) u = fmaxf(-limit, fminf(u, limit));
    act[(int64_t)row*F + j] = (g/(1.f+expf(-g))) * u;
  }
}
__global__ void w4a16_fc2_scatter(
    const float* __restrict__ act, const int* __restrict__ sel, const float* __restrict__ rw,
    const uint8_t* __restrict__ Wd, const uint8_t* __restrict__ SFd, const float* __restrict__ gld,
    float* __restrict__ out, int M, int U, int D, int F, int expert_base, int E_local){
  const int lane = threadIdx.x & 31;
  const int i = blockIdx.x * (blockDim.x>>5) + (threadIdx.x>>5);
  const int row = blockIdx.y;
  if(i >= D || row >= M*U) return;
  const int token = row / U;
  const int e = sel[row] - expert_base;
  if(e < 0 || e >= E_local) return;
  const float w = rw[row];
  const float* __restrict__ a = act + (int64_t)row*F;
  const int kb = F / SFVec, nbytes = F >> 1;
  const int64_t wbase = (int64_t)e*D*F/2, sbase = (int64_t)e*D*kb;
  const float de = gld[e];
  const uint8_t* wdr = Wd + wbase + (int64_t)i*nbytes;
  const uint8_t* sdr = SFd + sbase + (int64_t)i*kb;
  float acc = 0.f;
  const int nblk = F / 16;
  for(int b = lane; b < nblk; b += 32){
    const int byte0 = b*8, k0 = b*16;
    uint32_t d0,d1; memcpy(&d0,wdr+byte0,4); memcpy(&d1,wdr+byte0+4,4);
    uint32_t xh[8];
    #pragma unroll
    for(int t=0;t<8;t++) xh[t]=w4dec_packf16x2(a[k0+t*2], a[k0+t*2+1]);
    float dd = w4dec::fp4_dot8_sum_prescale(d0,d1, xh[0],xh[1],xh[2],xh[3],xh[4],xh[5],xh[6],xh[7]);
    acc += ldexpf(dd,14) * (ue4m3_decode(sdr[b]) * de);
  }
  acc = warp_reduce_sum(acc);
  if(lane == 0) atomicAdd(&out[(int64_t)token*D + i], w * acc);
}

// ===== ORTHODOX SINGLE-WEIGHT-SET HP DECODE (reads the FUSED fc1/fc2 layout) ===
// Under DSV4_MOE_FUSED the grouped dq_gate/dq_up (+ simple SFs) are freed once the
// fused repack folds them into fc1. These decode kernels read gate/up from the FUSED
// fc1 [E][2*inter][D/2] (rows [0,inter)=UP, [inter,2*inter)=GATE), down from fc2
// [E][D][F/2] (= dq_down alias), with the FUSED SWIZZLED_128x4 ue4m3 block scales +
// the per-expert g_common (fc1) / g_down (fc2) globals. Effective weight scale
// == ue4m3(simple * g_proj/g_common) * g_common == the prefill fused scale -> the
// MoE output is numerically the prefill fused output (lossless, single weight set).

// Mirror of dsv4-moe-fused-run.cu::sf_swizzled_index (computeSFIndex / SWIZZLED_128x4).
// Returns the byte offset (within one expert's SF block) of the ue4m3 scale for
// (rowIdx in [0,nRows_fused), colIdx = K-block in [0,totalColumn)).
__device__ __forceinline__ int dsv4_sf_swizzled_index(int rowIdx, int colIdx, int totalColumn){
  const int kColumnGroup0Size = 4;
  const int kRowGroup0Size = 32;
  const int kRowGroup1Size = kRowGroup0Size * 4; // 128
  int paddedColumn = ((totalColumn + 3) / 4) * 4;
  int columnIdxInGroup0 = colIdx % kColumnGroup0Size;
  int columnGroupIdx    = colIdx / kColumnGroup0Size;
  const int columnGroupStride = kColumnGroup0Size * kRowGroup1Size; // 512
  int rowIdxInGroup0 = rowIdx % kRowGroup0Size;
  int rowIdxInGroup1 = (rowIdx % kRowGroup1Size) / kRowGroup0Size;
  int rowGroupIdx    = rowIdx / kRowGroup1Size;
  const int rowGroup1Stride = kColumnGroup0Size;                    // 4
  const int rowGroup0Stride = kColumnGroup0Size * rowGroup1Stride;  // 16
  int rowGroupStride = kRowGroup1Size * paddedColumn;
  return columnIdxInGroup0 + columnGroupIdx * columnGroupStride
       + rowIdxInGroup0 * rowGroup0Stride + rowIdxInGroup1 * rowGroup1Stride
       + rowGroupIdx * rowGroupStride;
}

// Stage 1 (fused layout): gate/up + swiglu, WARP-PER-OUTPUT.
//   fc1: [E][2*inter][D/2] e2m1; UP at row j, GATE at row inter+j.
//   fc1_sf: per-expert stride sf1_stride = padUp(2*inter,128)*padUp(D/16,4);
//           scale byte at sf_off(e) + dsv4_sf_swizzled_index(row_fused, c/16, D/16).
//   global g_common[e] applied to BOTH gate and up (== prefill convention).
__global__ void dec_gate_up_swiglu_fused(
    const float* __restrict__ hidden, const int* __restrict__ sel,
    const uint8_t* __restrict__ fc1_w, const uint8_t* __restrict__ fc1_sf,
    const float* __restrict__ g_common,
    float* __restrict__ act, int M, int U, int D, int F, int inter,
    int sf1_stride, int sf1_cols, float limit, int expert_base, int E_local, int vec16){
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int warps_per_block = blockDim.x >> 5;
  const int j = blockIdx.x * warps_per_block + warp;   // output row in [0,F)  (F==inter)
  const int row = blockIdx.y;                          // token*U + slot
  if(j >= F || row >= M*U) return;
  const int token = row / U;
  // [ep2-dp] GLOBAL->LOCAL remap + skip remote experts (zero their act; other rank covers them).
  const int e = sel[row] - expert_base;
  if(e < 0 || e >= E_local){ act[(int64_t)row * F + j] = 0.f; return; }
  const float* __restrict__ x = hidden + (int64_t)token * D;
  const int nbytes = D >> 1;                            // packed bytes per weight row
  const int rows_fused = 2 * inter;
  const int64_t wbase = (int64_t)e * rows_fused * nbytes;
  const int64_t sfbase = (int64_t)e * sf1_stride;
  const float gc = g_common[e];
  // UP at fused row j, GATE at fused row inter+j.
  const int up_row   = j;
  const int gate_row = inter + j;
  const uint8_t* __restrict__ wur = fc1_w + wbase + (int64_t)up_row   * nbytes;
  const uint8_t* __restrict__ wgr = fc1_w + wbase + (int64_t)gate_row * nbytes;
  float accg = 0.f, accu = 0.f;
  if (vec16) {
    // [DSV4_MOE_VEC16] 16-byte (uint4) weight loads. The scalar loop below pulls ONE BYTE per lane,
    // so a warp asks the memory system for 32 B at a time -- not enough requests in flight to fill
    // DRAM. That, not the inner math, is what pinned this GEVM at 129 GB/s: swapping in b12x's own
    // fp4_dot8 dot product changed the MoE time by 0.05 ms (16.49 -> 16.53). With uint4 loads a warp
    // pulls 512 B per step. nbytes is 2048 (fc1) / 1024 (fc2) and every base is 16 B aligned.
    // 32 elements per chunk span exactly 2 SF blocks (SFVec = 16), so the swizzled scale lookup --
    // which the scalar loop redid on every one of its 64 iterations -- happens twice.
    const int n16 = nbytes >> 4;
    for (int p = lane; p < n16; p += 32) {
      const uint4 wg4 = reinterpret_cast<const uint4 *>(wgr)[p];
      const uint4 wu4 = reinterpret_cast<const uint4 *>(wur)[p];
      const int   c0   = 32 * p;
      const int   blk0 = c0 >> 4;
      const float sg0 = ue4m3_decode(fc1_sf[sfbase + dsv4_sf_swizzled_index(gate_row, blk0,     sf1_cols)]) * gc;
      const float su0 = ue4m3_decode(fc1_sf[sfbase + dsv4_sf_swizzled_index(up_row,   blk0,     sf1_cols)]) * gc;
      const float sg1 = ue4m3_decode(fc1_sf[sfbase + dsv4_sf_swizzled_index(gate_row, blk0 + 1, sf1_cols)]) * gc;
      const float su1 = ue4m3_decode(fc1_sf[sfbase + dsv4_sf_swizzled_index(up_row,   blk0 + 1, sf1_cols)]) * gc;
      const uint32_t gw[4] = { wg4.x, wg4.y, wg4.z, wg4.w };
      const uint32_t uw[4] = { wu4.x, wu4.y, wu4.z, wu4.w };
      #pragma unroll
      for (int q = 0; q < 4; q++) {
        const float sg = (q < 2) ? sg0 : sg1;   // 4 bytes = 8 elements per word
        const float su = (q < 2) ? su0 : su1;
        #pragma unroll
        for (int t = 0; t < 4; t++) {
          const uint8_t bg = (uint8_t) (gw[q] >> (8 * t));
          const uint8_t bu = (uint8_t) (uw[q] >> (8 * t));
          const int     c  = c0 + 8 * q + 2 * t;
          const float   x0 = x[c], x1 = x[c + 1];
          accg += e2m1_decode(bg & 0xF) * sg * x0 + e2m1_decode(bg >> 4) * sg * x1;
          accu += e2m1_decode(bu & 0xF) * su * x0 + e2m1_decode(bu >> 4) * su * x1;
        }
      }
    }
  } else {
  for(int p = lane; p < nbytes; p += 32){
    int c = 2 * p;
    int blk = c / SFVec;                               // SF K-block index (colIdx)
    float sg = ue4m3_decode(fc1_sf[sfbase + dsv4_sf_swizzled_index(gate_row, blk, sf1_cols)]) * gc;
    float su = ue4m3_decode(fc1_sf[sfbase + dsv4_sf_swizzled_index(up_row,   blk, sf1_cols)]) * gc;
    uint8_t bg = wgr[p], bu = wur[p];
    float x0 = x[c], x1 = x[c+1];
    accg += e2m1_decode(bg & 0xF) * sg * x0 + e2m1_decode(bg >> 4) * sg * x1;
    accu += e2m1_decode(bu & 0xF) * su * x0 + e2m1_decode(bu >> 4) * su * x1;
  }
  }
  accg = warp_reduce_sum(accg);
  accu = warp_reduce_sum(accu);
  if(lane == 0){
    float g = accg; if(limit > 0.f) g = fminf(g, limit);
    float u = accu; if(limit > 0.f) u = fmaxf(-limit, fminf(u, limit));
    float sv = g / (1.f + expf(-g));
    act[(int64_t)row * F + j] = sv * u;
  }
}

// Stage 2 (fused layout): down + router-weighted scatter-add.
//   fc2: [E][D][F/2] e2m1 (== dq_down). fc2_sf: per-expert stride
//   sf2_stride = padUp(D,128)*padUp(F/16,4); scale byte at sf_off(e) +
//   dsv4_sf_swizzled_index(i, c/16, F/16). global g_down[e].
__global__ void dec_down_scatter_fused(
    const float* __restrict__ act, const int* __restrict__ sel,
    const float* __restrict__ rw, const uint8_t* __restrict__ fc2_w,
    const uint8_t* __restrict__ fc2_sf, const float* __restrict__ g_down,
    float* __restrict__ out, int M, int U, int D, int F,
    int sf2_stride, int sf2_cols, int expert_base, int E_local, int vec16){
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int warps_per_block = blockDim.x >> 5;
  const int i = blockIdx.x * warps_per_block + warp;   // output row in [0,D)
  const int row = blockIdx.y;
  if(i >= D || row >= M*U) return;
  const int token = row / U;
  // [ep2-dp] GLOBAL->LOCAL remap + skip remote experts (other rank scatters them; AllReduce sums).
  const int e = sel[row] - expert_base;
  if(e < 0 || e >= E_local) return;
  const float w = rw[row];
  const float* __restrict__ a = act + (int64_t)row * F;
  const int nbytes = F >> 1;
  const int64_t wbase  = (int64_t)e * D * nbytes;
  const int64_t sfbase = (int64_t)e * sf2_stride;
  const float de = g_down[e];
  const uint8_t* __restrict__ wdr = fc2_w + wbase + (int64_t)i * nbytes;
  float acc = 0.f;
  if (vec16) {
    // [DSV4_MOE_VEC16] see dec_gate_up_swiglu_fused -- one byte per lane cannot fill DRAM.
    const int n16 = nbytes >> 4;
    for (int p = lane; p < n16; p += 32) {
      const uint4 wd4 = reinterpret_cast<const uint4 *>(wdr)[p];
      const int   c0   = 32 * p;
      const int   blk0 = c0 >> 4;
      const float sd0 = ue4m3_decode(fc2_sf[sfbase + dsv4_sf_swizzled_index(i, blk0,     sf2_cols)]) * de;
      const float sd1 = ue4m3_decode(fc2_sf[sfbase + dsv4_sf_swizzled_index(i, blk0 + 1, sf2_cols)]) * de;
      const uint32_t dw[4] = { wd4.x, wd4.y, wd4.z, wd4.w };
      #pragma unroll
      for (int q = 0; q < 4; q++) {
        const float sd = (q < 2) ? sd0 : sd1;
        #pragma unroll
        for (int t = 0; t < 4; t++) {
          const uint8_t bd = (uint8_t) (dw[q] >> (8 * t));
          const int     c  = c0 + 8 * q + 2 * t;
          acc += e2m1_decode(bd & 0xF) * sd * a[c] + e2m1_decode(bd >> 4) * sd * a[c + 1];
        }
      }
    }
  } else {
  for(int p = lane; p < nbytes; p += 32){
    int c = 2 * p;
    int blk = c / SFVec;
    float sd = ue4m3_decode(fc2_sf[sfbase + dsv4_sf_swizzled_index(i, blk, sf2_cols)]) * de;
    uint8_t bd = wdr[p];
    acc += e2m1_decode(bd & 0xF) * sd * a[c] + e2m1_decode(bd >> 4) * sd * a[c+1];
  }
  }
  acc = warp_reduce_sum(acc);
  if(lane == 0){
    atomicAdd(&out[(int64_t)token * D + i], w * acc);
  }
}

} // namespace dsv4_moe_grouped_detail

// ===== public load adapter ===================================================
using namespace dsv4_moe_grouped_detail;

// Fused-layer accessor (dsv4-moe-fused-run.cu). Under DSV4_MOE_FUSED the grouped
// gate/up weights + simple SFs are freed after the fused repack; the HP decode then
// reads the SAME fused fc1/fc2 + swizzled SF + g_common/g_down via these pointers.
// Returns false if the fused cache has no entry for the layer (grouped buffers live).
#ifdef DSV4_MOE_FUSED_CUTLASS
// Strong reference (the fused static lib is linked): forces archive extraction so
// the symbol is non-null and the decode path can share the fused weight set.
extern "C" bool dsv4_moe_fused_get_layer(
        int il, int* E, int* hidden, int* inter,
        const void** fc1_w, const void** fc2_w,
        const void** fc1_sf, const void** fc2_sf,
        const float** g_common, const float** g_down);
#else
// CUTLASS fused lib not built -> no fused layer ever exists; decode uses grouped
// buffers. Provide a stub so the single code path below compiles + always falls
// through to the grouped kernels.
static inline bool dsv4_moe_fused_get_layer(
        int, int*, int*, int*, const void**, const void**,
        const void**, const void**, const float**, const float**) { return false; }
#endif

// ---- host blob builder (no device needed) -----------------------------------
// Produces the exact 9-section NVFP4 registry blob for one layer / one rank from
// rank-local-sliced MXFP4 bytes. Used both by the OFFLINE tool and (internally)
// by the online converter so the two are byte-identical by construction.
void dsv4_moe_grouped_convert_layer(const void * gate_mxfp4,
                                    const void * up_mxfp4,
                                    const void * down_mxfp4,
                                    int n_expert,
                                    int n_embd,
                                    int n_ff_half,
                                    dsv4_moe_grouped_blob_header * hdr,
                                    std::vector<uint8_t> & out){
  const int n_gu=n_ff_half, k_gu=n_embd;   // gate/up: [n_ff_half, n_embd]
  const int n_d =n_embd,    k_d =n_ff_half;// down   : [n_embd,    n_ff_half]
  const int blocks_per_expert_gu=(n_gu*k_gu)/32;
  const int blocks_per_expert_d =(n_d *k_d )/32;

  const host_block_mxfp4* G =reinterpret_cast<const host_block_mxfp4*>(gate_mxfp4);
  const host_block_mxfp4* U =reinterpret_cast<const host_block_mxfp4*>(up_mxfp4);
  const host_block_mxfp4* Dn=reinterpret_cast<const host_block_mxfp4*>(down_mxfp4);

  std::vector<std::vector<uint8_t>>  qg(n_expert), qu(n_expert), qd(n_expert);
  std::vector<std::vector<ElementSF>> sg(n_expert), su(n_expert), sd(n_expert);
  std::vector<float> glg(n_expert), glu(n_expert), gld(n_expert);

  for(int e=0;e<n_expert;e++){
    glg[e]=convert_expert_mxfp4_to_nvfp4(G  + (size_t)e*blocks_per_expert_gu, n_gu,k_gu, qg[e],sg[e]);
    glu[e]=convert_expert_mxfp4_to_nvfp4(U  + (size_t)e*blocks_per_expert_gu, n_gu,k_gu, qu[e],su[e]);
    gld[e]=convert_expert_mxfp4_to_nvfp4(Dn + (size_t)e*blocks_per_expert_d , n_d ,k_d , qd[e],sd[e]);
  }

  // pack e2m1 per tensor into a single contiguous host buffer (= what the device holds)
  auto cat_q=[&](const std::vector<std::vector<uint8_t>>& q, std::vector<uint8_t>& dst){
    size_t total=0; for(auto& v:q) total+=v.size();
    dst.resize(total); size_t off=0; for(auto& v:q){ memcpy(dst.data()+off,v.data(),v.size()); off+=v.size(); }
  };
  std::vector<uint8_t> blob_qg, blob_qu, blob_qd;
  cat_q(qg,blob_qg); cat_q(qu,blob_qu); cat_q(qd,blob_qd);

  std::vector<ElementSF> sfb_g, sfb_u, sfb_d;
  int sfb_words_gu = build_sfb_swizzled_host(sg, n_gu,k_gu, sfb_g);
  (void)             build_sfb_swizzled_host(su, n_gu,k_gu, sfb_u);
  int sfb_words_d  = build_sfb_swizzled_host(sd, n_d ,k_d , sfb_d);

  // ElementSF (float_ue4m3_t) is a 1-byte storage type; copy raw storage bytes.
  static_assert(sizeof(ElementSF)==1, "ue4m3 SFB element must be 1 byte");

  hdr->n_expert=n_expert; hdr->n_embd=n_embd; hdr->n_ff_half=n_ff_half;
  hdr->sfb_words_gu=sfb_words_gu; hdr->sfb_words_d=sfb_words_d; hdr->_pad=0;

  auto append=[&](const void* p,size_t n){ const uint8_t* b=(const uint8_t*)p; out.insert(out.end(),b,b+n); };
  size_t before=out.size();
  append(blob_qg.data(), blob_qg.size());
  append(blob_qu.data(), blob_qu.size());
  append(blob_qd.data(), blob_qd.size());
  append(sfb_g.data(), sfb_g.size()*sizeof(ElementSF));
  append(sfb_u.data(), sfb_u.size()*sizeof(ElementSF));
  append(sfb_d.data(), sfb_d.size()*sizeof(ElementSF));
  append(glg.data(), glg.size()*4);
  append(glu.data(), glu.size()*4);
  append(gld.data(), gld.size()*4);
  hdr->blob_bytes = out.size()-before;
}

// ---- upload a pre-converted blob into the device registry (NO conversion) ----
void dsv4_moe_grouped_set_expert_weights_blob(int il,
                                              const dsv4_moe_grouped_blob_header * hdr,
                                              const void * blob,
                                              size_t blob_size){
  std::lock_guard<std::mutex> lk(g_reg_mu);
  if(g_registry.count(il)) return; // one-time

  const int n_expert=hdr->n_expert, n_embd=hdr->n_embd, n_ff_half=hdr->n_ff_half;
  const int n_gu=n_ff_half, k_gu=n_embd;
  const int n_d =n_embd,    k_d =n_ff_half;

  const size_t qg_bytes  = (size_t)n_expert*(n_gu*k_gu)/2;
  const size_t qd_bytes  = (size_t)n_expert*(n_d *k_d )/2;
  const size_t sfg_bytes = (size_t)n_expert*hdr->sfb_words_gu*sizeof(ElementSF);
  const size_t sfd_bytes = (size_t)n_expert*hdr->sfb_words_d *sizeof(ElementSF);
  const size_t gl_bytes  = (size_t)n_expert*4;
  const size_t expect = qg_bytes*2 + qd_bytes + sfg_bytes*2 + sfd_bytes + gl_bytes*3;
  if(expect != blob_size || (hdr->blob_bytes && hdr->blob_bytes != blob_size)){
    fprintf(stderr,"[dsv4-moe-grouped] blob size mismatch layer %d: got %zu expect %zu (hdr %llu)\n",
            il, blob_size, expect, (unsigned long long)hdr->blob_bytes); abort();
  }

  LayerWeights* L=new LayerWeights();
  L->n_expert=n_expert; L->n_embd=n_embd; L->n_ff_exp=n_ff_half;
  L->sfb_words_gu=hdr->sfb_words_gu; L->sfb_words_d=hdr->sfb_words_d;

  const uint8_t* p=(const uint8_t*)blob;
  auto up_dev=[&](size_t nbytes,void** dptr)->const uint8_t*{
    DSV4_CK(cudaMalloc(dptr,nbytes));
    DSV4_CK(cudaMemcpy(*dptr,p,nbytes,cudaMemcpyHostToDevice));
    return p+nbytes;
  };
  // [fc1-fused-layout] DSV4_MOE_FC1_FUSED=1: store gate/up as ONE buffer in the CUTLASS
  // fused-fc1 interleave (per expert: up rows then gate rows) instead of two tensors.
  // The fused prefill runner then ALIASES this buffer (zero repack malloc/concat/free ->
  // kills the per-layer +1.07GB fragmentation AND the decode use-after-free), and the
  // grouped decode/GEMM paths read the same bytes with expert stride 2x (gu_estride).
  static const bool fc1_fused = getenv("DSV4_MOE_FC1_FUSED") != nullptr;
  if (fc1_fused) {
    const size_t per_e = (size_t)(n_gu*k_gu)/2;            // packed bytes per expert per tensor
    static std::vector<uint8_t> stage;                      // reused across layers (host)
    stage.resize(qg_bytes*2);
    const uint8_t* gsec = p;                                // blob section [0] = gate
    const uint8_t* usec = p + qg_bytes;                     // blob section [1] = up
    for (int e = 0; e < n_expert; e++) {
      memcpy(stage.data() + (size_t)e*2*per_e,         usec + (size_t)e*per_e, per_e); // up first
      memcpy(stage.data() + (size_t)e*2*per_e + per_e, gsec + (size_t)e*per_e, per_e); // then gate
    }
    void* dfused = nullptr;
    DSV4_CK(cudaMalloc(&dfused, qg_bytes*2));
    DSV4_CK(cudaMemcpy(dfused, stage.data(), qg_bytes*2, cudaMemcpyHostToDevice));
    L->dq_up   = reinterpret_cast<decltype(L->dq_up)>((uint8_t*)dfused);          // base = expert 0 up rows
    L->dq_gate = reinterpret_cast<decltype(L->dq_gate)>((uint8_t*)dfused + per_e); // expert 0 gate rows
    L->gu_estride = (int64_t)2*per_e;
    p += qg_bytes*2;                                        // consumed sections [0]+[1]
  } else {
    p=up_dev(qg_bytes ,(void**)&L->dq_gate);
    p=up_dev(qg_bytes ,(void**)&L->dq_up);
  }
  p=up_dev(qd_bytes ,(void**)&L->dq_down);
  // [ep2-dp][mem] Under EP the grouped CUTLASS prefill path is HARD-GUARDED OFF (EP-unsafe
  // histogram/OOB — see the dispatch comment) and every sub-min_m M takes the HP path, which
  // reads only the SIMPLE (un-swizzled) SFs. The swizzled dsf_gate/up/down are therefore DEAD
  // under EP: skip the device upload entirely (~8.6GB across 43 layers on DSV4-Flash) — this
  // is the headroom the fused runner's own swizzled SFs need on big prefills.
  if (g_ep) {
    L->dsf_gate = nullptr; L->dsf_up = nullptr; L->dsf_down = nullptr;
    p += sfg_bytes*2 + sfd_bytes;   // blob sections [3][4][5] stay host-side (simple SFs built below)
  } else {
    p=up_dev(sfg_bytes,(void**)&L->dsf_gate);
    p=up_dev(sfg_bytes,(void**)&L->dsf_up);
    p=up_dev(sfd_bytes,(void**)&L->dsf_down);
  }
  p=up_dev(gl_bytes ,(void**)&L->dglobal_gate);
  p=up_dev(gl_bytes ,(void**)&L->dglobal_up);
  p=up_dev(gl_bytes ,(void**)&L->dglobal_down);

  L->gg_gate.alloc(n_expert, n_gu, k_gu, L->sfb_words_gu);
  L->gg_up.alloc  (n_expert, n_gu, k_gu, L->sfb_words_gu);
  if (L->gu_estride) { L->gg_gate.b_estride_bytes = L->gu_estride; L->gg_up.b_estride_bytes = L->gu_estride; }
  L->gg_down.alloc(n_expert, n_d , k_d , L->sfb_words_d);

  // ---- HIGH-PRECISION DECODE: build simple (un-swizzled) tight SFB copies ----
  // Need the swizzled host bytes; re-read them from the blob (sections [3],[4],[5]).
  {
    const uint8_t* base=(const uint8_t*)blob;
    const ElementSF* sfg=(const ElementSF*)(base + qg_bytes*2 + qd_bytes);
    const ElementSF* sfu=(const ElementSF*)((const uint8_t*)sfg + sfg_bytes);
    const ElementSF* sfd=(const ElementSF*)((const uint8_t*)sfu + sfg_bytes);
    std::vector<uint8_t> simp_g, simp_u, simp_d;
    unswizzle_sfb_to_simple(sfg, n_expert, n_gu, k_gu, hdr->sfb_words_gu, simp_g);
    unswizzle_sfb_to_simple(sfu, n_expert, n_gu, k_gu, hdr->sfb_words_gu, simp_u);
    unswizzle_sfb_to_simple(sfd, n_expert, n_d , k_d , hdr->sfb_words_d , simp_d);
    auto up_simple=[&](const std::vector<uint8_t>& v,uint8_t** dptr){
      DSV4_CK(cudaMalloc(dptr, v.size()));
      DSV4_CK(cudaMemcpy(*dptr, v.data(), v.size(), cudaMemcpyHostToDevice)); };
    up_simple(simp_g,&L->dsf_gate_simple);
    up_simple(simp_u,&L->dsf_up_simple);
    up_simple(simp_d,&L->dsf_down_simple);
  }

  g_registry[il]=L;
  fprintf(stderr,"[dsv4-moe-grouped] layer %d NVFP4 weights loaded FROM SIDECAR (E=%d n_embd=%d n_ff_half=%d)\n",
          il,n_expert,n_embd,n_ff_half);
}

void dsv4_moe_grouped_set_expert_weights(int il,
                                         const void * gate_mxfp4,
                                         const void * up_mxfp4,
                                         const void * down_mxfp4,
                                         int n_expert,
                                         int n_embd,
                                         int n_ff_exp){
  // Convert on the host, then upload via the blob path: guarantees the online and
  // OFFLINE (sidecar) registries are byte-identical.
  dsv4_moe_grouped_blob_header hdr{};
  std::vector<uint8_t> blob;
  dsv4_moe_grouped_convert_layer(gate_mxfp4, up_mxfp4, down_mxfp4,
                                 n_expert, n_embd, n_ff_exp, &hdr, blob);
  dsv4_moe_grouped_set_expert_weights_blob(il, &hdr, blob.data(), blob.size());
}

bool dsv4_moe_grouped_have_layer(int il){
  std::lock_guard<std::mutex> lk(g_reg_mu);
  return g_registry.count(il)>0;
}

// Number of registered grouped layers (for the decode warm-up gate).
int dsv4_grouped_layer_count(){
  std::lock_guard<std::mutex> lk(g_reg_mu);
  return (int)g_registry.size();
}

// Grouped-decode HP buffers warmed? One non-captured decode pass touches every layer once (graphs
// were deferred), allocating all d_act_decode buffers; after >= n_layers decode-shaped calls they
// are all sized -> warmed -> the engine re-enables decode graphs. (Fixes the fused-prefill ->
// grouped-decode mid-capture cudaMalloc = "internal operation failed" crash at long context.)
extern "C" bool dsv4_moe_grouped_decode_warmed(void){
  const int n = dsv4_grouped_layer_count();
  return g_grouped_decode_warm_count.load(std::memory_order_relaxed) >= (n > 0 ? n : 1);
}

// ---- accessor for the FUSED MoE op (dsv4-moe-fused.cu) -----------------------
// The flashinfer CUTLASS fused runner consumes the SAME per-layer NVFP4 registry
// this file owns. Rather than expose g_registry/LayerWeights (CUTLASS types) to
// the fused TU, hand it the raw device pointers + dims it needs. The fused op is
// responsible for any re-layout (gate||up concat into fused fc1, scale re-swizzle
// to the fused kernel's tile-atom, global reconciliation) -- see the port doc.
//
// Returns false if the layer is not registered. The *_simple scale pointers are
// the UN-SWIZZLED tight [E][n][k/16] ue4m3 (uint8) layout (built by
// unswizzle_sfb_to_simple); they are currently populated for the HP-decode path
// and are the right starting point for the fused kernel's own swizzler.
extern "C" bool dsv4_moe_grouped_get_layer_nvfp4(
        int il,
        int* n_expert, int* n_embd, int* n_ff_exp,
        const void** dq_gate,  const void** dq_up,  const void** dq_down,
        const void** sf_gate_simple, const void** sf_up_simple, const void** sf_down_simple,
        const float** global_gate, const float** global_up, const float** global_down) {
  std::lock_guard<std::mutex> lk(g_reg_mu);
  auto it = g_registry.find(il);
  if (it == g_registry.end() || !it->second) return false;
  LayerWeights* L = it->second;
  if (n_expert) *n_expert = L->n_expert;
  if (n_embd)   *n_embd   = L->n_embd;
  if (n_ff_exp) *n_ff_exp = L->n_ff_exp;     // == n_ff_half (this rank's slice)
  if (dq_gate)  *dq_gate  = (const void*)L->dq_gate;
  if (dq_up)    *dq_up    = (const void*)L->dq_up;
  if (dq_down)  *dq_down  = (const void*)L->dq_down;
  if (sf_gate_simple) *sf_gate_simple = (const void*)L->dsf_gate_simple;
  if (sf_up_simple)   *sf_up_simple   = (const void*)L->dsf_up_simple;
  if (sf_down_simple) *sf_down_simple = (const void*)L->dsf_down_simple;
  if (global_gate) *global_gate = L->dglobal_gate;
  if (global_up)   *global_up   = L->dglobal_up;
  if (global_down) *global_down = L->dglobal_down;
  return true;
}

// Free the grouped-path buffers that the FUSED path supersedes after it has built
// its own fused-format copies. The fused fc1 weights are a rearrangement of
// dq_gate+dq_up (freeable); the fused fc1 SF is rebuilt from dsf_gate/up_simple
// (freeable). The fused path ALIASES dq_down (fc2 weights) + dglobal_* + keeps
// dsf_down_simple is also rebuilt -> freeable. Down weights MUST be kept (aliased).
// Only valid when DSV4_MOE_FUSED commits to the fused path (no grouped fallback for
// this layer afterward). Returns false if layer not found.
// [fc1-fused-layout] per-expert byte stride of dq_gate/dq_up (0 = legacy separate tensors).
// The fused runner uses this to detect the alias-able layout (== n_ff*n_embd bytes).
extern "C" int64_t dsv4_moe_grouped_gu_estride(int il) {
  std::lock_guard<std::mutex> lk(g_reg_mu);
  auto it = g_registry.find(il);
  return (it == g_registry.end() || !it->second) ? 0 : it->second->gu_estride;
}

extern "C" bool dsv4_moe_grouped_free_superseded_by_fused(int il) {
  std::lock_guard<std::mutex> lk(g_reg_mu);
  auto it = g_registry.find(il);
  if (it == g_registry.end() || !it->second) return false;
  LayerWeights* L = it->second;
  // [fc1-fused-layout] the fused runner ALIASES dq_up/dq_gate (one shared fc1 buffer):
  // weights are NOT superseded (shared) — but once fused is live, decode routes to the
  // fused-aware HP kernels (dec_*_fused read fc1_sf/fc2_sf), so the tight SIMPLE SFs
  // ARE superseded: free them (~8.6GB/rank across 43 layers) to make room for the
  // fused swizzled SFs the repack just built (+~9GB). Net repack cost ≈ +0.4GB.
  if (L->gu_estride) {
    auto FS=[](uint8_t*&p){ if(p){ cudaFree(p); p=nullptr; } };
    FS(L->dsf_gate_simple); FS(L->dsf_up_simple); FS(L->dsf_down_simple);
    return true;
  }
  auto F=[](void*&p){ if(p){ cudaFree(p); p=nullptr; } };
  // gate/up packed weights (concatenated into fused fc1) -> free
  F((void*&)L->dq_gate); F((void*&)L->dq_up);
  // gate/up/down simple SF (re-swizzled into fused SF) -> free
  F((void*&)L->dsf_gate_simple); F((void*&)L->dsf_up_simple); F((void*&)L->dsf_down_simple);
  // NOTE: keep dq_down (fused fc2 aliases it), dglobal_* (fused reads them),
  // and the grouped gg_*/swizzled dsf_* (harmless; could also free but small).
  return true;
}

void dsv4_moe_grouped_free_all(void){
  std::lock_guard<std::mutex> lk(g_reg_mu);
  for(auto& kv : g_registry){
    LayerWeights* L=kv.second; if(!L) continue;
    auto F=[](void*p){ if(p) cudaFree(p); };
    F(L->dq_gate);F(L->dq_up);F(L->dq_down);
    F(L->dsf_gate);F(L->dsf_up);F(L->dsf_down);
    F(L->dglobal_gate);F(L->dglobal_up);F(L->dglobal_down);
    F(L->dsf_gate_simple);F(L->dsf_up_simple);F(L->dsf_down_simple);
    F(L->d_act_decode);
    // prefill arena
    F(L->pf_hist);F(L->pf_cursor);F(L->pf_indptr);F(L->pf_sfa_off_D);F(L->pf_sfa_off_F);
    F(L->pf_vals);F(L->pf_grp_of_row);F(L->pf_gscaleX);F(L->pf_gscaleAct);F(L->pf_absmax_part);
    F(L->pf_Xs_full);F(L->pf_Xs);F(L->pf_SFx_tight);F(L->pf_gate);F(L->pf_up);
    F(L->pf_act_full);F(L->pf_act);F(L->pf_SFa_tight);F(L->pf_down);F(L->pf_SFx);F(L->pf_SFa);
    L->gg_gate.free_all(); L->gg_up.free_all(); L->gg_down.free_all();
    delete L;
  }
  g_registry.clear();
  dsv4_moe_grouped_drain_retired();   // free any deferred (retired) grow allocations
}

// ===== the op ================================================================
// dst : F32 [n_embd, n_tokens]              moe_out
// src0: F32 [n_embd, n_tokens]              hidden states (this layer's ffn input)
// src1: I32 [n_expert_used, n_tokens]       selected expert ids
// src2: F32 [n_expert_used, n_tokens]       router weights (already normalized/scaled)
// op_params[0] = il
bool ggml_cuda_op_dsv4_moe_grouped(ggml_backend_cuda_context & ctx, ggml_tensor * dst){
  ggml_tensor * hidden = dst->src[0];
  ggml_tensor * sel    = dst->src[1];
  ggml_tensor * rw     = dst->src[2];

  GGML_ASSERT(hidden->type == GGML_TYPE_F32);
  GGML_ASSERT(sel->type    == GGML_TYPE_I32);
  GGML_ASSERT(rw->type     == GGML_TYPE_F32);
  GGML_ASSERT(dst->type    == GGML_TYPE_F32);

  const int il = ggml_get_op_params_i32(dst, 0);
  // DSV4 per-layer SwiGLU clamp limit (float bits in op_params[1]); 0 => no clamp.
  // Matches the reference routed-expert path: gate clamped to (-INF, limit], up to [-limit, limit].
  const float swiglu_limit = ggml_get_op_params_f32(dst, 1);
  LayerWeights* L=nullptr;
  { std::lock_guard<std::mutex> lk(g_reg_mu); auto it=g_registry.find(il); if(it!=g_registry.end()) L=it->second; }
  GGML_ASSERT(L && "dsv4_moe_grouped: layer weights not registered");

  const int D = (int)hidden->ne[0];          // n_embd
  const int M = (int)hidden->ne[1];          // n_tokens
  const int U = (int)sel->ne[0];             // n_expert_used
  const int E = L->n_expert;
  const int F = L->n_ff_exp;
  GGML_ASSERT(D==L->n_embd);
  const int rows = M*U;
  const int kbD = D/SFVec, kbF = F/SFVec;

  // [ep2-dp] EP shard params for this op's kernels. Under EP this rank holds local experts
  // [g_ep_base, g_ep_base+g_ep_n_local); sel[] are GLOBAL ids. The HP-DECODE kernels below remap
  // GLOBAL->LOCAL and SKIP remote experts (the other rank covers them; the PARTIAL AllReduce sums).
  // Non-EP => ep_base=0, ep_E_local=E => the range check is a no-op (byte-identical).
  // NOTE: the large-M CUTLASS GROUPED prefill path further below is NOT EP-aware; under EP the graph
  // routes large M to the FUSED op (runMoe) and only small M (decode + the fused-live HP band) here,
  // so the grouped CUTLASS prefill is never reached under EP. Guard it explicitly to be safe.
  const int ep_base    = g_ep ? g_ep_base    : 0;
  const int ep_E_local = g_ep ? g_ep_n_local : E;

  cudaStream_t s = ctx.stream();

  const float * d_hidden = (const float*) hidden->data;
  const int   * d_sel    = (const int*)   sel->data;
  const float * d_rw     = (const float*) rw->data;
  float       * d_out    = (float*)       dst->data;

  // [FUSED-LIVE FREED-WEIGHT OOB FIX] Under DSV4_MOE_FUSED, once the FUSED path has run for
  // this layer it FREES the grouped per-projection gate/up weights (dq_gate, dq_up) +
  // simple SFs (dsv4_moe_grouped_free_superseded_by_fused) to reclaim ~2 GB/layer. The graph
  // routes M >= DSV4_MOE_FUSED_MIN_M (default 256) to the fused op and M <= 16 to the HP-decode
  // path below (which is ALREADY fused-aware: it reads the fused fc1/fc2 via the dec_*_fused
  // kernels). But a prefill ubatch in the BAND 16 < M < 256 — e.g. the small last chunk of a
  // long prompt when total_tokens % n_ubatch lands in (16,256) — fell through to the grouped
  // CUTLASS PREFILL path (below), which reads the now-FREED L->dq_gate / L->dq_up (== nullptr)
  // -> illegal memory access. This is EAGER (no graph) and DATA-DEPENDENT (the crash chunk
  // varies with prompt length % n_ubatch -> "~5th chunk, 5k-9k run to run"). It is NOT a graph
  // or CUTLASS-runner bug (the fused runner is compute-sanitizer-clean over 60 imbalanced
  // M<=1024 chunks). FIX: when the FUSED weights are live for this layer, route the ENTIRE
  // small-prefill band (M < DSV4_MOE_FUSED_MIN_M) through the fused-aware HP path below, which
  // never touches the freed grouped gate/up. The fast grouped CUTLASS prefill path is unchanged
  // for non-fused runs (fused_live == false) and for M >= min_m (handled by the fused op itself).
  static const int DSV4_FUSED_MIN_M = []{
      if (getenv("DSV4_MOE_FUSED_DECODE") != nullptr) return 1;
      const char* e = getenv("DSV4_MOE_FUSED_MIN_M"); return e ? atoi(e) : 256;
  }();
  bool prefill_fused_live = false;
  {
      int t_E=0,t_h=0,t_i=0; const void *p0,*p1,*p2,*p3; const float *p4,*p5;
      prefill_fused_live = dsv4_moe_fused_get_layer(il,&t_E,&t_h,&t_i,&p0,&p1,&p2,&p3,&p4,&p5);
  }

  // ===== HIGH-PRECISION DECODE PATH (small M) ================================
  // For decode / MTP-verify (small batch) the NVFP4 4-bit ACTIVATION is too coarse
  // and flips borderline tokens (CJK corruption). Use FP32 activations with on-the-fly
  // dequantized NVFP4 weights. Threshold mirrors the mmvq mmid small-batch idea.
  static const int DSV4_DECODE_MAX = []{ const char*e=getenv("DSV4_MOE_DECODE_MAX"); return e?atoi(e):16; }();
  // HP path fires for: genuine decode/verify (M<=DSV4_DECODE_MAX), OR any small-prefill band
  // chunk (M<DSV4_FUSED_MIN_M) on a FUSED-LIVE layer (the freed-weight OOB fix above). Both use
  // the fused-aware kernels when prefill_fused_live, so neither reads the freed grouped gate/up.
  // NOTE: hp_small_prefill bypasses DSV4_MOE_NO_HP_DECODE — when fused is live the grouped CUTLASS
  // prefill path below is UNUSABLE (it reads the freed dq_gate/dq_up), so the fused-aware HP path is
  // the ONLY correct route for M<min_m on a fused-live layer, regardless of that debug toggle.
  //
  // [ep2-dp] EP CORRECTNESS: the grouped CUTLASS prefill path below is EP-UNSAFE (its on-device
  // routing histograms sel[] over E LOCAL bins but sel carries GLOBAL ids -> remote ids corrupt the
  // histogram + read OOB weights; it is hard-guarded to abort under g_ep). So under EP, EVERY M below
  // the fused threshold (the whole small-prefill band 16<M<min_m AND, crucially, the SHORT-PROMPT case
  // where M<min_m and fused was NEVER built so prefill_fused_live is false) MUST take the EP-aware HP
  // path -- otherwise a short prompt (M in (16,255]) falls through to the guarded CUTLASS path, writes
  // NO moe_out (return false), and the residual stream corrupts -> "<<<<" repetition. The HP path uses
  // the fused-aware kernels when fused is live, else the non-fused EP-aware kernels reading the still-
  // valid grouped dq_gate/dq_up (build_fused_layer -- which frees them -- was never called for a short
  // prompt). Both are EP-aware (GLOBAL->LOCAL remap + remote-skip). Large M (>=min_m) still routes to
  // the fast fused op upstream and never reaches this op. [ep2-dp]
  const bool hp_small_prefill = (prefill_fused_live || g_ep) && (M < DSV4_FUSED_MIN_M);
  if (hp_small_prefill || (M <= DSV4_DECODE_MAX && getenv("DSV4_MOE_NO_HP_DECODE")==nullptr)) {
    // Persistent per-layer activation scratch.  For the REAL decode regime
    // (M small) we size to a small constant graph-capture ceiling (16*U*F) so
    // the buffer never grows mid-capture -> CUDA-graph-safe.  When the HP path
    // is force-enabled for PREFILL (DSV4_MOE_DECODE_MAX set huge, M up to 512+),
    // that prefill batch is NOT graph-captured, so we simply size to the exact
    // per-call need.  The OLD formula (DSV4_DECODE_MAX*U*F) allocated GBs/layer
    // when DSV4_DECODE_MAX was set to 99999 -> OOM/crash; that's fixed here.
    {
      const int graph_ceiling = DSV4_DECODE_MAX < 16 ? DSV4_DECODE_MAX : 16;
      size_t want = (size_t)graph_ceiling * U * F;         // small graph-safe ceiling
      size_t need = (size_t)rows * F;                      // exact need for this call
      if (want < need) want = need;                        // grow to fit large (prefill) M
      // [FUSED-LIVE band capture-safety] When this is the small-prefill band (16<M<min_m on a
      // fused-live layer), pin the buffer to the WORST-CASE band size ((min_m-1)*U*F) ONCE so it
      // never regrows for a different band M (a mid-capture cudaMalloc would be illegal). The band
      // is the rare partial last chunk, so the one-time over-alloc (~min_m*U*F floats) is cheap.
      if (hp_small_prefill) {
          size_t band_cap = (size_t)(DSV4_FUSED_MIN_M - 1) * U * F;
          if (want < band_cap) want = band_cap;
      }
      if (L->d_act_decode == nullptr || L->act_decode_elems < want) {
        // CAPTURE-SAFETY (fused-prefill -> grouped-decode transition crash): under
        // DSV4_MOE_FUSED, PREFILL uses the FUSED op, so the GROUPED decode buffer
        // d_act_decode is never warmed during prefill -> the FIRST decode allocates it.
        // If that first decode is being CUDA-graph-CAPTURED, the cudaMalloc here is illegal
        // mid-capture -> "an internal operation failed" at long context. The engine's grouped
        // graph-gate (ggml-cuda.cu) now disables graphs for grouped-decode until warmed (see
        // dsv4_moe_grouped_decode_warmed), so this alloc lands OUTSIDE capture. Belt-and-suspenders:
        // also bail clearly if somehow still capturing with an unready buffer.
        cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
        cudaStreamIsCapturing(s, &cap);
        if (cap != cudaStreamCaptureStatusNone) {
          fprintf(stderr, "[dsv4-grouped] FATAL: decode act buffer (il=%d) unsized during graph "
                          "capture; cannot cudaMalloc mid-capture. The grouped graph-gate should "
                          "have deferred capture until warmed.\n", il);
          return false; // surface as a launch error rather than an opaque capture failure
        }
        std::lock_guard<std::mutex> lk(g_reg_mu);          // serialize first-touch alloc
        if (L->d_act_decode == nullptr || L->act_decode_elems < want) {
          // CUDA-graph-safe grow: alloc new, retire old (defer-free at next sync),
          // never cudaFree the live buffer inline (it may be in a captured graph).
          float* n_act=nullptr;
          DSV4_CK(cudaMalloc((void**)&n_act, want * sizeof(float)));
          dsv4_retire(L->d_act_decode);
          L->d_act_decode = n_act;
          L->act_decode_elems = want;
        }
      }
    }
    // Decode-shaped call with a ready buffer -> count toward warm-up (graphs re-enable once every
    // layer has been touched once outside capture). Monotonic; repeated calls keep it warmed.
    if (M <= DSV4_DECODE_MAX) g_grouped_decode_warm_count.fetch_add(1, std::memory_order_relaxed);
    float* d_act = L->d_act_decode;
    DSV4_CK(cudaMemsetAsync(d_out, 0, (size_t)M*D*4, s));
    // WARP-PER-OUTPUT re-tiling: one warp computes one output element so the K
    // (= D for gate/up, F for down) reduction is coalesced across the 32 lanes,
    // and the grid spans ALL output rows x (token*U+slot) -> hundreds of blocks
    // that fill the GB10 SMs instead of the old ~6 blocks (1 per token*expert).
    const int WPB = 4;                                     // warps per block (128 thr)
    const int blk = WPB * 32;
    dim3 g_gu((F + WPB - 1) / WPB, rows);                  // gate/up: F output rows
    dim3 g_dn((D + WPB - 1) / WPB, rows);                  // down:    D output rows

    // ORTHODOX SINGLE-WEIGHT-SET: if the FUSED path built this layer, the grouped
    // gate/up weights + simple SFs were freed (real memory reclaimed). Read gate/up
    // from the fused fc1 slices + fc2 + the fused swizzled SF + g_common/g_down.
    // There is exactly ONE weight set; nothing here reads the freed dq_gate/dq_up.
    int  fu_E=0, fu_hidden=0, fu_inter=0;
    const void *fc1_w=nullptr,*fc2_w=nullptr,*fc1_sf=nullptr,*fc2_sf=nullptr;
    const float *fu_gcommon=nullptr,*fu_gdown=nullptr;
    const bool fused_live =
        dsv4_moe_fused_get_layer(il, &fu_E, &fu_hidden, &fu_inter,
            &fc1_w, &fc2_w, &fc1_sf, &fc2_sf, &fu_gcommon, &fu_gdown);
    if (fused_live) {
      GGML_ASSERT(fu_E==E && fu_hidden==D && fu_inter==F &&
                  "fused/grouped decode dim mismatch");
      // Per-expert SWIZZLED_128x4 SF strides + unpadded K-block col counts (mirror
      // of dsv4-moe-fused-run.cu build: padN=padUp(rows,128), padC=padUp(cols,4)).
      // [DSV4_MOE_VEC16] 16-byte weight loads in the decode GEVM (see the kernels). Default ON
      // once measured; DSV4_MOE_VEC16=0 restores the byte-load loop for an A/B.
      static const bool moe_vec16 = !getenv("DSV4_MOE_VEC16_OFF");
      auto padUp=[](int x,int a){ return (x + a - 1)/a*a; };
      const int colsD = D / SFVec, colsF = F / SFVec;
      const int sf1_stride = padUp(2*F,128) * padUp(colsD,4); // fc1: rows=2*inter, K=D
      const int sf2_stride = padUp(D,  128) * padUp(colsF,4); // fc2: rows=D,       K=F
      dec_gate_up_swiglu_fused<<<g_gu, blk, 0, s>>>(
          d_hidden, d_sel,
          reinterpret_cast<const uint8_t*>(fc1_w),
          reinterpret_cast<const uint8_t*>(fc1_sf),
          fu_gcommon, d_act, M, U, D, F, /*inter=*/F,
          sf1_stride, colsD, swiglu_limit, ep_base, ep_E_local, moe_vec16);
      dec_down_scatter_fused<<<g_dn, blk, 0, s>>>(
          d_act, d_sel, d_rw,
          reinterpret_cast<const uint8_t*>(fc2_w),
          reinterpret_cast<const uint8_t*>(fc2_sf),
          fu_gdown, d_out, M, U, D, F, sf2_stride, colsF, ep_base, ep_E_local, moe_vec16);
      return true;
    }

    static const bool w4a16_dec = getenv("DSV4_MOE_W4A16_DECODE") != nullptr;
    if (w4a16_dec) {
      // FAITHFUL b12x fp4_dot8 decode variant (A/B vs dec_* below).
      w4a16_fc1_swiglu<<<g_gu, blk, 0, s>>>(
          d_hidden, d_sel,
          reinterpret_cast<const uint8_t*>(L->dq_gate),
          reinterpret_cast<const uint8_t*>(L->dq_up),
          L->dsf_gate_simple, L->dsf_up_simple,
          L->dglobal_gate, L->dglobal_up, d_act, M, U, D, F, swiglu_limit, ep_base, ep_E_local,
          L->gu_estride ? L->gu_estride : (int64_t)F*D/2);
      w4a16_fc2_scatter<<<g_dn, blk, 0, s>>>(
          d_act, d_sel, d_rw,
          reinterpret_cast<const uint8_t*>(L->dq_down),
          L->dsf_down_simple, L->dglobal_down, d_out, M, U, D, F, ep_base, ep_E_local);
      return true;
    }
    dec_gate_up_swiglu<<<g_gu, blk, 0, s>>>(
        d_hidden, d_sel,
        reinterpret_cast<const uint8_t*>(L->dq_gate),
        reinterpret_cast<const uint8_t*>(L->dq_up),
        L->dsf_gate_simple, L->dsf_up_simple,
        L->dglobal_gate, L->dglobal_up, d_act, M, U, D, F, swiglu_limit, ep_base, ep_E_local,
        L->gu_estride ? L->gu_estride : (int64_t)F*D/2);
    dec_down_scatter<<<g_dn, blk, 0, s>>>(
        d_act, d_sel, d_rw,
        reinterpret_cast<const uint8_t*>(L->dq_down),
        L->dsf_down_simple, L->dglobal_down, d_out, M, U, D, F, ep_base, ep_E_local);
    // NO sync, NO free: result feeds the graph; scratch is persistent.
    return true;
  }

  // [ep2-dp] EP GUARD: the large-M CUTLASS grouped prefill path below is NOT EP-aware (its on-device
  // routing histograms sel[] over E local bins but sel carries GLOBAL ids [0,256) -> remote ids index
  // out of the local histogram and the GEMM reads the wrong/OOB expert weights). Under EP the graph
  // routes all large M to the FUSED op (runMoe, native EP), so this path must never be reached with EP.
  // If it is (misconfig), fail loudly rather than silently corrupt. (g_ep==0 default => no change.)
  if (g_ep) {
    fprintf(stderr, "[dsv4-moe-grouped] FATAL: EP reached the grouped CUTLASS prefill path (il=%d M=%d). "
                    "Large-M EP must run on the fused op; check DSV4_MOE_FUSED + min_m routing.\n", il, M);
    return false;
  }

  // ===== CAPTURE-SAFE PREFILL PATH (large M, CUTLASS grouped GEMM) ============
  // Every capture-breaker the old prefill path had (thrust::sort_by_key, D2H copies
  // + cudaStreamSynchronize for the host histogram/indptr/absmax, per-call
  // cudaMalloc/cudaFree, host_problem_shapes) is replaced by pure on-device kernels
  // + a persistent per-layer arena. The MATH (routing/clamp/dequant/quant scales)
  // is bit-identical: routing groups by expert (intra-group order is irrelevant),
  // the absmax/gscale are the same reduction done on-device, and the GEMMs are the
  // same dense per-group ops. -> the WHOLE op is now CUDA-graph capturable.
  //
  // Persistent arena sized to rows_cap = max(floor*U, rows), GROWN ONCE if a larger
  // prefill batch arrives. The arena is ~0.26 MB/token PER LAYER; a too-large floor
  // multiplies that across ~58 layers and can trip the box memory cap (silent OOM-kill).
  // Default floor = 0 -> size to the ACTUAL prefill rows (= ubatch*U, typically 256*U);
  // grow-once covers any bigger chunk. Set DSV4_MOE_PREFILL_MAX>0 to pre-size to that
  // many tokens (avoids the one-time grow / re-warmup when a known max ubatch is used).
  // PRE-SIZE: the server publishes its max prefill ubatch (n_ubatch) into
  // DSV4_MOE_PREFILL_MAX at startup (tools/server/server.cpp), BEFORE any graph
  // capture. Since every prefill ubatch M <= n_ubatch, sizing the arena once to
  // n_ubatch*U rows means the grow guard below NEVER re-fires after capture ->
  // no malloc/free after capture -> a replayed prefill graph always sees the same
  // arena addresses -> graphs stay ON and capture-safe. The env is an OVERRIDE
  // here; the value originates from the actual server n_ubatch, not a user knob.
  static const int DSV4_PREFILL_MAX = []{ const char*e=getenv("DSV4_MOE_PREFILL_MAX"); return e?atoi(e):0; }();
  {
    int want_rows = rows; int cap_rows = std::max(DSV4_PREFILL_MAX*U, want_rows);
    // SFA word capacity: worst-case padded sum = rows_cap + E*128 padding rows.
    size_t pad_rows_cap = (size_t)cap_rows + (size_t)E*128;
    size_t sfx_words_cap = pad_rows_cap*kbD;
    size_t sfa_words_cap = pad_rows_cap*kbF;
    if (!L->pf_inited || L->pf_rows_cap < cap_rows ||
        L->pf_sfx_words_cap < sfx_words_cap || L->pf_sfa_words_cap < sfa_words_cap) {
      std::lock_guard<std::mutex> lk(g_reg_mu);           // serialize first-touch / grow
      if (!L->pf_inited || L->pf_rows_cap < cap_rows ||
          L->pf_sfx_words_cap < sfx_words_cap || L->pf_sfa_words_cap < sfa_words_cap) {
        // ROBUST GROW (CUDA-graph-safe): allocate the NEW arena into locals FIRST,
        // then swap the live pointers and RETIRE the old ones via dsv4_retire (freed
        // at the next device synchronize, never inline). A captured graph that still
        // references the old addresses replays validly until the stream drains, so
        // there is NO use-after-free even if a forced large batch grows post-capture.
        const int NPARTS = 512;                            // absmax reduction grid
        int *n_hist=nullptr,*n_cursor=nullptr,*n_indptr=nullptr,*n_sfa_off_D=nullptr,*n_sfa_off_F=nullptr;
        int *n_vals=nullptr,*n_grp_of_row=nullptr; float *n_gscaleX=nullptr,*n_gscaleAct=nullptr,*n_absmax_part=nullptr;
        ElementInput *n_Xs_full=nullptr,*n_Xs=nullptr; ElementSF *n_SFx_tight=nullptr;
        ElementD *n_gate=nullptr,*n_up=nullptr; ElementInput *n_act_full=nullptr,*n_act=nullptr;
        ElementSF *n_SFa_tight=nullptr; ElementD *n_down=nullptr; ElementSF *n_SFx=nullptr,*n_SFa=nullptr;
        DSV4_CK(cudaMalloc((void**)&n_hist,        (size_t)E*4));
        DSV4_CK(cudaMalloc((void**)&n_cursor,      (size_t)E*4));
        DSV4_CK(cudaMalloc((void**)&n_indptr,      (size_t)(E+1)*4));
        DSV4_CK(cudaMalloc((void**)&n_sfa_off_D,   (size_t)(E+1)*4));
        DSV4_CK(cudaMalloc((void**)&n_sfa_off_F,   (size_t)(E+1)*4));
        DSV4_CK(cudaMalloc((void**)&n_vals,        (size_t)cap_rows*4));
        DSV4_CK(cudaMalloc((void**)&n_grp_of_row,  (size_t)cap_rows*4));
        DSV4_CK(cudaMalloc((void**)&n_gscaleX,     sizeof(float)));
        DSV4_CK(cudaMalloc((void**)&n_gscaleAct,   sizeof(float)));
        DSV4_CK(cudaMalloc((void**)&n_absmax_part, (size_t)NPARTS*4));
        DSV4_CK(cudaMalloc((void**)&n_Xs_full,     (size_t)cap_rows*D));
        DSV4_CK(cudaMalloc((void**)&n_Xs,          (size_t)cap_rows*D/2));
        DSV4_CK(cudaMalloc((void**)&n_SFx_tight,   (size_t)cap_rows*kbD*sizeof(ElementSF)));
        DSV4_CK(cudaMalloc((void**)&n_gate,        (size_t)cap_rows*F*2));
        DSV4_CK(cudaMalloc((void**)&n_up,          (size_t)cap_rows*F*2));
        DSV4_CK(cudaMalloc((void**)&n_act_full,    (size_t)cap_rows*F));
        DSV4_CK(cudaMalloc((void**)&n_act,         (size_t)cap_rows*F/2));
        DSV4_CK(cudaMalloc((void**)&n_SFa_tight,   (size_t)cap_rows*kbF*sizeof(ElementSF)));
        DSV4_CK(cudaMalloc((void**)&n_down,        (size_t)cap_rows*D*2));
        DSV4_CK(cudaMalloc((void**)&n_SFx,         sfx_words_cap*sizeof(ElementSF)));
        DSV4_CK(cudaMalloc((void**)&n_SFa,         sfa_words_cap*sizeof(ElementSF)));
        // retire the OLD live arena (defer-free at next synchronize) BEFORE swapping
        dsv4_retire(L->pf_hist);dsv4_retire(L->pf_cursor);dsv4_retire(L->pf_indptr);
        dsv4_retire(L->pf_sfa_off_D);dsv4_retire(L->pf_sfa_off_F);
        dsv4_retire(L->pf_vals);dsv4_retire(L->pf_grp_of_row);dsv4_retire(L->pf_gscaleX);
        dsv4_retire(L->pf_gscaleAct);dsv4_retire(L->pf_absmax_part);
        dsv4_retire(L->pf_Xs_full);dsv4_retire(L->pf_Xs);dsv4_retire(L->pf_SFx_tight);
        dsv4_retire(L->pf_gate);dsv4_retire(L->pf_up);dsv4_retire(L->pf_act_full);
        dsv4_retire(L->pf_act);dsv4_retire(L->pf_SFa_tight);dsv4_retire(L->pf_down);
        dsv4_retire(L->pf_SFx);dsv4_retire(L->pf_SFa);
        // swap in the new arena
        L->pf_hist=n_hist; L->pf_cursor=n_cursor; L->pf_indptr=n_indptr;
        L->pf_sfa_off_D=n_sfa_off_D; L->pf_sfa_off_F=n_sfa_off_F;
        L->pf_vals=n_vals; L->pf_grp_of_row=n_grp_of_row; L->pf_gscaleX=n_gscaleX;
        L->pf_gscaleAct=n_gscaleAct; L->pf_absmax_part=n_absmax_part;
        L->pf_Xs_full=n_Xs_full; L->pf_Xs=n_Xs; L->pf_SFx_tight=n_SFx_tight;
        L->pf_gate=n_gate; L->pf_up=n_up; L->pf_act_full=n_act_full;
        L->pf_act=n_act; L->pf_SFa_tight=n_SFa_tight; L->pf_down=n_down;
        L->pf_SFx=n_SFx; L->pf_SFa=n_SFa;
        L->pf_absmax_nparts = NPARTS;
        L->pf_rows_cap = cap_rows;
        L->pf_sfx_words_cap = sfx_words_cap;
        L->pf_sfa_words_cap = sfa_words_cap;
        L->pf_inited = true;
      }
    }
  }
  int  *d_vals       = L->pf_vals;
  int  *d_indptr     = L->pf_indptr;
  int  *d_sfa_off_D  = L->pf_sfa_off_D, *d_sfa_off_F = L->pf_sfa_off_F;
  int  *d_grp_of_row = L->pf_grp_of_row;
  ElementInput *d_Xs_full = L->pf_Xs_full; ElementInput *d_Xs = L->pf_Xs;
  ElementSF *d_SFx_tight  = L->pf_SFx_tight;
  ElementD  *d_gate = L->pf_gate, *d_up = L->pf_up;
  ElementInput *d_act_full = L->pf_act_full, *d_act = L->pf_act;
  ElementSF *d_SFa_tight = L->pf_SFa_tight;
  ElementD  *d_down = L->pf_down;
  ElementSF *d_SFx = L->pf_SFx, *d_SFa = L->pf_SFa;

  // SFA tile-atom layout: built from the FIXED arena capacity (M-extent only changes
  // tile COUNT; the per-(row,col) atom mapping scatter_sfa uses is M-independent),
  // so this host value is STABLE across calls -> safe to bake into the captured graph.
  const int lsa_rows = std::max(L->pf_rows_cap,128);
  auto lsa_x=Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(make_shape(lsa_rows,F,D,1));
  auto lsa_a=Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(make_shape(lsa_rows,D,F,1));

  // ---- on-device routing: histogram -> indptr (+cursor) -> stable scatter ----
  DSV4_CK(cudaMemsetAsync(L->pf_hist,0,(size_t)E*4,s));
  { int nt=256,nb=(rows+nt-1)/nt; hist_expert<<<nb,nt,0,s>>>(d_sel,L->pf_hist,rows); }
  excl_scan_indptr<<<1,1,0,s>>>(L->pf_hist,d_indptr,L->pf_cursor,E);
  { int nt=256,nb=(rows+nt-1)/nt; counting_scatter<<<nb,nt,0,s>>>(d_sel,d_indptr,L->pf_cursor,d_vals,d_grp_of_row,rows,E); }
  // per-group SFA atom-aligned offsets (ceil(m/128)*128 * kb), on-device.
  compute_sfa_off<<<1,1,0,s>>>(d_indptr,d_sfa_off_D,E,kbD);
  compute_sfa_off<<<1,1,0,s>>>(d_indptr,d_sfa_off_F,E,kbF);

  // ---- SFA buffers: zero ONLY the words this call uses (upper bound from cap) ----
  // The needed word count is data-dependent (sfa_off[E]); we conservatively clear the
  // worst-case padded span for THIS rows so the GEMM never reads stale SF. This is a
  // memset on the stream -> capture-safe.
  { size_t pad_rows = (size_t)rows + (size_t)E*128;
    DSV4_CK(cudaMemsetAsync(d_SFx,0,pad_rows*kbD*sizeof(ElementSF),s));
    DSV4_CK(cudaMemsetAsync(d_SFa,0,pad_rows*kbF*sizeof(ElementSF),s)); }

  // ---- on-device hidden absmax -> gscaleX (device float) ----
  { int nparts = L->pf_absmax_nparts; int nt=256;
    absmax_partial<<<nparts,nt,nt*4,s>>>(d_hidden,L->pf_absmax_part,(int64_t)M*D);
    absmax_finalize<<<1,256,256*4,s>>>(L->pf_absmax_part,nparts,L->pf_gscaleX); }

  DSV4_CK(cudaMemsetAsync(d_out,0,(size_t)M*D*4,s));
  gather_quant<<<rows,64,64*4,s>>>(d_hidden,d_vals,d_Xs_full,d_SFx_tight,rows,D,U,L->pf_gscaleX);
  { int64_t ne=(int64_t)rows*D; int nt=256; int64_t nb=((ne/2)+nt-1)/nt;
    pack_e2m1<<<nb,nt,0,s>>>(d_Xs_full,reinterpret_cast<uint8_t*>(d_Xs),ne); }
  scatter_sfa<<<((size_t)rows*kbD+255)/256,256,0,s>>>(d_SFx_tight,d_SFx,rows,kbD,lsa_x,d_grp_of_row,d_indptr,d_sfa_off_D);

  L->gg_gate.run(d_Xs,L->dq_gate,d_SFx,L->dsf_gate,d_gate,d_indptr,d_sfa_off_D,L->dglobal_gate,L->pf_gscaleX,s);
  L->gg_up.run  (d_Xs,L->dq_up,  d_SFx,L->dsf_up,  d_up,  d_indptr,d_sfa_off_D,L->dglobal_up,  L->pf_gscaleX,s);

  // ---- on-device act absmax (over CLAMPED swiglu) -> gscaleAct (device float) ----
  { int nparts = L->pf_absmax_nparts; int nt=256;
    swiglu_absmax_partial<<<nparts,nt,nt*4,s>>>(d_gate,d_up,L->pf_absmax_part,(int64_t)rows*F,swiglu_limit);
    absmax_finalize<<<1,256,256*4,s>>>(L->pf_absmax_part,nparts,L->pf_gscaleAct); }

  swiglu_quant<<<rows,64,64*4,s>>>(d_gate,d_up,d_act_full,d_SFa_tight,rows,F,L->pf_gscaleAct,swiglu_limit);
  { int64_t ne=(int64_t)rows*F; int nt=256; int64_t nb=((ne/2)+nt-1)/nt;
    pack_e2m1<<<nb,nt,0,s>>>(d_act_full,reinterpret_cast<uint8_t*>(d_act),ne); }
  scatter_sfa<<<((size_t)rows*kbF+255)/256,256,0,s>>>(d_SFa_tight,d_SFa,rows,kbF,lsa_a,d_grp_of_row,d_indptr,d_sfa_off_F);

  L->gg_down.run(d_act,L->dq_down,d_SFa,L->dsf_down,d_down,d_indptr,d_sfa_off_F,L->dglobal_down,L->pf_gscaleAct,s);
  scatter_add<<<rows,256,0,s>>>(d_down,d_vals,d_rw,d_out,rows,D,U);

  // NO sync, NO free: result feeds the graph; ALL scratch is persistent.
  return true;
}

bool ggml_cuda_op_dsv4_moe_grouped_supported(void){
  return true;
}
