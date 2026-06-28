// Timing harness: measure runMoe TFLOP/s per tactic at the DSV4 model config.
// Based on dsv4_fused_moe_repro.cu. arg: [E] [iters] [tactic] [M]
#include "cutlass_fused_moe_kernels.cuh"
#include "moe_kernels.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <memory>
namespace tk  = tensorrt_llm::kernels;
namespace tkc = tensorrt_llm::kernels::cutlass_kernels;
#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
  fprintf(stderr,"CUDA err %s @ %s:%d\n",cudaGetErrorString(e),__FILE__,__LINE__); exit(1);} }while(0)
static int padUp(int x,int a){ return (x+a-1)/a*a; }

int main(int argc,char**argv){
    const int E=(argc>1)?atoi(argv[1]):256;
    const int iters=(argc>2)?atoi(argv[2]):50;
    const int tac=(argc>3)?atoi(argv[3]):0;
    const int M=(argc>4)?atoi(argv[4]):1024;
    const int U=6, hidden=4096, inter=2048;
    cudaStream_t stream; CK(cudaStreamCreate(&stream));
    const int hbytes=hidden/2, colsD=hidden/16, colsF=inter/16;
    uint8_t *fc1_w,*fc2_w,*fc1_sf,*fc2_sf;
    CK(cudaMalloc(&fc1_w,(size_t)E*2*inter*hbytes)); CK(cudaMemset(fc1_w,0,(size_t)E*2*inter*hbytes));
    CK(cudaMalloc(&fc2_w,(size_t)E*hidden*(inter/2))); CK(cudaMemset(fc2_w,0,(size_t)E*hidden*(inter/2)));
    int padN1=padUp(2*inter,128),padC1=padUp(colsD,4);
    CK(cudaMalloc(&fc1_sf,(size_t)E*padN1*padC1)); CK(cudaMemset(fc1_sf,0,(size_t)E*padN1*padC1));
    int padN2=padUp(hidden,128),padC2=padUp(colsF,4);
    CK(cudaMalloc(&fc2_sf,(size_t)E*padN2*padC2)); CK(cudaMemset(fc2_sf,0,(size_t)E*padN2*padC2));
    float *g_common,*g_down,*fc2ag,*sl,*a1,*a2,*f1ag;
    CK(cudaMalloc(&g_common,E*4));CK(cudaMalloc(&g_down,E*4));CK(cudaMalloc(&a1,E*4));CK(cudaMalloc(&a2,E*4));
    CK(cudaMalloc(&fc2ag,4));CK(cudaMalloc(&sl,E*4));CK(cudaMalloc(&f1ag,4));
    CK(cudaMemset(g_common,0,E*4));CK(cudaMemset(g_down,0,E*4));CK(cudaMemset(a1,0,E*4));CK(cudaMemset(a2,0,E*4));
    CK(cudaMemset(fc2ag,0,4));CK(cudaMemset(sl,0,E*4));CK(cudaMemset(f1ag,0,4));
    __nv_bfloat16 *d_hidden,*d_out; CK(cudaMalloc(&d_hidden,(size_t)M*hidden*2));CK(cudaMemset(d_hidden,0,(size_t)M*hidden*2));
    CK(cudaMalloc(&d_out,(size_t)M*hidden*2));
    int *d_sel,*d_src2dst; float*d_weights;
    CK(cudaMalloc(&d_sel,(size_t)U*M*4));CK(cudaMalloc(&d_weights,(size_t)U*M*4));CK(cudaMemset(d_weights,0,(size_t)U*M*4));
    CK(cudaMalloc(&d_src2dst,(size_t)U*M*4));
    std::vector<int> sel(U*M); std::mt19937 rng(7);
    std::uniform_int_distribution<int> uni(0,E-1);
    for(size_t i=0;i<sel.size();++i) sel[i]=uni(rng);
    CK(cudaMemcpy(d_sel,sel.data(),(size_t)U*M*4,cudaMemcpyHostToDevice));

    auto runner=std::make_unique<tkc::CutlassMoeFCRunner<__nv_fp4_e2m1,__nv_fp4_e2m1,__nv_bfloat16,__nv_bfloat16>>();
    auto tactics=runner->getTactics();
    int ti=tac; if(ti<0||ti>=(int)tactics.size())ti=0;
    runner->setTactic(tactics[ti],tactics[ti]);
    fprintf(stderr,"[bench] tactic[%d] tileID=%d M=%d E=%d\n",ti,tactics[ti].getTileConfigAsInt(),M,E);
    tkc::MOEParallelismConfig pc(1,0,1,0);
    auto qp=tkc::QuantParams::FP4(f1ag,
        reinterpret_cast<tkc::TmaWarpSpecializedGroupedGemmInput::NVFP4ElementSF const*>(fc1_sf),a1,
        fc2ag,reinterpret_cast<tkc::TmaWarpSpecializedGroupedGemmInput::NVFP4ElementSF const*>(fc2_sf),a2,false,false);
    tkc::ActivationParams act(tkc::ActivationType::SwigluBias,nullptr,nullptr,sl);
    tkc::MoeMinLatencyParams mlp{}; tk::LoraParams lora{};
    size_t ws=runner->getWorkspaceSize(M,hidden,inter,E,U,tkc::ActivationType::Swiglu,pc,false,false,false,false);
    char* d_ws; CK(cudaMalloc(&d_ws,ws));
    auto run=[&](){ runner->runMoe(d_hidden,nullptr,d_sel,d_weights,fc1_w,nullptr,act,fc2_w,nullptr,qp,
        M,hidden,inter,E,U,d_ws,d_out,d_src2dst,pc,false,false,lora,false,false,mlp,false,stream); };
    // warmup
    for(int i=0;i<5;i++) run();
    CK(cudaStreamSynchronize(stream));
    cudaEvent_t e0,e1; cudaEventCreate(&e0);cudaEventCreate(&e1);
    cudaEventRecord(e0,stream);
    for(int i=0;i<iters;i++) run();
    cudaEventRecord(e1,stream); CK(cudaStreamSynchronize(stream));
    float ms=0; cudaEventElapsedTime(&ms,e0,e1);
    double per=ms/iters;
    // FLOPs: 2 GEMMs. fc1: M*top? No — grouped: total rows = M*U tokens routed.
    // fc1: (M*U) x (2*inter) x hidden ; fc2: (M*U) x hidden x inter. *2 for MAC.
    double rows=(double)M*U;
    double flop=2.0*rows*(2.0*inter)*hidden + 2.0*rows*hidden*inter;
    double tflops=flop/(per*1e-3)/1e12;
    fprintf(stderr,"[bench] tactic %d: %.3f ms/call  %.1f eff TFLOP/s (rows=%.0f)\n",ti,per,tflops,rows);
    return 0;
}
