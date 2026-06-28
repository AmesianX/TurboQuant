// Standalone repro for the DSV4 fused MoE multi-chunk prefill OOB.
// Mirrors dsv4-moe-fused-run.cu's build_fused_layer buffer sizing AND the per-call
// runMoe path: sizes the shared workspace ONCE to tok_cap=1024 (as our op does),
// then drives runMoe over N sequential M=1024 chunks with IMBALANCED routing.
// Run under: compute-sanitizer --tool memcheck ./fused_moe_repro
//
// We allocate ZEROED fp4 weights + swizzled SF at the SAME sizes our build_fused_layer
// uses; values don't matter for a memcheck bounds run, only buffer extents do.

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

int main(int argc, char** argv){
    // DSV4 config
    const int E        = (argc>1)?atoi(argv[1]):256;   // experts
    const int U        = 6;                              // experts_per_token (top_k)
    const int hidden   = 4096;                           // n_embd (DSV4-Flash, from gguf)
    const int inter    = 2048;                           // n_ff_exp (per-expert, from gguf)
    const int UB       = 1024;                           // ubatch / tok_cap
    const int NCHUNK   = (argc>2)?atoi(argv[2]):10;
    const long tok_cap = UB;

    fprintf(stderr,"[repro] E=%d U=%d hidden=%d inter=%d UB=%d chunks=%d\n",E,U,hidden,inter,UB,NCHUNK);

    cudaStream_t stream; CK(cudaStreamCreate(&stream));

    const int hbytes = hidden/2;          // fp4 packs 2 nibbles/byte
    const int colsD  = hidden/16;         // fc1 K-block SF cols
    const int colsF  = inter/16;          // fc2 K-block SF cols

    // ---- weights (zeroed, exact build_fused_layer sizes) ----
    uint8_t *fc1_w=nullptr,*fc2_w=nullptr,*fc1_sf=nullptr,*fc2_sf=nullptr;
    CK(cudaMalloc(&fc1_w,(size_t)E*2*inter*hbytes)); CK(cudaMemset(fc1_w,0,(size_t)E*2*inter*hbytes));
    CK(cudaMalloc(&fc2_w,(size_t)E*hidden*(inter/2))); CK(cudaMemset(fc2_w,0,(size_t)E*hidden*(inter/2)));
    const int padN_fc1=padUp(2*inter,128), padC_fc1=padUp(colsD,4);
    CK(cudaMalloc(&fc1_sf,(size_t)E*padN_fc1*padC_fc1)); CK(cudaMemset(fc1_sf,0,(size_t)E*padN_fc1*padC_fc1));
    const int padN_fc2=padUp(hidden,128), padC_fc2=padUp(colsF,4);
    CK(cudaMalloc(&fc2_sf,(size_t)E*padN_fc2*padC_fc2)); CK(cudaMemset(fc2_sf,0,(size_t)E*padN_fc2*padC_fc2));

    // per-expert globals/alphas/limits
    float *g_common=nullptr,*g_down=nullptr,*fc2_act_global=nullptr,*swiglu_limit=nullptr;
    float *fc1_alpha=nullptr,*fc2_alpha=nullptr,*fc1_act_global=nullptr,*act_part=nullptr;
    CK(cudaMalloc(&g_common,(size_t)E*4)); CK(cudaMalloc(&g_down,(size_t)E*4));
    CK(cudaMalloc(&fc1_alpha,(size_t)E*4)); CK(cudaMalloc(&fc2_alpha,(size_t)E*4));
    CK(cudaMalloc(&swiglu_limit,(size_t)E*4));
    CK(cudaMalloc(&fc2_act_global,4)); CK(cudaMalloc(&fc1_act_global,4));
    const int act_nparts=256; CK(cudaMalloc(&act_part,(size_t)act_nparts*4));
    { std::vector<float> ones(E,1.0f); CK(cudaMemcpy(g_common,ones.data(),E*4,cudaMemcpyHostToDevice));
      CK(cudaMemcpy(g_down,ones.data(),E*4,cudaMemcpyHostToDevice));
      std::vector<float> lim(E,8.0f); CK(cudaMemcpy(swiglu_limit,lim.data(),E*4,cudaMemcpyHostToDevice));
      float one=1.0f; CK(cudaMemcpy(fc2_act_global,&one,4,cudaMemcpyHostToDevice));
      CK(cudaMemcpy(fc1_act_global,&one,4,cudaMemcpyHostToDevice));
      std::vector<float> al(E,0.001f); CK(cudaMemcpy(fc1_alpha,al.data(),E*4,cudaMemcpyHostToDevice));
      CK(cudaMemcpy(fc2_alpha,al.data(),E*4,cudaMemcpyHostToDevice)); }

    // activations + outputs (bf16), sized to tok_cap (as our scratch does)
    __nv_bfloat16 *d_hidden=nullptr,*d_out=nullptr;
    CK(cudaMalloc(&d_hidden,(size_t)tok_cap*hidden*sizeof(__nv_bfloat16)));
    CK(cudaMalloc(&d_out,(size_t)tok_cap*hidden*sizeof(__nv_bfloat16)));
    CK(cudaMemset(d_hidden,0,(size_t)tok_cap*hidden*sizeof(__nv_bfloat16)));

    // routing buffers sized to tok_cap
    int   *d_sel=nullptr,*d_src2dst=nullptr; float* d_weights=nullptr;
    CK(cudaMalloc(&d_sel,(size_t)U*tok_cap*sizeof(int)));
    CK(cudaMalloc(&d_weights,(size_t)U*tok_cap*sizeof(float)));
    CK(cudaMalloc(&d_src2dst,(size_t)U*tok_cap*sizeof(int)));
    { std::vector<float> w(U*tok_cap,1.0f/U); CK(cudaMemcpy(d_weights,w.data(),U*tok_cap*sizeof(float),cudaMemcpyHostToDevice)); }

    // ---- runner + workspace sized ONCE to tok_cap (mirrors our op) ----
    auto runner = std::make_unique<tkc::CutlassMoeFCRunner<__nv_fp4_e2m1,__nv_fp4_e2m1,__nv_bfloat16,__nv_bfloat16>>();
    auto tactics = runner->getTactics();
    if (tactics.empty()){ fprintf(stderr,"no tactics\n"); return 1; }
    int ti = (argc>3)?atoi(argv[3]):0; if(ti<0||ti>=(int)tactics.size())ti=0;
    runner->setTactic(tactics[ti],tactics[ti]);
    fprintf(stderr,"[repro] tactic[%d]=%s\n",ti,tactics[ti].toString().c_str());

    tkc::MOEParallelismConfig pc(1,0,1,0);
    // arg5: pf_max (0 = production high-water path, i.e. DSV4_MOE_PREFILL_MAX unset)
    const int pf_max = (argc>5)?atoi(argv[5]):UB;
    char* d_workspace=nullptr; size_t workspace_cap=0; long tok_cap_hw=0;

    auto qp = tkc::QuantParams::FP4(
        fc1_act_global,
        reinterpret_cast<tkc::TmaWarpSpecializedGroupedGemmInput::NVFP4ElementSF const*>(fc1_sf),
        fc1_alpha,
        fc2_act_global,
        reinterpret_cast<tkc::TmaWarpSpecializedGroupedGemmInput::NVFP4ElementSF const*>(fc2_sf),
        fc2_alpha, false,false);
    tkc::ActivationParams act(tkc::ActivationType::SwigluBias, nullptr, nullptr, swiglu_limit);
    tkc::MoeMinLatencyParams mlp{};
    tk::LoraParams lora{};

    // optional: vary M per chunk to mirror real prefill (first chunk small, partial last, etc.)
    // arg4: 1 => first chunk M=17 (warmup-like) then 1024s ; 2 => ramp 128,256,...,1024 ; else all 1024
    const int mmode = (argc>4)?atoi(argv[4]):0;

    std::mt19937 rng(1234);
    for (int c=0;c<NCHUNK;c++){
        int M = UB; // full chunk
        if (mmode==1) M = (c==0)? 17 : UB;
        else if (mmode==2) M = ((c%8)+1)*128; // 128..1024 cycling
        else if (mmode==3) M = (c==NCHUNK-1)? 533 : UB; // partial last chunk
        else if (mmode==5){ std::uniform_int_distribution<int> mu(256,1024); M = mu(rng); } // random M each chunk
        // IMBALANCED routing: each chunk picks a different "hot" expert that gets a big share.
        std::vector<int> sel(U*M);
        int hot = (c*37) % E;                     // rotates each chunk -> different distribution
        std::uniform_int_distribution<int> uni(0,E-1);
        std::bernoulli_distribution hotpick(0.6); // 60% of slots -> the hot expert
        for (int t=0;t<M;t++) for(int u=0;u<U;u++){
            int e = hotpick(rng)? hot : uni(rng);
            sel[(long)t*U+u]=e;   // [n_tokens, n_expert_used] row-major == [U,n_tokens] ggml col-major
        }
        // extra-skew chunk to push one expert way past the average:
        if (c==4 || c==7){ for (int t=0;t<M;t++) for(int u=0;u<U;u++) sel[(long)t*U+u]=hot; }
        CK(cudaMemcpy(d_sel,sel.data(),(size_t)U*M*sizeof(int),cudaMemcpyHostToDevice));

        // ==== EXACT production tok_cap + workspace grow (dsv4-moe-fused-run.cu:341-380) ====
        long chunk_tok_cap = pf_max > M ? pf_max : M;
        if (chunk_tok_cap < tok_cap_hw) chunk_tok_cap = tok_cap_hw;
        tok_cap_hw = chunk_tok_cap;
        size_t need = runner->getWorkspaceSize((int)chunk_tok_cap, hidden, inter, E, U,
            tkc::ActivationType::Swiglu, pc, false,false,false,false);
        if (workspace_cap < need){
            if (d_workspace) cudaFree(d_workspace);  // eager: free+realloc (production retires)
            CK(cudaMalloc(&d_workspace, need)); workspace_cap=need;
            fprintf(stderr,"[repro]   workspace grow -> tok_cap=%ld %.2f MiB\n",chunk_tok_cap,need/1048576.0);
        }

        runner->runMoe(
            d_hidden, nullptr, d_sel, d_weights,
            fc1_w, nullptr, act, fc2_w, nullptr, qp,
            M, hidden, inter, E, U,
            d_workspace, d_out, d_src2dst,
            pc, false,false, lora, false,false, mlp, false, stream);
        cudaError_t e = cudaStreamSynchronize(stream);
        fprintf(stderr,"[repro] chunk %d M=%d hot=%d : %s\n",c,M,hot,cudaGetErrorString(e));
        if (e!=cudaSuccess){ fprintf(stderr,"[repro] CRASH at chunk %d\n",c); return 2; }
    }
    fprintf(stderr,"[repro] ALL %d chunks OK\n",NCHUNK);
    return 0;
}
