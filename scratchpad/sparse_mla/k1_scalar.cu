// Stage 1: correctness baseline. One CTA per query token. Plain indexed loads
// (no TMA), scalar QK^T + online softmax + PV (no MMA). Validates the reference,
// data layout, idx addressing, and online-softmax math before the hard parts.
#include "ref.h"

__device__ __forceinline__ float bf16_to_f32_dev(bf16_t b){ uint32_t u=((uint32_t)b)<<16; float f; memcpy(&f,&u,4); return f; }

// 1 CTA, blockDim = HQ*? -> use 256 threads. Each thread owns a subset of (head,d) for output.
// Simpler: grid=1 CTA per query; we test a SINGLE query here (the per-query CTA).
// smem: gathered K [TOPK*D] is 512*512*2 = 512KB -> too big. So stream K row-by-row from global.
// For the scalar baseline we don't stage K in smem; we recompute per head. This is just a numeric ref.
__global__ void k1(const float* __restrict__ Q, const bf16_t* __restrict__ K,
                   const int* __restrict__ idx, float scale, float* __restrict__ O){
    // one block. threads cooperate over heads. Each warp handles a few heads.
    int t = threadIdx.x; int nt = blockDim.x;
    __shared__ float s_score[TOPK];
    for(int h=0; h<HQ; h++){
        const float* q = Q + (size_t)h*D;
        // QK^T: each thread does a subset of the TOPK keys
        for(int j=t; j<TOPK; j+=nt){
            const bf16_t* k = K + (size_t)idx[j]*D;
            float acc=0; for(int d=0; d<D; d++) acc += q[d]*bf16_to_f32_dev(k[d]);
            s_score[j]=acc*scale;
        }
        __syncthreads();
        // softmax (thread 0 reduces — fine for the baseline)
        __shared__ float s_mx, s_sum;
        if(t==0){ float mx=-INFINITY; for(int j=0;j<TOPK;j++) if(s_score[j]>mx)mx=s_score[j];
            float sum=0; for(int j=0;j<TOPK;j++){ s_score[j]=expf(s_score[j]-mx); sum+=s_score[j]; }
            s_mx=mx; s_sum=sum; }
        __syncthreads();
        float inv=1.0f/s_sum;
        // PV: each thread owns output dims d = t, t+nt, ...
        for(int d=t; d<D; d+=nt){
            float acc=0; for(int j=0;j<TOPK;j++){ const bf16_t* v=K+(size_t)idx[j]*D; acc += s_score[j]*bf16_to_f32_dev(v[d]); }
            O[(size_t)h*D+d]=acc*inv;
        }
        __syncthreads();
    }
}

int main(){
    cudaSetDevice(0); cudaDeviceProp pr; cudaGetDeviceProperties(&pr,0);
    printf("GPU %s sm_%d%d  D=%d HQ=%d TOPK=%d\n",pr.name,pr.major,pr.minor,D,HQ,TOPK);
    Problem p = make_problem(2048, 1234);
    auto ref = ref_attn(p);
    float *dQ,*dO; bf16_t* dK; int* dI;
    cudaMalloc(&dQ,p.Q.size()*4); cudaMalloc(&dO,(size_t)HQ*D*4);
    cudaMalloc(&dK,p.Kbf.size()*2); cudaMalloc(&dI,TOPK*4);
    cudaMemcpy(dQ,p.Q.data(),p.Q.size()*4,cudaMemcpyHostToDevice);
    cudaMemcpy(dK,p.Kbf.data(),p.Kbf.size()*2,cudaMemcpyHostToDevice);
    cudaMemcpy(dI,p.idx.data(),TOPK*4,cudaMemcpyHostToDevice);
    k1<<<1,256>>>(dQ,dK,dI,p.scale,dO);
    cudaError_t e=cudaDeviceSynchronize();
    printf("sync: %s\n",cudaGetErrorString(e)); if(e)return 1;
    std::vector<float> out((size_t)HQ*D); cudaMemcpy(out.data(),dO,out.size()*4,cudaMemcpyDeviceToHost);
    printf("cos=%.6f  rel_err=%.3e\n", cosine(out,ref), rel_err(out,ref));
    printf("ref[0..3]=%.4f %.4f %.4f %.4f\n",ref[0],ref[1],ref[2],ref[3]);
    printf("out[0..3]=%.4f %.4f %.4f %.4f\n",out[0],out[1],out[2],out[3]);
    return 0;
}
