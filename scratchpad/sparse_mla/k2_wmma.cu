// Stage 2: WMMA tensor-core sparse-MLA. One CTA per query token.
// QK^T and PV as bf16 WMMA 16x16x16, flash online-softmax over key-blocks.
// K gathered into smem by index (plain indexed load; TMA comes in stage 3). MLA: V==K.
// Shape: M=HQ=64 heads, D=512 head_dim, TOPK=512 selected keys.
//
// smem budget (GB10 sm_121 max dynamic ~227KB):
//   sQ  [HQ x D] bf16   = 64KB   (persistent across blocks)
//   sK  [KB x D] bf16   = 64KB   (current key block)
//   sP  [HQ x KB] bf16  = 8KB
//   sS  [HQ x KB] f32   = 16KB
//   sO  : NOT in smem (too big). O kept in registers? -> 64x512 too large.
// => Resolve by keeping O in smem but reusing sQ's space is wrong (Q needed each block).
//    Instead: O[HQ x D] f32 lives in smem; we DROP the f32 sS (reuse sP region as f32 via union of size).
//    Budget: sQ 64 + sK 64 + sO(f32 128)=256 > 227. Still over.
//    FINAL: split D into two halves is unnecessary — instead keep O in GLOBAL and accumulate there
//    across blocks (each CTA owns one query's O row; no cross-CTA race). Only Q,K,P,S in smem.
//      sQ 64 + sK 64 + sP 8 + sS 16 = 152KB. OK.
//    O accumulation in global needs rescale-in-place across blocks -> fine, single CTA owns it.
#include "ref.h"
#include <mma.h>
using namespace nvcuda;

#define KB 16     // keys per flash block. GB10 dynamic-smem cap = 99KB:
                  //   sQ 64KB + sK(KB*D*2) + sP(HQ*KB*2) + sS(HQ*KB*4).
                  //   KB=16 -> 64+16+2+4 = 86KB. KB=32 -> 64+32+4+8 = 108KB (over). So KB<=16.
#define WARPS 8
#define NWMMA_M (HQ/16)   // 4
#define NWMMA_N (KB/16)   // 4
#define NWMMA_D (D/16)    // 32

__device__ __forceinline__ __nv_bfloat16 bf16raw(bf16_t b){ __nv_bfloat16 o; memcpy(&o,&b,2); return o; }

extern __shared__ char smem[];

__global__ void __launch_bounds__(256) k2(
        const float* __restrict__ Q, const bf16_t* __restrict__ K,
        const int* __restrict__ idx, float scale, float* __restrict__ O){
    const int t = threadIdx.x, warp = t>>5, lane = t&31;
    __nv_bfloat16* sQ = (__nv_bfloat16*)smem;            // [HQ*D]
    __nv_bfloat16* sK = sQ + HQ*D;                       // [KB*D]
    __nv_bfloat16* sP = sK + KB*D;                       // [HQ*KB]
    float*         sS = (float*)(sP + HQ*KB);            // [HQ*KB]
    __shared__ float m_run[HQ], l_run[HQ];
    __shared__ float wscratch[WARPS][16*16];

    for(int i=t; i<HQ*D; i+=256) sQ[i] = __float2bfloat16(Q[i]);
    for(int i=t; i<HQ*D; i+=256) O[i]=0.f;               // O accumulated in global
    if(t<HQ){ m_run[t]=-INFINITY; l_run[t]=0.f; }
    __syncthreads();

    for(int kb0=0; kb0<TOPK; kb0+=KB){
        // 1) gather Kblk into smem
        for(int r=warp; r<KB; r+=WARPS){
            const bf16_t* krow = K + (size_t)idx[kb0+r]*D;
            for(int d=lane; d<D; d+=32) sK[r*D+d] = bf16raw(krow[d]);
        }
        __syncthreads();

        // 2) S = Q @ K^T  (WMMA over D)
        for(int tile = warp; tile < NWMMA_M*NWMMA_N; tile += WARPS){
            int mt = tile / NWMMA_N, nt = tile % NWMMA_N;
            wmma::fragment<wmma::accumulator,16,16,16,float> acc; wmma::fill_fragment(acc,0.f);
            for(int dt=0; dt<NWMMA_D; dt++){
                wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major> a;
                wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::col_major> b;
                wmma::load_matrix_sync(a, sQ + (mt*16)*D + dt*16, D);
                wmma::load_matrix_sync(b, sK + (nt*16)*D + dt*16, D);
                wmma::mma_sync(acc,a,b,acc);
            }
            wmma::store_matrix_sync(sS + (mt*16)*KB + nt*16, acc, KB, wmma::mem_row_major);
        }
        __syncthreads();

        // 3) online softmax + build P + rescale O(global)
        for(int h=warp; h<HQ; h+=WARPS){
            float* srow = sS + h*KB;
            float bmx=-INFINITY; for(int j=lane;j<KB;j+=32){ float v=srow[j]*scale; if(v>bmx)bmx=v; }
            for(int o=16;o>0;o>>=1) bmx=fmaxf(bmx,__shfl_xor_sync(0xffffffff,bmx,o));
            float m_old=m_run[h]; float m_new=fmaxf(m_old,bmx);
            float corr = (m_old==-INFINITY)?0.f:expf(m_old - m_new);
            float bsum=0; for(int j=lane;j<KB;j+=32){ float p=expf(srow[j]*scale - m_new); sP[h*KB+j]=__float2bfloat16(p); bsum+=p; }
            for(int o=16;o>0;o>>=1) bsum+=__shfl_xor_sync(0xffffffff,bsum,o);
            if(lane==0){ m_run[h]=m_new; l_run[h]=l_run[h]*corr + bsum; }
            for(int d=lane; d<D; d+=32) O[h*D+d]*=corr;
        }
        __syncthreads();

        // 4) O += P @ V  (V==K)
        for(int tile = warp; tile < NWMMA_M*NWMMA_D; tile += WARPS){
            int mt = tile / NWMMA_D, dt = tile % NWMMA_D;
            wmma::fragment<wmma::accumulator,16,16,16,float> acc; wmma::fill_fragment(acc,0.f);
            for(int kt=0; kt<NWMMA_N; kt++){
                wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major> a;
                wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::row_major> b;
                wmma::load_matrix_sync(a, sP + (mt*16)*KB + kt*16, KB);
                wmma::load_matrix_sync(b, sK + (kt*16)*D + dt*16, D);
                wmma::mma_sync(acc,a,b,acc);
            }
            wmma::store_matrix_sync(wscratch[warp], acc, 16, wmma::mem_row_major);
            for(int e=lane; e<256; e+=32){ int rr=e/16, cc=e%16; O[(mt*16+rr)*D + dt*16 + cc] += wscratch[warp][e]; }
        }
        __syncthreads();
    }

    for(int h=warp; h<HQ; h+=WARPS){ float inv=1.0f/l_run[h];
        for(int d=lane; d<D; d+=32) O[h*D+d]*=inv; }
}

int main(){
    cudaSetDevice(0); cudaDeviceProp pr; cudaGetDeviceProperties(&pr,0);
    printf("GPU %s sm_%d%d  D=%d HQ=%d TOPK=%d KB=%d\n",pr.name,pr.major,pr.minor,D,HQ,TOPK,KB);
    Problem p = make_problem(2048, 1234);
    auto ref = ref_attn(p);
    float *dQ,*dO; bf16_t* dK; int* dI;
    cudaMalloc(&dQ,p.Q.size()*4); cudaMalloc(&dO,(size_t)HQ*D*4);
    cudaMalloc(&dK,p.Kbf.size()*2); cudaMalloc(&dI,TOPK*4);
    cudaMemcpy(dQ,p.Q.data(),p.Q.size()*4,cudaMemcpyHostToDevice);
    cudaMemcpy(dK,p.Kbf.data(),p.Kbf.size()*2,cudaMemcpyHostToDevice);
    cudaMemcpy(dI,p.idx.data(),TOPK*4,cudaMemcpyHostToDevice);

    size_t smem = (size_t)HQ*D*2 + KB*D*2 + HQ*KB*2 + HQ*KB*4;
    printf("requested dynamic smem = %zu bytes (%.1f KB)\n", smem, smem/1024.0);
    cudaError_t se=cudaFuncSetAttribute(k2, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    printf("setattr=%s\n",cudaGetErrorString(se));
    k2<<<1,256,smem>>>(dQ,dK,dI,p.scale,dO);
    printf("launch=%s\n",cudaGetErrorString(cudaGetLastError()));
    cudaError_t e=cudaDeviceSynchronize();
    printf("sync: %s\n",cudaGetErrorString(e)); if(e)return 1;
    std::vector<float> out((size_t)HQ*D); cudaMemcpy(out.data(),dO,out.size()*4,cudaMemcpyDeviceToHost);
    printf("cos=%.6f  rel_err=%.3e\n", cosine(out,ref), rel_err(out,ref));
    printf("ref[0..3]=%.4f %.4f %.4f %.4f\n",ref[0],ref[1],ref[2],ref[3]);
    printf("out[0..3]=%.4f %.4f %.4f %.4f\n",out[0],out[1],out[2],out[3]);
    return 0;
}
