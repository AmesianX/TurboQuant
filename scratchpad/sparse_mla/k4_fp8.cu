// Stage 3b: FP8-E4M3-B128 latent dequant in the TMA-gather tile loader.
// The DSV4 compressed-KV latent is stored FP8-E4M3 with a per-(row, 128-col-block) scale.
// TMA gather4 pulls the FP8 rows (1 byte/elem); the staging copy dequants FP8->bf16 using
// the block scale, landing bf16 into sK for the WMMA QK^T/PV (identical to stage 3a).
// FP8 row = D=512 bytes; TMA box width cap 256 elems => 2 col-tiles of 256 bytes each.
#include "ref.h"
#include <mma.h>
#include <cudaTypedefs.h>
#include <cuda_fp8.h>
using namespace nvcuda;

#define KB 16
#define WARPS 8
#define NWMMA_M (HQ/16)
#define NWMMA_N (KB/16)
#define NWMMA_D (D/16)
#define TMA_W 256              // 256 FP8 elems/tile (256 bytes). D=512 => NTILE=2.
#define NTILE (D/TMA_W)

extern __shared__ char smem[];

__device__ __forceinline__ void cp_gather4(uint32_t dst_s, const CUtensorMap* tm,
        int c0, int r0,int r1,int r2,int r3, uint32_t bar_s){
    asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes "
        "[%0],[%1,{%2,%3,%4,%5,%6}],[%7];"
        ::"r"(dst_s),"l"(tm),"r"(c0),"r"(r0),"r"(r1),"r"(r2),"r"(r3),"r"(bar_s):"memory");
}
__device__ __forceinline__ float fp8e4m3_to_f32(uint8_t v){ __nv_fp8_e4m3 f; memcpy(&f,&v,1); return (float)f; }

__global__ void __launch_bounds__(256) k4(
        const __grid_constant__ CUtensorMap tmK,    // [n_comp x D] FP8 (uint8), 256-wide box, gather4
        const float* __restrict__ Q, const int* __restrict__ idx,
        const float* __restrict__ Kscale,           // [n_comp x NBLK]
        float scale, float* __restrict__ O){
    const int t=threadIdx.x, warp=t>>5, lane=t&31;
    __nv_bfloat16* sQ=(__nv_bfloat16*)smem;
    __nv_bfloat16* sK=sQ+HQ*D;
    __nv_bfloat16* sP=sK+KB*D;
    float*         sS=(float*)(sP+HQ*KB);
    __shared__ float m_run[HQ], l_run[HQ];
    __shared__ float wscratch[WARPS][256];
    __shared__ alignas(128) uint8_t sStage[4*TMA_W];      // FP8 staging (256B/row x 4 rows)
    __shared__ alignas(8) uint64_t bar;

    for(int i=t;i<HQ*D;i+=256) sQ[i]=__float2bfloat16(Q[i]);
    for(int i=t;i<HQ*D;i+=256) O[i]=0.f;
    if(t<HQ){ m_run[t]=-INFINITY; l_run[t]=0.f; }
    uint32_t bar_s=(uint32_t)__cvta_generic_to_shared(&bar);
    __syncthreads();

    for(int kb0=0; kb0<TOPK; kb0+=KB){
        for(int g=0; g<KB/4; g++){
            for(int ct=0; ct<NTILE; ct++){
                if(t==0){
                    asm volatile("mbarrier.init.shared.b64 [%0],1;"::"r"(bar_s));
                    asm volatile("fence.proxy.async.shared::cta;");
                    uint32_t dst_s=(uint32_t)__cvta_generic_to_shared(sStage);
                    int r0=idx[kb0+g*4+0], r1=idx[kb0+g*4+1], r2=idx[kb0+g*4+2], r3=idx[kb0+g*4+3];
                    cp_gather4(dst_s,&tmK, ct*TMA_W, r0,r1,r2,r3, bar_s);
                    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;"::"r"(bar_s),"r"((uint32_t)(4*TMA_W*1)));
                    asm volatile("{.reg .pred p;L:mbarrier.try_wait.parity.shared::cta.b64 p,[%0],0;@!p bra L;}"::"r"(bar_s));
                }
                __syncthreads();
                // dequant FP8 staged [4 x TMA_W] -> bf16 sK[(g*4+i)*D + ct*TMA_W + c], block scale.
                for(int e=t; e<4*TMA_W; e+=256){
                    int i=e/TMA_W, c=e%TMA_W;            // row-in-group, col-in-tile
                    int dcol=ct*TMA_W + c;                // absolute D column
                    int row=idx[kb0+g*4+i];
                    int blk=dcol/B128;
                    float sc=Kscale[(size_t)row*NBLK+blk];
                    float v=fp8e4m3_to_f32(sStage[e])*sc;
                    sK[(g*4+i)*D + dcol]=__float2bfloat16(v);
                }
                __syncthreads();
            }
        }

        for(int tile=warp; tile<NWMMA_M*NWMMA_N; tile+=WARPS){
            int mt=tile/NWMMA_N, nt=tile%NWMMA_N;
            wmma::fragment<wmma::accumulator,16,16,16,float> acc; wmma::fill_fragment(acc,0.f);
            for(int dt=0;dt<NWMMA_D;dt++){
                wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major> a;
                wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::col_major> b;
                wmma::load_matrix_sync(a,sQ+(mt*16)*D+dt*16,D);
                wmma::load_matrix_sync(b,sK+(nt*16)*D+dt*16,D);
                wmma::mma_sync(acc,a,b,acc);
            }
            wmma::store_matrix_sync(sS+(mt*16)*KB+nt*16,acc,KB,wmma::mem_row_major);
        }
        __syncthreads();

        for(int h=warp; h<HQ; h+=WARPS){
            float* srow=sS+h*KB;
            float bmx=-INFINITY; for(int j=lane;j<KB;j+=32){ float v=srow[j]*scale; if(v>bmx)bmx=v; }
            for(int o=16;o>0;o>>=1) bmx=fmaxf(bmx,__shfl_xor_sync(0xffffffff,bmx,o));
            float m_old=m_run[h]; float m_new=fmaxf(m_old,bmx);
            float corr=(m_old==-INFINITY)?0.f:expf(m_old-m_new);
            float bsum=0; for(int j=lane;j<KB;j+=32){ float pp=expf(srow[j]*scale-m_new); sP[h*KB+j]=__float2bfloat16(pp); bsum+=pp; }
            for(int o=16;o>0;o>>=1) bsum+=__shfl_xor_sync(0xffffffff,bsum,o);
            if(lane==0){ m_run[h]=m_new; l_run[h]=l_run[h]*corr+bsum; }
            for(int d=lane;d<D;d+=32) O[h*D+d]*=corr;
        }
        __syncthreads();

        for(int tile=warp; tile<NWMMA_M*NWMMA_D; tile+=WARPS){
            int mt=tile/NWMMA_D, dt=tile%NWMMA_D;
            wmma::fragment<wmma::accumulator,16,16,16,float> acc; wmma::fill_fragment(acc,0.f);
            for(int kt=0;kt<NWMMA_N;kt++){
                wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major> a;
                wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::row_major> b;
                wmma::load_matrix_sync(a,sP+(mt*16)*KB+kt*16,KB);
                wmma::load_matrix_sync(b,sK+(kt*16)*D+dt*16,D);
                wmma::mma_sync(acc,a,b,acc);
            }
            wmma::store_matrix_sync(wscratch[warp],acc,16,wmma::mem_row_major);
            for(int e=lane;e<256;e+=32){ int rr=e/16,cc=e%16; O[(mt*16+rr)*D+dt*16+cc]+=wscratch[warp][e]; }
        }
        __syncthreads();
    }
    for(int h=warp;h<HQ;h+=WARPS){ float inv=1.0f/l_run[h]; for(int d=lane;d<D;d+=32) O[h*D+d]*=inv; }
}

int main(){
    cudaSetDevice(0); cudaDeviceProp pr; cudaGetDeviceProperties(&pr,0);
    printf("GPU %s sm_%d%d  D=%d HQ=%d TOPK=%d KB=%d B128=%d NBLK=%d\n",pr.name,pr.major,pr.minor,D,HQ,TOPK,KB,B128,NBLK);
    Problem p=make_problem(2048,1234);
    auto ref     = ref_attn(p);       // bf16 (true) reference
    auto ref_fp8 = ref_attn_fp8(p);   // FP8-dequant reference (isolates kernel from quant)
    float *dQ,*dO,*dKs; uint8_t* dK8; int* dI;
    cudaMalloc(&dQ,p.Q.size()*4); cudaMalloc(&dO,(size_t)HQ*D*4);
    cudaMalloc(&dK8,p.Kfp8.size()); cudaMalloc(&dKs,p.Kscale.size()*4); cudaMalloc(&dI,TOPK*4);
    cudaMemcpy(dQ,p.Q.data(),p.Q.size()*4,cudaMemcpyHostToDevice);
    cudaMemcpy(dK8,p.Kfp8.data(),p.Kfp8.size(),cudaMemcpyHostToDevice);
    cudaMemcpy(dKs,p.Kscale.data(),p.Kscale.size()*4,cudaMemcpyHostToDevice);
    cudaMemcpy(dI,p.idx.data(),TOPK*4,cudaMemcpyHostToDevice);

    CUtensorMap tm{};
    uint64_t dims[2]={(uint64_t)D,(uint64_t)p.n_comp};
    uint64_t str[1]={(uint64_t)D*1};                       // FP8: 1 byte/elem
    uint32_t box[2]={TMA_W,1};
    uint32_t es[2]={1,1};
    CUresult r=cuTensorMapEncodeTiled(&tm,CU_TENSOR_MAP_DATA_TYPE_UINT8,2,dK8,dims,str,box,es,
        CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_NONE,CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if(r){const char*s;cuGetErrorString(r,&s);printf("encode FAIL: %s\n",s);return 2;}

    size_t smem=(size_t)HQ*D*2 + KB*D*2 + HQ*KB*2 + HQ*KB*4;
    printf("smem=%zu (%.1f KB)\n",smem,smem/1024.0);
    cudaFuncSetAttribute(k4,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem);
    k4<<<1,256,smem>>>(tm,dQ,dI,dKs,p.scale,dO);
    printf("launch=%s\n",cudaGetErrorString(cudaGetLastError()));
    cudaError_t e=cudaDeviceSynchronize();
    printf("sync=%s\n",cudaGetErrorString(e)); if(e)return 1;
    std::vector<float> out((size_t)HQ*D); cudaMemcpy(out.data(),dO,out.size()*4,cudaMemcpyDeviceToHost);
    printf("vs FP8-dequant ref:  cos=%.6f  rel_err=%.3e   (isolates kernel)\n",cosine(out,ref_fp8),rel_err(out,ref_fp8));
    printf("vs bf16 (true) ref:  cos=%.6f  rel_err=%.3e   (kernel + FP8 quant)\n",cosine(out,ref),rel_err(out,ref));
    printf("out[0..3]=%.4f %.4f %.4f %.4f\n",out[0],out[1],out[2],out[3]);
    printf("ref[0..3]=%.4f %.4f %.4f %.4f\n",ref[0],ref[1],ref[2],ref[3]);
    return 0;
}
