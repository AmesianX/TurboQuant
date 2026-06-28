// Stage 4 (occupancy): cut smem/CTA to raise CTA/SM above 1.
// k4 used sQ[64x512]bf16=64KB persistent -> 86KB/CTA -> 1 CTA/SM (48 concurrent on GB10).
// k5: DROP persistent sQ. Keep Q in global f32; per QK^T head-tile, stage the 16-row Q-tile
// (16x512 bf16 = 16KB) on demand. Budget: sQtile16KB + sK16KB + sP2KB + sS4KB = 38KB -> 2 CTA/SM.
// Q re-staged per flash-block (32x) but Q traffic (64x512x2=64KB/query) << K traffic. Same numerics.
#include "ref.h"
#include <mma.h>
#include <cudaTypedefs.h>
#include <cuda_fp8.h>
using namespace nvcuda;

#define KB 16
#define WARPS 8
#define NWMMA_M (HQ/16)   // 4
#define NWMMA_N (KB/16)   // 1
#define NWMMA_D (D/16)    // 32
#define TMA_W 256
#define NTILE (D/TMA_W)

extern __shared__ char smem[];
__device__ __forceinline__ void cp_gather4(uint32_t d,const CUtensorMap* tm,int c0,int r0,int r1,int r2,int r3,uint32_t b){
  asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes [%0],[%1,{%2,%3,%4,%5,%6}],[%7];"
    ::"r"(d),"l"(tm),"r"(c0),"r"(r0),"r"(r1),"r"(r2),"r"(r3),"r"(b):"memory");}
__device__ __forceinline__ float f8(uint8_t v){__nv_fp8_e4m3 f;memcpy(&f,&v,1);return (float)f;}

__global__ void __launch_bounds__(256,2) k5(
        const __grid_constant__ CUtensorMap tmK, const float* __restrict__ Q,
        const int* __restrict__ idx, const float* __restrict__ Kscale, float scale, float* __restrict__ O){
    const int t=threadIdx.x, warp=t>>5, lane=t&31;
    __nv_bfloat16* sQt=(__nv_bfloat16*)smem;          // [HQ*D] bf16? NO -> we still need all heads for QK^T.
    // Correction: QK^T needs ALL 64 head-rows of Q against the KB keys. So Q-tile must cover all heads
    // for the current key-block. That's still [64 x 512] = 64KB. Streaming per-block doesn't shrink it
    // because every block touches all 64 heads.  => The smem floor for Q is fundamentally 64KB IF we
    // stage all heads. ALTERNATIVE: store Q in smem as FP8 (32KB) and dequant per WMMA tile. Do that.
    // sQ8: FP8 [HQ*D] = 32KB ; sK bf16 [KB*D]=16KB ; sP 2KB ; sS 4KB ; sQtile bf16 [16*16] tiny scratch.
    uint8_t* sQ8=(uint8_t*)smem;                       // [HQ*D] FP8 = 32KB
    float*   sQs=(float*)(sQ8+HQ*D);                   // [HQ*NBLK] per-(head,block) Q scale = 64*4*4=1KB
    __nv_bfloat16* sK=(__nv_bfloat16*)(sQs+HQ*NBLK);   // [KB*D] = 16KB
    __nv_bfloat16* sP=sK+KB*D;                          // [HQ*KB] = 2KB
    float*   sS=(float*)(sP+HQ*KB);                     // [HQ*KB] = 4KB
    __nv_bfloat16* sQb=(__nv_bfloat16*)(sS+HQ*KB);      // bf16 dequant scratch [HQ*D]? too big.
    // We need bf16 Q for WMMA load. Dequant a 16x16 Q sub-tile into a tiny per-warp buffer at use time.
    __shared__ float m_run[HQ], l_run[HQ];
    __shared__ float wscratch[WARPS][256];
    __shared__ __nv_bfloat16 qtile[WARPS][16*16];      // per-warp bf16 Q sub-tile for WMMA
    __shared__ alignas(128) uint8_t sStage[4*TMA_W];
    __shared__ alignas(8) uint64_t bar;
    (void)sQt;(void)sQb;

    // Quantize Q -> FP8 in smem once (per-(head,128-block) scale, like the K cache).
    for(int h=t; h<HQ; h+=256){
        for(int blk=0; blk<NBLK; blk++){
            float amax=0; for(int c=0;c<B128;c++) amax=fmaxf(amax,fabsf(Q[h*D+blk*B128+c]));
            float sc=(amax>0)?amax/448.0f:1.0f; sQs[h*NBLK+blk]=sc;
            for(int c=0;c<B128;c++){ __nv_fp8_e4m3 q=(__nv_fp8_e4m3)(Q[h*D+blk*B128+c]/sc); memcpy(&sQ8[h*D+blk*B128+c],&q,1);}
        }
    }
    for(int i=t;i<HQ*D;i+=256) O[i]=0.f;
    if(t<HQ){ m_run[t]=-INFINITY; l_run[t]=0.f; }
    uint32_t bar_s=(uint32_t)__cvta_generic_to_shared(&bar);
    __syncthreads();

    for(int kb0=0; kb0<TOPK; kb0+=KB){
        // gather + FP8 dequant K -> bf16 sK
        for(int g=0;g<KB/4;g++) for(int ct=0;ct<NTILE;ct++){
            if(t==0){ asm volatile("mbarrier.init.shared.b64 [%0],1;"::"r"(bar_s)); asm volatile("fence.proxy.async.shared::cta;");
                uint32_t ds=(uint32_t)__cvta_generic_to_shared(sStage); int r0=idx[kb0+g*4],r1=idx[kb0+g*4+1],r2=idx[kb0+g*4+2],r3=idx[kb0+g*4+3];
                cp_gather4(ds,&tmK,ct*TMA_W,r0,r1,r2,r3,bar_s);
                asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;"::"r"(bar_s),"r"((uint32_t)(4*TMA_W)));
                asm volatile("{.reg .pred p;L:mbarrier.try_wait.parity.shared::cta.b64 p,[%0],0;@!p bra L;}"::"r"(bar_s)); }
            __syncthreads();
            for(int e=t;e<4*TMA_W;e+=256){ int i=e/TMA_W,c=e%TMA_W,dcol=ct*TMA_W+c,row=idx[kb0+g*4+i],blk=dcol/B128;
                sK[(g*4+i)*D+dcol]=__float2bfloat16(f8(sStage[e])*Kscale[(size_t)row*NBLK+blk]); }
            __syncthreads();
        }

        // QK^T: per (head-tile mt) x (key-tile nt=0). Dequant Q sub-tile (16 heads x 16 dim) from FP8 just-in-time.
        for(int tile=warp; tile<NWMMA_M*NWMMA_N; tile+=WARPS){
            int mt=tile/NWMMA_N; // nt=0
            wmma::fragment<wmma::accumulator,16,16,16,float> acc; wmma::fill_fragment(acc,0.f);
            for(int dt=0;dt<NWMMA_D;dt++){
                // dequant Q[mt*16 .. +16][dt*16 .. +16] FP8 -> qtile bf16
                for(int e=lane;e<256;e+=32){ int rr=e/16,cc=e%16; int hh=mt*16+rr, dd=dt*16+cc;
                    qtile[warp][e]=__float2bfloat16(f8(sQ8[hh*D+dd])*sQs[hh*NBLK+(dd/B128)]); }
                __syncwarp();
                wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major> a;
                wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::col_major> b;
                wmma::load_matrix_sync(a,qtile[warp],16);
                wmma::load_matrix_sync(b,sK+dt*16,D);            // key-tile nt=0
                wmma::mma_sync(acc,a,b,acc);
            }
            wmma::store_matrix_sync(sS+(mt*16)*KB,acc,KB,wmma::mem_row_major);
        }
        __syncthreads();

        for(int h=warp; h<HQ; h+=WARPS){
            float* sr=sS+h*KB;
            float bmx=-INFINITY; for(int j=lane;j<KB;j+=32){ float v=sr[j]*scale; if(v>bmx)bmx=v; }
            for(int o=16;o>0;o>>=1) bmx=fmaxf(bmx,__shfl_xor_sync(0xffffffff,bmx,o));
            float mo=m_run[h],mn=fmaxf(mo,bmx),corr=(mo==-INFINITY)?0.f:expf(mo-mn);
            float bs=0; for(int j=lane;j<KB;j+=32){ float pp=expf(sr[j]*scale-mn); sP[h*KB+j]=__float2bfloat16(pp); bs+=pp; }
            for(int o=16;o>0;o>>=1) bs+=__shfl_xor_sync(0xffffffff,bs,o);
            if(lane==0){ m_run[h]=mn; l_run[h]=l_run[h]*corr+bs; }
            for(int d=lane;d<D;d+=32) O[h*D+d]*=corr;
        }
        __syncthreads();

        for(int tile=warp; tile<NWMMA_M*NWMMA_D; tile+=WARPS){
            int mt=tile/NWMMA_D, dt=tile%NWMMA_D;
            wmma::fragment<wmma::accumulator,16,16,16,float> acc; wmma::fill_fragment(acc,0.f);
            // contract over KB=16 -> 1 k-tile
            wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major> a;
            wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::row_major> b;
            wmma::load_matrix_sync(a,sP+(mt*16)*KB,KB);
            wmma::load_matrix_sync(b,sK+dt*16,D);
            wmma::mma_sync(acc,a,b,acc);
            wmma::store_matrix_sync(wscratch[warp],acc,16,wmma::mem_row_major);
            for(int e=lane;e<256;e+=32){ int rr=e/16,cc=e%16; O[(mt*16+rr)*D+dt*16+cc]+=wscratch[warp][e]; }
        }
        __syncthreads();
    }
    for(int h=warp;h<HQ;h+=WARPS){ float inv=1.0f/l_run[h]; for(int d=lane;d<D;d+=32) O[h*D+d]*=inv; }
}

int main(){
    cudaSetDevice(0); cudaDeviceProp pr; cudaGetDeviceProperties(&pr,0);
    int NC=getenv("NC")?atoi(getenv("NC")):16384;
    Problem p=make_problem(NC,1234);
    auto ref=ref_attn(p); auto ref_fp8=ref_attn_fp8(p);
    float *dQ,*dO,*dKs; uint8_t* dK8; int* dI;
    int NQ=getenv("NQ")?atoi(getenv("NQ")):1;
    cudaMalloc(&dQ,p.Q.size()*4); cudaMalloc(&dO,(size_t)NQ*HQ*D*4);
    cudaMalloc(&dK8,p.Kfp8.size()); cudaMalloc(&dKs,p.Kscale.size()*4); cudaMalloc(&dI,TOPK*4);
    cudaMemcpy(dQ,p.Q.data(),p.Q.size()*4,cudaMemcpyHostToDevice);
    cudaMemcpy(dK8,p.Kfp8.data(),p.Kfp8.size(),cudaMemcpyHostToDevice);
    cudaMemcpy(dKs,p.Kscale.data(),p.Kscale.size()*4,cudaMemcpyHostToDevice);
    cudaMemcpy(dI,p.idx.data(),TOPK*4,cudaMemcpyHostToDevice);
    CUtensorMap tm{}; uint64_t dims[2]={(uint64_t)D,(uint64_t)p.n_comp}; uint64_t str[1]={(uint64_t)D}; uint32_t box[2]={TMA_W,1}; uint32_t es[2]={1,1};
    cuTensorMapEncodeTiled(&tm,CU_TENSOR_MAP_DATA_TYPE_UINT8,2,dK8,dims,str,box,es,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_NONE,CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    size_t smem=(size_t)HQ*D /*FP8 Q*/ + HQ*NBLK*4 /*Qscale*/ + KB*D*2 + HQ*KB*2 + HQ*KB*4;
    printf("smem=%zu (%.1f KB)  NC=%d NQ=%d\n",smem,smem/1024.0,NC,NQ);
    cudaError_t se=cudaFuncSetAttribute(k5,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem);
    int maxb; cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxb,k5,256,smem);
    printf("setattr=%s  occupancy=%d CTA/SM\n",cudaGetErrorString(se),maxb);
    k5<<<NQ,256,smem>>>(tm,dQ,dI,dKs,p.scale,dO);
    printf("launch=%s\n",cudaGetErrorString(cudaGetLastError()));
    if(cudaDeviceSynchronize()){printf("sync=%s\n",cudaGetErrorString(cudaGetLastError()));return 1;}
    if(NQ==1){ std::vector<float> out((size_t)HQ*D); cudaMemcpy(out.data(),dO,out.size()*4,cudaMemcpyDeviceToHost);
        printf("vs FP8-dequant ref: cos=%.6f rel=%.3e\n",cosine(out,ref_fp8),rel_err(out,ref_fp8));
        printf("vs bf16 true ref:   cos=%.6f rel=%.3e\n",cosine(out,ref),rel_err(out,ref)); }
    else { cudaEvent_t a,b;cudaEventCreate(&a);cudaEventCreate(&b);cudaEventRecord(a);
        int IT=10; for(int i=0;i<IT;i++) k5<<<NQ,256,smem>>>(tm,dQ,dI,dKs,p.scale,dO);
        cudaEventRecord(b);cudaEventSynchronize(b);float ms;cudaEventElapsedTime(&ms,a,b);ms/=IT;
        printf("%.3f ms/launch => %.0f queries/s\n",ms,NQ/(ms/1000.0)); }
    return 0;
}
