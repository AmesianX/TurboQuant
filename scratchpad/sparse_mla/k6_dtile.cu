// Stage 4b (occupancy via D-tiling): cut smem to <=46KB -> 2 CTA/SM on GB10 (100KB/SM).
// Key change vs k4/k5: never stage the full [KB x 512] K block. Instead tile D into DT=128 chunks
// and stage only [KB x DT] K at a time. QK^T accumulates the [64 x KB] score over the 4 D-tiles;
// PV accumulates O over the 4 D-tiles. Q kept FP8 in smem (32KB). MLA: V==K.
// Budget: sQ8 32KB + sKt[KB*DT]bf16 (16*128*2=4KB) + sP 2KB + sS 4KB + scales ~1KB = ~43KB dyn.
#include "ref.h"
#include <mma.h>
#include <cudaTypedefs.h>
#include <cuda_fp8.h>
using namespace nvcuda;

#define KB 16
#define WARPS 8
#define DT 128                 // D-tile width
#define NDT (D/DT)             // 4
#define NWMMA_M (HQ/16)        // 4
#define NWMMA_DT (DT/16)       // 8 (16-wide WMMA k-steps within a D-tile)
#define TMA_W 256              // FP8 gather tile width (bytes). DT=128 < 256, gather covers 2 D-tiles? No:
                               // we gather the FULL row's 512 bytes as 2x256 still, but into a [KB x D]?
                               // To keep sK small we gather per D-tile: box width = DT=128 (128 FP8 bytes).
extern __shared__ char smem[];
__device__ __forceinline__ void cp_gather4(uint32_t d,const CUtensorMap* tm,int c0,int r0,int r1,int r2,int r3,uint32_t b){
  asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes [%0],[%1,{%2,%3,%4,%5,%6}],[%7];"
    ::"r"(d),"l"(tm),"r"(c0),"r"(r0),"r"(r1),"r"(r2),"r"(r3),"r"(b):"memory");}
__device__ __forceinline__ float f8(uint8_t v){__nv_fp8_e4m3 f;memcpy(&f,&v,1);return (float)f;}

__global__ void __launch_bounds__(256,2) k6(
        const __grid_constant__ CUtensorMap tmK, const float* __restrict__ Q,
        const int* __restrict__ idx, const float* __restrict__ Kscale, float scale, float* __restrict__ O){
    const int t=threadIdx.x, warp=t>>5, lane=t&31;
    uint8_t* sQ8=(uint8_t*)smem;                       // [HQ*D] FP8 = 32KB
    float*   sQs=(float*)(sQ8+HQ*D);                   // [HQ*NBLK] = 1KB
    __nv_bfloat16* sKt=(__nv_bfloat16*)(sQs+HQ*NBLK);  // [KB*DT] bf16 = 4KB (current D-tile)
    __nv_bfloat16* sP=sKt+KB*DT;                        // [HQ*KB] = 2KB
    float*   sS=(float*)(sP+HQ*KB);                     // [HQ*KB] = 4KB
    __shared__ float m_run[HQ], l_run[HQ];
    // O kept in GLOBAL (per-CTA), accumulated across D-tiles+blocks. Keep static smem minimal
    // so the 43KB dynamic actually buys 2 CTA/SM (static counts against the 100KB/SM too).
    __shared__ float wscratch[WARPS][256];             // 8KB
    __shared__ __nv_bfloat16 qtile[WARPS][16*16];      // 4KB
    __shared__ alignas(128) uint8_t sStage[4*DT];      // 0.5KB
    __shared__ alignas(8) uint64_t bar;

    for(int h=t; h<HQ; h+=256) for(int blk=0; blk<NBLK; blk++){
        float amax=0; for(int c=0;c<B128;c++) amax=fmaxf(amax,fabsf(Q[h*D+blk*B128+c]));
        float sc=(amax>0)?amax/448.0f:1.0f; sQs[h*NBLK+blk]=sc;
        for(int c=0;c<B128;c++){ __nv_fp8_e4m3 q=(__nv_fp8_e4m3)(Q[h*D+blk*B128+c]/sc); memcpy(&sQ8[h*D+blk*B128+c],&q,1);} }
    for(int i=t;i<HQ*D;i+=256) O[i]=0.f;
    if(t<HQ){ m_run[t]=-INFINITY; l_run[t]=0.f; }
    uint32_t bar_s=(uint32_t)__cvta_generic_to_shared(&bar);
    __syncthreads();

    for(int kb0=0; kb0<TOPK; kb0+=KB){
        // ---- QK^T over D-tiles: accumulate sS[64 x KB] ----
        for(int i=t;i<HQ*KB;i+=256) sS[i]=0.f;
        __syncthreads();
        for(int dtile=0; dtile<NDT; dtile++){
            // gather [KB x DT] FP8 of this D-tile into sKt (DT=128 FP8 bytes = 1 gather tile of 128)
            for(int g=0; g<KB/4; g++){
                if(t==0){ asm volatile("mbarrier.init.shared.b64 [%0],1;"::"r"(bar_s)); asm volatile("fence.proxy.async.shared::cta;");
                    uint32_t ds=(uint32_t)__cvta_generic_to_shared(sStage); int r0=idx[kb0+g*4],r1=idx[kb0+g*4+1],r2=idx[kb0+g*4+2],r3=idx[kb0+g*4+3];
                    cp_gather4(ds,&tmK,dtile*DT,r0,r1,r2,r3,bar_s);
                    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;"::"r"(bar_s),"r"((uint32_t)(4*DT)));
                    asm volatile("{.reg .pred p;L:mbarrier.try_wait.parity.shared::cta.b64 p,[%0],0;@!p bra L;}"::"r"(bar_s)); }
                __syncthreads();
                for(int e=t;e<4*DT;e+=256){ int i=e/DT,c=e%DT,dcol=dtile*DT+c,row=idx[kb0+g*4+i],blk=dcol/B128;
                    sKt[(g*4+i)*DT+c]=__float2bfloat16(f8(sStage[e])*Kscale[(size_t)row*NBLK+blk]); }
                __syncthreads();
            }
            // accumulate QK^T contribution of this D-tile into sS
            for(int mt=warp; mt<NWMMA_M; mt+=WARPS){
                wmma::fragment<wmma::accumulator,16,16,16,float> acc; wmma::fill_fragment(acc,0.f);
                for(int kk=0; kk<NWMMA_DT; kk++){
                    for(int e=lane;e<256;e+=32){ int rr=e/16,cc=e%16; int hh=mt*16+rr, dd=dtile*DT+kk*16+cc;
                        qtile[warp][e]=__float2bfloat16(f8(sQ8[hh*D+dd])*sQs[hh*NBLK+(dd/B128)]); }
                    __syncwarp();
                    wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major> a;
                    wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::col_major> b;
                    wmma::load_matrix_sync(a,qtile[warp],16);
                    wmma::load_matrix_sync(b,sKt+kk*16,DT);
                    wmma::mma_sync(acc,a,b,acc);
                }
                // add into sS[mt*16 .. ][0..KB)
                wmma::store_matrix_sync(wscratch[warp],acc,16,wmma::mem_row_major);
                for(int e=lane;e<16*KB;e+=32){ int rr=e/KB,cc=e%KB; sS[(mt*16+rr)*KB+cc]+=wscratch[warp][rr*16+cc]; }
            }
            __syncthreads();
        }

        // ---- online softmax over KB ----
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

        // ---- PV over D-tiles: O[:, dtile] += P @ V[:, dtile] ----
        for(int dtile=0; dtile<NDT; dtile++){
            // re-gather this D-tile of K (= V) into sKt
            for(int g=0; g<KB/4; g++){
                if(t==0){ asm volatile("mbarrier.init.shared.b64 [%0],1;"::"r"(bar_s)); asm volatile("fence.proxy.async.shared::cta;");
                    uint32_t ds=(uint32_t)__cvta_generic_to_shared(sStage); int r0=idx[kb0+g*4],r1=idx[kb0+g*4+1],r2=idx[kb0+g*4+2],r3=idx[kb0+g*4+3];
                    cp_gather4(ds,&tmK,dtile*DT,r0,r1,r2,r3,bar_s);
                    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;"::"r"(bar_s),"r"((uint32_t)(4*DT)));
                    asm volatile("{.reg .pred p;L:mbarrier.try_wait.parity.shared::cta.b64 p,[%0],0;@!p bra L;}"::"r"(bar_s)); }
                __syncthreads();
                for(int e=t;e<4*DT;e+=256){ int i=e/DT,c=e%DT,dcol=dtile*DT+c,row=idx[kb0+g*4+i],blk=dcol/B128;
                    sKt[(g*4+i)*DT+c]=__float2bfloat16(f8(sStage[e])*Kscale[(size_t)row*NBLK+blk]); }
                __syncthreads();
            }
            for(int tile=warp; tile<NWMMA_M*(DT/16); tile+=WARPS){
                int mt=tile/(DT/16), dd=tile%(DT/16);   // output [mt*16 heads x dd*16 cols within D-tile]
                wmma::fragment<wmma::accumulator,16,16,16,float> acc; wmma::fill_fragment(acc,0.f);
                wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major> a;
                wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::row_major> b;
                wmma::load_matrix_sync(a,sP+(mt*16)*KB,KB);     // P [16 heads x KB]
                wmma::load_matrix_sync(b,sKt+dd*16,DT);         // V [KB x 16 cols]
                wmma::mma_sync(acc,a,b,acc);
                wmma::store_matrix_sync(wscratch[warp],acc,16,wmma::mem_row_major);
                for(int e=lane;e<256;e+=32){ int rr=e/16,cc=e%16; O[(mt*16+rr)*D + dtile*DT + dd*16 + cc]+=wscratch[warp][e]; }
            }
            __syncthreads();
        }
    }
    for(int h=warp;h<HQ;h+=WARPS){ float inv=1.0f/l_run[h]; for(int d=lane;d<D;d+=32) O[h*D+d]*=inv; }
}

int main(){
    cudaSetDevice(0); cudaDeviceProp pr; cudaGetDeviceProperties(&pr,0);
    int NC=getenv("NC")?atoi(getenv("NC")):16384; int NQ=getenv("NQ")?atoi(getenv("NQ")):1;
    Problem p=make_problem(NC,1234);
    auto ref=ref_attn(p); auto ref_fp8=ref_attn_fp8(p);
    float *dQ,*dO,*dKs; uint8_t* dK8; int* dI;
    cudaMalloc(&dQ,p.Q.size()*4); cudaMalloc(&dO,(size_t)NQ*HQ*D*4);
    cudaMalloc(&dK8,p.Kfp8.size()); cudaMalloc(&dKs,p.Kscale.size()*4); cudaMalloc(&dI,TOPK*4);
    cudaMemcpy(dQ,p.Q.data(),p.Q.size()*4,cudaMemcpyHostToDevice);
    cudaMemcpy(dK8,p.Kfp8.data(),p.Kfp8.size(),cudaMemcpyHostToDevice);
    cudaMemcpy(dKs,p.Kscale.data(),p.Kscale.size()*4,cudaMemcpyHostToDevice);
    cudaMemcpy(dI,p.idx.data(),TOPK*4,cudaMemcpyHostToDevice);
    CUtensorMap tm{}; uint64_t dims[2]={(uint64_t)D,(uint64_t)p.n_comp}; uint64_t str[1]={(uint64_t)D}; uint32_t box[2]={DT,1}; uint32_t es[2]={1,1};
    CUresult r=cuTensorMapEncodeTiled(&tm,CU_TENSOR_MAP_DATA_TYPE_UINT8,2,dK8,dims,str,box,es,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_NONE,CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if(r){const char*s;cuGetErrorString(r,&s);printf("encode FAIL %s\n",s);return 2;}
    size_t smem=(size_t)HQ*D + HQ*NBLK*4 + KB*DT*2 + HQ*KB*2 + HQ*KB*4;
    printf("smem=%zu (%.1f KB)  NC=%d NQ=%d\n",smem,smem/1024.0,NC,NQ);
    cudaFuncSetAttribute(k6,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem);
    int maxb; cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxb,k6,256,smem);
    printf("occupancy=%d CTA/SM\n",maxb);
    k6<<<NQ,256,smem>>>(tm,dQ,dI,dKs,p.scale,dO);
    printf("launch=%s\n",cudaGetErrorString(cudaGetLastError()));
    if(cudaDeviceSynchronize()){printf("sync FAIL=%s\n",cudaGetErrorString(cudaGetLastError()));return 1;}
    if(NQ==1){ std::vector<float> out((size_t)HQ*D); cudaMemcpy(out.data(),dO,out.size()*4,cudaMemcpyDeviceToHost);
        printf("vs FP8-dequant ref: cos=%.6f rel=%.3e\n",cosine(out,ref_fp8),rel_err(out,ref_fp8));
        printf("vs bf16 true ref:   cos=%.6f rel=%.3e\n",cosine(out,ref),rel_err(out,ref)); }
    else { cudaEvent_t a,b;cudaEventCreate(&a);cudaEventCreate(&b);cudaEventRecord(a);
        int IT=10; for(int i=0;i<IT;i++) k6<<<NQ,256,smem>>>(tm,dQ,dI,dKs,p.scale,dO);
        cudaEventRecord(b);cudaEventSynchronize(b);float ms;cudaEventElapsedTime(&ms,a,b);ms/=IT;
        printf("%.3f ms/launch => %.0f queries/s\n",ms,NQ/(ms/1000.0)); }
    return 0;
}
