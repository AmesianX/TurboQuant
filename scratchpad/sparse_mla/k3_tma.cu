// Stage 3a: TMA Gather4 K-loop replacing the plain indexed gather of stage 2.
// The 512-dim latent (1024B bf16) exceeds the TMA box-width cap (256 elems / 512B),
// so each gathered row is pulled as TWO 256-wide column tiles (proven in wide.cu).
// Gather KB=16 keys per flash block = 4 gather4 ops per 256-col tile x 2 tiles = 8 gather4.
// Then identical WMMA QK^T + online softmax + PV as stage 2. MLA: V==K.
#include "ref.h"
#include <mma.h>
#include <cudaTypedefs.h>
using namespace nvcuda;

#define KB 16
#define WARPS 8
#define NWMMA_M (HQ/16)
#define NWMMA_N (KB/16)
#define NWMMA_D (D/16)
#define TMA_W 256              // TMA box width in bf16 elems (512B). D=512 => 2 tiles.
#define NTILE (D/TMA_W)        // 2

extern __shared__ char smem[];

__device__ __forceinline__ void cp_gather4(uint32_t dst_s, const CUtensorMap* tm,
        int c0, int r0,int r1,int r2,int r3, uint32_t bar_s){
    asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes "
        "[%0],[%1,{%2,%3,%4,%5,%6}],[%7];"
        ::"r"(dst_s),"l"(tm),"r"(c0),"r"(r0),"r"(r1),"r"(r2),"r"(r3),"r"(bar_s):"memory");
}

__global__ void __launch_bounds__(256) k3(
        const __grid_constant__ CUtensorMap tmK,    // [n_comp x D] bf16, 2D tensor map (256-wide box, gather4)
        const float* __restrict__ Q, const int* __restrict__ idx,
        float scale, float* __restrict__ O){
    const int t=threadIdx.x, warp=t>>5, lane=t&31;
    __nv_bfloat16* sQ=(__nv_bfloat16*)smem;           // [HQ*D]
    __nv_bfloat16* sK=sQ+HQ*D;                        // [KB*D]
    __nv_bfloat16* sP=sK+KB*D;                        // [HQ*KB]
    float*         sS=(float*)(sP+HQ*KB);             // [HQ*KB]
    __shared__ float m_run[HQ], l_run[HQ];
    __shared__ float wscratch[WARPS][256];
    // static, explicitly 128B-aligned staging for TMA gather4 (matches proven g2/perf2/wide).
    __shared__ alignas(128) __nv_bfloat16 sStage[4*TMA_W];
    __shared__ alignas(8) uint64_t bar;

    for(int i=t;i<HQ*D;i+=256) sQ[i]=__float2bfloat16(Q[i]);
    for(int i=t;i<HQ*D;i+=256) O[i]=0.f;
    if(t<HQ){ m_run[t]=-INFINITY; l_run[t]=0.f; }
    uint32_t bar_s=(uint32_t)__cvta_generic_to_shared(&bar);
    __syncthreads();

    for(int kb0=0; kb0<TOPK; kb0+=KB){
        // 1) TMA Gather4 the KB=16 rows x D=512 (2 col-tiles of 256) into sK.
        //    gather4 pulls 4 rows at a time -> KB/4 = 4 groups, x NTILE=2 col-tiles = 8 gather4 ops.
        //    gather4 writes the 4 rows CONTIGUOUSLY (stride TMA_W) to a 128B-aligned dst, so we
        //    gather each col-tile into a contiguous [4 x TMA_W] staging region then scatter into
        //    sK[r*D + ct*TMA_W]. (Correct-first; fuse later.)
        for(int g=0; g<KB/4; g++){
            for(int ct=0; ct<NTILE; ct++){
                if(t==0){
                    // re-init barrier each gather4 -> always wait phase 0 (matches g2.cu/perf2.cu).
                    asm volatile("mbarrier.init.shared.b64 [%0],1;"::"r"(bar_s));
                    asm volatile("fence.proxy.async.shared::cta;");
                    uint32_t dst_s=(uint32_t)__cvta_generic_to_shared(sStage);
                    int r0=idx[kb0+g*4+0], r1=idx[kb0+g*4+1], r2=idx[kb0+g*4+2], r3=idx[kb0+g*4+3];
                    cp_gather4(dst_s,&tmK, ct*TMA_W, r0,r1,r2,r3, bar_s);
                    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;"::"r"(bar_s),"r"((uint32_t)(4*TMA_W*2)));
                    asm volatile("{.reg .pred p;L:mbarrier.try_wait.parity.shared::cta.b64 p,[%0],0;@!p bra L;}"::"r"(bar_s));
                }
                __syncthreads();
                // copy staged [4 x TMA_W] into sK[(g*4+i)*D + ct*TMA_W + c]
                for(int e=t; e<4*TMA_W; e+=256){ int i=e/TMA_W, c=e%TMA_W; sK[(g*4+i)*D + ct*TMA_W + c]=sStage[e]; }
                __syncthreads();
            }
        }

        // 2) S = Q @ K^T
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

        // 3) online softmax
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

        // 4) O += P @ V
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
    printf("GPU %s sm_%d%d  D=%d HQ=%d TOPK=%d KB=%d TMA_W=%d NTILE=%d\n",pr.name,pr.major,pr.minor,D,HQ,TOPK,KB,TMA_W,NTILE);
    Problem p=make_problem(2048,1234);
    auto ref=ref_attn(p);
    float *dQ,*dO; bf16_t* dK; int* dI;
    cudaMalloc(&dQ,p.Q.size()*4); cudaMalloc(&dO,(size_t)HQ*D*4);
    cudaMalloc(&dK,p.Kbf.size()*2); cudaMalloc(&dI,TOPK*4);
    cudaMemcpy(dQ,p.Q.data(),p.Q.size()*4,cudaMemcpyHostToDevice);
    cudaMemcpy(dK,p.Kbf.data(),p.Kbf.size()*2,cudaMemcpyHostToDevice);
    cudaMemcpy(dI,p.idx.data(),TOPK*4,cudaMemcpyHostToDevice);

    // TMA map: [width=D, height=n_comp] bf16, box {TMA_W,1} for gather4 (height implicitly 4).
    CUtensorMap tm{};
    uint64_t dims[2]={(uint64_t)D,(uint64_t)p.n_comp};
    uint64_t str[1]={(uint64_t)D*2};
    uint32_t box[2]={TMA_W,1};
    uint32_t es[2]={1,1};
    CUresult r=cuTensorMapEncodeTiled(&tm,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,dK,dims,str,box,es,
        CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_NONE,CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if(r){const char*s;cuGetErrorString(r,&s);printf("encode FAIL: %s\n",s);return 2;}

    size_t smem=(size_t)HQ*D*2 + KB*D*2 + HQ*KB*2 + HQ*KB*4;
    printf("smem=%zu (%.1f KB)\n",smem,smem/1024.0);
    cudaError_t se=cudaFuncSetAttribute(k3,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem);
    printf("setattr=%s\n",cudaGetErrorString(se));
    k3<<<1,256,smem>>>(tm,dQ,dI,p.scale,dO);
    printf("launch=%s\n",cudaGetErrorString(cudaGetLastError()));
    cudaError_t e=cudaDeviceSynchronize();
    printf("sync=%s\n",cudaGetErrorString(e)); if(e)return 1;
    std::vector<float> out((size_t)HQ*D); cudaMemcpy(out.data(),dO,out.size()*4,cudaMemcpyDeviceToHost);
    printf("cos=%.6f  rel_err=%.3e\n",cosine(out,ref),rel_err(out,ref));
    printf("ref[0..3]=%.4f %.4f %.4f %.4f\n",ref[0],ref[1],ref[2],ref[3]);
    printf("out[0..3]=%.4f %.4f %.4f %.4f\n",out[0],out[1],out[2],out[3]);
    return 0;
}
