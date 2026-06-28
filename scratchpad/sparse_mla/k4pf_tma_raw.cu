// TMA harness WITH a raw window (n_raw>0), mirroring the in-graph total = n_raw + top_k logical keys.
// raw rows [0,n_raw): krow=pos (contiguous, always valid). comp rows: krow = n_raw + idx (gathered).
// Gate: cos vs per-query dense over (raw rows 0..n_raw-1) + (selected comp rows n_raw+sel).
#include "ref.h"
#include <mma.h>
#include <cudaTypedefs.h>
using namespace nvcuda;
#define KB 16
#define WARPS 8
#define NM (HQ/16)
#define NN (KB/16)
#define ND (D/16)
#define TMA_W 256
#define NTILE (D/TMA_W)
extern __shared__ char smem[];
__device__ __forceinline__ void cp_gather4(uint32_t dst_s,const CUtensorMap* tm,int c0,int r0,int r1,int r2,int r3,uint32_t bar_s){
    asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes "
        "[%0],[%1,{%2,%3,%4,%5,%6}],[%7];"::"r"(dst_s),"l"(tm),"r"(c0),"r"(r0),"r"(r1),"r"(r2),"r"(r3),"r"(bar_s):"memory");
}
__global__ void __launch_bounds__(256) k(const __grid_constant__ CUtensorMap tmK,const float* Qall,const int* kv_idx,
        int n_raw,int top_k,int kv_row_elems,int n_comp_view,int n_kv,float scale,float* Oall){
    const int t=threadIdx.x,warp=t>>5,lane=t&31;int it=blockIdx.x;
    const float* Q=Qall+(size_t)it*HQ*D; float* O=Oall+(size_t)it*HQ*D;
    const int* idx_q=kv_idx+(size_t)it*kv_row_elems;
    __nv_bfloat16* sQ=(__nv_bfloat16*)smem;__nv_bfloat16* sK=sQ+HQ*D;__nv_bfloat16* sP=sK+KB*D;float* sS=(float*)(sP+HQ*KB);
    __shared__ float m_run[HQ],l_run[HQ];__shared__ float wscratch[WARPS][256];__shared__ int sValid[KB];__shared__ int sRow[KB];
    __shared__ alignas(128) __nv_bfloat16 sStage[4*TMA_W];__shared__ alignas(8) uint64_t bar;
    for(int i=t;i<HQ*D;i+=256)sQ[i]=__float2bfloat16(Q[i]);
    for(int i=t;i<HQ*D;i+=256)O[i]=0.f;if(t<HQ){m_run[t]=-INFINITY;l_run[t]=0.f;}
    uint32_t bar_s=(uint32_t)__cvta_generic_to_shared(&bar);
    int total=n_raw+top_k; __syncthreads();
    for(int kb0=0;kb0<total;kb0+=KB){
        for(int r=t;r<KB;r+=256){
            int pos=kb0+r;int krow=0;bool valid=false;
            if(pos<total){ if(pos<n_raw){krow=pos;valid=true;} else {int id=idx_q[pos-n_raw];if(id>=0&&id<n_comp_view){krow=n_raw+id;valid=true;}} }
            if(valid&&(krow<0||krow>=n_kv)){valid=false;krow=0;}
            sValid[r]=valid?1:0;sRow[r]=krow;
        }
        __syncthreads();
        for(int g=0;g<KB/4;g++)for(int ct=0;ct<NTILE;ct++){
            if(t==0){
                asm volatile("mbarrier.init.shared.b64 [%0],1;"::"r"(bar_s));
                asm volatile("fence.proxy.async.shared::cta;");
                uint32_t dst_s=(uint32_t)__cvta_generic_to_shared(sStage);
                cp_gather4(dst_s,&tmK,ct*TMA_W,sRow[g*4+0],sRow[g*4+1],sRow[g*4+2],sRow[g*4+3],bar_s);
                asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;"::"r"(bar_s),"r"((uint32_t)(4*TMA_W*2)));
                asm volatile("{.reg .pred p;L:mbarrier.try_wait.parity.shared::cta.b64 p,[%0],0;@!p bra L;}"::"r"(bar_s));
            }
            __syncthreads();
            for(int e=t;e<4*TMA_W;e+=256){int i=e/TMA_W,c=e%TMA_W;sK[(g*4+i)*D+ct*TMA_W+c]=sStage[e];}
            __syncthreads();
        }
        for(int tile=warp;tile<NM*NN;tile+=WARPS){int mt=tile/NN,nt=tile%NN;
            wmma::fragment<wmma::accumulator,16,16,16,float> acc;wmma::fill_fragment(acc,0.f);
            for(int dt=0;dt<ND;dt++){wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major> a;wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::col_major> b;
                wmma::load_matrix_sync(a,sQ+(mt*16)*D+dt*16,D);wmma::load_matrix_sync(b,sK+(nt*16)*D+dt*16,D);wmma::mma_sync(acc,a,b,acc);}
            wmma::store_matrix_sync(sS+(mt*16)*KB+nt*16,acc,KB,wmma::mem_row_major);}
        __syncthreads();
        for(int h=warp;h<HQ;h+=WARPS){float* sr=sS+h*KB;
            float bmx=-INFINITY;for(int j=lane;j<KB;j+=32){float v=sValid[j]?sr[j]*scale:-INFINITY;if(v>bmx)bmx=v;}
            for(int o=16;o>0;o>>=1)bmx=fmaxf(bmx,__shfl_xor_sync(0xffffffff,bmx,o));
            float mo=m_run[h],mn=fmaxf(mo,bmx),corr=(mo==-INFINITY)?0.f:expf(mo-mn);
            float bs=0;for(int j=lane;j<KB;j+=32){float pp=sValid[j]?expf(sr[j]*scale-mn):0.f;sP[h*KB+j]=__float2bfloat16(pp);bs+=pp;}
            for(int o=16;o>0;o>>=1)bs+=__shfl_xor_sync(0xffffffff,bs,o);
            if(lane==0){m_run[h]=mn;l_run[h]=l_run[h]*corr+bs;}
            for(int d=lane;d<D;d+=32)O[h*D+d]*=corr;}
        __syncthreads();
        for(int tile=warp;tile<NM*ND;tile+=WARPS){int mt=tile/ND,dt=tile%ND;
            wmma::fragment<wmma::accumulator,16,16,16,float> acc;wmma::fill_fragment(acc,0.f);
            for(int kt=0;kt<NN;kt++){wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major> a;wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::row_major> b;
                wmma::load_matrix_sync(a,sP+(mt*16)*KB+kt*16,KB);wmma::load_matrix_sync(b,sK+(kt*16)*D+dt*16,D);wmma::mma_sync(acc,a,b,acc);}
            wmma::store_matrix_sync(wscratch[warp],acc,16,wmma::mem_row_major);
            for(int e=lane;e<256;e+=32){int rr=e/16,cc=e%16;O[(mt*16+rr)*D+dt*16+cc]+=wscratch[warp][e];}}
        __syncthreads();
    }
    for(int h=warp;h<HQ;h+=WARPS){float inv=(l_run[h]>0.f)?1.0f/l_run[h]:0.f;for(int d=lane;d<D;d+=32)O[h*D+d]*=inv;}
}
static std::vector<float> dense_q(const std::vector<float>& Q,const Problem& p,const std::vector<int>& rows){
    std::vector<float> O((size_t)HQ*D,0.f);int nv=rows.size();std::vector<float> Kf((size_t)nv*D);
    for(int j=0;j<nv;j++){const bf16_t* kr=&p.Kbf[(size_t)rows[j]*D];for(int d=0;d<D;d++)Kf[(size_t)j*D+d]=bf16_to_f32(kr[d]);}
    std::vector<float> s(nv);
    for(int h=0;h<HQ;h++){const float* q=&Q[(size_t)h*D];float mx=-INFINITY;
        for(int j=0;j<nv;j++){float a=0;const float* k=&Kf[(size_t)j*D];for(int d=0;d<D;d++)a+=q[d]*k[d];s[j]=a*p.scale;if(s[j]>mx)mx=s[j];}
        float sum=0;for(int j=0;j<nv;j++){s[j]=expf(s[j]-mx);sum+=s[j];}float inv=1.f/sum;float* o=&O[(size_t)h*D];
        for(int j=0;j<nv;j++){float w=s[j]*inv;const float* v=&Kf[(size_t)j*D];for(int d=0;d<D;d++)o[d]+=w*v[d];}}
    return O;
}
int main(){
    cudaSetDevice(0);cudaDeviceProp pr;cudaGetDeviceProperties(&pr,0);
    int M=512,n_raw=64,n_comp_view=2500,top_k=512;
    int n_kv=((n_raw+n_comp_view+255)/256)*256;          // raw window + comp + pad
    Problem p=make_problem(n_kv,7);int kv_row_elems=n_comp_view;
    std::vector<float> Qall((size_t)M*HQ*D);std::vector<int> kvbuf((size_t)kv_row_elems*M);
    std::mt19937 rng(123);std::vector<std::vector<int>> rowsel(M);
    for(int it=0;it<M;it++){
        for(size_t i=0;i<(size_t)HQ*D;i++)Qall[(size_t)it*HQ*D+i]=p.Q[i]*(1.f+0.001f*it);
        std::vector<int> all(n_comp_view);for(int i=0;i<n_comp_view;i++)all[i]=i;std::shuffle(all.begin(),all.end(),rng);
        for(int j=0;j<n_comp_view;j++)kvbuf[(size_t)it*kv_row_elems+j]=all[j];
        std::vector<int> rs; for(int r=0;r<n_raw;r++)rs.push_back(r);             // raw rows 0..n_raw-1
        for(int j=0;j<top_k;j++)rs.push_back(n_raw+all[j]);                       // selected comp abs rows
        rowsel[it]=rs;
    }
    float *dQ,*dO;bf16_t* dK;int* dKv;
    cudaMalloc(&dQ,Qall.size()*4);cudaMalloc(&dO,(size_t)M*HQ*D*4);cudaMalloc(&dK,p.Kbf.size()*2);cudaMalloc(&dKv,kvbuf.size()*4);
    cudaMemcpy(dQ,Qall.data(),Qall.size()*4,cudaMemcpyHostToDevice);
    cudaMemcpy(dK,p.Kbf.data(),p.Kbf.size()*2,cudaMemcpyHostToDevice);
    cudaMemcpy(dKv,kvbuf.data(),kvbuf.size()*4,cudaMemcpyHostToDevice);
    CUtensorMap tm{};uint64_t dims[2]={(uint64_t)D,(uint64_t)n_kv};uint64_t str[1]={(uint64_t)D*2};
    uint32_t box[2]={TMA_W,1};uint32_t es[2]={1,1};
    cuTensorMapEncodeTiled(&tm,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,dK,dims,str,box,es,
        CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_NONE,CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    size_t smem=(size_t)HQ*D*2+KB*D*2+HQ*KB*2+HQ*KB*4;cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem);
    printf("GPU %s sm_%d%d  RAW-WINDOW: M=%d n_raw=%d n_comp=%d top_k=%d n_kv=%d total=%d\n",pr.name,pr.major,pr.minor,M,n_raw,n_comp_view,top_k,n_kv,n_raw+top_k);
    k<<<M,256,smem>>>(tm,dQ,dKv,n_raw,top_k,kv_row_elems,n_comp_view,n_kv,p.scale,dO);
    cudaError_t e=cudaDeviceSynchronize();printf("sync=%s\n",cudaGetErrorString(e));if(e)return 1;
    std::vector<float> out((size_t)M*HQ*D);cudaMemcpy(out.data(),dO,out.size()*4,cudaMemcpyDeviceToHost);
    double worst=1.0;int wi=-1;
    for(int it:{0,M/2,M-1}){
        std::vector<float> q(Qall.begin()+(size_t)it*HQ*D,Qall.begin()+(size_t)(it+1)*HQ*D);
        auto ref=dense_q(q,p,rowsel[it]);
        std::vector<float> o(out.begin()+(size_t)it*HQ*D,out.begin()+(size_t)(it+1)*HQ*D);
        double c=cosine(o,ref);printf("token %4d: cos=%.6f rel=%.3e\n",it,c,rel_err(o,ref));if(c<worst){worst=c;wi=it;}
    }
    printf("WORST cos=%.6f @ token %d  (>=0.999 = PASS)\n",worst,wi);
    return worst>=0.999?0:2;
}
