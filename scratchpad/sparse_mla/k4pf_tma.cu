// PREFILL-shape TMA-Gather4 harness that MIRRORS the in-graph kernel design exactly:
//   - K cache is bf16 (k_all materialized), gathered via cp.async.bulk.tensor...gather4 (2x256-wide).
//   - per-token NON-CONTIGUOUS kv_idx (row stride = kv_row_elems = n_comp_view, NOT top_k).
//   - validity guard: comp slot valid only if idx in [0,n_comp_view) AND krow in [0,n_kv); invalid
//     slots gather a SAFE row 0 (never OOB) and are masked to -inf in softmax.
//   - tensor-map built over the (address-stable) K buffer, [width=D, height=n_kv], stride D*2.
// Gate: cos vs per-query dense over the valid selected rows, at PREFILL (M=1024,n_comp=3000) and
// SHORT (n_comp=7 < top_k => mostly padding) shapes.
#include "ref.h"
#include <mma.h>
#include <cudaTypedefs.h>
using namespace nvcuda;
#define KB 16
#define WARPS 8
#define NM (HQ/16)
#define NN (KB/16)
#define ND (D/16)
#define TMA_W 256              // bf16 elems/tile (512B). D=512 => NTILE=2.
#define NTILE (D/TMA_W)
extern __shared__ char smem[];

__device__ __forceinline__ void cp_gather4(uint32_t dst_s, const CUtensorMap* tm,
        int c0, int r0,int r1,int r2,int r3, uint32_t bar_s){
    asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes "
        "[%0],[%1,{%2,%3,%4,%5,%6}],[%7];"
        ::"r"(dst_s),"l"(tm),"r"(c0),"r"(r0),"r"(r1),"r"(r2),"r"(r3),"r"(bar_s):"memory");
}

// One CTA per query token (grid = M). n_raw=0 (comp segment only); raw window handled separately
// in-graph (it's the same gather4 with contiguous krow=pos). kv_idx row stride = kv_row_elems.
__global__ void __launch_bounds__(256) k4pf_tma(
        const __grid_constant__ CUtensorMap tmK,   // [width=D, height=n_kv] bf16, box {TMA_W,1} gather4
        const float* Qall, const int* kv_idx, int top_k, int kv_row_elems,
        int n_comp_view, int n_kv, float scale, float* Oall, int M){
    const int t=threadIdx.x,warp=t>>5,lane=t&31; int it=blockIdx.x;
    const float* Q = Qall + (size_t)it*HQ*D;
    float*       O = Oall + (size_t)it*HQ*D;
    const int*   idx_q = kv_idx + (size_t)it*kv_row_elems;
    __nv_bfloat16* sQ=(__nv_bfloat16*)smem; __nv_bfloat16* sK=sQ+HQ*D; __nv_bfloat16* sP=sK+KB*D; float* sS=(float*)(sP+HQ*KB);
    __shared__ float m_run[HQ],l_run[HQ]; __shared__ float wscratch[WARPS][256]; __shared__ int sValid[KB];
    __shared__ alignas(128) __nv_bfloat16 sStage[4*TMA_W];   // TMA gather4 dst (128B-aligned static)
    __shared__ alignas(8) uint64_t bar;
    __shared__ int sRow[KB];                                  // resolved (clamped) gather row per slot
    for(int i=t;i<HQ*D;i+=256) sQ[i]=__float2bfloat16(Q[i]);
    for(int i=t;i<HQ*D;i+=256) O[i]=0.f; if(t<HQ){m_run[t]=-INFINITY;l_run[t]=0.f;}
    uint32_t bar_s=(uint32_t)__cvta_generic_to_shared(&bar);
    __syncthreads();
    for(int kb0=0;kb0<top_k;kb0+=KB){
        // resolve validity + clamped gather row for this block's KB slots
        for(int r=t;r<KB;r+=256){
            int pos=kb0+r; int krow=0; bool valid=false;
            if(pos<top_k){ int id=idx_q[pos]; if(id>=0 && id<n_comp_view){ krow=id; valid=true; } }
            if(valid && (krow<0||krow>=n_kv)){ valid=false; krow=0; }
            sValid[r]=valid?1:0; sRow[r]=krow;   // invalid -> krow=0 (safe in-bounds gather, masked)
        }
        __syncthreads();
        // TMA gather4: KB rows x D, as KB/4 groups x NTILE col-tiles. Single-issue (t==0), re-init bar.
        for(int g=0; g<KB/4; g++){
            for(int ct=0; ct<NTILE; ct++){
                if(t==0){
                    asm volatile("mbarrier.init.shared.b64 [%0],1;"::"r"(bar_s));
                    asm volatile("fence.proxy.async.shared::cta;");
                    uint32_t dst_s=(uint32_t)__cvta_generic_to_shared(sStage);
                    int r0=sRow[g*4+0],r1=sRow[g*4+1],r2=sRow[g*4+2],r3=sRow[g*4+3];
                    cp_gather4(dst_s,&tmK, ct*TMA_W, r0,r1,r2,r3, bar_s);
                    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;"::"r"(bar_s),"r"((uint32_t)(4*TMA_W*2)));
                    asm volatile("{.reg .pred p;L:mbarrier.try_wait.parity.shared::cta.b64 p,[%0],0;@!p bra L;}"::"r"(bar_s));
                }
                __syncthreads();
                for(int e=t;e<4*TMA_W;e+=256){ int i=e/TMA_W,c=e%TMA_W; sK[(g*4+i)*D + ct*TMA_W + c]=sStage[e]; }
                __syncthreads();
            }
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

static std::vector<float> dense_q(const std::vector<float>& Q, const Problem& p, const std::vector<int>& sel){
    std::vector<float> O((size_t)HQ*D,0.f); int nv=sel.size(); if(nv==0)return O;
    std::vector<float> Kf((size_t)nv*D);
    for(int j=0;j<nv;j++){const bf16_t* kr=&p.Kbf[(size_t)sel[j]*D];for(int d=0;d<D;d++)Kf[(size_t)j*D+d]=bf16_to_f32(kr[d]);}
    std::vector<float> s(nv);
    for(int h=0;h<HQ;h++){const float* q=&Q[(size_t)h*D];float mx=-INFINITY;
        for(int j=0;j<nv;j++){float a=0;const float* k=&Kf[(size_t)j*D];for(int d=0;d<D;d++)a+=q[d]*k[d];s[j]=a*p.scale;if(s[j]>mx)mx=s[j];}
        float sum=0;for(int j=0;j<nv;j++){s[j]=expf(s[j]-mx);sum+=s[j];}float inv=1.f/sum;float* o=&O[(size_t)h*D];
        for(int j=0;j<nv;j++){float w=s[j]*inv;const float* v=&Kf[(size_t)j*D];for(int d=0;d<D;d++)o[d]+=w*v[d];}}
    return O;
}

static int run(int M, int n_comp_view, const char* tag){
    int top_k=512;
    int n_kv = ((n_comp_view+255)/256)*256; if(n_kv<256)n_kv=256;
    Problem p = make_problem(n_kv, 7);
    int kv_row_elems = n_comp_view;
    std::vector<float> Qall((size_t)M*HQ*D);
    std::vector<int> kvbuf((size_t)kv_row_elems*M);
    std::mt19937 rng(123);
    std::vector<std::vector<int>> sels(M);
    int n_valid = (n_comp_view<top_k)? n_comp_view : top_k;
    for(int it=0; it<M; it++){
        for(size_t i=0;i<(size_t)HQ*D;i++) Qall[(size_t)it*HQ*D+i]=p.Q[i]*(1.0f+0.001f*it);
        std::vector<int> all(n_comp_view); for(int i=0;i<n_comp_view;i++) all[i]=i;
        std::shuffle(all.begin(),all.end(),rng);
        for(int j=0;j<n_comp_view;j++) kvbuf[(size_t)it*kv_row_elems + j]=all[j];
        // top_k slots: first n_valid real (in [0,n_comp_view)); rest PADDING (>= n_comp_view, many OOB)
        // -> mirrors the SHORT-context crash scenario. (When n_comp_view>=top_k, all top_k are valid.)
        sels[it].assign(all.begin(), all.begin()+n_valid);
    }
    // If n_comp_view < top_k, pad the kv row out to top_k with OOB indices so the kernel reads them.
    std::vector<int> kvfull; int row_elems = kv_row_elems;
    if(n_comp_view < top_k){
        row_elems = top_k;
        kvfull.assign((size_t)row_elems*M, 0);
        for(int it=0; it<M; it++){
            for(int j=0;j<top_k;j++){
                if(j<n_valid) kvfull[(size_t)it*row_elems+j]=kvbuf[(size_t)it*kv_row_elems+j];
                else kvfull[(size_t)it*row_elems+j]= n_comp_view + ((j*131)%(n_kv*2)); // padding, many >= n_kv
            }
        }
    }
    const int* kvptr = (n_comp_view<top_k)? kvfull.data() : kvbuf.data();
    size_t kvcount = (n_comp_view<top_k)? kvfull.size() : kvbuf.size();
    float *dQ,*dO; bf16_t* dK; int* dKv;
    cudaMalloc(&dQ,Qall.size()*4); cudaMalloc(&dO,(size_t)M*HQ*D*4); cudaMalloc(&dK,p.Kbf.size()*2); cudaMalloc(&dKv,kvcount*4);
    cudaMemcpy(dQ,Qall.data(),Qall.size()*4,cudaMemcpyHostToDevice);
    cudaMemcpy(dK,p.Kbf.data(),p.Kbf.size()*2,cudaMemcpyHostToDevice);
    cudaMemcpy(dKv,kvptr,kvcount*4,cudaMemcpyHostToDevice);
    CUtensorMap tm{};
    uint64_t dims[2]={(uint64_t)D,(uint64_t)n_kv};
    uint64_t str[1]={(uint64_t)D*2};
    uint32_t box[2]={TMA_W,1}; uint32_t es[2]={1,1};
    CUresult r=cuTensorMapEncodeTiled(&tm,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,dK,dims,str,box,es,
        CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_NONE,CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if(r){const char*s;cuGetErrorString(r,&s);printf("[%s] encode FAIL: %s\n",tag,s);return 2;}
    size_t smem=(size_t)HQ*D*2+KB*D*2+HQ*KB*2+HQ*KB*4; cudaFuncSetAttribute(k4pf_tma,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem);
    printf("[%s] M=%d n_comp_view=%d top_k=%d n_kv=%d row_elems=%d n_valid=%d\n",tag,M,n_comp_view,top_k,n_kv,row_elems,n_valid);
    k4pf_tma<<<M,256,smem>>>(tm,dQ,dKv,top_k,row_elems,n_comp_view,n_kv,p.scale,dO,M);
    cudaError_t e=cudaDeviceSynchronize();
    printf("[%s] sync=%s\n",tag,cudaGetErrorString(e)); if(e){printf("[%s] CRASH\n",tag);return 1;}
    std::vector<float> out((size_t)M*HQ*D); cudaMemcpy(out.data(),dO,out.size()*4,cudaMemcpyDeviceToHost);
    double worst=1.0; int worst_it=-1;
    for(int it : {0, M/2, M-1}){
        std::vector<float> q(Qall.begin()+(size_t)it*HQ*D, Qall.begin()+(size_t)(it+1)*HQ*D);
        auto ref=dense_q(q,p,sels[it]);
        std::vector<float> o(out.begin()+(size_t)it*HQ*D, out.begin()+(size_t)(it+1)*HQ*D);
        double c=cosine(o,ref); printf("[%s] token %4d: cos=%.6f rel=%.3e\n",tag,it,c,rel_err(o,ref));
        if(c<worst){worst=c;worst_it=it;}
    }
    printf("[%s] WORST cos=%.6f @ token %d  (>=0.999 = PASS)\n",tag,worst,worst_it);
    cudaFree(dQ);cudaFree(dO);cudaFree(dK);cudaFree(dKv);
    return worst>=0.999?0:2;
}

int main(){
    cudaSetDevice(0); cudaDeviceProp pr; cudaGetDeviceProperties(&pr,0);
    printf("GPU %s sm_%d%d  TMA-Gather4 prefill harness\n",pr.name,pr.major,pr.minor);
    int rc=0;
    rc |= run(1024, 3000, "PREFILL");   // M=1024, n_comp=3000, top_k=512 (all valid)
    rc |= run(64,   7,    "SHORT");     // n_comp=7 < top_k => mostly padding (the crash guard)
    return rc;
}
