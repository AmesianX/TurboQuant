// Prefill-shape harness: M query tokens, each with its OWN top_k list laid out in a
// NON-CONTIGUOUS kv_idx (mirrors ggml_argsort_top_k = a VIEW of width top_k over an argsort buffer
// of width argsort_w = n_comp_view, so row stride = n_comp_view, NOT top_k). This reproduces the
// PREFILL crash: a kernel that uses it*top_k reads the wrong row / runs off the buffer for late
// tokens. The fixed kernel uses it*kv_row_elems (= argsort_w). Proves cos vs per-query dense.
#include "ref.h"
#include <mma.h>
using namespace nvcuda;
#define KB 16
#define WARPS 8
#define NM (HQ/16)
#define NN (KB/16)
#define ND (D/16)
__device__ __forceinline__ float bf16d(bf16_t b){ uint32_t u=((uint32_t)b)<<16; float f; memcpy(&f,&u,4); return f; }
extern __shared__ char smem[];

// Mirror of the in-graph kernel's per-token core (no raw window: n_raw=0). grid = M (one CTA/query).
// kv_idx: [argsort_w, M] i32, row stride = kv_row_elems = argsort_w. Each token's list = kv_idx + it*kv_row_elems.
__global__ void k4pf(const float* Qall, const bf16_t* K, const int* kv_idx, int top_k, int kv_row_elems,
                     int n_comp_view, int n_kv, float scale, float* Oall, int M){
    const int t=threadIdx.x,warp=t>>5,lane=t&31; int it=blockIdx.x;
    const float* Q = Qall + (size_t)it*HQ*D;     // this query's Q [HQ x D]
    float*       O = Oall + (size_t)it*HQ*D;
    const int*   idx_q = kv_idx + (size_t)it*kv_row_elems;
    __nv_bfloat16* sQ=(__nv_bfloat16*)smem; __nv_bfloat16* sK=sQ+HQ*D; __nv_bfloat16* sP=sK+KB*D; float* sS=(float*)(sP+HQ*KB);
    __shared__ float m_run[HQ],l_run[HQ]; __shared__ float wscratch[WARPS][256]; __shared__ int sValid[KB];
    for(int i=t;i<HQ*D;i+=256) sQ[i]=__float2bfloat16(Q[i]);
    for(int i=t;i<HQ*D;i+=256) O[i]=0.f; if(t<HQ){m_run[t]=-INFINITY;l_run[t]=0.f;}
    __syncthreads();
    for(int kb0=0;kb0<top_k;kb0+=KB){
        for(int r=warp;r<KB;r+=WARPS){
            int pos=kb0+r; int krow=0; bool valid=false;
            if(pos<top_k){ int id=idx_q[pos]; if(id>=0 && id<n_comp_view){ krow=id; valid=true; } }
            if(valid && (krow<0||krow>=n_kv)){ valid=false; krow=0; }
            if(lane==0) sValid[r]=valid?1:0;
            const bf16_t* kp=K+(size_t)krow*D;
            for(int d=lane;d<D;d+=32){ float kv=valid?bf16d(kp[d]):0.f; sK[r*D+d]=__float2bfloat16(kv);}
        }
        __syncthreads();
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
    // dense attention for one query Q[HQ*D] over the selected rows sel.
    std::vector<float> O((size_t)HQ*D,0.f); int nv=sel.size(); std::vector<float> Kf((size_t)nv*D);
    for(int j=0;j<nv;j++){const bf16_t* kr=&p.Kbf[(size_t)sel[j]*D];for(int d=0;d<D;d++)Kf[(size_t)j*D+d]=bf16_to_f32(kr[d]);}
    std::vector<float> s(nv);
    for(int h=0;h<HQ;h++){const float* q=&Q[(size_t)h*D];float mx=-INFINITY;
        for(int j=0;j<nv;j++){float a=0;const float* k=&Kf[(size_t)j*D];for(int d=0;d<D;d++)a+=q[d]*k[d];s[j]=a*p.scale;if(s[j]>mx)mx=s[j];}
        float sum=0;for(int j=0;j<nv;j++){s[j]=expf(s[j]-mx);sum+=s[j];}float inv=1.f/sum;float* o=&O[(size_t)h*D];
        for(int j=0;j<nv;j++){float w=s[j]*inv;const float* v=&Kf[(size_t)j*D];for(int d=0;d<D;d++)o[d]+=w*v[d];}}
    return O;
}

int main(){
    cudaSetDevice(0);
    int M = getenv("M")?atoi(getenv("M")):1024;           // prefill ubatch queries
    int n_comp_view = getenv("NCV")?atoi(getenv("NCV")):3000; // > top_k=512 (sparse engages)
    int top_k=512;
    int n_kv = ((n_comp_view+255)/256)*256;               // padded k_all length
    Problem p = make_problem(n_kv, 7);                     // cache rows
    int kv_row_elems = n_comp_view;                        // ggml argsort row stride = n_comp_view
    // Build per-token Q (distinct) + a NON-CONTIGUOUS kv_idx [n_comp_view, M]: each column it has its
    // own top_k selection in [0,n_comp_view), the rest of the row (top_k..n_comp_view) is argsort tail.
    std::vector<float> Qall((size_t)M*HQ*D);
    std::vector<int> kvbuf((size_t)kv_row_elems*M);
    std::mt19937 rng(123);
    std::vector<std::vector<int>> sels(M);
    for(int it=0; it<M; it++){
        for(size_t i=0;i<(size_t)HQ*D;i++) Qall[(size_t)it*HQ*D+i]=p.Q[i]* (1.0f+0.001f*it); // distinct-ish
        std::vector<int> all(n_comp_view); for(int i=0;i<n_comp_view;i++) all[i]=i;
        std::shuffle(all.begin(),all.end(),rng);
        for(int j=0;j<n_comp_view;j++) kvbuf[(size_t)it*kv_row_elems + j]=all[j];   // full argsort row
        sels[it].assign(all.begin(), all.begin()+top_k);                            // valid top_k
    }
    float *dQ,*dO; bf16_t* dK; int* dKv;
    cudaMalloc(&dQ,Qall.size()*4); cudaMalloc(&dO,(size_t)M*HQ*D*4); cudaMalloc(&dK,p.Kbf.size()*2); cudaMalloc(&dKv,kvbuf.size()*4);
    cudaMemcpy(dQ,Qall.data(),Qall.size()*4,cudaMemcpyHostToDevice);
    cudaMemcpy(dK,p.Kbf.data(),p.Kbf.size()*2,cudaMemcpyHostToDevice);
    cudaMemcpy(dKv,kvbuf.data(),kvbuf.size()*4,cudaMemcpyHostToDevice);
    size_t smem=(size_t)HQ*D*2+KB*D*2+HQ*KB*2+HQ*KB*4; cudaFuncSetAttribute(k4pf,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem);
    printf("PREFILL shape: M=%d queries, n_comp_view=%d, top_k=%d, n_kv(padded)=%d, kv_row_elems=%d\n",M,n_comp_view,top_k,n_kv,kv_row_elems);
    k4pf<<<M,256,smem>>>(dQ,dK,dKv,top_k,kv_row_elems,n_comp_view,n_kv,p.scale,dO,M);
    cudaError_t e=cudaDeviceSynchronize();
    printf("sync=%s\n",cudaGetErrorString(e)); if(e){printf("CRASH\n");return 1;}
    std::vector<float> out((size_t)M*HQ*D); cudaMemcpy(out.data(),dO,out.size()*4,cudaMemcpyDeviceToHost);
    // check cos for a few tokens (first, middle, last — last is where it*top_k would run off-buffer)
    double worst=1.0; int worst_it=-1;
    for(int it : {0, M/2, M-1}){
        std::vector<float> q(Qall.begin()+(size_t)it*HQ*D, Qall.begin()+(size_t)(it+1)*HQ*D);
        auto ref=dense_q(q,p,sels[it]);
        std::vector<float> o(out.begin()+(size_t)it*HQ*D, out.begin()+(size_t)(it+1)*HQ*D);
        double c=cosine(o,ref); printf("token %4d: cos=%.6f rel=%.3e\n",it,c,rel_err(o,ref));
        if(c<worst){worst=c;worst_it=it;}
    }
    printf("WORST cos=%.6f @ token %d  (>=0.999 = PASS)\n",worst,worst_it);
    return worst>=0.999 ? 0 : 2;
}
