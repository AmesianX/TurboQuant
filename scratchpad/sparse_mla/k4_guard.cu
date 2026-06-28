// Guard test: n_comp < top_k case (the in-graph crash scenario). top_k slots where only n_valid
// are real comp rows in [0,n_comp_view); the rest are PADDING indices (>= n_comp_view, like a tiny
// argsort over few rows padded out). The kernel must (a) NEVER read k_all OOB, (b) mask invalid
// slots to -inf so the result == dense attention over ONLY the n_valid real keys.
// Mirrors fattn-sparse-mla.cu's validity guard (idx in [0,n_comp_view) AND krow in [0,n_kv)).
#include "ref.h"
#include <mma.h>
#include <cuda_fp8.h>
using namespace nvcuda;
#define KB 16
#define WARPS 8
#define NM (HQ/16)
#define NN (KB/16)
#define ND (D/16)
__device__ __forceinline__ float f8(uint8_t v){__nv_fp8_e4m3 f;memcpy(&f,&v,1);return (float)f;}
__device__ __forceinline__ float bf16d(bf16_t b){ uint32_t u=((uint32_t)b)<<16; float f; memcpy(&f,&u,4); return f; }
extern __shared__ char smem[];

// Simplified standalone mirror of the in-graph kernel's gather+guard (no raw window: n_raw=0).
// K cache is bf16 here (option-a path gathers bf16 from k_all). idx may contain OOB padding.
__global__ void k4g(const float* Q, const bf16_t* K, const int* idx, int top_k, int n_comp_view,
                    int n_kv, float scale, float* O){
    const int t=threadIdx.x,warp=t>>5,lane=t&31;
    __nv_bfloat16* sQ=(__nv_bfloat16*)smem; __nv_bfloat16* sK=sQ+HQ*D; __nv_bfloat16* sP=sK+KB*D; float* sS=(float*)(sP+HQ*KB);
    __shared__ float m_run[HQ],l_run[HQ]; __shared__ float wscratch[WARPS][256]; __shared__ int sValid[KB];
    for(int i=t;i<HQ*D;i+=256) sQ[i]=__float2bfloat16(Q[i]);
    for(int i=t;i<HQ*D;i+=256) O[i]=0.f; if(t<HQ){m_run[t]=-INFINITY;l_run[t]=0.f;}
    __syncthreads();
    for(int kb0=0;kb0<top_k;kb0+=KB){
        for(int r=warp;r<KB;r+=WARPS){
            int pos=kb0+r; int krow=0; bool valid=false;
            if(pos<top_k){ int id=idx[pos]; if(id>=0 && id<n_comp_view){ krow=id; valid=true; } }
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

// reference: dense attention over the n_valid REAL selected rows only.
static std::vector<float> ref_valid(const Problem& p, const std::vector<int>& sel){
    std::vector<float> O((size_t)HQ*D,0.f); int nv=sel.size();
    std::vector<float> Kf((size_t)nv*D);
    for(int j=0;j<nv;j++){const bf16_t* kr=&p.Kbf[(size_t)sel[j]*D];for(int d=0;d<D;d++)Kf[(size_t)j*D+d]=bf16_to_f32(kr[d]);}
    std::vector<float> s(nv);
    for(int h=0;h<HQ;h++){const float* q=&p.Q[(size_t)h*D];float mx=-INFINITY;
        for(int j=0;j<nv;j++){float acc=0;const float* k=&Kf[(size_t)j*D];for(int d=0;d<D;d++)acc+=q[d]*k[d];s[j]=acc*p.scale;if(s[j]>mx)mx=s[j];}
        float sum=0;for(int j=0;j<nv;j++){s[j]=expf(s[j]-mx);sum+=s[j];}float inv=1.f/sum;float* o=&O[(size_t)h*D];
        for(int j=0;j<nv;j++){float w=s[j]*inv;const float* v=&Kf[(size_t)j*D];for(int d=0;d<D;d++)o[d]+=w*v[d];}}
    return O;
}

int main(){
    cudaSetDevice(0);
    int n_valid = getenv("NV")?atoi(getenv("NV")):7;   // tiny visible comp count (decode short ctx)
    int n_comp_view = n_valid;                          // valid index range [0, n_comp_view)
    // k_all comp segment is padded; n_kv = padded length. Allocate that many rows.
    int n_kv = ((n_comp_view+255)/256)*256; if(n_kv<256)n_kv=256;
    Problem p = make_problem(n_kv, 4242);               // cache rows present up to n_kv
    // top_k=512 slots: first n_valid are real distinct rows in [0,n_comp_view); rest are PADDING
    // indices that are OUT OF RANGE (n_comp_view .. n_kv-1 and even >= n_kv) — must be guarded.
    int top_k=512; std::vector<int> idx(top_k);
    std::vector<int> sel;
    for(int j=0;j<top_k;j++){
        if(j<n_valid){ idx[j]=j%n_comp_view; sel.push_back(idx[j]); }
        else idx[j] = n_comp_view + (j*131)%(n_kv*2);   // padding: in [n_comp_view, 2*n_kv) -> many >= n_kv (OOB) too
    }
    auto ref = ref_valid(p, sel);
    float *dQ,*dO;bf16_t* dK;int* dI;
    cudaMalloc(&dQ,p.Q.size()*4);cudaMalloc(&dO,(size_t)HQ*D*4);cudaMalloc(&dK,p.Kbf.size()*2);cudaMalloc(&dI,top_k*4);
    cudaMemcpy(dQ,p.Q.data(),p.Q.size()*4,cudaMemcpyHostToDevice);cudaMemcpy(dK,p.Kbf.data(),p.Kbf.size()*2,cudaMemcpyHostToDevice);cudaMemcpy(dI,idx.data(),top_k*4,cudaMemcpyHostToDevice);
    size_t smem=(size_t)HQ*D*2+KB*D*2+HQ*KB*2+HQ*KB*4; cudaFuncSetAttribute(k4g,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem);
    printf("n_valid=%d n_comp_view=%d n_kv=%d top_k=%d (padding idx mostly >= n_comp_view, many >= n_kv)\n",n_valid,n_comp_view,n_kv,top_k);
    k4g<<<1,256,smem>>>(dQ,dK,dI,top_k,n_comp_view,n_kv,p.scale,dO);
    cudaError_t e=cudaDeviceSynchronize();
    printf("sync=%s\n",cudaGetErrorString(e)); if(e){printf("CRASH (guard failed)\n");return 1;}
    std::vector<float> out((size_t)HQ*D);cudaMemcpy(out.data(),dO,out.size()*4,cudaMemcpyDeviceToHost);
    printf("cos(vs dense over %d valid keys)=%.6f  rel_err=%.3e\n",n_valid,cosine(out,ref),rel_err(out,ref));
    return 0;
}
