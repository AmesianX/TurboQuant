// Shared reference + problem setup for the DSV4 sparse-MLA standalone kernel.
// Problem shape (MQA): 1 latent KV head, HQ=64 Q heads, D=512 head_dim,
// per query token: TOPK=512 selected compressed-KV rows (indices into the comp cache).
// Reference = dense QK^T over the SELECTED rows -> softmax -> P*V.  The model's
// designed sparsity IS the top-512, so sparse-over-selected == dense+(-inf mask
// over non-selected): the standalone gate compares the kernel against this reference.
#pragma once
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>
#include <random>
#include <algorithm>
#include <cstring>

#define HQ   64      // Q heads
#define D    512     // head dim (== latent dim, MLA mirrored K==V)
#define TOPK 512     // selected keys per query
typedef unsigned short bf16_t;

static inline float bf16_to_f32(bf16_t b){ uint32_t u=((uint32_t)b)<<16; float f; memcpy(&f,&u,4); return f; }
static inline bf16_t f32_to_bf16(float f){ uint32_t u; memcpy(&u,&f,4); uint32_t r=(u>>16); // round-to-nearest-even
    uint32_t round_bit=(u>>15)&1, sticky=(u&0x7fff)!=0; if(round_bit&&(sticky||(r&1))) r++; return (bf16_t)r; }

// FP8 E4M3 (no inf, max 448) encode/decode — matches DSV4 latent KV storage (B128 block scale).
static inline float fp8_e4m3_to_f32(uint8_t v){
    uint32_t s=(v>>7)&1, e=(v>>3)&0xF, m=v&0x7; float sign=s?-1.f:1.f;
    if(e==0){ if(m==0) return s?-0.f:0.f; return sign*ldexpf((float)m,-6-3+1)*4.f /*2^-9 * m*/ * (1.0f); }
    if(e==0xF && m==0x7) return sign* 448.0f; // S1111.111 = 448 in e4m3 (no inf)
    return sign*ldexpf(1.0f + (float)m/8.0f, (int)e-7);
}
static inline uint8_t f32_to_fp8_e4m3(float f){
    if(f==0) return 0; uint32_t s=f<0?1:0; float a=fabsf(f); if(a>448.f)a=448.f;
    int e; float m=frexpf(a,&e); // a = m*2^e, m in [0.5,1)
    // normalize to 1.x form: a = 1.f * 2^(E), E=e-1
    int E=e-1; float mant=a/ldexpf(1.0f,E); // in [1,2)
    if(E< -6){ // subnormal
        float scaled=a/ldexpf(1.0f,-6); int mm=(int)lroundf(scaled*8.0f); if(mm>7)mm=7;
        return (uint8_t)((s<<7)|(0<<3)|(mm&7));
    }
    int Eb=E+7; if(Eb>15)Eb=15;
    int mm=(int)lroundf((mant-1.0f)*8.0f); if(mm>7){mm=0;Eb++;} if(Eb>15){Eb=15;mm=7;}
    return (uint8_t)((s<<7)|((Eb&0xF)<<3)|(mm&7));
}

#define B128 128             // FP8 block-scale group size along D (D=512 => 4 blocks/row)
#define NBLK (D/B128)

struct Problem {
    int n_comp;              // total comp-cache rows (>= TOPK)
    std::vector<float>  Q;   // [HQ*D]
    std::vector<bf16_t> Kbf; // [n_comp*D]  (bf16 cache variant)
    std::vector<uint8_t> Kfp8;  // [n_comp*D]  FP8-E4M3 quantized cache (B128 block-scaled)
    std::vector<float>  Kscale; // [n_comp*NBLK]  per-(row,block) dequant scale
    std::vector<int>    idx; // [TOPK] selected absolute comp rows
    float scale;             // softmax scale 1/sqrt(D)
};

static Problem make_problem(int n_comp, unsigned seed){
    Problem p; p.n_comp=n_comp; p.scale=1.0f/sqrtf((float)D);
    std::mt19937 rng(seed); std::normal_distribution<float> nd(0.f,1.0f);
    p.Q.resize(HQ*D); for(auto&x:p.Q) x=nd(rng)*0.5f;
    p.Kbf.resize((size_t)n_comp*D);
    for(size_t i=0;i<p.Kbf.size();i++) p.Kbf[i]=f32_to_bf16(nd(rng)*0.5f);
    // FP8-E4M3 B128 block-scaled quantization of the SAME underlying values.
    p.Kfp8.resize((size_t)n_comp*D);
    p.Kscale.resize((size_t)n_comp*NBLK);
    for(int row=0; row<n_comp; row++){
        for(int blk=0; blk<NBLK; blk++){
            float amax=0; for(int c=0;c<B128;c++){ float v=bf16_to_f32(p.Kbf[(size_t)row*D+blk*B128+c]); amax=fmaxf(amax,fabsf(v)); }
            float sc = (amax>0)? amax/448.0f : 1.0f;          // scale so max maps to e4m3 max (448)
            p.Kscale[(size_t)row*NBLK+blk]=sc;
            for(int c=0;c<B128;c++){ float v=bf16_to_f32(p.Kbf[(size_t)row*D+blk*B128+c]);
                p.Kfp8[(size_t)row*D+blk*B128+c]=f32_to_fp8_e4m3(v/sc); }
        }
    }
    // distinct random TOPK rows
    std::vector<int> all(n_comp); for(int i=0;i<n_comp;i++) all[i]=i;
    std::shuffle(all.begin(),all.end(),rng); p.idx.assign(all.begin(),all.begin()+TOPK);
    return p;
}

// Reference: dense attention over the SELECTED rows (f32 throughout).
// O[h,d] = sum_j softmax_j( scale * Q[h,:]·K[idx[j],:] ) * V[idx[j],d],  V==K.
static std::vector<float> ref_attn(const Problem& p){
    std::vector<float> O((size_t)HQ*D,0.f);
    std::vector<float> Kf((size_t)TOPK*D);
    for(int j=0;j<TOPK;j++){ const bf16_t* kr=&p.Kbf[(size_t)p.idx[j]*D];
        for(int d=0;d<D;d++) Kf[(size_t)j*D+d]=bf16_to_f32(kr[d]); }
    std::vector<float> s(TOPK);
    for(int h=0;h<HQ;h++){
        const float* q=&p.Q[(size_t)h*D];
        float mx=-INFINITY;
        for(int j=0;j<TOPK;j++){ float acc=0; const float* k=&Kf[(size_t)j*D];
            for(int d=0;d<D;d++) acc+=q[d]*k[d]; s[j]=acc*p.scale; if(s[j]>mx)mx=s[j]; }
        float sum=0; for(int j=0;j<TOPK;j++){ s[j]=expf(s[j]-mx); sum+=s[j]; }
        float inv=1.0f/sum;
        float* o=&O[(size_t)h*D];
        for(int j=0;j<TOPK;j++){ float w=s[j]*inv; const float* v=&Kf[(size_t)j*D];
            for(int d=0;d<D;d++) o[d]+=w*v[d]; }
    }
    return O;
}

// Reference over the FP8-DEQUANTIZED selected rows (isolates kernel error from quant error).
static std::vector<float> ref_attn_fp8(const Problem& p){
    Problem q=p; // dequant FP8 -> bf16 cache, then reuse ref_attn
    for(int row=0; row<p.n_comp; row++)
        for(int blk=0; blk<NBLK; blk++){ float sc=p.Kscale[(size_t)row*NBLK+blk];
            for(int c=0;c<B128;c++){ float v=fp8_e4m3_to_f32(p.Kfp8[(size_t)row*D+blk*B128+c])*sc;
                q.Kbf[(size_t)row*D+blk*B128+c]=f32_to_bf16(v); } }
    return ref_attn(q);
}

static double cosine(const std::vector<float>&a, const std::vector<float>&b){
    double dot=0,na=0,nb=0; for(size_t i=0;i<a.size();i++){dot+=(double)a[i]*b[i];na+=(double)a[i]*a[i];nb+=(double)b[i]*b[i];}
    return dot/(sqrt(na)*sqrt(nb)+1e-30);
}
static double rel_err(const std::vector<float>&a, const std::vector<float>&b){
    double num=0,den=0; for(size_t i=0;i<a.size();i++){double d=(double)a[i]-b[i];num+=d*d;den+=(double)b[i]*b[i];}
    return sqrt(num/(den+1e-30));
}
