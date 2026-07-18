// Decode GEVM (dec_gate_up_swiglu_fused) bandwidth isolation bench.
// Variants: 0=current vec16, 1=no-SF (const scale), 2=no-ALU (raw uint4 sum),
//           3=ILP, 4=bit-trick half2, 5=4-way accum, 6=e2m1 LUT. RESULT: all ALU tricks fail —
// V0=73.6 GB/s-equiv is the scalar-FFMA compute floor; V2 no-ALU=246 is the bandwidth ceiling.
// 3.3x gap = instruction throughput (per-nibble dequant+FFMA), NOT bandwidth/latency/accum-chain.
// bit-trick(37) + LUT(35) SLOWER (LDS/half-rate). Only fix = tensor-core MMA (poor M=1 util).
//   nvcc -arch=sm_121a -O3 turboquant/dsv4_dec_gevm_bench.cu -o /tmp/gevm && /tmp/gevm
// Report: effective GB/s over the fc1 weight bytes actually read (gate+up rows).
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>

#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
  fprintf(stderr,"CUDA err %s @ %d\n",cudaGetErrorString(e),__LINE__); exit(1);} }while(0)

__device__ __forceinline__ float e2m1_decode(uint8_t nib){
  const uint32_t e = (nib >> 1) & 0x3;
  const uint32_t m =  nib       & 0x1;
  const uint32_t norm = (((e - 1u) + 127u) << 23) | (m << 22);
  const uint32_t sub  = m ? 0x3F000000u : 0u;
  uint32_t bits = (e != 0u) ? norm : sub;
  bits |= ((uint32_t) (nib & 0x8)) << 28;
  float f; memcpy(&f,&bits,4); return f;
}
// ue4m3 (bench-equivalent instruction cost)
__device__ __forceinline__ float ue4m3_decode(uint8_t s){
  const uint32_t e = (s >> 3) & 0xF;
  const uint32_t m =  s       & 0x7;
  uint32_t bits = e ? (((e - 7u + 127u) << 23) | (m << 20)) : (m << 20);
  float f; memcpy(&f,&bits,4); return f;
}
__device__ __forceinline__ int dsv4_sf_swizzled_index(int rowIdx, int colIdx, int totalColumn){
  int paddedColumn = ((totalColumn + 3) / 4) * 4;
  int columnIdxInGroup0 = colIdx % 4;
  int columnGroupIdx    = colIdx / 4;
  int rowIdxInGroup0 = rowIdx % 32;
  int rowIdxInGroup1 = (rowIdx % 128) / 32;
  int rowGroupIdx    = rowIdx / 128;
  return columnIdxInGroup0 + columnGroupIdx * 512
       + rowIdxInGroup0 * 16 + rowIdxInGroup1 * 4
       + rowGroupIdx * 128 * paddedColumn;
}
__device__ __forceinline__ float warp_reduce_sum(float v){
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

// variant via template param
template<int V>
__global__ void dec_fc1_bench(
    const float* __restrict__ hidden, const int* __restrict__ sel,
    const uint8_t* __restrict__ fc1_w, const uint8_t* __restrict__ fc1_sf,
    const float* __restrict__ g_common,
    float* __restrict__ act, int M, int U, int D, int F, int inter,
    int sf1_stride, int sf1_cols, int E_local){
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int j = blockIdx.x * (blockDim.x>>5) + warp;
  const int row = blockIdx.y;
  if(j >= F || row >= M*U) return;
  const int token = row / U;
  const int e = sel[row];
  if(e < 0 || e >= E_local){ act[(int64_t)row*F + j] = 0.f; return; }
  const float* __restrict__ x = hidden + (int64_t)token * D;
  const int nbytes = D >> 1;
  const int64_t wbase = (int64_t)e * (2*(int64_t)inter) * nbytes;
  const int64_t sfbase = (int64_t)e * sf1_stride;
  const float gc = g_common[e];
  const uint8_t* __restrict__ wur = fc1_w + wbase + (int64_t)j * nbytes;
  const uint8_t* __restrict__ wgr = fc1_w + wbase + (int64_t)(inter+j) * nbytes;
  float accg = 0.f, accu = 0.f;
  const int n16 = nbytes >> 4;
  if (V == 2) {
    // no-ALU: raw uint4 sum of both rows (pure weight-stream ceiling)
    uint4 s = {0,0,0,0};
    for (int p = lane; p < n16; p += 32) {
      const uint4 a = reinterpret_cast<const uint4*>(wgr)[p];
      const uint4 b = reinterpret_cast<const uint4*>(wur)[p];
      s.x += a.x + b.x; s.y += a.y + b.y; s.z += a.z + b.z; s.w += a.w + b.w;
    }
    accg = (float)(s.x + s.y + s.z + s.w);
    accu = 0.f;
  } else if (V == 3) {
    // 2x uint4 per lane per iter: lane covers p and p+32 in one iter (deeper MLP)
    for (int p0 = lane; p0 < n16; p0 += 64) {
      const int p1 = p0 + 32;
      uint4 g4a = reinterpret_cast<const uint4*>(wgr)[p0];
      uint4 u4a = reinterpret_cast<const uint4*>(wur)[p0];
      uint4 g4b = (p1<n16)? reinterpret_cast<const uint4*>(wgr)[p1] : make_uint4(0,0,0,0);
      uint4 u4b = (p1<n16)? reinterpret_cast<const uint4*>(wur)[p1] : make_uint4(0,0,0,0);
      #pragma unroll
      for (int h = 0; h < 2; h++) {
        const uint4 wg4 = h? g4b : g4a;
        const uint4 wu4 = h? u4b : u4a;
        const int p = h? p1 : p0;
        if (p >= n16) break;
        const int c0 = 32*p, blk0 = c0 >> 4;
        const float sg0 = ue4m3_decode(fc1_sf[sfbase + dsv4_sf_swizzled_index(inter+j, blk0,   sf1_cols)]) * gc;
        const float su0 = ue4m3_decode(fc1_sf[sfbase + dsv4_sf_swizzled_index(j,       blk0,   sf1_cols)]) * gc;
        const float sg1 = ue4m3_decode(fc1_sf[sfbase + dsv4_sf_swizzled_index(inter+j, blk0+1, sf1_cols)]) * gc;
        const float su1 = ue4m3_decode(fc1_sf[sfbase + dsv4_sf_swizzled_index(j,       blk0+1, sf1_cols)]) * gc;
        const uint32_t gw[4] = { wg4.x, wg4.y, wg4.z, wg4.w };
        const uint32_t uw[4] = { wu4.x, wu4.y, wu4.z, wu4.w };
        #pragma unroll
        for (int q = 0; q < 4; q++) {
          const float sg = (q < 2) ? sg0 : sg1;
          const float su = (q < 2) ? su0 : su1;
          #pragma unroll
          for (int t = 0; t < 4; t++) {
            const uint8_t bg = (uint8_t)(gw[q] >> (8*t));
            const uint8_t bu = (uint8_t)(uw[q] >> (8*t));
            const int c = c0 + 8*q + 2*t;
            const float x0 = x[c], x1 = x[c+1];
            accg += e2m1_decode(bg & 0xF) * sg * x0 + e2m1_decode(bg >> 4) * sg * x1;
            accu += e2m1_decode(bu & 0xF) * su * x0 + e2m1_decode(bu >> 4) * su * x1;
          }
        }
      }
    }
  } else {
    // V0 current vec16 / V1 no-SF
    for (int p = lane; p < n16; p += 32) {
      const uint4 wg4 = reinterpret_cast<const uint4*>(wgr)[p];
      const uint4 wu4 = reinterpret_cast<const uint4*>(wur)[p];
      const int c0 = 32*p, blk0 = c0 >> 4;
      float sg0, su0, sg1, su1;
      if (V == 1) { sg0=su0=sg1=su1=gc; }
      else {
        sg0 = ue4m3_decode(fc1_sf[sfbase + dsv4_sf_swizzled_index(inter+j, blk0,   sf1_cols)]) * gc;
        su0 = ue4m3_decode(fc1_sf[sfbase + dsv4_sf_swizzled_index(j,       blk0,   sf1_cols)]) * gc;
        sg1 = ue4m3_decode(fc1_sf[sfbase + dsv4_sf_swizzled_index(inter+j, blk0+1, sf1_cols)]) * gc;
        su1 = ue4m3_decode(fc1_sf[sfbase + dsv4_sf_swizzled_index(j,       blk0+1, sf1_cols)]) * gc;
      }
      const uint32_t gw[4] = { wg4.x, wg4.y, wg4.z, wg4.w };
      const uint32_t uw[4] = { wu4.x, wu4.y, wu4.z, wu4.w };
      #pragma unroll
      for (int q = 0; q < 4; q++) {
        const float sg = (q < 2) ? sg0 : sg1;
        const float su = (q < 2) ? su0 : su1;
        #pragma unroll
        for (int t = 0; t < 4; t++) {
          const uint8_t bg = (uint8_t)(gw[q] >> (8*t));
          const uint8_t bu = (uint8_t)(uw[q] >> (8*t));
          const int c = c0 + 8*q + 2*t;
          const float x0 = x[c], x1 = x[c+1];
          accg += e2m1_decode(bg & 0xF) * sg * x0 + e2m1_decode(bg >> 4) * sg * x1;
          accu += e2m1_decode(bu & 0xF) * su * x0 + e2m1_decode(bu >> 4) * su * x1;
        }
      }
    }
  }
  if (V == 4) {
    // bit-trick: byte(2 nibbles) -> f16x2 prescale in ~3 ops (vs per-nibble float rebuild),
    // half2 fma chain; scale/x folded once per SF block. Prescale const 2^-14 folded into scale.
    for (int p = lane; p < n16; p += 32) {
      const uint4 wg4 = reinterpret_cast<const uint4*>(wgr)[p];
      const uint4 wu4 = reinterpret_cast<const uint4*>(wur)[p];
      const int c0 = 32*p, blk0 = c0 >> 4;
      const float sg0 = ue4m3_decode(fc1_sf[sfbase + dsv4_sf_swizzled_index(inter+j, blk0,   sf1_cols)]) * gc;
      const float su0 = ue4m3_decode(fc1_sf[sfbase + dsv4_sf_swizzled_index(j,       blk0,   sf1_cols)]) * gc;
      const float sg1 = ue4m3_decode(fc1_sf[sfbase + dsv4_sf_swizzled_index(inter+j, blk0+1, sf1_cols)]) * gc;
      const float su1 = ue4m3_decode(fc1_sf[sfbase + dsv4_sf_swizzled_index(j,       blk0+1, sf1_cols)]) * gc;
      const uint32_t gw[4] = { wg4.x, wg4.y, wg4.z, wg4.w };
      const uint32_t uw[4] = { wu4.x, wu4.y, wu4.z, wu4.w };
      __half2 ag = __float2half2_rn(0.f), au = __float2half2_rn(0.f);
      #pragma unroll
      for (int q = 0; q < 4; q++) {
        #pragma unroll
        for (int t = 0; t < 4; t++) {
          const uint8_t bg = (uint8_t)(gw[q] >> (8*t));
          const uint8_t bu = (uint8_t)(uw[q] >> (8*t));
          const int c = c0 + 8*q + 2*t;
          // byte -> f16x2 prescale (FP4 * 2^-14): place nibbles at [12:15], mask+shr3
          uint32_t ig = ((uint32_t)(bg & 0xF) << 12) | ((uint32_t)(bg >> 4) << 28);
          uint32_t iu = ((uint32_t)(bu & 0xF) << 12) | ((uint32_t)(bu >> 4) << 28);
          ig = (ig & 0x80008000u) | ((ig & 0x70007000u) >> 3);
          iu = (iu & 0x80008000u) | ((iu & 0x70007000u) >> 3);
          __half2 hg = *reinterpret_cast<__half2*>(&ig);
          __half2 hu = *reinterpret_cast<__half2*>(&iu);
          __half2 xx = __floats2half2_rn(x[c], x[c+1]);
          ag = __hfma2(hg, xx, ag);
          au = __hfma2(hu, xx, au);
        }
      }
      // NOTE: microbench applies a single block scale approximation (sg0/su0) — enough to
      // measure the ALU/bandwidth of the dequant path, not for numerical parity.
      accg += (__low2float(ag)+__high2float(ag)) * ldexpf(sg0,14);
      accu += (__low2float(au)+__high2float(au)) * ldexpf(su0,14);
    }
  }
  if (V == 5) {
    // 4-way accumulator split to break the FMA dependency chain (the real bottleneck: V0/V3
    // pin at 73 GB/s while the no-ALU stream ceiling is 246). Each of the 4 uint32 words in a
    // uint4 feeds its own accg[w]/accu[w]; independent chains hide FMA latency.
    float ag[4] = {0,0,0,0}, au[4] = {0,0,0,0};
    for (int p = lane; p < n16; p += 32) {
      const uint4 wg4 = reinterpret_cast<const uint4*>(wgr)[p];
      const uint4 wu4 = reinterpret_cast<const uint4*>(wur)[p];
      const int c0 = 32*p, blk0 = c0 >> 4;
      const float sg0 = ue4m3_decode(fc1_sf[sfbase + dsv4_sf_swizzled_index(inter+j, blk0,   sf1_cols)]) * gc;
      const float su0 = ue4m3_decode(fc1_sf[sfbase + dsv4_sf_swizzled_index(j,       blk0,   sf1_cols)]) * gc;
      const float sg1 = ue4m3_decode(fc1_sf[sfbase + dsv4_sf_swizzled_index(inter+j, blk0+1, sf1_cols)]) * gc;
      const float su1 = ue4m3_decode(fc1_sf[sfbase + dsv4_sf_swizzled_index(j,       blk0+1, sf1_cols)]) * gc;
      const uint32_t gw[4] = { wg4.x, wg4.y, wg4.z, wg4.w };
      const uint32_t uw[4] = { wu4.x, wu4.y, wu4.z, wu4.w };
      #pragma unroll
      for (int q = 0; q < 4; q++) {
        const float sg = (q < 2) ? sg0 : sg1;
        const float su = (q < 2) ? su0 : su1;
        #pragma unroll
        for (int t = 0; t < 4; t++) {
          const uint8_t bg = (uint8_t)(gw[q] >> (8*t));
          const uint8_t bu = (uint8_t)(uw[q] >> (8*t));
          const int c = c0 + 8*q + 2*t;
          const float x0 = x[c], x1 = x[c+1];
          ag[q] += e2m1_decode(bg & 0xF) * sg * x0 + e2m1_decode(bg >> 4) * sg * x1;
          au[q] += e2m1_decode(bu & 0xF) * su * x0 + e2m1_decode(bu >> 4) * su * x1;
        }
      }
    }
    accg = (ag[0]+ag[1])+(ag[2]+ag[3]);
    accu = (au[0]+au[1])+(au[2]+au[3]);
  }
  if (V == 6) {
    // instruction-reduction: 16-entry e2m1 LUT in shared mem (1 LDS vs ~5 ALU per nibble).
    // The dequant ALU, not bandwidth (246 ceiling) or accumulator chains, pins V0 at 73 GB/s.
    __shared__ float lut[16];
    if (threadIdx.x < 16) {
      const uint8_t nib = threadIdx.x;
      const uint32_t e = (nib >> 1) & 0x3, m = nib & 0x1;
      const uint32_t norm = (((e - 1u) + 127u) << 23) | (m << 22);
      const uint32_t sub  = m ? 0x3F000000u : 0u;
      uint32_t bits = (e != 0u) ? norm : sub; bits |= ((uint32_t)(nib & 0x8)) << 28;
      float f; memcpy(&f,&bits,4); lut[nib]=f;
    }
    __syncthreads();
    for (int p = lane; p < n16; p += 32) {
      const uint4 wg4 = reinterpret_cast<const uint4*>(wgr)[p];
      const uint4 wu4 = reinterpret_cast<const uint4*>(wur)[p];
      const int c0 = 32*p, blk0 = c0 >> 4;
      const float sg0 = ue4m3_decode(fc1_sf[sfbase + dsv4_sf_swizzled_index(inter+j, blk0,   sf1_cols)]) * gc;
      const float su0 = ue4m3_decode(fc1_sf[sfbase + dsv4_sf_swizzled_index(j,       blk0,   sf1_cols)]) * gc;
      const float sg1 = ue4m3_decode(fc1_sf[sfbase + dsv4_sf_swizzled_index(inter+j, blk0+1, sf1_cols)]) * gc;
      const float su1 = ue4m3_decode(fc1_sf[sfbase + dsv4_sf_swizzled_index(j,       blk0+1, sf1_cols)]) * gc;
      const uint32_t gw[4] = { wg4.x, wg4.y, wg4.z, wg4.w };
      const uint32_t uw[4] = { wu4.x, wu4.y, wu4.z, wu4.w };
      #pragma unroll
      for (int q = 0; q < 4; q++) {
        const float sg = (q < 2) ? sg0 : sg1;
        const float su = (q < 2) ? su0 : su1;
        #pragma unroll
        for (int t = 0; t < 4; t++) {
          const uint8_t bg = (uint8_t)(gw[q] >> (8*t));
          const uint8_t bu = (uint8_t)(uw[q] >> (8*t));
          const int c = c0 + 8*q + 2*t;
          const float x0 = x[c], x1 = x[c+1];
          accg += lut[bg & 0xF] * sg * x0 + lut[bg >> 4] * sg * x1;
          accu += lut[bu & 0xF] * su * x0 + lut[bu >> 4] * su * x1;
        }
      }
    }
  }
  accg = warp_reduce_sum(accg); accu = warp_reduce_sum(accu);
  if(lane == 0){
    float g = accg, u = accu;
    act[(int64_t)row*F + j] = (g/(1.f+expf(-g))) * u;
  }
}

int main(int argc, char** argv){
  const int D=4096, F=2048, inter=2048;
  const int M=1, U=8, E_local=4;      // decode: 1 token, top-8 slots, ~4 local experts (EP=2)
  const int iters = (argc>1)? atoi(argv[1]) : 200;
  const int nbytes = D/2;
  const int64_t wsz = (int64_t)E_local * 2*inter * nbytes;
  const int colsD = D/16;
  auto padUp=[](int x,int a){ return (x+a-1)/a*a; };
  const int sf1_stride = padUp(2*F,128)*padUp(colsD,4);
  uint8_t *fc1_w, *fc1_sf; float *hid, *gcm, *act; int *sel;
  CK(cudaMalloc(&fc1_w, wsz));
  CK(cudaMalloc(&fc1_sf, (int64_t)E_local*sf1_stride));
  CK(cudaMalloc(&hid, (int64_t)M*D*4));
  CK(cudaMalloc(&gcm, E_local*4));
  CK(cudaMalloc(&act, (int64_t)M*U*F*4));
  CK(cudaMalloc(&sel, M*U*4));
  // init: nonzero patterns
  {
    uint8_t* h = (uint8_t*)malloc(wsz); for(int64_t i=0;i<wsz;i++) h[i]=(uint8_t)((i*37+11)&0xFF);
    CK(cudaMemcpy(fc1_w,h,wsz,cudaMemcpyHostToDevice)); free(h);
    int hsel[8] = {0,1,2,3,0,1,2,3};  // 8 slots over 4 local experts (worst-ish spread)
    CK(cudaMemcpy(sel,hsel,sizeof hsel,cudaMemcpyHostToDevice));
    float hx[4096]; for(int i=0;i<4096;i++) hx[i]=0.001f*((i%7)-3);
    CK(cudaMemcpy(hid,hx,sizeof hx,cudaMemcpyHostToDevice));
    float hg[4]={1e-4f,1e-4f,1e-4f,1e-4f}; CK(cudaMemcpy(gcm,hg,sizeof hg,cudaMemcpyHostToDevice));
    CK(cudaMemset(fc1_sf,0x40,(int64_t)E_local*sf1_stride));
  }
  const int WPB=4, blk=WPB*32;
  dim3 grid((F+WPB-1)/WPB, M*U);
  // bytes actually read per launch: rows*2 weight rows of nbytes (gate+up) per output row j
  const double bytes = (double)M*U*F*2*nbytes;
  cudaEvent_t e0,e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  auto run=[&](int v, const char* name){
    for(int w=0;w<20;w++){ // warmup
      switch(v){
        case 0: dec_fc1_bench<0><<<grid,blk>>>(hid,sel,fc1_w,fc1_sf,gcm,act,M,U,D,F,inter,sf1_stride,colsD,E_local); break;
        case 1: dec_fc1_bench<1><<<grid,blk>>>(hid,sel,fc1_w,fc1_sf,gcm,act,M,U,D,F,inter,sf1_stride,colsD,E_local); break;
        case 2: dec_fc1_bench<2><<<grid,blk>>>(hid,sel,fc1_w,fc1_sf,gcm,act,M,U,D,F,inter,sf1_stride,colsD,E_local); break;
        case 3: dec_fc1_bench<3><<<grid,blk>>>(hid,sel,fc1_w,fc1_sf,gcm,act,M,U,D,F,inter,sf1_stride,colsD,E_local); break;
        case 4: dec_fc1_bench<4><<<grid,blk>>>(hid,sel,fc1_w,fc1_sf,gcm,act,M,U,D,F,inter,sf1_stride,colsD,E_local); break;
        case 5: dec_fc1_bench<5><<<grid,blk>>>(hid,sel,fc1_w,fc1_sf,gcm,act,M,U,D,F,inter,sf1_stride,colsD,E_local); break;
        case 6: dec_fc1_bench<6><<<grid,blk>>>(hid,sel,fc1_w,fc1_sf,gcm,act,M,U,D,F,inter,sf1_stride,colsD,E_local); break;
      }
    }
    CK(cudaEventRecord(e0));
    for(int i=0;i<iters;i++){
      switch(v){
        case 0: dec_fc1_bench<0><<<grid,blk>>>(hid,sel,fc1_w,fc1_sf,gcm,act,M,U,D,F,inter,sf1_stride,colsD,E_local); break;
        case 1: dec_fc1_bench<1><<<grid,blk>>>(hid,sel,fc1_w,fc1_sf,gcm,act,M,U,D,F,inter,sf1_stride,colsD,E_local); break;
        case 2: dec_fc1_bench<2><<<grid,blk>>>(hid,sel,fc1_w,fc1_sf,gcm,act,M,U,D,F,inter,sf1_stride,colsD,E_local); break;
        case 3: dec_fc1_bench<3><<<grid,blk>>>(hid,sel,fc1_w,fc1_sf,gcm,act,M,U,D,F,inter,sf1_stride,colsD,E_local); break;
        case 4: dec_fc1_bench<4><<<grid,blk>>>(hid,sel,fc1_w,fc1_sf,gcm,act,M,U,D,F,inter,sf1_stride,colsD,E_local); break;
        case 5: dec_fc1_bench<5><<<grid,blk>>>(hid,sel,fc1_w,fc1_sf,gcm,act,M,U,D,F,inter,sf1_stride,colsD,E_local); break;
        case 6: dec_fc1_bench<6><<<grid,blk>>>(hid,sel,fc1_w,fc1_sf,gcm,act,M,U,D,F,inter,sf1_stride,colsD,E_local); break;
      }
    }
    CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
    float ms=0; CK(cudaEventElapsedTime(&ms,e0,e1));
    const double gbs = bytes*iters/(ms*1e-3)/1e9;
    printf("V%d %-18s: %8.3f us/launch  %7.1f GB/s (weights only)\n", v, name, ms*1000.0/iters, gbs);
  };
  run(0,"current vec16");
  run(1,"no-SF");
  run(2,"no-ALU raw sum");
  run(3,"2x uint4 ILP");
  run(4,"bit-trick half2");
  run(5,"4-way accum");
  run(6,"e2m1 LUT");
  CK(cudaGetLastError());
  return 0;
}
