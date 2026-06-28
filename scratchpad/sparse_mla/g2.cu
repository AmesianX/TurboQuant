#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
// gather4 needs: smem dst 128B aligned, bar 8B aligned, tile box = {tile_cols, 1} with 4 rows gathered.
// Use COLS=32 (128 bytes/row for f32) so each gathered row is a full 128B TMA line.
__global__ void gatherKernel(const __grid_constant__ CUtensorMap tmap, float* out, int cols) {
    __shared__ alignas(128) float dst[4*32];
    __shared__ alignas(8) uint64_t bar;
    if (threadIdx.x == 0) {
        uint32_t dst_s = (uint32_t)__cvta_generic_to_shared(dst);
        uint32_t bar_s = (uint32_t)__cvta_generic_to_shared(&bar);
        asm volatile("mbarrier.init.shared.b64 [%0], 1;" :: "r"(bar_s));
        asm volatile("fence.proxy.async.shared::cta;");
        int c0=0,r0=0,r1=2,r2=4,r3=6;
        asm volatile(
            "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes "
            "[%0], [%1, {%2,%3,%4,%5,%6}], [%7];"
            :: "r"(dst_s), "l"(&tmap), "r"(c0),"r"(r0),"r"(r1),"r"(r2),"r"(r3), "r"(bar_s) : "memory");
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(bar_s), "r"((uint32_t)(4*cols*4)));
        uint32_t ph=0;
        asm volatile("{ .reg .pred p; LAB: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1; @!p bra LAB; }" :: "r"(bar_s), "r"(ph));
    }
    __syncthreads();
    if ((int)threadIdx.x < 4*cols) out[threadIdx.x] = dst[threadIdx.x];
}
int main(){
    cudaSetDevice(0); cudaDeviceProp p; cudaGetDeviceProperties(&p,0);
    printf("GPU %s sm_%d%d\n",p.name,p.major,p.minor);
    const int ROWS=8, COLS=32;
    float h[ROWS*COLS]; for(int i=0;i<ROWS*COLS;i++) h[i]=(float)i;
    float* d; cudaMalloc(&d,sizeof(h)); cudaMemcpy(d,h,sizeof(h),cudaMemcpyHostToDevice);
    CUtensorMap tmap{};
    uint64_t dims[2]={(uint64_t)COLS,(uint64_t)ROWS};   // {width=cols, height=rows}
    uint64_t strides[1]={(uint64_t)COLS*sizeof(float)};
    uint32_t boxdim[2]={(uint32_t)COLS,1};              // gather4: box height implicitly 4
    uint32_t elemstr[2]={1,1};
    CUresult r=cuTensorMapEncodeTiled(&tmap,CU_TENSOR_MAP_DATA_TYPE_FLOAT32,2,d,dims,strides,boxdim,elemstr,
        CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_NONE,CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if(r){const char*s;cuGetErrorString(r,&s);printf("encode FAIL %s\n",s);return 2;}
    float* dout; cudaMalloc(&dout,4*COLS*4);
    gatherKernel<<<1,128>>>(tmap,dout,COLS);
    cudaError_t e=cudaDeviceSynchronize();
    printf("sync: %s\n",cudaGetErrorString(e));
    if(!e){float o[128];cudaMemcpy(o,dout,4*COLS*4,cudaMemcpyDeviceToHost);
      printf("row0[0..3]=%.0f %.0f %.0f %.0f  row1(=src row2)[0]=%.0f  row2(=row4)[0]=%.0f  row3(=row6)[0]=%.0f\n",
        o[0],o[1],o[2],o[3],o[COLS],o[2*COLS],o[3*COLS]);
      printf("EXPECT row starts: 0, 64, 128, 192\n");}
    return e?1:0;
}
