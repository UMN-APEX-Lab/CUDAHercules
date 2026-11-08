#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>
// Full state-128 SSD, one shared Q/K group; deliberately simple CUDA baseline.
__global__ void mamba2_naive_kernel(const __nv_bfloat16* Q,const __nv_bfloat16* K,
    const __nv_bfloat16* V,__nv_bfloat16* O,const float* A,int H,int N) {
  int bh=blockIdx.x,b=bh/H;
  int row=blockIdx.y*blockDim.x+threadIdx.x;
  if(row>=N)return;
  float acc[64]={};float decay=1;
  for(int j=row;j>=0;--j) {
    float dot=0;
    for(int s=0;s<128;++s) dot+=float(Q[(size_t(b)*N+row)*128+s])*float(K[(size_t(b)*N+j)*128+s]);
    for(int d=0;d<64;++d) acc[d]+=dot*decay*float(V[(size_t(bh)*N+j)*64+d]);
    decay*=expf(A[size_t(bh)*N+j]);
  }
  for(int d=0;d<64;++d)O[(size_t(bh)*N+row)*64+d]=__float2bfloat16(acc[d]);
}
void Mamba2Forward(__nv_bfloat16* Q,__nv_bfloat16* K,__nv_bfloat16* V,
    __nv_bfloat16* O,float* A,int B,int H,int N) {
  mamba2_naive_kernel<<<dim3(B*H,(N+63)/64),64>>>(Q,K,V,O,A,H,N);
}
