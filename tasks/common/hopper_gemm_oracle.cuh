#pragma once
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
template<class A,class B> __global__ void kh_convert(const A* src,B* dst,size_t n) {
  size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i<n) dst[i]=B(float(src[i]));
}
// Untimed independent oracle, including explicit input decoding and output rounding.
template<class I,class O,bool TransposeB>
void kh_gemm_oracle(O* out,const I* a,const I* b,int M,int N,int K) {
  float *af,*bf,*cf;
  cudaMalloc(&af,size_t(M)*K*4);cudaMalloc(&bf,size_t(N)*K*4);cudaMalloc(&cf,size_t(M)*N*4);
  kh_convert<<<(size_t(M)*K+255)/256,256>>>(a,af,size_t(M)*K);
  kh_convert<<<(size_t(N)*K+255)/256,256>>>(b,bf,size_t(N)*K);
  cublasHandle_t handle;cublasCreate(&handle);cublasSetMathMode(handle,CUBLAS_PEDANTIC_MATH);
  float alpha=1,beta=0;
  auto err=cublasSgemm(handle,TransposeB?CUBLAS_OP_T:CUBLAS_OP_N,CUBLAS_OP_N,N,M,K,
      &alpha,bf,TransposeB?K:N,af,K,&beta,cf,N);
  if(err!=CUBLAS_STATUS_SUCCESS) {fprintf(stderr,"cuBLAS oracle: %d\n",int(err));exit(1);}
  kh_convert<<<(size_t(M)*N+255)/256,256>>>(cf,out,size_t(M)*N);
  auto status=cudaDeviceSynchronize();
  cublasDestroy(handle);cudaFree(af);cudaFree(bf);cudaFree(cf);
  if(status!=cudaSuccess) {fprintf(stderr,"oracle: %s\n",cudaGetErrorString(status));exit(1);}
}
