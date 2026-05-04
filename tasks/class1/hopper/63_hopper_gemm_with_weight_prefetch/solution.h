#pragma once
#include <cuda_runtime.h>
#include <cuda_fp8.h>

// Naive FP8 GEMM: A row-major E4M3 [M,K], B col-major E5M2 [K,N], D col-major E4M3 [M,N]
__global__ void naive_fp8_gemm_prefetch_kernel(
    int M, int N, int K, float alpha,
    __nv_fp8_e4m3 const *A, int lda,
    __nv_fp8_e5m2 const *B, int ldb,
    float beta,
    __nv_fp8_e4m3 *D, int ldd) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  if (i < M && j < N) {
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
      acc += float(A[i * lda + k]) * float(B[k + j * ldb]);
    }
    float val = alpha * acc;
    if (beta != 0.0f) {
      val += beta * float(D[i + j * ldd]);
    }
    D[i + j * ldd] = __nv_fp8_e4m3(val);
  }
}

cudaError_t HopperGemmPrefetch(
    int M, int N, int K, float alpha,
    __nv_fp8_e4m3 const *A, int lda,
    __nv_fp8_e5m2 const *B, int ldb,
    float beta,
    __nv_fp8_e4m3 *D, int ldd) {
  dim3 block(16, 16);
  dim3 grid((M + 15) / 16, (N + 15) / 16);
  naive_fp8_gemm_prefetch_kernel<<<grid, block>>>(M, N, K, alpha, A, lda, B, ldb, beta, D, ldd);
  return cudaGetLastError();
}
