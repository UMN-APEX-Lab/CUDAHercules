#pragma once
#include <cuda_runtime.h>

// Naive GEMM for testing the harness
// A: row-major [M, K] (lda = K), B: col-major [K, N] (ldb = K), C: col-major [M, N] (ldc = M)
__global__ void naive_gemm_kernel(
    int M, int N, int K, float alpha,
    float const *A, int lda, float const *B, int ldb,
    float beta, float *C, int ldc) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  if (i < M && j < N) {
    float acc = 0;
    for (int k = 0; k < K; ++k)
      acc += A[i * lda + k] * B[k + j * ldb];
    C[i + j * ldc] = alpha * acc + beta * C[i + j * ldc];
  }
}

cudaError_t BlackwellBf16x9Gemm(
    int M, int N, int K, float alpha,
    float const *A, int lda, float const *B, int ldb,
    float beta, float *C, int ldc) {
  dim3 block(16, 16);
  dim3 grid((M + 15) / 16, (N + 15) / 16);
  naive_gemm_kernel<<<grid, block>>>(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
  return cudaGetLastError();
}
