#pragma once
#include <cuda_runtime.h>

// Naive GEMM for testing the harness
// A row-major [M,K] (lda=K), B col-major [K,N] (ldb=K), C row-major [M,N] (ldc=N)
__global__ void naive_tf32_gemm_kernel(
    int M, int N, int K, float alpha,
    float const *A, int lda,
    float const *B, int ldb,
    float beta, float *C, int ldc) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  if (i < M && j < N) {
    float acc = 0.0f;
    for (int k = 0; k < K; ++k)
      acc += A[i * lda + k] * B[k + j * ldb];
    C[i * ldc + j] = alpha * acc + beta * C[i * ldc + j];
  }
}

cudaError_t Tf32Gemm(
    int M, int N, int K, float alpha,
    float const *A, int lda,
    float const *B, int ldb,
    float beta, float *C, int ldc) {
  dim3 block(16, 16);
  dim3 grid((M + 15) / 16, (N + 15) / 16);
  naive_tf32_gemm_kernel<<<grid, block>>>(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
  return cudaGetLastError();
}
