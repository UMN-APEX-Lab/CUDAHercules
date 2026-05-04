#pragma once
#include <cuda_runtime.h>

// Naive SYRK for testing the harness: C = alpha * A * A^T + beta * C (lower triangle only)
__global__ void naive_dsyrk_kernel(
    int N, int K, double alpha,
    double const *A, int lda,
    double beta, double *C, int ldc) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  if (i < N && j < N && i >= j) {
    double acc = 0;
    for (int k = 0; k < K; ++k)
      acc += A[i + k * lda] * A[j + k * lda];
    C[i + j * ldc] = alpha * acc + beta * C[i + j * ldc];
  }
}

cudaError_t Dsyrk(
    int N, int K, double alpha,
    double const *A, int lda,
    double beta, double *C, int ldc) {
  dim3 block(16, 16);
  dim3 grid((N + 15) / 16, (N + 15) / 16);
  naive_dsyrk_kernel<<<grid, block>>>(N, K, alpha, A, lda, beta, C, ldc);
  return cudaGetLastError();
}
