#pragma once
#include <cuda_runtime.h>

// Naive TRMM for testing: C = alpha * A * B, where A is lower-triangular
// Since upper triangle of A is zero, naive matmul naturally handles it.
__global__ void naive_dtrmm_kernel(
    int M, int N, double alpha,
    double const *A, int lda, double const *B, int ldb,
    double *C, int ldc) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  if (i < M && j < N) {
    double acc = 0;
    for (int k = 0; k < M; ++k)
      acc += A[i + k * lda] * B[k + j * ldb];
    C[i + j * ldc] = alpha * acc;
  }
}

cudaError_t Dtrmm(
    int M, int N, double alpha,
    double const *A, int lda, double const *B, int ldb,
    double *C, int ldc) {
  dim3 block(16, 16);
  dim3 grid((M + 15) / 16, (N + 15) / 16);
  naive_dtrmm_kernel<<<grid, block>>>(M, N, alpha, A, lda, B, ldb, C, ldc);
  return cudaGetLastError();
}
