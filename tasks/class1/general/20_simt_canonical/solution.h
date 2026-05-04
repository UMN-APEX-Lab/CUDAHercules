#pragma once
#include <cuda_runtime.h>

// Naive SGEMM for testing: D = alpha * A * B + beta * C
// A: row-major (M x K), B: column-major (K x N), C/D: row-major (M x N)
__global__ void naive_sgemm_rc_kernel(
    int M, int N, int K, float alpha,
    float const *A, int lda,
    float const *B, int ldb,
    float beta,
    float const *C, int ldc,
    float *D, int ldd) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  if (i < M && j < N) {
    float acc = 0.0f;
    for (int k = 0; k < K; ++k)
      acc += A[i * lda + k] * B[k + j * ldb];  // A row-major, B col-major
    D[i * ldd + j] = alpha * acc + beta * C[i * ldc + j];
  }
}

cudaError_t SimtCanonicalGemm(
    int M, int N, int K, float alpha,
    float const *A, int lda,
    float const *B, int ldb,
    float beta,
    float const *C, int ldc,
    float *D, int ldd) {
  dim3 block(16, 16);
  dim3 grid((M + 15) / 16, (N + 15) / 16);
  naive_sgemm_rc_kernel<<<grid, block>>>(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, D, ldd);
  return cudaGetLastError();
}
