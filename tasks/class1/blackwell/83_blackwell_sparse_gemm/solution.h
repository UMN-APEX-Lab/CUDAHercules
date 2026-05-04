#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Naive dense GEMM for testing the harness (ignores sparsity structure)
// A: row-major half [M,K], B: col-major half [K,N]
// C: col-major float [M,N], D: col-major float [M,N]
// D = alpha * A * B + beta * C
__global__ void naive_sparse_gemm_kernel(
    int M, int N, int K, float alpha,
    __half const *A, int lda,
    __half const *B, int ldb,
    float beta,
    float const *C, int ldc,
    float *D, int ldd) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  if (i < M && j < N) {
    float acc = 0;
    for (int k = 0; k < K; ++k) {
      // A row-major: A[i, k] = A[i * lda + k]
      // B col-major: B[k, j] = B[k + j * ldb]
      acc += __half2float(A[i * lda + k]) * __half2float(B[k + j * ldb]);
    }
    // C, D col-major: [i, j] = [i + j * ldc/ldd]
    D[i + j * ldd] = alpha * acc + beta * C[i + j * ldc];
  }
}

cudaError_t BlackwellSparseGemm(
    int M, int N, int K, float alpha,
    __half const *A, int lda,
    __half const *B, int ldb,
    float beta,
    float const *C, int ldc,
    float *D, int ldd) {
  dim3 block(16, 16);
  dim3 grid((M + 15) / 16, (N + 15) / 16);
  naive_sparse_gemm_kernel<<<grid, block>>>(
    M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, D, ldd);
  return cudaGetLastError();
}
