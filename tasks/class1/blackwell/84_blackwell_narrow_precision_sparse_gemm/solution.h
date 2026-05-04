#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "cutlass/numeric_types.h"

// Naive GEMM for testing the harness
// Takes dequantized float A (row-major) and B (col-major), produces BF16 D (row-major)
// D = alpha * A * B + beta * C
__global__ void naive_nvfp4_sparse_gemm_kernel(
    int M, int N, int K, float alpha,
    float const *A, int lda,
    float const *B, int ldb,
    float beta,
    cutlass::bfloat16_t const *C, int ldc,
    cutlass::bfloat16_t *D, int ldd) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  if (i < M && j < N) {
    float acc = 0;
    for (int k = 0; k < K; ++k) {
      // A row-major: A[i, k] = A[i * lda + k]
      // B col-major: B[k, j] = B[k + j * ldb]
      acc += A[i * lda + k] * B[k + j * ldb];
    }
    // C, D row-major: [i, j] = [i * ldc/ldd + j]
    float c_val = float(C[i * ldc + j]);
    D[i * ldd + j] = cutlass::bfloat16_t(alpha * acc + beta * c_val);
  }
}

cudaError_t BlackwellNvfp4SparseBf16Gemm(
    int M, int N, int K, float alpha,
    float const *A, int lda,
    float const *B, int ldb,
    float beta,
    cutlass::bfloat16_t const *C, int ldc,
    cutlass::bfloat16_t *D, int ldd) {
  dim3 block(16, 16);
  dim3 grid((M + 15) / 16, (N + 15) / 16);
  naive_nvfp4_sparse_gemm_kernel<<<grid, block>>>(
    M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, D, ldd);
  return cudaGetLastError();
}
