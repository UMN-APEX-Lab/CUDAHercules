#pragma once
#include <cuda_runtime.h>
#include <cstdint>

// Naive GEMM: A row-major [M,K] (int8), B col-major [K,N] (int8), D col-major [M,N] (int32)
// Accumulates in int32, alpha/beta are int32. Output is int32 (no clamping).
__global__ void naive_epilogue_swizzle_gemm_kernel(
    int M, int N, int K, int32_t alpha,
    int8_t const *A, int lda,
    int8_t const *B, int ldb,
    int32_t beta, int32_t *D, int ldd) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  if (i < M && j < N) {
    int32_t acc = 0;
    for (int k = 0; k < K; ++k) {
      acc += static_cast<int32_t>(A[i * lda + k]) * static_cast<int32_t>(B[k + j * ldb]);
    }
    D[i + j * ldd] = alpha * acc + beta * D[i + j * ldd];
  }
}

cudaError_t HopperEpilogueSwizzleGemm(
    int M, int N, int K, int32_t alpha,
    int8_t const *A, int lda,
    int8_t const *B, int ldb,
    int32_t beta, int32_t *D, int ldd) {
  dim3 block(16, 16);
  dim3 grid((M + 15) / 16, (N + 15) / 16);
  naive_epilogue_swizzle_gemm_kernel<<<grid, block>>>(M, N, K, alpha, A, lda, B, ldb, beta, D, ldd);
  return cudaGetLastError();
}
