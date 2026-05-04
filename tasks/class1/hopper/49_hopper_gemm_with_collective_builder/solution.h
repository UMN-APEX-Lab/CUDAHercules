#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Naive GEMM: A row-major [M,K] half, B col-major [K,N] half, D col-major [M,N] half
__global__ void naive_collective_gemm_kernel(
    int M, int N, int K, float alpha,
    __half const *A, int lda,
    __half const *B, int ldb,
    float beta, __half *D, int ldd) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  if (i < M && j < N) {
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
      acc += __half2float(A[i * lda + k]) * __half2float(B[k + j * ldb]);
    }
    float old = beta * __half2float(D[i + j * ldd]);
    D[i + j * ldd] = __float2half(alpha * acc + old);
  }
}

cudaError_t HopperCollectiveGemm(
    int M, int N, int K, float alpha,
    __half const *A, int lda,
    __half const *B, int ldb,
    float beta, __half *D, int ldd) {
  dim3 block(16, 16);
  dim3 grid((M + 15) / 16, (N + 15) / 16);
  naive_collective_gemm_kernel<<<grid, block>>>(M, N, K, alpha, A, lda, B, ldb, beta, D, ldd);
  return cudaGetLastError();
}
