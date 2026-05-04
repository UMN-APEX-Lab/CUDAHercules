#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Naive HGEMM for testing the harness
// A is row-major [M, K] (lda = K), B is column-major [K, N] (ldb = K),
// D is column-major [M, N] (ldd = M). FP16 input, FP16 output, FP32 accumulation.
__global__ void naive_collective_gemm_kernel(
    int M, int N, int K, float alpha,
    __half const *A, int lda, __half const *B, int ldb,
    float beta, __half *D, int ldd) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  if (i < M && j < N) {
    float acc = 0.0f;
    for (int k = 0; k < K; ++k)
      acc += __half2float(A[i * lda + k]) * __half2float(B[k + j * ldb]);
    float old = __half2float(D[i + j * ldd]);
    D[i + j * ldd] = __float2half(alpha * acc + beta * old);
  }
}

cudaError_t BlackwellCollectiveGemm(
    int M, int N, int K, float alpha,
    __half const *A, int lda, __half const *B, int ldb,
    float beta, __half *D, int ldd) {
  dim3 block(16, 16);
  dim3 grid((M + 15) / 16, (N + 15) / 16);
  naive_collective_gemm_kernel<<<grid, block>>>(M, N, K, alpha, A, lda, B, ldb, beta, D, ldd);
  return cudaGetLastError();
}
