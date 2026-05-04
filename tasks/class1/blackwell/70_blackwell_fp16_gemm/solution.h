#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Naive HGEMM for testing the harness
// A is row-major [M, K] (lda = K), B is column-major [K, N] (ldb = K),
// C is column-major [M, N] (ldc = M). FP16 input, FP32 output.
__global__ void naive_hgemm_kernel(
    int M, int N, int K, float alpha,
    __half const *A, int lda, __half const *B, int ldb,
    float beta, float *C, int ldc) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  if (i < M && j < N) {
    float acc = 0.0f;
    for (int k = 0; k < K; ++k)
      acc += __half2float(A[i * lda + k]) * __half2float(B[k + j * ldb]);
    C[i + j * ldc] = alpha * acc + beta * C[i + j * ldc];
  }
}

cudaError_t BlackwellHgemm(
    int M, int N, int K, float alpha,
    __half const *A, int lda, __half const *B, int ldb,
    float beta, float *C, int ldc) {
  dim3 block(16, 16);
  dim3 grid((M + 15) / 16, (N + 15) / 16);
  naive_hgemm_kernel<<<grid, block>>>(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
  return cudaGetLastError();
}
