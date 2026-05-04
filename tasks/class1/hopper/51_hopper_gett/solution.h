#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Naive GEMM kernel for testing the harness.
// D = alpha * A * B + beta * C
// A column-major [M, K] (lda=M), B column-major [K, N] (ldb=K),
// C column-major [M, N] (ldc=M), D column-major [M, N] (ldd=M).
// Half inputs (A, B, C), float output (D).
__global__ void naive_gett_kernel(
    int M, int N, int K, float alpha,
    __half const *A, int lda,
    __half const *B, int ldb,
    float beta,
    __half const *C, int ldc,
    float *D, int ldd) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  if (i < M && j < N) {
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
      // A col-major: A[i,k] = A[i + k * lda]
      // B col-major: B[k,j] = B[k + j * ldb]
      acc += __half2float(A[i + k * lda]) * __half2float(B[k + j * ldb]);
    }
    // C col-major: C[i,j] = C[i + j * ldc]
    // D col-major: D[i,j] = D[i + j * ldd]
    D[i + j * ldd] = alpha * acc + beta * __half2float(C[i + j * ldc]);
  }
}

cudaError_t HopperGett(
    int M, int N, int K, float alpha,
    __half const *A, int lda,
    __half const *B, int ldb,
    float beta,
    __half const *C, int ldc,
    float *D, int ldd) {
  dim3 block(16, 16);
  dim3 grid((M + 15) / 16, (N + 15) / 16);
  naive_gett_kernel<<<grid, block>>>(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, D, ldd);
  return cudaGetLastError();
}
