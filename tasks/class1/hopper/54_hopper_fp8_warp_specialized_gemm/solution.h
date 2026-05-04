#pragma once
#include <cuda_runtime.h>
#include <cuda_fp8.h>

// Naive FP8 GEMM: A row-major [M,K], B col-major [K,N], C col-major [M,N]
// FP8 E4M3 inputs, FP32 output
__global__ void naive_fp8_gemm_kernel(
    int M, int N, int K, float alpha,
    __nv_fp8_e4m3 const *A, int lda,
    __nv_fp8_e4m3 const *B, int ldb,
    float beta,
    float *C, int ldc) {

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  if (i < M && j < N) {
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
      float a_val = float(A[i * lda + k]);
      float b_val = float(B[k + j * ldb]);
      acc += a_val * b_val;
    }
    C[i + j * ldc] = alpha * acc + beta * C[i + j * ldc];
  }
}

cudaError_t HopperFp8Gemm(
    int M, int N, int K, float alpha,
    __nv_fp8_e4m3 const *A, int lda,
    __nv_fp8_e4m3 const *B, int ldb,
    float beta,
    float *C, int ldc) {
  dim3 block(16, 16);
  dim3 grid((M + 15) / 16, (N + 15) / 16);
  naive_fp8_gemm_kernel<<<grid, block>>>(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
  return cudaGetLastError();
}
