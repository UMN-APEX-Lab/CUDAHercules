#pragma once
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>

// Naive FP8 GEMM for testing the harness
// A row-major [M,K], B col-major [K,N], C row-major [M,N]
// FP8 E4M3 inputs, BF16 output, FP32 accumulation.
__global__ void naive_fp8_gemm_kernel(
    int M, int N, int K, float alpha,
    __nv_fp8_e4m3 const *A, int lda,
    __nv_fp8_e4m3 const *B, int ldb,
    float beta, __nv_bfloat16 *C, int ldc) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  if (i < M && j < N) {
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
      acc += float(A[i * lda + k]) * float(B[k + j * ldb]);
    }
    float c_val = __bfloat162float(C[i * ldc + j]);
    C[i * ldc + j] = __float2bfloat16(alpha * acc + beta * c_val);
  }
}

cudaError_t BlackwellGeforceFp8Gemm(
    int M, int N, int K, float alpha,
    __nv_fp8_e4m3 const *A, int lda,
    __nv_fp8_e4m3 const *B, int ldb,
    float beta, __nv_bfloat16 *C, int ldc) {
  dim3 block(16, 16);
  dim3 grid((M + 15) / 16, (N + 15) / 16);
  naive_fp8_gemm_kernel<<<grid, block>>>(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
  return cudaGetLastError();
}
