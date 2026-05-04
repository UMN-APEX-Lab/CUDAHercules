#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>

// Naive BF16 GEMM kernel: A col-major, B row-major, D col-major
__global__ void naive_gemm_bf16_kernel(
    int M, int N, int K, float alpha,
    __nv_bfloat16 const *A, int lda,
    __nv_bfloat16 const *B, int ldb,
    float beta,
    __nv_bfloat16 const *C, int ldc,
    __nv_bfloat16 *D, int ldd) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  if (i < M && j < N) {
    float acc = 0;
    for (int k = 0; k < K; ++k) {
      acc += __bfloat162float(A[i + k * lda]) * __bfloat162float(B[k * ldb + j]);
    }
    D[i + j * ldd] = __float2bfloat16(alpha * acc + beta * __bfloat162float(C[i + j * ldc]));
  }
}

// Naive reduction kernel: for each column k of A (col-major), sum over rows
__global__ void naive_reduction_kernel(
    int M, int K,
    __nv_bfloat16 const *A, int lda,
    float *reduction) {
  int k = threadIdx.x + blockIdx.x * blockDim.x;
  if (k < K) {
    float sum = 0;
    for (int m = 0; m < M; ++m) {
      sum += __bfloat162float(A[m + k * lda]);
    }
    reduction[k] = sum;
  }
}

cudaError_t GemmWithReduction(
    int M, int N, int K, float alpha,
    __nv_bfloat16 const *A, int lda,
    __nv_bfloat16 const *B, int ldb,
    float beta,
    __nv_bfloat16 const *C, int ldc,
    __nv_bfloat16 *D, int ldd,
    float *row_reduction) {
  // GEMM
  {
    dim3 block(16, 16);
    dim3 grid((M + 15) / 16, (N + 15) / 16);
    naive_gemm_bf16_kernel<<<grid, block>>>(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, D, ldd);
  }
  // Reduction
  {
    dim3 block(256);
    dim3 grid((K + 255) / 256);
    naive_reduction_kernel<<<grid, block>>>(M, K, A, lda, row_reduction);
  }
  return cudaGetLastError();
}
