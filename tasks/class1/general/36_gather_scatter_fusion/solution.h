#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Naive gather-scatter GEMM kernel
// Row-major layout. Half input A/B, float accumulate, float output.
// D[scatter_D[i], j] = alpha * sum_k A[gather_A[i], k] * B[k, j] + beta * C[gather_C[i], j]
__global__ void naive_gather_scatter_gemm_kernel(
    int M, int N, int K, float alpha,
    __half const *A, int lda, int const *gather_A,
    __half const *B, int ldb,
    float beta,
    float const *C, int ldc, int const *gather_C,
    float *D, int ldd, int const *scatter_D) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;  // logical row [0, M)
  int j = blockIdx.y * blockDim.y + threadIdx.y;  // column [0, N)

  if (i < M && j < N) {
    int a_row = gather_A[i];
    int c_row = gather_C[i];
    int d_row = scatter_D[i];

    float acc = 0;
    for (int k = 0; k < K; ++k)
      acc += __half2float(A[a_row * lda + k]) * __half2float(B[k * ldb + j]);

    D[d_row * ldd + j] = alpha * acc + beta * C[c_row * ldc + j];
  }
}

cudaError_t GatherScatterGemm(
    int M, int N, int K, float alpha,
    __half const *A, int lda, int const *gather_A,
    __half const *B, int ldb,
    float beta,
    float const *C, int ldc, int const *gather_C,
    float *D, int ldd, int const *scatter_D) {

  dim3 block(16, 16);
  dim3 grid((M + 15) / 16, (N + 15) / 16);
  naive_gather_scatter_gemm_kernel<<<grid, block>>>(
    M, N, K, alpha, A, lda, gather_A, B, ldb,
    beta, C, ldc, gather_C, D, ldd, scatter_D);
  return cudaGetLastError();
}
