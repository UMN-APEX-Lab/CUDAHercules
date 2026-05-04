#pragma once
#include <cuda_runtime.h>

// Naive batched strided SGEMM for testing the harness
__global__ void naive_batched_sgemm_kernel(
    int m, int n, int k, float alpha,
    float const *A, int lda, long long int batch_stride_A,
    float const *B, int ldb, long long int batch_stride_B,
    float *C, int ldc, long long int batch_stride_C,
    float beta, int batch_count) {

  int row = threadIdx.x + blockIdx.x * blockDim.x;
  int col = threadIdx.y + blockIdx.y * blockDim.y;
  int batch = blockIdx.z;

  if (row < m && col < n && batch < batch_count) {
    float const *A_batch = A + batch * batch_stride_A;
    float const *B_batch = B + batch * batch_stride_B;
    float *C_batch = C + batch * batch_stride_C;

    float acc = 0.0f;
    for (int kk = 0; kk < k; ++kk) {
      // Column-major: A[row, kk] = A[row + kk * lda]
      // Column-major: B[kk, col] = B[kk + col * ldb]
      acc += A_batch[row + kk * lda] * B_batch[kk + col * ldb];
    }
    C_batch[row + col * ldc] = alpha * acc + beta * C_batch[row + col * ldc];
  }
}

cudaError_t StridedBatchedSgemm(
    int m, int n, int k, float alpha,
    float const *A, int lda, long long int batch_stride_A,
    float const *B, int ldb, long long int batch_stride_B,
    float *C, int ldc, long long int batch_stride_C,
    float beta, int batch_count) {

  dim3 block(16, 16);
  dim3 grid((m + 15) / 16, (n + 15) / 16, batch_count);
  naive_batched_sgemm_kernel<<<grid, block>>>(
    m, n, k, alpha, A, lda, batch_stride_A, B, ldb, batch_stride_B,
    C, ldc, batch_stride_C, beta, batch_count);
  return cudaGetLastError();
}
