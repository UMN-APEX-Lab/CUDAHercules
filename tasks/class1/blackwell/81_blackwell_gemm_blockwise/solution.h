#pragma once
#include <cuda_runtime.h>
#include <cuda_fp8.h>

// Naive blockwise-scaled FP8 GEMM for testing the harness
// A: row-major [M,K], B: col-major [K,N], D: col-major [M,N]
// scale_A: [M, ceil(K/block_size)], scale_B: [N, ceil(K/block_size)]
__global__ void naive_blockwise_gemm_kernel(
    int M, int N, int K, float alpha,
    __nv_fp8_e4m3 const *A, int lda, float const *scale_A,
    __nv_fp8_e4m3 const *B, int ldb, float const *scale_B,
    float beta, __nv_fp8_e4m3 *D, int ldd,
    int block_size, int k_blocks) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  if (i < M && j < N) {
    float acc = 0;
    for (int k = 0; k < K; ++k) {
      int kb = k / block_size;
      float a_val = float(A[i * lda + k]) * scale_A[i * k_blocks + kb];
      float b_val = float(B[k + j * ldb]) * scale_B[j * k_blocks + kb];
      acc += a_val * b_val;
    }
    float result = alpha * acc;
    D[i + j * ldd] = __nv_fp8_e4m3(result);
  }
}

cudaError_t BlackwellBlockwiseGemm(
    int M, int N, int K, float alpha,
    __nv_fp8_e4m3 const *A, int lda, float const *scale_A,
    __nv_fp8_e4m3 const *B, int ldb, float const *scale_B,
    float beta, __nv_fp8_e4m3 *D, int ldd,
    int block_size) {
  int k_blocks = (K + block_size - 1) / block_size;
  dim3 block(16, 16);
  dim3 grid((M + 15) / 16, (N + 15) / 16);
  naive_blockwise_gemm_kernel<<<grid, block>>>(
    M, N, K, alpha, A, lda, scale_A, B, ldb, scale_B,
    beta, D, ldd, block_size, k_blocks);
  return cudaGetLastError();
}
