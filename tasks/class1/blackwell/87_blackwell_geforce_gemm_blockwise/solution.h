#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

// Naive blockwise-scaled FP8 GEMM for testing the harness
// A: row-major [M, K] FP8 E4M3, B: col-major [K, N] FP8 E4M3
// scale_A: [M, ceil(K/block_size)] float, scale_B: [N, ceil(K/block_size)] float
// D: row-major [M, N] BF16
__global__ void naive_fp8_blockwise_gemm_kernel(
    int M, int N, int K, float alpha,
    __nv_fp8_e4m3 const *A, int lda,
    float const *scale_A,
    __nv_fp8_e4m3 const *B, int ldb,
    float const *scale_B,
    float beta,
    __nv_bfloat16 *D, int ldd,
    int block_size) {

  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < M && col < N) {
    float acc = 0.0f;
    int k_blocks = (K + block_size - 1) / block_size;
    for (int k = 0; k < K; ++k) {
      int kb = k / block_size;
      float a_val = float(A[row * lda + k]) * scale_A[row * k_blocks + kb];
      float b_val = float(B[k + col * ldb]) * scale_B[col * k_blocks + kb];
      acc += a_val * b_val;
    }
    D[row * ldd + col] = __float2bfloat16(alpha * acc + beta * __bfloat162float(D[row * ldd + col]));
  }
}

cudaError_t BlackwellGeForceBlockwiseGemm(
    int M, int N, int K, float alpha,
    __nv_fp8_e4m3 const *A, int lda,
    float const *scale_A,
    __nv_fp8_e4m3 const *B, int ldb,
    float const *scale_B,
    float beta,
    __nv_bfloat16 *D, int ldd,
    int block_size) {

  dim3 block(16, 16);
  dim3 grid((M + block.x - 1) / block.x, (N + block.y - 1) / block.y);
  naive_fp8_blockwise_gemm_kernel<<<grid, block>>>(
    M, N, K, alpha, A, lda, scale_A, B, ldb, scale_B, beta, D, ldd, block_size);
  return cudaGetLastError();
}
