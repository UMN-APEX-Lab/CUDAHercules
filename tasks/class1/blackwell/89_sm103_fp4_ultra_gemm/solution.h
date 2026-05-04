#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "cutlass/numeric_types.h"

// Naive FP4 ultra GEMM for testing the harness
// A: row-major [M, K] in FP4 E2M1 (one per byte), B: col-major [K, N] in FP4 E2M1 (one per byte)
// A_scales: [M, ceil(K/32)] float, B_scales: [N, ceil(K/32)] float
// D: row-major [M, N] BF16
__global__ void naive_fp4_ultra_gemm_kernel(
    int M, int N, int K, float alpha,
    cutlass::float_e2m1_t const *A, int lda,
    float const *A_scales, int scale_lda,
    cutlass::float_e2m1_t const *B, int ldb,
    float const *B_scales, int scale_ldb,
    float beta,
    __nv_bfloat16 *D, int ldd) {

  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < M && col < N) {
    float acc = 0.0f;
    int group_size = 32;
    for (int k = 0; k < K; ++k) {
      int kg = k / group_size;
      float a_val = float(A[row * lda + k]) * A_scales[row * scale_lda + kg];
      float b_val = float(B[k + col * ldb]) * B_scales[col * scale_ldb + kg];
      acc += a_val * b_val;
    }
    D[row * ldd + col] = __float2bfloat16(alpha * acc + beta * __bfloat162float(D[row * ldd + col]));
  }
}

cudaError_t Fp4UltraGemm(
    int M, int N, int K, float alpha,
    cutlass::float_e2m1_t const *A, int lda,
    float const *A_scales, int scale_lda,
    cutlass::float_e2m1_t const *B, int ldb,
    float const *B_scales, int scale_ldb,
    float beta,
    __nv_bfloat16 *D, int ldd) {

  dim3 block(16, 16);
  dim3 grid((M + block.x - 1) / block.x, (N + block.y - 1) / block.y);
  naive_fp4_ultra_gemm_kernel<<<grid, block>>>(
    M, N, K, alpha, A, lda, A_scales, scale_lda, B, ldb, B_scales, scale_ldb, beta, D, ldd);
  return cudaGetLastError();
}
