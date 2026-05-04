#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "cutlass/numeric_types.h"

// Naive GEMM for block-scaled FP4: A row-major, B col-major, D row-major (BF16 output)
// Each FP4 element is stored as one byte (unpacked). Scale factors are per-group (group_size=32).
__global__ void naive_narrow_gemm_kernel(
    int M, int N, int K, float alpha,
    cutlass::float_e2m1_t const *A, int lda,
    float const *A_scales, int scale_lda,
    cutlass::float_e2m1_t const *B, int ldb,
    float const *B_scales, int scale_ldb,
    float beta,
    __nv_bfloat16 *D, int ldd) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < M && j < N) {
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
      int kg = k / 32;
      float a_val = float(A[i * lda + k]) * A_scales[i * scale_lda + kg];
      float b_val = float(B[k + j * ldb]) * B_scales[j * scale_ldb + kg];
      acc += a_val * b_val;
    }
    D[i * ldd + j] = __float2bfloat16(alpha * acc + beta * __bfloat162float(D[i * ldd + j]));
  }
}

cudaError_t BlackwellNarrowPrecisionGemm(
    int M, int N, int K,
    float alpha,
    cutlass::float_e2m1_t const *A, int lda,
    float const *A_scales, int scale_lda,
    cutlass::float_e2m1_t const *B, int ldb,
    float const *B_scales, int scale_ldb,
    float beta,
    __nv_bfloat16 *D, int ldd) {
  dim3 block(16, 16);
  dim3 grid((M + 15) / 16, (N + 15) / 16);
  naive_narrow_gemm_kernel<<<grid, block>>>(
    M, N, K, alpha, A, lda, A_scales, scale_lda,
    B, ldb, B_scales, scale_ldb, beta, D, ldd);
  return cudaGetLastError();
}
