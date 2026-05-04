#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include "cutlass/numeric_types.h"

// Naive mixed-precision block-scaled GEMM for testing the harness
// A: row-major [M, K] FP8 E4M3 with per-group scales (group=32)
// B: col-major [K, N] FP6 E3M2 with per-group scales (group=32), one element per byte
// D: row-major [M, N] BF16
__global__ void naive_mxfp_gemm_kernel(
    int M, int N, int K, float alpha,
    __nv_fp8_e4m3 const *A, int lda,
    float const *A_scales, int scale_lda,
    cutlass::float_e3m2_t const *B, int ldb,
    float const *B_scales, int scale_ldb,
    float beta, __nv_bfloat16 *D, int ldd) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  if (i < M && j < N) {
    float acc = 0;
    for (int k = 0; k < K; ++k) {
      int kg = k / 32;
      float a_val = float(A[i * lda + k]) * A_scales[i * scale_lda + kg];
      float b_val = float(B[k + j * ldb]) * B_scales[j * scale_ldb + kg];
      acc += a_val * b_val;
    }
    D[i * ldd + j] = __float2bfloat16(alpha * acc + beta * __bfloat162float(D[i * ldd + j]));
  }
}

cudaError_t MxfpBf16Gemm(
    int M, int N, int K, float alpha,
    __nv_fp8_e4m3 const *A, int lda,
    float const *A_scales, int scale_lda,
    cutlass::float_e3m2_t const *B, int ldb,
    float const *B_scales, int scale_ldb,
    float beta, __nv_bfloat16 *D, int ldd) {
  dim3 block(16, 16);
  dim3 grid((M + 15) / 16, (N + 15) / 16);
  naive_mxfp_gemm_kernel<<<grid, block>>>(
      M, N, K, alpha, A, lda, A_scales, scale_lda,
      B, ldb, B_scales, scale_ldb, beta, D, ldd);
  return cudaGetLastError();
}
