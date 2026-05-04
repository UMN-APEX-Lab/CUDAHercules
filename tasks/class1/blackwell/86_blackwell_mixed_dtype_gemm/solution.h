#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>

// Naive mixed-dtype GEMM: BF16 A * (INT4-as-int8 B with BF16 blockwise scales) -> BF16 D
// A: row-major [M, K], B: col-major [K, N] stored as int8_t (one element per byte),
// B_scales: [N, ceil(K/128)] row-major BF16 scales, D: row-major [M, N] BF16
__global__ void naive_mixed_dtype_gemm_kernel(
    int M, int N, int K, float alpha,
    __nv_bfloat16 const *A, int lda,
    int8_t const *B, int ldb,
    __nv_bfloat16 const *B_scales, int scale_ldb,
    float beta,
    __nv_bfloat16 *D, int ldd) {

  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < M && col < N) {
    float acc = 0.0f;
    int group_size = 128;
    for (int k = 0; k < K; ++k) {
      float a_val = __bfloat162float(A[row * lda + k]);
      float b_val = float(B[k + col * ldb]);
      int kg = k / group_size;
      float scale = __bfloat162float(B_scales[col * scale_ldb + kg]);
      acc += a_val * b_val * scale;
    }
    D[row * ldd + col] = __float2bfloat16(alpha * acc + beta * __bfloat162float(D[row * ldd + col]));
  }
}

cudaError_t BlackwellMixedDtypeGemm(
    int M, int N, int K, float alpha,
    __nv_bfloat16 const *A, int lda,
    int8_t const *B, int ldb,
    __nv_bfloat16 const *B_scales, int scale_ldb,
    float beta,
    __nv_bfloat16 *D, int ldd) {

  dim3 block(16, 16);
  dim3 grid((M + block.x - 1) / block.x, (N + block.y - 1) / block.y);
  naive_mixed_dtype_gemm_kernel<<<grid, block>>>(
    M, N, K, alpha, A, lda, B, ldb, B_scales, scale_ldb, beta, D, ldd);
  return cudaGetLastError();
}
