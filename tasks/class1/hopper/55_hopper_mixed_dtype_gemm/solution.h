#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

// Naive mixed-dtype GEMM with per-group scale and zero-point
// A: half, row-major [M, K] (lda = K)
// B: int8, col-major [K, N] (ldb = K)
// scale_B: half, row-major [N, K/group_size]
// zero_B:  half, row-major [N, K/group_size]
// D: half, row-major [M, N] (ldd = N)
//
// Dequantization: B_dq[k,n] = scale_B[n, k/group_size] * (B[k,n] - zero_B[n, k/group_size])
__global__ void naive_mixed_dtype_gemm_kernel(
    int M, int N, int K,
    float alpha,
    __half const *A, int lda,
    int8_t const *B, int ldb,
    __half const *scale_B,
    __half const *zero_B,
    int group_size,
    float beta,
    __half *D, int ldd) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < M && j < N) {
    int num_groups = (K + group_size - 1) / group_size;
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
      int g = k / group_size;
      float s = __half2float(scale_B[j * num_groups + g]);
      float z = __half2float(zero_B[j * num_groups + g]);
      float a_val = __half2float(A[i * lda + k]);
      float b_val = s * (float(B[k + j * ldb]) - z);
      acc += a_val * b_val;
    }
    float d_val = alpha * acc + beta * __half2float(D[i * ldd + j]);
    D[i * ldd + j] = __float2half(d_val);
  }
}

cudaError_t HopperMixedDtypeGemm(
    int M, int N, int K,
    float alpha,
    __half const *A, int lda,
    int8_t const *B, int ldb,
    __half const *scale_B,
    __half const *zero_B,
    int group_size,
    float beta,
    __half *D, int ldd) {

  dim3 block(16, 16);
  dim3 grid((M + 15) / 16, (N + 15) / 16);
  naive_mixed_dtype_gemm_kernel<<<grid, block>>>(
    M, N, K, alpha, A, lda, B, ldb, scale_B, zero_B, group_size, beta, D, ldd);
  return cudaGetLastError();
}
