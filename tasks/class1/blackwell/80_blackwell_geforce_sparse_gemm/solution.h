#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

// Naive dense GEMM for testing the harness (ignores sparsity structure)
// A: row-major FP8 E4M3 [M,K] (2:4 sparse with zeros), B: col-major FP8 E4M3 [K,N]
// C: row-major BF16 [M,N], D: row-major BF16 [M,N]
// D = alpha * A * B + beta * C
__global__ void naive_mxfp8_bf16_sparse_gemm_kernel(
    int M, int N, int K, float alpha,
    __nv_fp8_e4m3 const *A, int lda,
    __nv_fp8_e4m3 const *B, int ldb,
    float beta,
    __nv_bfloat16 const *C, int ldc,
    __nv_bfloat16 *D, int ldd) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  if (i < M && j < N) {
    float acc = 0;
    for (int k = 0; k < K; ++k) {
      // A row-major: A[i, k] = A[i * lda + k]
      // B col-major: B[k, j] = B[k + j * ldb]
      acc += float(A[i * lda + k]) * float(B[k + j * ldb]);
    }
    // C, D row-major: [i, j] = [i * ldc/ldd + j]
    D[i * ldd + j] = __float2bfloat16(alpha * acc + beta * __bfloat162float(C[i * ldc + j]));
  }
}

cudaError_t BlackwellGeForceSparseMxfp8Bf16Gemm(
    int M, int N, int K, float alpha,
    __nv_fp8_e4m3 const *A, int lda,
    __nv_fp8_e4m3 const *B, int ldb,
    float beta,
    __nv_bfloat16 const *C, int ldc,
    __nv_bfloat16 *D, int ldd) {
  dim3 block(16, 16);
  dim3 grid((M + 15) / 16, (N + 15) / 16);
  naive_mxfp8_bf16_sparse_gemm_kernel<<<grid, block>>>(
    M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, D, ldd);
  return cudaGetLastError();
}
