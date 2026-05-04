#pragma once
#include <cuda_runtime.h>
#include <cuda_fp8.h>

// Naive FP8 blockwise scaling GEMM:
// A row-major E4M3 [M,K], B col-major E4M3 [K,N], D col-major float [M,N]
// scale_A: per-block scale factors for A, scale_B: per-block for B
// block_size determines the scaling granularity along K
__global__ void naive_fp8_blockwise_gemm_kernel(
    int M, int N, int K, float alpha,
    __nv_fp8_e4m3 const *A, int lda, float const *scale_A,
    __nv_fp8_e4m3 const *B, int ldb, float const *scale_B,
    float *D, int ldd,
    int block_size) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  if (i < M && j < N) {
    float acc = 0.0f;
    int num_blocks = (K + block_size - 1) / block_size;
    for (int kb = 0; kb < num_blocks; ++kb) {
      int k_start = kb * block_size;
      int k_end = min(k_start + block_size, K);
      float sa = scale_A[i * num_blocks + kb];
      float sb = scale_B[j * num_blocks + kb];
      float block_acc = 0.0f;
      for (int k = k_start; k < k_end; ++k) {
        block_acc += float(A[i * lda + k]) * float(B[k + j * ldb]);
      }
      acc += sa * sb * block_acc;
    }
    // D is col-major: D[i + j * ldd]
    D[i + j * ldd] = alpha * acc;
  }
}

cudaError_t HopperFp8BlockwiseGemm(
    int M, int N, int K, float alpha,
    __nv_fp8_e4m3 const *A, int lda, float const *scale_A,
    __nv_fp8_e4m3 const *B, int ldb, float const *scale_B,
    float *D, int ldd,
    int block_size) {
  dim3 block(16, 16);
  dim3 grid((M + 15) / 16, (N + 15) / 16);
  naive_fp8_blockwise_gemm_kernel<<<grid, block>>>(
    M, N, K, alpha, A, lda, scale_A, B, ldb, scale_B, D, ldd, block_size);
  return cudaGetLastError();
}
