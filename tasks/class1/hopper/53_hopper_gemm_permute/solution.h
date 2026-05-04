#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Naive strided batched GEMM for testing the harness.
// D = alpha * A * B + beta * D  (beta = 0 typically)
// A row-major [batch, M, K] (lda = K), batch_stride_A = M*K
// B row-major [batch, K, N] (ldb = N), batch_stride_B = K*N
// D row-major [batch, M, N] (ldd = N), batch_stride_D = M*N
// Half-precision inputs/outputs.
__global__ void naive_batched_gemm_kernel(
    int M, int N, int K, int batch, float alpha,
    __half const *A, int lda,
    __half const *B, int ldb,
    float beta,
    __half *D, int ldd) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int b = blockIdx.z;
  if (i >= M || j >= N || b >= batch) return;

  int64_t offA = int64_t(b) * M * K;
  int64_t offB = int64_t(b) * K * N;
  int64_t offD = int64_t(b) * M * N;

  float acc = 0.0f;
  for (int k = 0; k < K; ++k) {
    // A row-major: A[i, k] = A[offA + i * lda + k]
    // B row-major: B[k, j] = B[offB + k * ldb + j]
    acc += __half2float(A[offA + i * lda + k]) * __half2float(B[offB + k * ldb + j]);
  }
  // D row-major: D[i, j] = D[offD + i * ldd + j]
  float old_val = beta * __half2float(D[offD + i * ldd + j]);
  D[offD + i * ldd + j] = __float2half(alpha * acc + old_val);
}

cudaError_t HopperGemmPermute(
    int M, int N, int K, int batch, float alpha,
    __half const *A, int lda,
    __half const *B, int ldb,
    float beta,
    __half *D, int ldd) {
  dim3 block(16, 16);
  dim3 grid((M + 15) / 16, (N + 15) / 16, batch);
  naive_batched_gemm_kernel<<<grid, block>>>(M, N, K, batch, alpha, A, lda, B, ldb, beta, D, ldd);
  return cudaGetLastError();
}
