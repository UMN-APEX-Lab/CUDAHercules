#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Naive GEMM + row-wise softmax for testing the harness
__global__ void naive_gemm_softmax_kernel(
    int M, int N, int K,
    float alpha, float beta,
    __half const *A, int lda,
    __half const *B, int ldb,
    __half const *C, int ldc,
    __half *D, int ldd,
    __half *Softmax, int lds) {

  int m = blockIdx.x * blockDim.x + threadIdx.x;
  if (m >= M) return;

  // GEMM: D[m,:] = alpha * A[m,:] @ B[:,:] + beta * C[m,:]
  for (int n = 0; n < N; ++n) {
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
      acc += __half2float(A[m * lda + k]) * __half2float(B[k + n * ldb]);
    }
    float val = alpha * acc + beta * __half2float(C[m * ldc + n]);
    D[m * ldd + n] = __float2half(val);
  }

  // Row-wise softmax
  float max_val = -1e30f;
  for (int n = 0; n < N; ++n) {
    max_val = fmaxf(max_val, __half2float(D[m * ldd + n]));
  }
  float sum_exp = 0.0f;
  for (int n = 0; n < N; ++n) {
    sum_exp += expf(__half2float(D[m * ldd + n]) - max_val);
  }
  for (int n = 0; n < N; ++n) {
    float s = expf(__half2float(D[m * ldd + n]) - max_val) / sum_exp;
    Softmax[m * lds + n] = __float2half(s);
  }
}

cudaError_t gemm_softmax_solution(
    int M, int N, int K,
    float alpha, float beta,
    __half const *A, int lda,
    __half const *B, int ldb,
    __half const *C, int ldc,
    __half *D, int ldd,
    __half *Softmax, int lds,
    int batch_count,
    int64_t batch_stride_A,
    int64_t batch_stride_B,
    int64_t batch_stride_C,
    int64_t batch_stride_D,
    int64_t batch_stride_Softmax) {

  int threads = 256;
  int blocks = (M + threads - 1) / threads;

  for (int b = 0; b < batch_count; ++b) {
    naive_gemm_softmax_kernel<<<blocks, threads>>>(
      M, N, K, alpha, beta,
      A + b * batch_stride_A, lda,
      B + b * batch_stride_B, ldb,
      C + b * batch_stride_C, ldc,
      D + b * batch_stride_D, ldd,
      Softmax + b * batch_stride_Softmax, lds);
  }

  return cudaGetLastError();
}
