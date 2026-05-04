#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>

// Naive GEMM + TopK + Softmax for testing the harness
// A row-major [M,K], B col-major [K,N], D row-major [M,N]
// For each row: compute GEMM, keep top-K values, zero out rest, softmax over top-K
__global__ void naive_gemm_topk_softmax_kernel(
    int M, int N, int K, float alpha,
    __half const *A, int lda,
    __half const *B, int ldb,
    __half *D, int ldd,
    int topk) {

  int m = blockIdx.x * blockDim.x + threadIdx.x;
  if (m >= M) return;

  // Step 1: Compute GEMM row
  extern __shared__ float shmem[];
  // Use global memory for row storage since N can be large
  // Each thread works on its own row

  // Compute GEMM for this row into D
  for (int n = 0; n < N; ++n) {
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
      acc += __half2float(A[m * lda + k]) * __half2float(B[k + n * ldb]);
    }
    D[m * ldd + n] = __float2half(alpha * acc);
  }

  // Step 2: Find top-K values using insertion sort
  // Store top-K values and indices
  float topk_vals[64]; // max topk=64
  int topk_count = 0;

  for (int n = 0; n < N; ++n) {
    float val = __half2float(D[m * ldd + n]);
    if (topk_count < topk) {
      // Insert into topk_vals maintaining sorted order (descending)
      int pos = topk_count;
      for (int t = 0; t < topk_count; ++t) {
        if (val > topk_vals[t]) { pos = t; break; }
      }
      for (int t = topk_count; t > pos; --t) topk_vals[t] = topk_vals[t-1];
      topk_vals[pos] = val;
      topk_count++;
    } else if (val > topk_vals[topk - 1]) {
      int pos = topk - 1;
      for (int t = 0; t < topk - 1; ++t) {
        if (val > topk_vals[t]) { pos = t; break; }
      }
      for (int t = topk - 1; t > pos; --t) topk_vals[t] = topk_vals[t-1];
      topk_vals[pos] = val;
    }
  }

  // Step 3: Compute softmax over top-K
  float max_val = topk_vals[0];
  float sum_exp = 0.0f;
  for (int t = 0; t < topk; ++t) {
    sum_exp += expf(topk_vals[t] - max_val);
  }

  // Step 4: Apply to output - zero non-top-K, softmax top-K
  float threshold = topk_vals[topk - 1];
  for (int n = 0; n < N; ++n) {
    float val = __half2float(D[m * ldd + n]);
    if (val >= threshold) {
      D[m * ldd + n] = __float2half(expf(val - max_val) / sum_exp);
    } else {
      D[m * ldd + n] = __float2half(0.0f);
    }
  }
}

cudaError_t HopperGemmTopkSoftmax(
    int M, int N, int K, float alpha,
    __half const *A, int lda,
    __half const *B, int ldb,
    __half *D, int ldd,
    int topk) {

  int threads = 256;
  int blocks = (M + threads - 1) / threads;
  naive_gemm_topk_softmax_kernel<<<blocks, threads>>>(
      M, N, K, alpha, A, lda, B, ldb, D, ldd, topk);
  return cudaGetLastError();
}
