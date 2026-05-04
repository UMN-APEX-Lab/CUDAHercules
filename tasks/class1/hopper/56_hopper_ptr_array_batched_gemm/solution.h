#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Naive batched GEMM with pointer arrays
// A row-major [M, K] (lda = K), B col-major [K, N] (ldb = K), D col-major [M, N] (ldd = M)
__global__ void naive_batched_gemm_kernel(
    int M, int N, int K, float alpha,
    __half const *A, int lda,
    __half const *B, int ldb,
    float beta,
    __half *D, int ldd) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < M && j < N) {
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
      float a_val = __half2float(A[i * lda + k]);
      float b_val = __half2float(B[k + j * ldb]);
      acc += a_val * b_val;
    }
    float d_val = alpha * acc + beta * __half2float(D[i + j * ldd]);
    D[i + j * ldd] = __float2half(d_val);
  }
}

// Helper kernel to extract pointers from device array and launch per-batch
__global__ void dispatch_batched_gemm_kernel(
    int M, int N, int K, int batch, float alpha,
    __half const **A_ptrs, int lda,
    __half const **B_ptrs, int ldb,
    float beta,
    __half **D_ptrs, int ldd,
    __half const **out_A, __half const **out_B, __half **out_D) {
  // Copy pointers to output arrays (accessible from host via memcpy)
  int idx = threadIdx.x;
  if (idx < batch) {
    out_A[idx] = A_ptrs[idx];
    out_B[idx] = B_ptrs[idx];
    out_D[idx] = D_ptrs[idx];
  }
}

cudaError_t HopperPtrArrayBatchedGemm(
    int M, int N, int K, int batch,
    float alpha,
    __half const **A_ptrs, int lda,
    __half const **B_ptrs, int ldb,
    float beta,
    __half **D_ptrs, int ldd) {

  // Copy device pointer arrays to host
  std::vector<__half const *> h_A(batch), h_B(batch);
  std::vector<__half *> h_D(batch);
  cudaMemcpy(h_A.data(), A_ptrs, batch * sizeof(__half const *), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_B.data(), B_ptrs, batch * sizeof(__half const *), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_D.data(), D_ptrs, batch * sizeof(__half *), cudaMemcpyDeviceToHost);

  dim3 block(16, 16);
  dim3 grid((M + 15) / 16, (N + 15) / 16);

  for (int b = 0; b < batch; ++b) {
    naive_batched_gemm_kernel<<<grid, block>>>(
      M, N, K, alpha, h_A[b], lda, h_B[b], ldb, beta, h_D[b], ldd);
  }

  return cudaGetLastError();
}
