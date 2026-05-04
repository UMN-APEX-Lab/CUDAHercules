#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

// Naive grouped mixed-dtype GEMM for testing the harness
// A: row-major half [M x K], B: column-major int8 [K x N], D: row-major half [M x N]
// D = alpha * A(half) * B(int8->half) + beta * D
__global__ void naive_mixed_grouped_gemm_kernel(
    int M, int N, int K,
    float alpha, float beta,
    __half const *A, int lda,
    int8_t const *B, int ldb,
    __half *D, int ldd) {

  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < M && col < N) {
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
      // A is row-major: A[row, k] = A[row * lda + k]
      // B is column-major: B[k, col] = B[k + col * ldb]
      acc += __half2float(A[row * lda + k]) * float(B[k + col * ldb]);
    }
    float old_val = __half2float(D[row * ldd + col]);
    D[row * ldd + col] = __float2half(alpha * acc + beta * old_val);
  }
}

cudaError_t HopperMixedGroupedGemm(
    int problem_count,
    int const *Ms, int const *Ns, int const *Ks,
    float alpha,
    __half const **A_ptrs, int const *ldas,
    int8_t const **B_ptrs, int const *ldbs,
    float beta,
    __half **D_ptrs, int const *ldds) {

  // Copy problem info from device to host
  std::vector<int> h_Ms(problem_count), h_Ns(problem_count), h_Ks(problem_count);
  std::vector<int> h_ldas(problem_count), h_ldbs(problem_count), h_ldds(problem_count);
  std::vector<__half const*> h_A_ptrs(problem_count);
  std::vector<int8_t const*> h_B_ptrs(problem_count);
  std::vector<__half*> h_D_ptrs(problem_count);

  cudaMemcpy(h_Ms.data(), Ms, problem_count * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_Ns.data(), Ns, problem_count * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_Ks.data(), Ks, problem_count * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_ldas.data(), ldas, problem_count * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_ldbs.data(), ldbs, problem_count * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_ldds.data(), ldds, problem_count * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_A_ptrs.data(), A_ptrs, problem_count * sizeof(void*), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_B_ptrs.data(), B_ptrs, problem_count * sizeof(void*), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_D_ptrs.data(), D_ptrs, problem_count * sizeof(void*), cudaMemcpyDeviceToHost);

  for (int p = 0; p < problem_count; ++p) {
    int M = h_Ms[p], N = h_Ns[p], K = h_Ks[p];
    dim3 block(16, 16);
    dim3 grid((M + 15) / 16, (N + 15) / 16);
    naive_mixed_grouped_gemm_kernel<<<grid, block>>>(
      M, N, K, alpha, beta,
      h_A_ptrs[p], h_ldas[p],
      h_B_ptrs[p], h_ldbs[p],
      h_D_ptrs[p], h_ldds[p]);
  }
  return cudaGetLastError();
}
