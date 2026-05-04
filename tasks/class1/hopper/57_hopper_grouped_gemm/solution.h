#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

// Naive grouped FP8 GEMM for testing the harness
// A[i] row-major, B[i] col-major, D[i] col-major (matching CUTLASS ColumnMajor output)
__global__ void naive_grouped_fp8_gemm_kernel(
    int M, int N, int K, float alpha, float beta,
    __nv_fp8_e4m3 const *A, int lda,
    __nv_fp8_e4m3 const *B, int ldb,
    __half *D, int ldd) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < M && j < N) {
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
      float a_val = float(A[i * lda + k]);
      float b_val = float(B[k + j * ldb]);
      acc += a_val * b_val;
    }
    // ColumnMajor output: D[i,j] = D[i + j * ldd]
    D[i + j * M] = __float2half(alpha * acc);
  }
}

cudaError_t HopperGroupedGemm(
    int problem_count,
    int const *Ms, int const *Ns, int const *Ks,
    float alpha,
    __nv_fp8_e4m3 const **A_ptrs, int const *ldas,
    __nv_fp8_e4m3 const **B_ptrs, int const *ldbs,
    float beta,
    __half **D_ptrs, int const *ldds) {

  // Copy problem info to host
  std::vector<int> h_Ms(problem_count), h_Ns(problem_count), h_Ks(problem_count);
  std::vector<int> h_ldas(problem_count), h_ldbs(problem_count), h_ldds(problem_count);
  std::vector<__nv_fp8_e4m3 const *> h_A(problem_count);
  std::vector<__nv_fp8_e4m3 const *> h_B(problem_count);
  std::vector<__half *> h_D(problem_count);

  cudaMemcpy(h_Ms.data(), Ms, problem_count * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_Ns.data(), Ns, problem_count * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_Ks.data(), Ks, problem_count * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_ldas.data(), ldas, problem_count * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_ldbs.data(), ldbs, problem_count * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_ldds.data(), ldds, problem_count * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_A.data(), A_ptrs, problem_count * sizeof(__nv_fp8_e4m3 const *), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_B.data(), B_ptrs, problem_count * sizeof(__nv_fp8_e4m3 const *), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_D.data(), D_ptrs, problem_count * sizeof(__half *), cudaMemcpyDeviceToHost);

  for (int p = 0; p < problem_count; ++p) {
    int M = h_Ms[p], N = h_Ns[p], K = h_Ks[p];
    if (M == 0 || N == 0 || K == 0) continue;
    dim3 block(16, 16);
    dim3 grid((M + 15) / 16, (N + 15) / 16);
    naive_grouped_fp8_gemm_kernel<<<grid, block>>>(
        M, N, K, alpha, beta,
        h_A[p], h_ldas[p], h_B[p], h_ldbs[p],
        h_D[p], h_ldds[p]);
  }
  return cudaGetLastError();
}
