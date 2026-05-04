#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

// Naive grouped GEMM: A row-major FP8, B col-major FP8, D col-major FP16
// D_g = alpha * A_g * B_g + beta * D_g
__global__ void naive_grouped_gemm_kernel(
    int M, int N, int K,
    float alpha, float beta,
    __nv_fp8_e4m3 const *A, int lda,
    __nv_fp8_e4m3 const *B, int ldb,
    __half *D, int ldd) {

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  if (i < M && j < N) {
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
      acc += float(A[i * lda + k]) * float(B[k + j * ldb]);
    }
    D[i + j * ldd] = __float2half(alpha * acc + beta * __half2float(D[i + j * ldd]));
  }
}

cudaError_t BlackwellGroupedGemm(
    int num_groups,
    int const *Ms, int const *Ns, int const *Ks,
    float alpha, float beta,
    __nv_fp8_e4m3 const * const *A_ptrs, int const *ldas,
    __nv_fp8_e4m3 const * const *B_ptrs, int const *ldbs,
    __half * const *D_ptrs, int const *ldds) {

  // Copy pointers and sizes from device to host
  std::vector<int> h_Ms(num_groups), h_Ns(num_groups), h_Ks(num_groups);
  std::vector<int> h_ldas(num_groups), h_ldbs(num_groups), h_ldds(num_groups);
  std::vector<__nv_fp8_e4m3 const*> h_A(num_groups);
  std::vector<__nv_fp8_e4m3 const*> h_B(num_groups);
  std::vector<__half*> h_D(num_groups);

  cudaMemcpy(h_Ms.data(), Ms, sizeof(int) * num_groups, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_Ns.data(), Ns, sizeof(int) * num_groups, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_Ks.data(), Ks, sizeof(int) * num_groups, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_ldas.data(), ldas, sizeof(int) * num_groups, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_ldbs.data(), ldbs, sizeof(int) * num_groups, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_ldds.data(), ldds, sizeof(int) * num_groups, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_A.data(), A_ptrs, sizeof(__nv_fp8_e4m3*) * num_groups, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_B.data(), B_ptrs, sizeof(__nv_fp8_e4m3*) * num_groups, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_D.data(), D_ptrs, sizeof(__half*) * num_groups, cudaMemcpyDeviceToHost);

  for (int g = 0; g < num_groups; ++g) {
    dim3 block(16, 16);
    dim3 grid((h_Ms[g] + 15) / 16, (h_Ns[g] + 15) / 16);
    naive_grouped_gemm_kernel<<<grid, block>>>(
      h_Ms[g], h_Ns[g], h_Ks[g], alpha, beta,
      h_A[g], h_ldas[g], h_B[g], h_ldbs[g], h_D[g], h_ldds[g]);
  }
  return cudaGetLastError();
}
