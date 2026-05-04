#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "cutlass/numeric_types.h"

// Naive FP4 grouped GEMM for testing the harness
// A row-major FP4 [M,K], B col-major FP4 [K,N], D row-major FP16 [M,N]
// Scale factors: A_scale [M, ceil(K/32)], B_scale [N, ceil(K/32)]
__global__ void naive_fp4_gemm_kernel(
    int M, int N, int K,
    float alpha, float beta,
    cutlass::float_e2m1_t const *A, int lda,
    float const *A_scale, int scale_lda,
    cutlass::float_e2m1_t const *B, int ldb,
    float const *B_scale, int scale_ldb,
    __half *D, int ldd) {

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  if (i < M && j < N) {
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
      int kg = k / 32;
      float a_val = float(A[i * lda + k]) * A_scale[i * scale_lda + kg];
      float b_val = float(B[k + j * ldb]) * B_scale[j * scale_ldb + kg];
      acc += a_val * b_val;
    }
    D[i * ldd + j] = __float2half(alpha * acc + beta * __half2float(D[i * ldd + j]));
  }
}

cudaError_t Fp4UltraGroupedGemm(
    int num_groups,
    int const *Ms, int const *Ns, int const *Ks,
    float alpha, float beta,
    cutlass::float_e2m1_t const * const *A_ptrs, int const *ldas,
    float const * const *A_scale_ptrs, int const *scale_ldas,
    cutlass::float_e2m1_t const * const *B_ptrs, int const *ldbs,
    float const * const *B_scale_ptrs, int const *scale_ldbs,
    __half * const *D_ptrs, int const *ldds) {

  // Copy group parameters from device to host
  std::vector<int> h_Ms(num_groups), h_Ns(num_groups), h_Ks(num_groups);
  std::vector<int> h_ldas(num_groups), h_ldbs(num_groups), h_ldds(num_groups);
  std::vector<int> h_sldas(num_groups), h_sldbs(num_groups);
  std::vector<cutlass::float_e2m1_t const*> h_A(num_groups), h_B(num_groups);
  std::vector<float const*> h_sA(num_groups), h_sB(num_groups);
  std::vector<__half*> h_D(num_groups);

  cudaMemcpy(h_Ms.data(), Ms, sizeof(int)*num_groups, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_Ns.data(), Ns, sizeof(int)*num_groups, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_Ks.data(), Ks, sizeof(int)*num_groups, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_ldas.data(), ldas, sizeof(int)*num_groups, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_ldbs.data(), ldbs, sizeof(int)*num_groups, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_ldds.data(), ldds, sizeof(int)*num_groups, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_sldas.data(), scale_ldas, sizeof(int)*num_groups, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_sldbs.data(), scale_ldbs, sizeof(int)*num_groups, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_A.data(), A_ptrs, sizeof(void*)*num_groups, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_B.data(), B_ptrs, sizeof(void*)*num_groups, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_sA.data(), A_scale_ptrs, sizeof(void*)*num_groups, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_sB.data(), B_scale_ptrs, sizeof(void*)*num_groups, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_D.data(), D_ptrs, sizeof(void*)*num_groups, cudaMemcpyDeviceToHost);

  for (int g = 0; g < num_groups; ++g) {
    dim3 block(16, 16);
    dim3 grid((h_Ms[g] + 15) / 16, (h_Ns[g] + 15) / 16);
    naive_fp4_gemm_kernel<<<grid, block>>>(
      h_Ms[g], h_Ns[g], h_Ks[g], alpha, beta,
      h_A[g], h_ldas[g], h_sA[g], h_sldas[g],
      h_B[g], h_ldbs[g], h_sB[g], h_sldbs[g],
      h_D[g], h_ldds[g]);
  }

  return cudaGetLastError();
}
