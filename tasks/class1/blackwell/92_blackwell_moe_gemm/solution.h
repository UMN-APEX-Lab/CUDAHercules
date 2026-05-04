#pragma once
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <vector>

// Naive MoE GEMM: A row-major FP8, B col-major FP8, D col-major FP16
__global__ void naive_moe_gemm_kernel(
    int M, int N, int K,
    float alpha,
    __nv_fp8_e4m3 const *A, int lda,
    __nv_fp8_e4m3 const *B, int ldb,
    float beta,
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

cudaError_t BlackwellMoeGemm(
    int num_experts,
    int const *Ms, int const *Ns, int const *Ks,
    float alpha, float beta,
    __nv_fp8_e4m3 const * const *A_ptrs, int const *ldas,
    __nv_fp8_e4m3 const * const *B_ptrs, int const *ldbs,
    __half * const *D_ptrs, int const *ldds) {

  // Copy parameters from device to host
  std::vector<int> h_Ms(num_experts), h_Ns(num_experts), h_Ks(num_experts);
  std::vector<int> h_ldas(num_experts), h_ldbs(num_experts), h_ldds(num_experts);
  std::vector<__nv_fp8_e4m3 const*> h_A(num_experts), h_B(num_experts);
  std::vector<__half*> h_D(num_experts);

  cudaMemcpy(h_Ms.data(), Ms, sizeof(int)*num_experts, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_Ns.data(), Ns, sizeof(int)*num_experts, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_Ks.data(), Ks, sizeof(int)*num_experts, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_ldas.data(), ldas, sizeof(int)*num_experts, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_ldbs.data(), ldbs, sizeof(int)*num_experts, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_ldds.data(), ldds, sizeof(int)*num_experts, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_A.data(), A_ptrs, sizeof(void*)*num_experts, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_B.data(), B_ptrs, sizeof(void*)*num_experts, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_D.data(), D_ptrs, sizeof(void*)*num_experts, cudaMemcpyDeviceToHost);

  for (int e = 0; e < num_experts; ++e) {
    dim3 block(16, 16);
    dim3 grid((h_Ms[e] + 15) / 16, (h_Ns[e] + 15) / 16);
    naive_moe_gemm_kernel<<<grid, block>>>(
      h_Ms[e], h_Ns[e], h_Ks[e], alpha,
      h_A[e], h_ldas[e], h_B[e], h_ldbs[e],
      beta, h_D[e], h_ldds[e]);
  }

  return cudaGetLastError();
}
