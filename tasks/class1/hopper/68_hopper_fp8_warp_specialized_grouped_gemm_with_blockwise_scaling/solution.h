#pragma once
#include <cuda_runtime.h>
#include <cuda_fp8.h>

// Naive FP8 blockwise scaling grouped GEMM kernel
// Processes one problem at a time. A row-major, B col-major, D col-major.
__global__ void naive_fp8_blockwise_grouped_gemm_kernel(
    int M, int N, int K, float alpha,
    __nv_fp8_e4m3 const *A, int lda,
    float const *scale_A,
    __nv_fp8_e4m3 const *B, int ldb,
    float const *scale_B,
    __nv_fp8_e4m3 *D, int ldd,
    int block_size) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  if (i < M && j < N) {
    float acc = 0.0f;
    int num_blocks = (K + block_size - 1) / block_size;
    for (int kb = 0; kb < num_blocks; ++kb) {
      int k_start = kb * block_size;
      int k_end = k_start + block_size;
      if (k_end > K) k_end = K;
      float sa = scale_A[i * num_blocks + kb];
      float sb = scale_B[j * num_blocks + kb];
      float block_acc = 0.0f;
      for (int k = k_start; k < k_end; ++k) {
        block_acc += float(A[i * lda + k]) * float(B[k + j * ldb]);
      }
      acc += sa * sb * block_acc;
    }
    // D col-major: D[i + j * ldd]
    D[i + j * ldd] = __nv_fp8_e4m3(alpha * acc);
  }
}

// Host-side wrapper kernel that reads from device pointer arrays
__global__ void dispatch_grouped_gemm(
    int problem_count,
    int const *Ms, int const *Ns, int const *Ks,
    float alpha,
    __nv_fp8_e4m3 const **A_ptrs, int const *ldas,
    float const **scale_A_ptrs,
    __nv_fp8_e4m3 const **B_ptrs, int const *ldbs,
    float const **scale_B_ptrs,
    __nv_fp8_e4m3 **D_ptrs, int const *ldds,
    int block_size,
    int problem_idx) {
  int M = Ms[problem_idx], N = Ns[problem_idx], K = Ks[problem_idx];
  int lda = ldas[problem_idx], ldb = ldbs[problem_idx], ldd = ldds[problem_idx];
  __nv_fp8_e4m3 const *A = A_ptrs[problem_idx];
  __nv_fp8_e4m3 const *B = B_ptrs[problem_idx];
  __nv_fp8_e4m3 *D = D_ptrs[problem_idx];
  float const *sA = scale_A_ptrs[problem_idx];
  float const *sB = scale_B_ptrs[problem_idx];

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  if (i < M && j < N) {
    float acc = 0.0f;
    int num_blocks = (K + block_size - 1) / block_size;
    for (int kb = 0; kb < num_blocks; ++kb) {
      int k_start = kb * block_size;
      int k_end = k_start + block_size;
      if (k_end > K) k_end = K;
      float sa = sA[i * num_blocks + kb];
      float sb = sB[j * num_blocks + kb];
      float block_acc = 0.0f;
      for (int k = k_start; k < k_end; ++k) {
        block_acc += float(A[i * lda + k]) * float(B[k + j * ldb]);
      }
      acc += sa * sb * block_acc;
    }
    D[i + j * ldd] = __nv_fp8_e4m3(alpha * acc);
  }
}

cudaError_t HopperFp8BlockwiseGroupedGemm(
    int problem_count,
    int const *Ms, int const *Ns, int const *Ks,
    float alpha,
    __nv_fp8_e4m3 const **A_ptrs, int const *ldas,
    float const **scale_A_ptrs,
    __nv_fp8_e4m3 const **B_ptrs, int const *ldbs,
    float const **scale_B_ptrs,
    __nv_fp8_e4m3 **D_ptrs, int const *ldds,
    int block_size) {
  // Read problem sizes from device
  std::vector<int> h_Ms(problem_count), h_Ns(problem_count);
  cudaMemcpy(h_Ms.data(), Ms, problem_count * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_Ns.data(), Ns, problem_count * sizeof(int), cudaMemcpyDeviceToHost);

  for (int p = 0; p < problem_count; ++p) {
    int M = h_Ms[p], N = h_Ns[p];
    if (M == 0 || N == 0) continue;
    dim3 block(16, 16);
    dim3 grid((M + 15) / 16, (N + 15) / 16);
    dispatch_grouped_gemm<<<grid, block>>>(
      problem_count, Ms, Ns, Ks, alpha,
      A_ptrs, ldas, scale_A_ptrs,
      B_ptrs, ldbs, scale_B_ptrs,
      D_ptrs, ldds, block_size, p);
  }
  return cudaGetLastError();
}
