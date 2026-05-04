#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Naive N-mode gather/scatter GEMM for testing the harness.
//
// N-mode gather/scatter:
// A row-major [M, K] (lda = K), no gather applied.
// B col-major [K, N_orig] (ldb = K), gather_indices[j] selects column for j in [0, index_size).
// D col-major [M, N_orig] (ldd = M), scatter_indices[j] selects column to write for j in [0, index_size).
//
// The logical GEMM computes C_logical[i, j] = sum_k A[i,k] * B[k, gather[j]]
// Then D[i, scatter[j]] = alpha * C_logical[i, j] + beta * D_old[i, scatter[j]]
//
// Parameters:
//   M: number of rows in A and D
//   index_size: number of gathered/scattered columns (logical N dimension)
//   K: contraction dimension
//   alpha, beta: scalars
//   A: [M, K] row-major
//   lda: leading dimension of A (= K)
//   B: [K, N_orig] col-major
//   ldb: leading dimension of B (= K)
//   gather_indices: array of index_size indices into B's N dimension
//   beta: scalar for accumulation
//   D: [M, N_orig] col-major output
//   ldd: leading dimension of D (= M)
//   scatter_indices: array of index_size indices into D's N dimension
__global__ void naive_gather_scatter_kernel(
    int M, int index_size, int K, float alpha,
    __half const *A, int lda,
    __half const *B, int ldb,
    int const *gather_indices,
    float beta,
    __half *D, int ldd,
    int const *scatter_indices) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= M || j >= index_size) return;

  int j_b = gather_indices[j];    // column index into B
  int j_d = scatter_indices[j];   // column index into D

  float acc = 0.0f;
  for (int k = 0; k < K; ++k) {
    // A row-major: A[i, k] = A[i * lda + k]
    // B col-major: B[k, j_b] = B[k + j_b * ldb]
    acc += __half2float(A[i * lda + k]) * __half2float(B[k + j_b * ldb]);
  }
  // D col-major: D[i, j_d] = D[i + j_d * ldd]
  float old_val = beta * __half2float(D[i + j_d * ldd]);
  D[i + j_d * ldd] = __float2half(alpha * acc + old_val);
}

cudaError_t HopperGatherScatterGemm(
    int M, int index_size, int K, float alpha,
    __half const *A, int lda,
    __half const *B, int ldb,
    int const *gather_indices,
    float beta,
    __half *D, int ldd,
    int const *scatter_indices) {
  dim3 block(16, 16);
  dim3 grid((M + 15) / 16, (index_size + 15) / 16);
  naive_gather_scatter_kernel<<<grid, block>>>(
    M, index_size, K, alpha, A, lda, B, ldb,
    gather_indices, beta, D, ldd, scatter_indices);
  return cudaGetLastError();
}
