#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

// Naive sparse GEMM for testing the harness
// Decompresses 2:4 sparse A and performs dense GEMM
// A_compressed: M x (K/2) row-major, A_metadata: encodes positions
// B: col-major [K, N] (ldb = K), D: row-major [M, N] (ldd = N)
__global__ void naive_sparse_gemm_kernel(
    int M, int N, int K, float alpha,
    __half const *A_compressed, uint16_t const *A_metadata,
    __half const *B, int ldb, float beta,
    __half *D, int ldd) {

  int m = blockIdx.x * blockDim.x + threadIdx.x;
  int n = blockIdx.y * blockDim.y + threadIdx.y;
  if (m >= M || n >= N) return;

  int K_comp = K / 2;
  // Metadata layout: each uint16_t encodes 4 groups of 4 elements
  // Each group uses 4 bits: 2 bits for idx0, 2 bits for idx1
  int meta_cols = K / 16;
  if (meta_cols == 0) meta_cols = 1;

  float acc = 0.0f;

  // Iterate over groups of 4 in K dimension
  for (int kg = 0; kg < K; kg += 4) {
    int group = kg / 4;
    int meta_word_idx = group / 4;
    int meta_shift = (group % 4) * 4;

    uint16_t meta_word = A_metadata[m * meta_cols + meta_word_idx];
    uint16_t bits = (meta_word >> meta_shift) & 0xF;
    int idx0 = bits & 0x3;
    int idx1 = (bits >> 2) & 0x3;

    int ci = (kg / 4) * 2;
    float a0 = __half2float(A_compressed[m * K_comp + ci]);
    float a1 = __half2float(A_compressed[m * K_comp + ci + 1]);

    float b0 = __half2float(B[(kg + idx0) + n * ldb]);
    float b1 = __half2float(B[(kg + idx1) + n * ldb]);

    acc += a0 * b0 + a1 * b1;
  }

  D[m * ldd + n] = __float2half(alpha * acc);
}

cudaError_t HopperSparseGemm(
    int M, int N, int K, float alpha,
    __half const *A_compressed, uint16_t const *A_metadata,
    __half const *B, int ldb, float beta,
    __half *D, int ldd) {

  dim3 block(16, 16);
  dim3 grid((M + 15) / 16, (N + 15) / 16);
  naive_sparse_gemm_kernel<<<grid, block>>>(
      M, N, K, alpha, A_compressed, A_metadata, B, ldb, beta, D, ldd);
  return cudaGetLastError();
}
