#pragma once
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>

// Baseline FP8 GEMM (plain CUDA-core kernel: shared-memory tiling, FP32 FMA) so that the harness
// can be exercised end to end; it is far from the tensor-core roofline.
//   A: row-major [M, K] (lda = K), B: column-major [K, N] (ldb = K),
//   C: row-major [M, N] (ldc = N), FP8 E4M3 inputs, FP32 accumulation, BF16 output.
//   C = alpha * A * B + beta * C   (C is not read when beta == 0)
#define TILE_M 64
#define TILE_N 64
#define TILE_K 16

__global__ void baseline_fp8_gemm_kernel(
    int M, int N, int K, float alpha,
    __nv_fp8_e4m3 const *A, int lda,
    __nv_fp8_e4m3 const *B, int ldb,
    float beta, __nv_bfloat16 *C, int ldc) {
  __shared__ float As[TILE_K][TILE_M + 4];
  __shared__ float Bs[TILE_K][TILE_N + 4];
  int tid = threadIdx.x;               // 256 threads
  int tx = tid % 16, ty = tid / 16;
  int row0 = blockIdx.y * TILE_M, col0 = blockIdx.x * TILE_N;
  float acc[4][4] = {};
  for (int k0 = 0; k0 < K; k0 += TILE_K) {
    // 64 rows x 16 k per operand: 4 elements per thread
    for (int e = 0; e < 4; ++e) {
      int idx = tid + e * 256;
      int r = idx / TILE_K, kk = idx % TILE_K;
      int gk = k0 + kk;
      As[kk][r] = (row0 + r < M && gk < K) ? float(A[size_t(row0 + r) * lda + gk]) : 0.f;
      Bs[kk][r] = (col0 + r < N && gk < K) ? float(B[gk + size_t(col0 + r) * ldb]) : 0.f;
    }
    __syncthreads();
#pragma unroll
    for (int kk = 0; kk < TILE_K; ++kk) {
      float a[4], b[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) { a[i] = As[kk][ty + 16 * i]; b[i] = Bs[kk][tx + 16 * i]; }
#pragma unroll
      for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] += a[i] * b[j];
    }
    __syncthreads();
  }
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) {
      int r = row0 + ty + 16 * i, c = col0 + tx + 16 * j;
      if (r < M && c < N) {
        float v = alpha * acc[i][j];
        if (beta != 0.f) v += beta * __bfloat162float(C[size_t(r) * ldc + c]);
        C[size_t(r) * ldc + c] = __float2bfloat16(v);
      }
    }
}

cudaError_t BlackwellFp8Gemm(
    int M, int N, int K, float alpha,
    __nv_fp8_e4m3 const *A, int lda,
    __nv_fp8_e4m3 const *B, int ldb,
    float beta, __nv_bfloat16 *C, int ldc) {
  dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
  baseline_fp8_gemm_kernel<<<grid, 256>>>(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
  return cudaGetLastError();
}
