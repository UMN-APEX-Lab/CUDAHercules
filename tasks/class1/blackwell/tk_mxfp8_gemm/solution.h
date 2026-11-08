#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>

// Baseline MXFP8 GEMM, C = (A * sA) * (B * sB)^T (plain CUDA-core kernel: shared-memory tiling, FP32
// FMA) so that the harness can be exercised end to end; it is far from the tensor-core roofline.
//   A: [M, K] E4M3 row-major, B: [N, K] E4M3 row-major, C: [M, N] BF16 row-major.
//   One E8M0 scale per 32 consecutive K elements; scale of (row r, k-block kb) lives at
//     ((r / 128) * (K / 128) + kb / 4) * 512 + (r % 32) * 16 + ((r / 32) % 4) * 4 + kb % 4
//   (the swizzled 128x4 tile layout; one 512-byte tile per 128 rows x 128 K).
#define TILE_M 64
#define TILE_N 64
#define TILE_K 16

__device__ __forceinline__ int mx_scale_index(int row, int kb, int K) {
  return ((row / 128) * (K / 128) + kb / 4) * 512 + (row % 32) * 16 + ((row / 32) % 4) * 4 + kb % 4;
}

__global__ void baseline_mxfp8_gemm_kernel(const __nv_fp8_e4m3 *A, const __nv_fp8_e4m3 *B,
                                           const __nv_fp8_e8m0 *A_scale, const __nv_fp8_e8m0 *B_scale,
                                           __nv_bfloat16 *C, int M, int N, int K) {
  __shared__ float As[TILE_K][TILE_M + 4];
  __shared__ float Bs[TILE_K][TILE_N + 4];
  int tid = threadIdx.x;               // 256 threads
  int tx = tid % 16, ty = tid / 16;
  int row0 = blockIdx.y * TILE_M, col0 = blockIdx.x * TILE_N;
  float acc[4][4] = {};
  for (int k0 = 0; k0 < K; k0 += TILE_K) {
    for (int e = 0; e < 4; ++e) {
      int idx = tid + e * 256;
      int r = idx / TILE_K, kk = idx % TILE_K;
      int gk = k0 + kk;
      float a = 0.f, b = 0.f;
      if (row0 + r < M && gk < K)
        a = float(A[size_t(row0 + r) * K + gk]) * float(A_scale[mx_scale_index(row0 + r, gk / 32, K)]);
      if (col0 + r < N && gk < K)
        b = float(B[size_t(col0 + r) * K + gk]) * float(B_scale[mx_scale_index(col0 + r, gk / 32, K)]);
      As[kk][r] = a;
      Bs[kk][r] = b;
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
      if (r < M && c < N) C[size_t(r) * N + c] = __float2bfloat16(acc[i][j]);
    }
}

void Mxfp8Gemm(__nv_fp8_e4m3 *A, __nv_fp8_e4m3 *B, __nv_fp8_e8m0 *A_scale, __nv_fp8_e8m0 *B_scale,
               __nv_bfloat16 *C, int M, int N, int K) {
  dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
  baseline_mxfp8_gemm_kernel<<<grid, 256>>>(A, B, A_scale, B_scale, C, M, N, K);
}
