#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

// Baseline W4A16 GEMM (plain CUDA-core kernel: shared-memory tiling, FP32 FMA) so that the
// harness can be exercised end to end; it is far from the roofline.
//   A: row-major [M, K] BF16 (lda = K)
//   B: packed INT4, column-major [K, N]: column j = K/2 bytes at B + j*ldb (ldb = K/2); element
//      (k, j) is the low nibble (even k) or high nibble (odd k) of byte k/2; two's complement.
//   B_scales: BF16 [N, K/128] row-major (scale of (k, j) at B_scales[j*scale_ldb + k/128])
//   D: row-major [M, N] BF16 (ldd = N);  D = alpha * A * (B * scale) + beta * D
#define TILE_M 64
#define TILE_N 64
#define TILE_K 16

__global__ void baseline_w4a16_kernel(
    int M, int N, int K, float alpha,
    __nv_bfloat16 const *A, int lda,
    uint8_t const *B, int ldb,
    __nv_bfloat16 const *B_scales, int scale_ldb,
    float beta, __nv_bfloat16 *D, int ldd) {
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
      As[kk][r] = (row0 + r < M && gk < K)
                      ? __bfloat162float(A[size_t(row0 + r) * lda + gk]) : 0.f;
      float b = 0.f;
      if (col0 + r < N && gk < K) {
        int byte = B[(gk >> 1) + size_t(col0 + r) * ldb];
        int q = (gk & 1) ? (byte >> 4) : (byte & 0xF);
        q = (q ^ 8) - 8;   // sign-extend the nibble
        float sc = __bfloat162float(B_scales[size_t(col0 + r) * scale_ldb + gk / 128]);
        b = float(q) * sc;
      }
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
      if (r < M && c < N) {
        float v = alpha * acc[i][j];
        if (beta != 0.f) v += beta * __bfloat162float(D[size_t(r) * ldd + c]);
        D[size_t(r) * ldd + c] = __float2bfloat16(v);
      }
    }
}

cudaError_t BlackwellMixedDtypeGemm(
    int M, int N, int K, float alpha,
    __nv_bfloat16 const *A, int lda,
    uint8_t const *B, int ldb,
    __nv_bfloat16 const *B_scales, int scale_ldb,
    float beta, __nv_bfloat16 *D, int ldd) {
  dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
  baseline_w4a16_kernel<<<grid, 256>>>(M, N, K, alpha, A, lda, B, ldb, B_scales, scale_ldb,
                                       beta, D, ldd);
  return cudaGetLastError();
}
