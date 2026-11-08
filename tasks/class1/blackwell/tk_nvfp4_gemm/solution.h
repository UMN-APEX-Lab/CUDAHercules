#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <stdint.h>

// Baseline NVFP4 GEMM, C = gA * gB * (A * sA) * (B * sB)^T (plain CUDA-core kernel: shared-memory
// tiling, FP32 FMA) so that the harness can be exercised end to end; far from the tensor-core roofline.
//   A: [M, K] E2M1 packed two per byte (element 2j in the low nibble), B: [N, K] likewise,
//   C: [M, N] BF16 row-major. One E4M3 scale per 16 consecutive K elements; scale of (row r,
//   k-block kb) lives at ((r / 128) * (K / 64) + kb / 4) * 512 + (r % 32) * 16 + ((r / 32) % 4) * 4 + kb % 4.
#define TILE_M 64
#define TILE_N 64
#define TILE_K 16

__device__ __forceinline__ int nv_scale_index(int row, int kb, int K) {
  return ((row / 128) * (K / 64) + kb / 4) * 512 + (row % 32) * 16 + ((row / 32) % 4) * 4 + kb % 4;
}
// E2M1: sign bit 3, magnitude codes 0..7 -> 0, .5, 1, 1.5, 2, 3, 4, 6
__device__ __forceinline__ float e2m1_to_float(unsigned nib) {
  const float lut[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
  float v = lut[nib & 7];
  return (nib & 8) ? -v : v;
}

__global__ void baseline_nvfp4_gemm_kernel(const uint8_t *A, const uint8_t *B, const __nv_fp8_e4m3 *A_scale,
                                           const __nv_fp8_e4m3 *B_scale, const float *ga, const float *gb, __nv_bfloat16 *C,
                                           int M, int N, int K) {
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
      if (row0 + r < M && gk < K) {
        unsigned byte = A[size_t(row0 + r) * (K / 2) + gk / 2];
        a = e2m1_to_float((gk & 1) ? (byte >> 4) : (byte & 0xF)) *
            float(A_scale[nv_scale_index(row0 + r, gk / 16, K)]);
      }
      if (col0 + r < N && gk < K) {
        unsigned byte = B[size_t(col0 + r) * (K / 2) + gk / 2];
        b = e2m1_to_float((gk & 1) ? (byte >> 4) : (byte & 0xF)) *
            float(B_scale[nv_scale_index(col0 + r, gk / 16, K)]);
      }
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
      if (r < M && c < N) C[size_t(r) * N + c] = __float2bfloat16(acc[i][j] * (*ga) * (*gb));
    }
}

void Nvfp4Gemm(__nv_fp4x2_e2m1 *A, __nv_fp4x2_e2m1 *B, __nv_fp8_e4m3 *A_scale, __nv_fp8_e4m3 *B_scale,
               float *A_scale_global, float *B_scale_global, __nv_bfloat16 *C, int M, int N, int K) {
  dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
  baseline_nvfp4_gemm_kernel<<<grid, 256>>>(reinterpret_cast<const uint8_t *>(A),
                                            reinterpret_cast<const uint8_t *>(B), A_scale, B_scale,
                                            A_scale_global, B_scale_global, C, M, N, K);
}
