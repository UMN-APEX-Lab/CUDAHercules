#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

// Simple SIMT solution for the FP8 blockwise-scaled GEMM (FP32 FMA micro-tile GEMM, 128x128 tile,
// 8x8 outputs per thread). Not tensor-core based: a real solution should use tcgen05.mma (kind::f8f6f4)
// with per-128-K-block promotion of the partial sums to FP32.
//
//   A  : [M, K]    FP8 E4M3, K contiguous          SFA[kb * M + m]            (FP32)
//   B  : [N, K]    FP8 E4M3, K contiguous          SFB[kb * ceil(N/128) + n/128]
//   D  : [M, N] row-major BF16
//   D[m,n] = sum_kb SFA[kb,m] * SFB[kb,n/128] * sum_{k in kb} A[m,k] * B[n,k]

namespace blk8 {
constexpr int BM = 128, BN = 128, BK = 32;

__global__ void __launch_bounds__(256) blockwise_gemm_kernel(int M, int N, int K, __nv_fp8_e4m3 const *A,
                                                            float const *SFA, __nv_fp8_e4m3 const *B,
                                                            float const *SFB, __nv_bfloat16 *D) {
  __shared__ __align__(16) float As[BK][BM];
  __shared__ __align__(16) float Bs[BK][BN];
  int t = threadIdx.x, tx = t % 16, ty = t / 16;
  int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
  int nbk = (N + 127) / 128;
  float acc[8][8] = {};    // final accumulators (scaled)
  float part[8][8];        // partial sums of the current 128-wide K block
  float sa[8];
  float sb = 1.f;
  int Kend = K;
  for (int k0 = 0; k0 < Kend; k0 += BK) {
    if ((k0 & 127) == 0) {
      // new scale block: read the scales of this thread's rows / the block's columns
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        int m = m0 + ty * 8 + i;
        sa[i] = m < M ? SFA[size_t(k0 / 128) * M + m] : 0.f;
      }
      sb = SFB[size_t(k0 / 128) * nbk + n0 / 128];
#pragma unroll
      for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j) part[i][j] = 0.f;
    }
    // load A / B tiles (BM x BK fp8): threads 0..127 -> A rows, 128..255 -> B rows, 32 bytes each
    {
      bool isA = t < 128;
      int r = isA ? t : t - 128;
      int row = (isA ? m0 : n0) + r;
      int rows = isA ? M : N;
      auto *S = isA ? As : Bs;
      if (row >= rows) {
        for (int k = 0; k < BK; ++k) S[k][r] = 0.f;
      } else {
        uint4 v0 = *reinterpret_cast<uint4 const *>(reinterpret_cast<uint8_t const *>(isA ? A : B) + size_t(row) * K + k0);
        uint4 v1 = *reinterpret_cast<uint4 const *>(reinterpret_cast<uint8_t const *>(isA ? A : B) + size_t(row) * K + k0 + 16);
        uint8_t const *b0 = reinterpret_cast<uint8_t const *>(&v0), *b1 = reinterpret_cast<uint8_t const *>(&v1);
#pragma unroll
        for (int k = 0; k < 16; ++k) {
          __nv_fp8_e4m3 q0, q1;
          q0.__x = b0[k]; q1.__x = b1[k];
          float f0 = float(q0), f1 = float(q1);
          S[k][r] = f0; S[16 + k][r] = f1;
        }
      }
    }
    __syncthreads();
#pragma unroll 8
    for (int kk = 0; kk < BK; ++kk) {
      int k = kk;
      float4 a0 = *reinterpret_cast<float4 *>(&As[k][ty * 8]), a1 = *reinterpret_cast<float4 *>(&As[k][ty * 8 + 4]);
      float4 b0 = *reinterpret_cast<float4 *>(&Bs[k][tx * 8]), b1 = *reinterpret_cast<float4 *>(&Bs[k][tx * 8 + 4]);
      float a[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
      float b[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
#pragma unroll
      for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j) part[i][j] = fmaf(a[i], b[j], part[i][j]);
    }
    __syncthreads();
    if ((k0 & 127) == 128 - BK) {   // end of the scale block: promote
#pragma unroll
      for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          acc[i][j] = fmaf(part[i][j], sa[i] * sb, acc[i][j]);
        }
    }
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    int m = m0 + ty * 8 + i;
    if (m >= M) continue;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      int n = n0 + tx * 8 + j;
      if (n < N) D[size_t(m) * N + n] = __float2bfloat16_rn(acc[i][j]);
    }
  }
}
}  // namespace blk8

cudaError_t BlackwellBlockwiseGemm(int M, int N, int K, __nv_fp8_e4m3 const *A, float const *SFA,
                                   __nv_fp8_e4m3 const *B, float const *SFB, __nv_bfloat16 *D) {
  dim3 grid((N + blk8::BN - 1) / blk8::BN, (M + blk8::BM - 1) / blk8::BM);
  blk8::blockwise_gemm_kernel<<<grid, 256>>>(M, N, K, A, SFA, B, SFB, D);
  return cudaGetLastError();
}
