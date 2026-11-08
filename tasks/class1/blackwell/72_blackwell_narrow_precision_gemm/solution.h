#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

// Reference-quality-but-simple SIMT solution for the NVFP4 x NVFP4 -> BF16 block-scaled GEMM.
// Not tensor-core based: it dequantizes FP4 values and UE4M3 scales into shared memory and runs an
// FP32 FMA micro-tile GEMM (128x128 tile, 8x8 per thread). A real solution should use
// tcgen05.mma ... kind::mxf4nvf4 with the block scales.
//
//   A, B : [rows, K/2] bytes, K-contiguous, low nibble = even k           (FP4 E2M1)
//   SFA/B: UE4M3 scale per 16 K elements in the swizzled 128x4 layout (see description.txt)
//   D    : [M, N] row-major BF16, D = alpha * (sum_k (a * sfa) * (b * sfb))

namespace nv4 {
constexpr int BM = 128, BN = 128, BK = 32;

__device__ __forceinline__ float e2m1(unsigned code) {
  const float mag[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
  float v = mag[code & 7];
  return (code & 8) ? -v : v;
}
__device__ __forceinline__ float ue4m3(unsigned b) {
  unsigned e = (b >> 3) & 15, m = b & 7;
  return e == 0 ? float(m) * (1.f / 512.f) : (1.f + float(m) * 0.125f) * exp2f(float(int(e) - 7));
}
__device__ __forceinline__ size_t sf_off(int r, int c, int kb_pad) {
  return (size_t(r / 128) * (kb_pad / 4) + c / 4) * 512 + (r % 32) * 16 + ((r / 32) % 4) * 4 + (c % 4);
}

// dequantize 16 packed bytes (32 FP4 values, 2 scale blocks) of one row into smem column `col`
__device__ __forceinline__ void load_row(float (*S)[BM], int col, uint8_t const *data, uint8_t const *sf,
                                         int row, int rows, int K, int k0, int kb_pad, bool is_a) {
  if (row >= rows) {
    for (int k = 0; k < BK; ++k) S[k][col] = 0.f;
    return;
  }
  uint4 raw = *reinterpret_cast<uint4 const *>(data + size_t(row) * (K / 2) + k0 / 2);
  uint8_t const *bytes = reinterpret_cast<uint8_t const *>(&raw);
#pragma unroll
  for (int blk = 0; blk < 2; ++blk) {
    int c = k0 / 16 + blk;
    float s = ue4m3(sf[sf_off(row, c, kb_pad)]);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      unsigned byte = bytes[blk * 8 + j];
      float lo = e2m1(byte), hi = e2m1(byte >> 4);
      lo *= s; hi *= s;
      S[blk * 16 + 2 * j][col] = lo;
      S[blk * 16 + 2 * j + 1][col] = hi;
    }
  }
}

__global__ void __launch_bounds__(256) nvfp4_gemm_kernel(int M, int N, int K, float alpha, uint8_t const *A,
                                                         uint8_t const *SFA, uint8_t const *B,
                                                         uint8_t const *SFB, __nv_bfloat16 *D) {
  __shared__ __align__(16) float As[BK][BM];
  __shared__ __align__(16) float Bs[BK][BN];
  int t = threadIdx.x, tx = t % 16, ty = t / 16;
  int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
  int kb_pad = (K / 16 + 3) / 4 * 4;
  float acc[8][8] = {};
  int Kend = K;
  for (int k0 = 0; k0 < Kend; k0 += BK) {
    if (t < 128) load_row(As, t, A, SFA, m0 + t, M, K, k0, kb_pad, true);
    else load_row(Bs, t - 128, B, SFB, n0 + t - 128, N, K, k0, kb_pad, false);
    __syncthreads();
#pragma unroll 8
    for (int k = 0; k < BK; ++k) {
      float4 a0 = *reinterpret_cast<float4 *>(&As[k][ty * 8]), a1 = *reinterpret_cast<float4 *>(&As[k][ty * 8 + 4]);
      float4 b0 = *reinterpret_cast<float4 *>(&Bs[k][tx * 8]), b1 = *reinterpret_cast<float4 *>(&Bs[k][tx * 8 + 4]);
      float a[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
      float b[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
#pragma unroll
      for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j) acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
    }
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    int m = m0 + ty * 8 + i;
    if (m >= M) continue;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      int n = n0 + tx * 8 + j;
      if (n < N) D[size_t(m) * N + n] = __float2bfloat16_rn(alpha * acc[i][j]);
    }
  }
}
}  // namespace nv4

cudaError_t BlackwellNvfp4Gemm(int M, int N, int K, float alpha, uint8_t const *A, uint8_t const *SFA,
                               uint8_t const *B, uint8_t const *SFB, __nv_bfloat16 *D) {
  dim3 grid((N + nv4::BN - 1) / nv4::BN, (M + nv4::BM - 1) / nv4::BM);
  nv4::nvfp4_gemm_kernel<<<grid, 256>>>(M, N, K, alpha, A, SFA, B, SFB, D);
  return cudaGetLastError();
}
