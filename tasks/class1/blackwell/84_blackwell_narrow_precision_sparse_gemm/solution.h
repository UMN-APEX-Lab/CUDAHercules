#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

// Simple SIMT solution for the 2:4-structured-sparse NVFP4 x NVFP4 -> BF16 block-scaled GEMM.
// Not tensor-core based: it expands the compressed A tile (values + metadata + scales) into a dense
// FP32 tile in shared memory and runs an FP32 FMA micro-tile GEMM (128x128 tile, 8x8 per thread).
// A real solution should use the sparse tcgen05.mma (kind::mxf4nvf4 with sparsity) instead.
// Formats: see description.txt.

namespace sp4 {
constexpr int BM = 128, BN = 128, BK = 32;   // BK = 32 logical K = 16 stored bytes of A / 2 scale blocks

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

// Expand the 32 logical K elements [k0, k0+32) of A row `row` into S[0..32)[col] (zeros where pruned).
__device__ __forceinline__ void load_a_row(float (*S)[BM], int col, uint8_t const *A, uint8_t const *E,
                                           uint8_t const *SFA, int row, int M, int Mp, int K, int k0, int kb_pad) {
  float v[32];
#pragma unroll
  for (int k = 0; k < 32; ++k) v[k] = 0.f;
  if (row < M) {
    uint4 raw = *reinterpret_cast<uint4 const *>(A + size_t(row) * (K / 4) + k0 / 4);   // 16 bytes = 16 kept pairs
    uint8_t const *ab = reinterpret_cast<uint8_t const *>(&raw);
    int g = k0 / 16;                                                                    // metadata groups g, g+1
    int sb = k0 / 32;                                                                   // scale column
    uint8_t e0 = E[size_t(g / 16) * Mp * 16 + size_t(row / 128) * 2048 + (row % 128) * 16 + (g % 16)];
    uint8_t e1 = E[size_t(g / 16) * Mp * 16 + size_t(row / 128) * 2048 + (row % 128) * 16 + ((g + 1) % 16)];
    // (g and g+1 are in the same 16-group because k0 is a multiple of 32 -> g is even)
    unsigned nib[4] = {e0 & 15u, e0 >> 4, e1 & 15u, e1 >> 4};
    float s0 = ue4m3(SFA[sf_off(row, sb, kb_pad)]), s1 = s0;
#pragma unroll
    for (int c = 0; c < 4; ++c) {                  // 4 chunks of 8 logical elements
      float s = c < 2 ? s0 : s1;
#pragma unroll
      for (int sl = 0; sl < 2; ++sl) {             // the two kept pairs of the chunk
        unsigned idx = (nib[c] >> (2 * sl)) & 3;
        unsigned b = ab[2 * c + sl];
        int pos = c * 8 + 2 * idx;
        v[pos] = e2m1(b) * s;
        v[pos + 1] = e2m1(b >> 4) * s;
      }
    }
  }
#pragma unroll
  for (int k = 0; k < 32; ++k) S[k][col] = v[k];
}

__device__ __forceinline__ void load_b_row(float (*S)[BN], int col, uint8_t const *B, uint8_t const *SFB,
                                           int row, int N, int K, int k0, int kb_pad) {
  if (row >= N) {
    for (int k = 0; k < BK; ++k) S[k][col] = 0.f;
    return;
  }
  uint4 raw = *reinterpret_cast<uint4 const *>(B + size_t(row) * (K / 2) + k0 / 2);
  uint8_t const *bytes = reinterpret_cast<uint8_t const *>(&raw);
  float s = ue4m3(SFB[sf_off(row, k0 / 32, kb_pad)]);
#pragma unroll
  for (int blk = 0; blk < 2; ++blk) {
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      unsigned byte = bytes[blk * 8 + j];
      S[blk * 16 + 2 * j][col] = e2m1(byte) * s;
      S[blk * 16 + 2 * j + 1][col] = e2m1(byte >> 4) * s;
    }
  }
}

__global__ void __launch_bounds__(256) sparse_nvfp4_gemm_kernel(int M, int N, int K, float alpha, uint8_t const *A,
                                                                uint8_t const *E, uint8_t const *SFA,
                                                                uint8_t const *B, uint8_t const *SFB,
                                                                __nv_bfloat16 *D) {
  __shared__ __align__(16) float As[BK][BM];
  __shared__ __align__(16) float Bs[BK][BN];
  int t = threadIdx.x, tx = t % 16, ty = t / 16;
  int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
  int kb_pad = (K / 32 + 3) / 4 * 4, Mp = (M + 127) / 128 * 128;
  float acc[8][8] = {};
  int Kend = K;
  for (int k0 = 0; k0 < Kend; k0 += BK) {
    if (t < 128) load_a_row(As, t, A, E, SFA, m0 + t, M, Mp, K, k0, kb_pad);
    else load_b_row(Bs, t - 128, B, SFB, n0 + t - 128, N, K, k0, kb_pad);
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
}  // namespace sp4

cudaError_t BlackwellNvfp4SparseGemm(int M, int N, int K, float alpha, uint8_t const *A, uint8_t const *E,
                                     uint8_t const *SFA, uint8_t const *B, uint8_t const *SFB,
                                     __nv_bfloat16 *D) {
  dim3 grid((N + sp4::BN - 1) / sp4::BN, (M + sp4::BM - 1) / sp4::BM);
  sp4::sparse_nvfp4_gemm_kernel<<<grid, 256>>>(M, N, K, alpha, A, E, SFA, B, SFB, D);
  return cudaGetLastError();
}
