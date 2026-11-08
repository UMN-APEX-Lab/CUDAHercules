#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Naive 3D convolution fprop (correctness baseline): tiled implicit GEMM with FP32 accumulation,
// the im2col gather is done on the fly while loading the shared-memory tiles.
//   input  NDHWC [N, D, H, W, C], filter KTRSC [K, T, R, S, C], output NDHWC [N, Z, P, Q, K]
//   out[n,z,p,q,k] = sum_{t,r,s,c} in[n, z*sd+t-pd, p*sh+r-ph, q*sw+s-pw, c] * flt[k,t,r,s,c]
#define NC_TILE 64
#define NC_KT 16

__global__ void naive_conv3d_fprop_kernel(
    int N, int D, int H, int W, int C, int K, int T, int R, int S, int Z, int P, int Q,
    int pad_d, int pad_h, int pad_w, int stride_d, int stride_h, int stride_w,
    __half const *input, __half const *filter, __half *output) {
  __shared__ float As[NC_KT][NC_TILE + 1];
  __shared__ float Bs[NC_KT][NC_TILE + 1];
  int tx = threadIdx.x % 16, ty = threadIdx.x / 16;
  long long M = (long long)N * Z * P * Q;
  int Kg = T * R * S * C;
  long long row0 = (long long)blockIdx.x * NC_TILE;
  int col0 = blockIdx.y * NC_TILE;

  // Each thread loads rows (threadIdx.x / 16) + 16 j, k offset threadIdx.x % 16.
  int kk = threadIdx.x % 16;
  int ld_r = threadIdx.x / 16;
  int base_d[4], base_h[4], base_w[4], base_n[4];
  bool row_ok[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    long long m = row0 + ld_r + 16 * j;
    row_ok[j] = m < M;
    long long mm = row_ok[j] ? m : 0;
    int q = int(mm % Q); mm /= Q;
    int p = int(mm % P); mm /= P;
    int z = int(mm % Z); mm /= Z;
    base_n[j] = int(mm);
    base_d[j] = z * stride_d - pad_d;
    base_h[j] = p * stride_h - pad_h;
    base_w[j] = q * stride_w - pad_w;
  }

  float acc[4][4] = {};
  for (int k0 = 0; k0 < Kg; k0 += NC_KT) {
    int kg = k0 + kk;
    int c = 0, s = 0, r = 0, t = 0;
    bool k_ok = kg < Kg;
    if (k_ok) {
      int tmp = kg;
      c = tmp % C; tmp /= C;
      s = tmp % S; tmp /= S;
      r = tmp % R; t = tmp / R;
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      float a = 0.f, b = 0.f;
      if (k_ok && row_ok[j]) {
        int d = base_d[j] + t, h = base_h[j] + r, w = base_w[j] + s;
        if (d >= 0 && d < D && h >= 0 && h < H && w >= 0 && w < W)
          a = __half2float(input[((((size_t)base_n[j] * D + d) * H + h) * W + w) * C + c]);
      }
      int col = col0 + ld_r + 16 * j;
      if (k_ok && col < K) b = __half2float(filter[(size_t)col * Kg + kg]);
      As[kk][ld_r + 16 * j] = a;
      Bs[kk][ld_r + 16 * j] = b;
    }
    __syncthreads();
#pragma unroll
    for (int k = 0; k < NC_KT; ++k)
#pragma unroll
      for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] += As[k][ty * 4 + i] * Bs[k][tx * 4 + j];
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      long long m = row0 + ty * 4 + i;
      int k = col0 + tx * 4 + j;
      if (m < M && k < K) output[m * K + k] = __float2half_rn(acc[i][j]);
    }
}

cudaError_t BlackwellConv3dFprop(
    int N, int D, int H, int W, int C,
    int K, int T, int R, int S,
    int pad_d, int pad_h, int pad_w,
    int stride_d, int stride_h, int stride_w,
    __half const *input, __half const *filter, __half *output) {
  int Z = (D + 2 * pad_d - T) / stride_d + 1;
  int P = (H + 2 * pad_h - R) / stride_h + 1;
  int Q = (W + 2 * pad_w - S) / stride_w + 1;
  long long M = (long long)N * Z * P * Q;
  dim3 grid((unsigned)((M + NC_TILE - 1) / NC_TILE), (K + NC_TILE - 1) / NC_TILE);
  naive_conv3d_fprop_kernel<<<grid, 256>>>(N, D, H, W, C, K, T, R, S, Z, P, Q, pad_d, pad_h,
                                           pad_w, stride_d, stride_h, stride_w, input, filter,
                                           output);
  return cudaGetLastError();
}
