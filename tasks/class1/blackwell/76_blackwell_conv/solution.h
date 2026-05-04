#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Naive 3D convolution fprop: NDHWC layout
// output[n][z][p][q][k] = sum_{t,r,s,c} input[n][d][h][w][c] * filter[k][t][r][s][c]
//   where d = z*stride_d + t - pad_d, h = p*stride_h + r - pad_h, w = q*stride_w + s - pad_w
__global__ void naive_conv3d_fprop_kernel(
    int N, int D, int H, int W, int C,
    int K, int T, int R, int S,
    int Z, int P, int Q,
    int pad_d, int pad_h, int pad_w,
    int stride_d, int stride_h, int stride_w,
    __half const *input, __half const *filter, __half *output) {

  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int total = N * Z * P * Q * K;
  if (idx >= total) return;

  // Decode linear index -> (n, z, p, q, k)
  int tmp = idx;
  int k = tmp % K; tmp /= K;
  int q = tmp % Q; tmp /= Q;
  int p = tmp % P; tmp /= P;
  int z = tmp % Z; tmp /= Z;
  int n = tmp;

  float acc = 0.0f;
  for (int t = 0; t < T; ++t)
    for (int r = 0; r < R; ++r)
      for (int s = 0; s < S; ++s) {
        int d = z * stride_d + t - pad_d;
        int h = p * stride_h + r - pad_h;
        int w = q * stride_w + s - pad_w;
        if (d >= 0 && d < D && h >= 0 && h < H && w >= 0 && w < W) {
          for (int c = 0; c < C; ++c) {
            int in_idx = ((((n * D + d) * H + h) * W + w) * C + c);
            int fi_idx = ((((k * T + t) * R + r) * S + s) * C + c);
            acc += __half2float(input[in_idx]) * __half2float(filter[fi_idx]);
          }
        }
      }

  output[idx] = __float2half(acc);
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
  int total = N * Z * P * Q * K;

  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  naive_conv3d_fprop_kernel<<<blocks, threads>>>(
    N, D, H, W, C, K, T, R, S, Z, P, Q,
    pad_d, pad_h, pad_w, stride_d, stride_h, stride_w,
    input, filter, output);
  return cudaGetLastError();
}
