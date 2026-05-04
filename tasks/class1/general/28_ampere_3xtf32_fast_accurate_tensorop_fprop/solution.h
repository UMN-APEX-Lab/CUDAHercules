#pragma once
#include <cuda_runtime.h>

// Naive Conv2D forward, NHWC layout, float32
__global__ void naive_conv2d_fprop_kernel(
    float const *input, float const *filter, float *output,
    int N, int C, int H, int W, int K, int R, int S,
    int P, int Q, int pad_h, int pad_w, int stride_h, int stride_w) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = N * P * Q * K;
  if (idx >= total) return;

  int k = idx % K; idx /= K;
  int q = idx % Q; idx /= Q;
  int p = idx % P; idx /= P;
  int n = idx;

  float acc = 0.0f;
  for (int r = 0; r < R; ++r)
    for (int s = 0; s < S; ++s)
      for (int c = 0; c < C; ++c) {
        int h = p * stride_h + r - pad_h;
        int w = q * stride_w + s - pad_w;
        if (h >= 0 && h < H && w >= 0 && w < W) {
          acc += input[((n * H + h) * W + w) * C + c]
               * filter[((k * R + r) * S + s) * C + c];
        }
      }
  output[((n * P + p) * Q + q) * K + k] = acc;
}

cudaError_t Conv2dFprop3xTF32(
    float const *input, float const *filter, float *output,
    int N, int C, int H, int W, int K, int R, int S,
    int pad_h, int pad_w, int stride_h, int stride_w) {
  int P = (H + 2 * pad_h - R) / stride_h + 1;
  int Q = (W + 2 * pad_w - S) / stride_w + 1;
  int total = N * P * Q * K;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  naive_conv2d_fprop_kernel<<<blocks, threads>>>(
      input, filter, output, N, C, H, W, K, R, S,
      P, Q, pad_h, pad_w, stride_h, stride_w);
  return cudaGetLastError();
}
