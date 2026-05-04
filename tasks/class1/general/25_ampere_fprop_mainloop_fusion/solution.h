#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Naive conv2d fprop with fused scale+bias+relu for testing the harness.
// Fused operation: transformed = relu(scale[c] * input[n][h][w][c] + bias[c])
// Output: output[n][p][q][k] = sum_{c,r,s} transformed[n][h][w][c] * filter[k][r][s][c]
// NHWC layout, fp16 input/filter, fp32 output.

__global__ void naive_fprop_scale_bias_kernel(
    __half const *input, __half const *filter,
    __half const *scale, __half const *bias,
    float *output,
    int N, int C, int H, int W, int K, int R, int S,
    int P, int Q, int pad_h, int pad_w, int stride_h, int stride_w) {

  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int total = N * P * Q * K;
  if (idx >= total) return;

  int k = idx % K;
  int q_idx = (idx / K) % Q;
  int p_idx = (idx / (K * Q)) % P;
  int n = idx / (K * Q * P);

  float acc = 0.0f;

  for (int r = 0; r < R; ++r) {
    for (int s = 0; s < S; ++s) {
      int h = p_idx * stride_h + r - pad_h;
      int w = q_idx * stride_w + s - pad_w;
      if (h >= 0 && h < H && w >= 0 && w < W) {
        for (int c = 0; c < C; ++c) {
          float x = __half2float(input[((n * H + h) * W + w) * C + c]);
          float sc = __half2float(scale[c]);
          float bi = __half2float(bias[c]);
          float transformed = sc * x + bi;
          // ReLU
          transformed = fmaxf(0.0f, transformed);
          float f = __half2float(filter[((k * R + r) * S + s) * C + c]);
          acc += transformed * f;
        }
      }
    }
  }

  output[((n * P + p_idx) * Q + q_idx) * K + k] = acc;
}

cudaError_t Conv2dFpropScaleBias(
    __half const *input, __half const *filter,
    __half const *scale, __half const *bias,
    float *output,
    int N, int C, int H, int W, int K, int R, int S,
    int pad_h, int pad_w, int stride_h, int stride_w) {

  int P = (H + 2 * pad_h - R) / stride_h + 1;
  int Q = (W + 2 * pad_w - S) / stride_w + 1;
  int total = N * P * Q * K;
  dim3 block(256);
  dim3 grid((total + block.x - 1) / block.x);
  naive_fprop_scale_bias_kernel<<<grid, block>>>(input, filter, scale, bias, output,
    N, C, H, W, K, R, S, P, Q, pad_h, pad_w, stride_h, stride_w);
  return cudaGetLastError();
}
