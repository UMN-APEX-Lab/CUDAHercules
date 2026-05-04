#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Naive conv2d wgrad with fused scale+bias+relu for testing the harness.
// Fused operation: transformed = relu(scale[c] * input[n][h][w][c] + bias[c])
// grad_weight[k][r][s][c] = sum_{n,p,q} grad_output[n][p][q][k] * transformed[n][h][w][c]
// NHWC layout, fp16 input/grad_output, fp32 grad_weight.

__global__ void naive_wgrad_scale_bias_kernel(
    __half const *input, __half const *grad_output,
    __half const *scale, __half const *bias,
    float *grad_weight,
    int N, int C, int H, int W, int K, int R, int S,
    int P, int Q, int pad_h, int pad_w, int stride_h, int stride_w) {

  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int total = K * R * S * C;
  if (idx >= total) return;

  int c = idx % C;
  int s_idx = (idx / C) % S;
  int r_idx = (idx / (C * S)) % R;
  int k = idx / (C * S * R);

  float acc = 0.0f;

  for (int n = 0; n < N; ++n) {
    for (int p = 0; p < P; ++p) {
      for (int q = 0; q < Q; ++q) {
        int h = p * stride_h + r_idx - pad_h;
        int w = q * stride_w + s_idx - pad_w;
        if (h >= 0 && h < H && w >= 0 && w < W) {
          float x = __half2float(input[((n * H + h) * W + w) * C + c]);
          float sc = __half2float(scale[c]);
          float bi = __half2float(bias[c]);
          float transformed = sc * x + bi;
          // ReLU
          transformed = fmaxf(0.0f, transformed);
          float go = __half2float(grad_output[((n * P + p) * Q + q) * K + k]);
          acc += go * transformed;
        }
      }
    }
  }

  grad_weight[((k * R + r_idx) * S + s_idx) * C + c] = acc;
}

cudaError_t Conv2dWgradScaleBias(
    __half const *input, __half const *grad_output,
    __half const *scale, __half const *bias,
    float *grad_weight,
    int N, int C, int H, int W, int K, int R, int S,
    int pad_h, int pad_w, int stride_h, int stride_w) {

  int total = K * R * S * C;
  dim3 block(256);
  dim3 grid((total + block.x - 1) / block.x);
  int P = (H + 2 * pad_h - R) / stride_h + 1;
  int Q = (W + 2 * pad_w - S) / stride_w + 1;
  naive_wgrad_scale_bias_kernel<<<grid, block>>>(input, grad_output, scale, bias, grad_weight,
    N, C, H, W, K, R, S, P, Q, pad_h, pad_w, stride_h, stride_w);
  return cudaGetLastError();
}
