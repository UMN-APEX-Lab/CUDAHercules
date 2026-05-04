#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Naive Conv2D forward propagation: half input, float output, NHWC layout
// output[n][p][q][k] = alpha * sum_{c,r,s} input[n][h][w][c] * filter[k][r][s][c] + beta * output[n][p][q][k]
__global__ void naive_conv2d_fprop_f16_kernel(
    __half const *input, __half const *filter, float *output,
    int N, int C, int H, int W, int K, int R, int S,
    int P, int Q, int pad_h, int pad_w, int stride_h, int stride_w,
    float alpha, float beta) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int total = N * P * Q * K;
  if (idx >= total) return;

  int k_idx = idx % K; int tmp = idx / K;
  int q_idx = tmp % Q; tmp /= Q;
  int p_idx = tmp % P; int n_idx = tmp / P;

  float acc = 0.0f;
  for (int c = 0; c < C; ++c)
    for (int r = 0; r < R; ++r)
      for (int s = 0; s < S; ++s) {
        int h = p_idx * stride_h + r - pad_h;
        int w = q_idx * stride_w + s - pad_w;
        if (h >= 0 && h < H && w >= 0 && w < W) {
          float inp = __half2float(input[((n_idx * H + h) * W + w) * C + c]);
          float flt = __half2float(filter[((k_idx * R + r) * S + s) * C + c]);
          acc += inp * flt;
        }
      }
  int out_idx = ((n_idx * P + p_idx) * Q + q_idx) * K + k_idx;
  output[out_idx] = alpha * acc + beta * output[out_idx];
}

cudaError_t Conv2dFpropF16(
    __half const *input, __half const *filter, float *output,
    int N, int C, int H, int W, int K, int R, int S,
    int pad_h, int pad_w, int stride_h, int stride_w,
    float alpha, float beta) {
  int P = (H + 2 * pad_h - R) / stride_h + 1;
  int Q = (W + 2 * pad_w - S) / stride_w + 1;
  int total = N * P * Q * K;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  naive_conv2d_fprop_f16_kernel<<<blocks, threads>>>(
    input, filter, output,
    N, C, H, W, K, R, S, P, Q,
    pad_h, pad_w, stride_h, stride_w, alpha, beta);
  return cudaGetLastError();
}
