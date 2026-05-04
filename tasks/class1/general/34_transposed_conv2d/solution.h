#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Naive transposed conv2d (deconvolution), NHWC layout, half precision
// output[n][h_out][w_out][k] = sum_{c,r,s} input[n][h][w][c] * filter[c][r][s][k]
// where h_out = h * stride_h + r - pad_h
__global__ void naive_transposed_conv2d_kernel(
    __half const *input, __half const *filter, float *output_fp32,
    int N, int C, int H, int W, int K, int R, int S,
    int H_out, int W_out, int pad_h, int pad_w, int stride_h, int stride_w) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = N * H_out * W_out * K;
  if (idx >= total) return;

  int k = idx % K; idx /= K;
  int w_out = idx % W_out; idx /= W_out;
  int h_out = idx % H_out; idx /= H_out;
  int n = idx;

  float acc = 0.0f;
  for (int c = 0; c < C; ++c)
    for (int r = 0; r < R; ++r)
      for (int s = 0; s < S; ++s) {
        // h_out = h * stride_h + r - pad_h  =>  h = (h_out - r + pad_h) / stride_h
        int h_val = h_out - r + pad_h;
        int w_val = w_out - s + pad_w;
        if (h_val >= 0 && (h_val % stride_h) == 0 &&
            w_val >= 0 && (w_val % stride_w) == 0) {
          int h = h_val / stride_h;
          int w = w_val / stride_w;
          if (h < H && w < W) {
            acc += __half2float(input[((n * H + h) * W + w) * C + c])
                 * __half2float(filter[((c * R + r) * S + s) * K + k]);
          }
        }
      }
  output_fp32[((n * H_out + h_out) * W_out + w_out) * K + k] = acc;
}

__global__ void fp32_to_half_kernel_tc(float const *src, __half *dst, int count) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) dst[i] = __float2half(src[i]);
}

cudaError_t TransposedConv2d(
    __half const *input, __half const *filter, __half *output,
    int N, int C, int H, int W, int K, int R, int S,
    int pad_h, int pad_w, int stride_h, int stride_w) {
  int H_out = (H - 1) * stride_h + R - 2 * pad_h;
  int W_out = (W - 1) * stride_w + S - 2 * pad_w;
  int output_count = N * H_out * W_out * K;

  float *d_output_fp32;
  cudaMalloc(&d_output_fp32, output_count * sizeof(float));

  int threads = 256;
  int blocks = (output_count + threads - 1) / threads;
  naive_transposed_conv2d_kernel<<<blocks, threads>>>(
      input, filter, d_output_fp32,
      N, C, H, W, K, R, S, H_out, W_out,
      pad_h, pad_w, stride_h, stride_w);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) { cudaFree(d_output_fp32); return err; }

  // Convert fp32 -> half
  blocks = (output_count + threads - 1) / threads;
  fp32_to_half_kernel_tc<<<blocks, threads>>>(d_output_fp32, output, output_count);
  err = cudaGetLastError();
  cudaFree(d_output_fp32);
  return err;
}
