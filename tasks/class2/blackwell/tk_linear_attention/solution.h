#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cmath>

// Straightforward FP32 CUDA-core implementation of the recurrence
//   S_t = exp(-slope) * S_{t-1} + k_t^T v_t,   o_t = q_t S_t        (S is D x D, FP32)
// One CTA per (batch, head); thread j owns column j of S (D FP32 registers) and walks the whole
// sequence sequentially. No tensor cores, no chunking: a baseline to beat.
//
//   Q, K, V, O: [B, H, N, D] bf16, row-major (O is written by this function)
//   slopes:     [H] fp32, decay rate per head (> 0)
// D must be 128.

#define LA_NAIVE_D 128
#define LA_NAIVE_TILE 32

__global__ void la_naive_kernel(const __nv_bfloat16 *__restrict__ Q, const __nv_bfloat16 *__restrict__ K,
                                const __nv_bfloat16 *__restrict__ V, __nv_bfloat16 *__restrict__ O,
                                const float *__restrict__ slopes, int H, int N) {
  __shared__ float qs[LA_NAIVE_TILE][LA_NAIVE_D];
  __shared__ float ks[LA_NAIVE_TILE][LA_NAIVE_D];
  const int bh = blockIdx.x;
  const int h = bh % H;
  const int j = threadIdx.x;
  const float decay = expf(-slopes[h]);
  const size_t base = size_t(bh) * N * LA_NAIVE_D;
  float S[LA_NAIVE_D];
#pragma unroll
  for (int d = 0; d < LA_NAIVE_D; d++) S[d] = 0.f;
  for (int t0 = 0; t0 < N; t0 += LA_NAIVE_TILE) {
    __syncthreads();
    for (int i = j; i < LA_NAIVE_TILE * LA_NAIVE_D; i += LA_NAIVE_D) {
      qs[i / LA_NAIVE_D][i % LA_NAIVE_D] = __bfloat162float(Q[base + size_t(t0) * LA_NAIVE_D + i]);
      ks[i / LA_NAIVE_D][i % LA_NAIVE_D] = __bfloat162float(K[base + size_t(t0) * LA_NAIVE_D + i]);
    }
    __syncthreads();
    for (int t = 0; t < LA_NAIVE_TILE; t++) {
      float vj = __bfloat162float(V[base + size_t(t0 + t) * LA_NAIVE_D + j]);
      float acc = 0.f;
#pragma unroll
      for (int d = 0; d < LA_NAIVE_D; d++) {
        S[d] = fmaf(S[d], decay, ks[t][d] * vj);
        acc = fmaf(qs[t][d], S[d], acc);
      }
      O[base + size_t(t0 + t) * LA_NAIVE_D + j] = __float2bfloat16(acc);
    }
  }
}

void LinearAttention(__nv_bfloat16 *Q, __nv_bfloat16 *K, __nv_bfloat16 *V, __nv_bfloat16 *O,
                     float *slopes, int B, int H, int N, int D) {
  la_naive_kernel<<<B * H, LA_NAIVE_D>>>(Q, K, V, O, slopes, H, N);
}
