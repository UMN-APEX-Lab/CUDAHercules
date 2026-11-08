#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>

// Straightforward FP32 CUDA-core implementation of causal Based attention:
//   O[n] = sum_{m <= n} (1 + s + s^2/2) * V[m],  s = (Q[n] . K[m]) / 4
// Quadratic in N: one thread per (b, h, n) computes all 64 output channels.
//   Q, K: [B, H, N, 16] bf16;  V, O: [B, H, N, 64] bf16 (O is written by this function)

#define BASED_NAIVE_DQK 16
#define BASED_NAIVE_DV 64
#define BASED_NAIVE_BLOCK 128

__global__ void based_naive_kernel(const __nv_bfloat16 *__restrict__ Q, const __nv_bfloat16 *__restrict__ K,
                                   const __nv_bfloat16 *__restrict__ V, __nv_bfloat16 *__restrict__ O, int N) {
  __shared__ float ks[BASED_NAIVE_BLOCK][BASED_NAIVE_DQK];
  __shared__ float vs[BASED_NAIVE_BLOCK][BASED_NAIVE_DV];
  const int bh = blockIdx.y;
  const int n = blockIdx.x * BASED_NAIVE_BLOCK + threadIdx.x;
  const size_t qk_base = size_t(bh) * N * BASED_NAIVE_DQK;
  const size_t v_base = size_t(bh) * N * BASED_NAIVE_DV;
  float q[BASED_NAIVE_DQK];
#pragma unroll
  for (int d = 0; d < BASED_NAIVE_DQK; d++) q[d] = __bfloat162float(Q[qk_base + size_t(n) * BASED_NAIVE_DQK + d]);
  float acc[BASED_NAIVE_DV];
#pragma unroll
  for (int d = 0; d < BASED_NAIVE_DV; d++) acc[d] = 0.f;
  const int last_tile = blockIdx.x;
  for (int t = 0; t <= last_tile; t++) {
    __syncthreads();
    const int m0 = t * BASED_NAIVE_BLOCK;
    for (int i = threadIdx.x; i < BASED_NAIVE_BLOCK * BASED_NAIVE_DQK; i += BASED_NAIVE_BLOCK)
      ks[i / BASED_NAIVE_DQK][i % BASED_NAIVE_DQK] = __bfloat162float(K[qk_base + size_t(m0) * BASED_NAIVE_DQK + i]);
    for (int i = threadIdx.x; i < BASED_NAIVE_BLOCK * BASED_NAIVE_DV; i += BASED_NAIVE_BLOCK)
      vs[i / BASED_NAIVE_DV][i % BASED_NAIVE_DV] = __bfloat162float(V[v_base + size_t(m0) * BASED_NAIVE_DV + i]);
    __syncthreads();
    const int mmax = (t == last_tile) ? threadIdx.x + 1 : BASED_NAIVE_BLOCK;
    for (int m = 0; m < mmax; m++) {
      float dot = 0.f;
#pragma unroll
      for (int d = 0; d < BASED_NAIVE_DQK; d++) dot = fmaf(q[d], ks[m][d], dot);
      float s = dot * 0.25f;
      float wgt = 1.f + s + 0.5f * s * s;
#pragma unroll
      for (int d = 0; d < BASED_NAIVE_DV; d++) acc[d] = fmaf(wgt, vs[m][d], acc[d]);
    }
  }
#pragma unroll
  for (int d = 0; d < BASED_NAIVE_DV; d++) O[v_base + size_t(n) * BASED_NAIVE_DV + d] = __float2bfloat16(acc[d]);
}

void BasedLinearAttention(__nv_bfloat16 *Q, __nv_bfloat16 *K, __nv_bfloat16 *V, __nv_bfloat16 *O,
                          int B, int H, int N) {
  based_naive_kernel<<<dim3(N / BASED_NAIVE_BLOCK, B * H), BASED_NAIVE_BLOCK>>>(Q, K, V, O, N);
}
