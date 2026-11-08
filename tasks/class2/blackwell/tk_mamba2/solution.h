#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cmath>

// Straightforward FP32 CUDA-core implementation of the Mamba-2 recurrence
//   S_t = exp(A_t) * S_{t-1} + K_t^T V_t,   O_t = Q_t S_t        (S is d_state x 64, FP32)
// One CTA per (batch, head); thread j owns column j of S and walks the sequence sequentially.
//   Q, K: [B, 1, N, d_state] bf16 (shared by all heads);  V, O: [B, H, N, 64] bf16;  A: [B, H, N] fp32
// d_state must be 64 or 128.

#define M2_NAIVE_DH 64
#define M2_NAIVE_TILE 32

template <int DS>
__global__ void m2_naive_kernel(const __nv_bfloat16 *__restrict__ Q, const __nv_bfloat16 *__restrict__ K,
                                const __nv_bfloat16 *__restrict__ V, __nv_bfloat16 *__restrict__ O,
                                const float *__restrict__ A, int H, int N) {
  __shared__ float qs[M2_NAIVE_TILE][DS];
  __shared__ float ks[M2_NAIVE_TILE][DS];
  const int b = blockIdx.x / H;
  const int j = threadIdx.x;
  const size_t qk_base = size_t(b) * N * DS;
  const size_t v_base = size_t(blockIdx.x) * N * M2_NAIVE_DH;
  const float *a = A + size_t(blockIdx.x) * N;
  float S[DS];
#pragma unroll
  for (int d = 0; d < DS; d++) S[d] = 0.f;
  for (int t0 = 0; t0 < N; t0 += M2_NAIVE_TILE) {
    __syncthreads();
    for (int i = j; i < M2_NAIVE_TILE * DS; i += M2_NAIVE_DH) {
      qs[i / DS][i % DS] = __bfloat162float(Q[qk_base + size_t(t0) * DS + i]);
      ks[i / DS][i % DS] = __bfloat162float(K[qk_base + size_t(t0) * DS + i]);
    }
    __syncthreads();
    for (int t = 0; t < M2_NAIVE_TILE; t++) {
      const float decay = expf(a[t0 + t]);
      const float vj = __bfloat162float(V[v_base + size_t(t0 + t) * M2_NAIVE_DH + j]);
      float acc = 0.f;
#pragma unroll
      for (int d = 0; d < DS; d++) {
        S[d] = fmaf(S[d], decay, ks[t][d] * vj);
        acc = fmaf(qs[t][d], S[d], acc);
      }
      O[v_base + size_t(t0 + t) * M2_NAIVE_DH + j] = __float2bfloat16(acc);
    }
  }
}

void Mamba2Forward(__nv_bfloat16 *Q, __nv_bfloat16 *K, __nv_bfloat16 *V, __nv_bfloat16 *O, float *A,
                   int B, int H, int N, int d_state) {
  if (d_state == 128) m2_naive_kernel<128><<<B * H, M2_NAIVE_DH>>>(Q, K, V, O, A, H, N);
  else                m2_naive_kernel<64><<<B * H, M2_NAIVE_DH>>>(Q, K, V, O, A, H, N);
}
