#pragma once
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <vector>

// Naive MoE grouped GEMM with DeepSeek-V3 block scaling (correctness baseline): one tiled
// FP32-accumulating kernel launch per non-empty expert. The per-expert counts live on the device,
// so this copies them to the host.
//
//   Y[r, n] = sum_kb X_scale[r, kb] * W_scale[e, n/128, kb] * ( sum_{k in block kb} X[r,k] * W[e,n,k] )
//   X [total_rows, K] FP8 E4M3, X_scale [total_rows, K/128] FP32 (one scale per token and 128 K),
//   W [E, N, K] FP8 E4M3, W_scale [E, N/128, K/128] FP32 (one scale per 128 x 128 block),
//   Y [total_rows, N] BF16; rows sorted by expert.

#define NM_TILE 64
#define NM_KT 16

__global__ void naive_moe_gemm_kernel(
    int M, int N, int K,
    __nv_fp8_e4m3 const *A, float const *As, __nv_fp8_e4m3 const *B, float const *Bs,
    __nv_bfloat16 *D) {
  __shared__ float Xs_[NM_KT][NM_TILE + 1];
  __shared__ float Ws_[NM_KT][NM_TILE + 1];
  int tx = threadIdx.x % 16, ty = threadIdx.x / 16;     // 16 x 16 threads, 4 x 4 outputs each
  int row0 = blockIdx.x * NM_TILE, col0 = blockIdx.y * NM_TILE;
  int KB = K / 128;
  float acc[4][4] = {};
  for (int kb = 0; kb < KB; ++kb) {
    float part[4][4] = {};
    for (int k0 = kb * 128; k0 < kb * 128 + 128; k0 += NM_KT) {
      for (int idx = threadIdx.x; idx < NM_TILE * NM_KT; idx += 256) {
        int r = idx / NM_KT, kk = idx % NM_KT;
        int gr = row0 + r, gc = col0 + r, gk = k0 + kk;
        Xs_[kk][r] = (gr < M) ? float(A[size_t(gr) * K + gk]) : 0.f;
        Ws_[kk][r] = (gc < N) ? float(B[size_t(gc) * K + gk]) : 0.f;
      }
      __syncthreads();
#pragma unroll
      for (int kk = 0; kk < NM_KT; ++kk)
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
          for (int j = 0; j < 4; ++j) part[i][j] += Xs_[kk][ty * 4 + i] * Ws_[kk][tx * 4 + j];
      __syncthreads();
    }
    int nb = (col0 + tx * 4) / 128;       // the 4 columns of a thread lie in one 128-block
    float ws = Bs[size_t(nb) * KB + kb];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      int r = row0 + ty * 4 + i;
      float xs = (r < M) ? As[size_t(r) * KB + kb] : 0.f;
#pragma unroll
      for (int j = 0; j < 4; ++j) acc[i][j] += part[i][j] * (xs * ws);
    }
  }
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      int r = row0 + ty * 4 + i, c = col0 + tx * 4 + j;
      if (r < M && c < N) D[size_t(r) * N + c] = __float2bfloat16_rn(acc[i][j]);
    }
}

cudaError_t BlackwellMoeGemm(
    int num_experts, int total_rows, int N, int K,
    __nv_fp8_e4m3 const *X, float const *X_scale, __nv_fp8_e4m3 const *W, float const *W_scale,
    int const *expert_offsets, int const *tokens_per_expert, __nv_bfloat16 *Y) {
  std::vector<int> cnt(num_experts), off(num_experts + 1);
  cudaMemcpy(cnt.data(), tokens_per_expert, sizeof(int) * num_experts, cudaMemcpyDeviceToHost);
  cudaMemcpy(off.data(), expert_offsets, sizeof(int) * (num_experts + 1), cudaMemcpyDeviceToHost);
  int KB = K / 128, NB = N / 128;
  for (int e = 0; e < num_experts; ++e) {
    if (cnt[e] <= 0) continue;
    dim3 grid((cnt[e] + NM_TILE - 1) / NM_TILE, (N + NM_TILE - 1) / NM_TILE);
    naive_moe_gemm_kernel<<<grid, 256>>>(cnt[e], N, K, X + size_t(off[e]) * K,
                                         X_scale + size_t(off[e]) * KB, W + size_t(e) * N * K,
                                         W_scale + size_t(e) * NB * KB, Y + size_t(off[e]) * N);
  }
  return cudaGetLastError();
}
