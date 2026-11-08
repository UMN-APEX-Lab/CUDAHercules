#pragma once
// Reference-free "naive but not slow" FMHA for the harness: one warp per query row, FP32 math,
// online softmax. Exact up to FP32 summation order, so it is also the comparator used to
// calibrate the tolerances. It is slow (CUDA cores), but finishes the whole test in a few minutes.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>
#include <stdint.h>

namespace naive_fmha {

__device__ __forceinline__ float2 load2(__half const *p) {
  return __half22float2(*reinterpret_cast<__half2 const *>(p));
}
__device__ __forceinline__ float2 load2(__nv_bfloat16 const *p) {
  return __bfloat1622float2(*reinterpret_cast<__nv_bfloat162 const *>(p));
}
__device__ __forceinline__ void store2(__half *p, float a, float b) {
  *reinterpret_cast<__half2 *>(p) = __floats2half2_rn(a, b);
}
__device__ __forceinline__ void store2(__nv_bfloat16 *p, float a, float b) {
  *reinterpret_cast<__nv_bfloat162 *>(p) = __floats2bfloat162_rn(a, b);
}

// D = 32 * DPL: every lane owns DPL consecutive head-dim elements of its query row.
template <typename T, int DPL>
__global__ void fmha_kernel(int Hq, int Hkv, int Sq, int Sk, int causal, float scale,
                            T const *Q, T const *K, T const *V, T *O, float *LSE) {
  int lane = threadIdx.x & 31;
  long long row = (long long)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  int q = int(row % Sq);
  long long bh = row / Sq;
  int h = int(bh % Hq);
  int b = int(bh / Hq);
  int hk = h / (Hq / Hkv);
  constexpr int D = 32 * DPL;

  float qv[DPL], acc[DPL];
  T const *qp = Q + (((long long)b * Sq + q) * Hq + h) * D + lane * DPL;
#pragma unroll
  for (int i = 0; i < DPL; i += 2) {
    float2 f = load2(qp + i);
    qv[i] = f.x; qv[i + 1] = f.y; acc[i] = 0.f; acc[i + 1] = 0.f;
  }
  float m = -INFINITY, l = 0.f;
  int kend = causal ? min(Sk, q + 1) : Sk;
  for (int j = 0; j < kend; ++j) {
    long long off = (((long long)b * Sk + j) * Hkv + hk) * D + lane * DPL;
    float s = 0.f;
#pragma unroll
    for (int i = 0; i < DPL; i += 2) {
      float2 f = load2(K + off + i);
      s += qv[i] * f.x + qv[i + 1] * f.y;
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    s *= scale;
    float mn = fmaxf(m, s);
    float alpha = __expf(m - mn);
    float p = __expf(s - mn);
    l = l * alpha + p;
#pragma unroll
    for (int i = 0; i < DPL; i += 2) {
      float2 f = load2(V + off + i);
      acc[i] = acc[i] * alpha + p * f.x;
      acc[i + 1] = acc[i + 1] * alpha + p * f.y;
    }
    m = mn;
  }
  float inv = 1.f / l;
  T *op = O + (((long long)b * Sq + q) * Hq + h) * D + lane * DPL;
#pragma unroll
  for (int i = 0; i < DPL; i += 2) store2(op + i, acc[i] * inv, acc[i + 1] * inv);
  if (lane == 0) LSE[(bh * Sq) + q] = m + logf(l);
}

template <typename T>
cudaError_t launch(int B, int Hq, int Hkv, int Sq, int Sk, int D, bool causal,
                   T const *Q, T const *K, T const *V, T *O, float *LSE) {
  long long rows = (long long)B * Hq * Sq;
  int warps = 4;
  float scale = 1.0f / sqrtf((float)D);
  dim3 grid((unsigned)((rows + warps - 1) / warps)), block(32 * warps);
  if (D == 128) fmha_kernel<T, 4><<<grid, block>>>(Hq, Hkv, Sq, Sk, causal, scale, Q, K, V, O, LSE);
  else if (D == 64) fmha_kernel<T, 2><<<grid, block>>>(Hq, Hkv, Sq, Sk, causal, scale, Q, K, V, O, LSE);
  else return cudaErrorInvalidValue;
  return cudaGetLastError();
}

}  // namespace naive_fmha

// Q, O: [batch, seq_q, num_q_heads, head_dim]; K, V: [batch, seq_k, num_kv_heads, head_dim];
// LSE: [batch, num_q_heads, seq_q] (FP32, natural log). Query head h uses KV head h / (Hq / Hkv).
cudaError_t BlackwellFmha(int batch, int num_q_heads, int num_kv_heads, int seq_q, int seq_k,
                          int head_dim, bool causal, __half const *Q, __half const *K,
                          __half const *V, __half *O, float *LSE) {
  return naive_fmha::launch<__half>(batch, num_q_heads, num_kv_heads, seq_q, seq_k, head_dim,
                                    causal, Q, K, V, O, LSE);
}

cudaError_t BlackwellFmha(int batch, int num_q_heads, int num_kv_heads, int seq_q, int seq_k,
                          int head_dim, bool causal, __nv_bfloat16 const *Q,
                          __nv_bfloat16 const *K, __nv_bfloat16 const *V, __nv_bfloat16 *O,
                          float *LSE) {
  return naive_fmha::launch<__nv_bfloat16>(batch, num_q_heads, num_kv_heads, seq_q, seq_k,
                                           head_dim, causal, Q, K, V, O, LSE);
}
