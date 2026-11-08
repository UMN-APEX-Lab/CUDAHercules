#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cmath>

// Straightforward FP32 CUDA-core implementation of Hedgehog attention (see description.txt).
// One CTA (128 threads) per (batch, head) walks the sequence token by token; thread j owns
// value column j of the linear-attention state (128 feature rows in registers).
//   Q, K, V, O: [B, H, N, 128] bf16;  Qmap, Kmap: [H, 128, 64] bf16;  alphas, betas: [H] fp32
// No tensor cores, no chunking: a baseline to beat.

#define HH_D 128
#define HH_NF 64
#define HH_BLK 64

__device__ __forceinline__ float hh_f(__nv_bfloat16 x) { return __bfloat162float(x); }

__global__ void hh_naive_kernel(const __nv_bfloat16 *__restrict__ Q, const __nv_bfloat16 *__restrict__ K,
                                const __nv_bfloat16 *__restrict__ V, __nv_bfloat16 *__restrict__ O,
                                const __nv_bfloat16 *__restrict__ Qmap, const __nv_bfloat16 *__restrict__ Kmap,
                                const float *__restrict__ alphas, const float *__restrict__ betas, int H, int N) {
  __shared__ float qs[HH_D], sc[2 * HH_BLK], zs[HH_NF], ezp[HH_NF], ezm[HH_NF], ksum[2 * HH_NF];
  __shared__ float red[2];
  const int bh = blockIdx.x, h = bh % H, t = threadIdx.x;
  const size_t base = size_t(bh) * N * HH_D;
  const __nv_bfloat16 *Qm = Qmap + size_t(h) * HH_D * HH_NF, *Km = Kmap + size_t(h) * HH_D * HH_NF;
  const float alpha = alphas[h], beta = betas[h];
  const float scale = 0.08838834764831845f;
  float S[2 * HH_NF];
#pragma unroll
  for (int f = 0; f < 2 * HH_NF; f++) S[f] = 0.f;
  ksum[t] = 0.f;
  __syncthreads();

  for (int n = 0; n < N; n++) {
    const int b = n / HH_BLK;
    const int mstart = b >= 1 ? (b - 1) * HH_BLK : 0;
    const int wlen = n - mstart + 1;
    qs[t] = hh_f(Q[base + size_t(n) * HH_D + t]);
    __syncthreads();
    if (t < wlen) {
      float s = 0.f;
      const __nv_bfloat16 *kr = K + base + size_t(mstart + t) * HH_D;
      for (int d = 0; d < HH_D; d++) s = fmaf(qs[d], hh_f(kr[d]), s);
      sc[t] = s * scale;
    }
    if (t < HH_NF) {
      float z = 0.f;
      for (int d = 0; d < HH_D; d++) z = fmaf(qs[d], hh_f(Qm[d * HH_NF + t]), z);
      zs[t] = z;
    }
    __syncthreads();
    float mx = -INFINITY;
    for (int m = 0; m < wlen; m++) mx = fmaxf(mx, sc[m]);
    float zmax = -INFINITY, zmin = INFINITY;
    for (int f = 0; f < HH_NF; f++) { zmax = fmaxf(zmax, zs[f]); zmin = fminf(zmin, zs[f]); }
    __syncthreads();
    if (t < wlen) sc[t] = expf(sc[t] - mx) * beta;
    if (t < HH_NF) { ezp[t] = expf(zs[t] - zmax); ezm[t] = expf(zmin - zs[t]); }
    __syncthreads();
    float acc = 0.f, den = 0.f;
    const __nv_bfloat16 *vp = V + base + size_t(mstart) * HH_D + t;
    for (int m = 0; m < wlen; m++) { den += sc[m]; acc = fmaf(sc[m], hh_f(vp[size_t(m) * HH_D]), acc); }
    if (b >= 1) {
      float sp = 0.f, sm = 0.f;
      for (int f = 0; f < HH_NF; f++) { sp += ezp[f]; sm += ezm[f]; }
      const float cp = alpha / sp, cm = alpha / sm;
      float lin = 0.f, nrm = 0.f;
#pragma unroll
      for (int f = 0; f < HH_NF; f++) {
        lin = fmaf(cp * ezp[f], S[f], lin);
        lin = fmaf(cm * ezm[f], S[HH_NF + f], lin);
        nrm = fmaf(cp * ezp[f], ksum[f], nrm);
        nrm = fmaf(cm * ezm[f], ksum[HH_NF + f], nrm);
      }
      acc += lin;
      den += nrm;
    }
    O[base + size_t(n) * HH_D + t] = __float2bfloat16(acc / den);
    __syncthreads();

    if (n % HH_BLK == HH_BLK - 1 && b >= 1) {   // fold block b-1 into the linear state
      for (int m = (b - 1) * HH_BLK; m < b * HH_BLK; m++) {
        qs[t] = hh_f(K[base + size_t(m) * HH_D + t]);
        __syncthreads();
        if (t < HH_NF) {
          float z = 0.f;
          for (int d = 0; d < HH_D; d++) z = fmaf(qs[d], hh_f(Km[d * HH_NF + t]), z);
          zs[t] = z;
        }
        __syncthreads();
        float zmax = -INFINITY, zmin = INFINITY;
        for (int f = 0; f < HH_NF; f++) { zmax = fmaxf(zmax, zs[f]); zmin = fminf(zmin, zs[f]); }
        if (t < HH_NF) { ezp[t] = expf(zs[t] - zmax); ezm[t] = expf(zmin - zs[t]); }
        __syncthreads();
        float sp = 0.f, sm = 0.f;
        for (int f = 0; f < HH_NF; f++) { sp += ezp[f]; sm += ezm[f]; }
        const float vj = hh_f(V[base + size_t(m) * HH_D + t]);
        const float cp = 1.f / sp, cm = 1.f / sm;
#pragma unroll
        for (int f = 0; f < HH_NF; f++) {
          S[f] = fmaf(cp * ezp[f], vj, S[f]);
          S[HH_NF + f] = fmaf(cm * ezm[f], vj, S[HH_NF + f]);
        }
        if (t < HH_NF) { ksum[t] += cp * ezp[t]; ksum[HH_NF + t] += cm * ezm[t]; }
        __syncthreads();
      }
    }
  }
}

void HedgehogAttention(__nv_bfloat16 *Q, __nv_bfloat16 *K, __nv_bfloat16 *V, __nv_bfloat16 *O,
                       __nv_bfloat16 *Q_map, __nv_bfloat16 *K_map, float *alphas, float *betas,
                       int B, int H, int N) {
  hh_naive_kernel<<<B * H, HH_D>>>(Q, K, V, O, Q_map, K_map, alphas, betas, H, N);
}
