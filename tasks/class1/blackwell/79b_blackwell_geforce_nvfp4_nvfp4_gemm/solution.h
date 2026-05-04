#pragma once
#include <cuda_runtime.h>
#include "cutlass/numeric_types.h"

// Naive GEMM for testing: dequant FP4 with per-group scales, accumulate in float,
// quantize output to FP4 with per-group output scale factors.
// A: row-major [M, K], B: col-major [K, N], D: row-major [M, N] in FP4
// A_scales: [M, ceil(K/32)], B_scales: [N, ceil(K/32)]
// D_scales: [M, ceil(N/16)]
static constexpr int kSolGroupSize = 32;
static constexpr int kSolOutputSFVec = 16;

// Pass 1: compute GEMM result in float into a temporary buffer
__global__ void gemm_float_kernel(
    int M, int N, int K, float alpha,
    cutlass::float_e2m1_t const *A, int lda,
    float const *A_scales, int scale_lda,
    cutlass::float_e2m1_t const *B, int ldb,
    float const *B_scales, int scale_ldb,
    float *D_float, int ldd) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  if (i < M && j < N) {
    float acc = 0;
    for (int k = 0; k < K; ++k) {
      int kg = k / kSolGroupSize;
      float a_val = float(A[i * lda + k]) * A_scales[i * scale_lda + kg];
      float b_val = float(B[k + j * ldb]) * B_scales[j * scale_ldb + kg];
      acc += a_val * b_val;
    }
    D_float[i * ldd + j] = alpha * acc;
  }
}

// Pass 2: compute per-group max abs for output scale factors
__global__ void compute_scales_kernel(
    int M, int N, float const *D_float, int ldd,
    float *D_scales, int scale_ldd) {
  int i = blockIdx.x;  // row
  int sg = threadIdx.x; // scale group index
  if (i < M && sg < scale_ldd) {
    int j_start = sg * kSolOutputSFVec;
    int j_end = min(j_start + kSolOutputSFVec, N);
    float max_abs = 0;
    for (int j = j_start; j < j_end; ++j) {
      max_abs = fmaxf(max_abs, fabsf(D_float[i * ldd + j]));
    }
    // max representable fp4 e2m1 = 6.0
    D_scales[i * scale_ldd + sg] = max_abs / 6.0f;
  }
}

// Pass 3: quantize to FP4 using computed scales
__global__ void quantize_kernel(
    int M, int N, float const *D_float, int ldd,
    float const *D_scales, int scale_ldd,
    cutlass::float_e2m1_t *D, int ldd_out) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  if (i < M && j < N) {
    int sg = j / kSolOutputSFVec;
    float scale = D_scales[i * scale_ldd + sg];
    float val = (scale > 0) ? D_float[i * ldd + j] / scale : 0.0f;
    D[i * ldd_out + j] = cutlass::float_e2m1_t(val);
  }
}

cudaError_t Nvfp4Nvfp4Gemm(
    int M, int N, int K, float alpha,
    cutlass::float_e2m1_t const *A, int lda,
    float const *A_scales, int scale_lda,
    cutlass::float_e2m1_t const *B, int ldb,
    float const *B_scales, int scale_ldb,
    float beta,
    cutlass::float_e2m1_t *D, int ldd,
    float *D_scales, int scale_ldd) {

  // Allocate temporary float buffer
  float *D_float;
  cudaMalloc(&D_float, M * N * sizeof(float));

  // Pass 1: GEMM in float
  {
    dim3 block(16, 16);
    dim3 grid((M + 15) / 16, (N + 15) / 16);
    gemm_float_kernel<<<grid, block>>>(M, N, K, alpha,
      A, lda, A_scales, scale_lda, B, ldb, B_scales, scale_ldb, D_float, ldd);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) { cudaFree(D_float); return err; }
  }

  // Pass 2: compute output scales
  {
    compute_scales_kernel<<<M, scale_ldd>>>(M, N, D_float, ldd, D_scales, scale_ldd);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) { cudaFree(D_float); return err; }
  }

  // Pass 3: quantize to FP4
  {
    dim3 block(16, 16);
    dim3 grid((M + 15) / 16, (N + 15) / 16);
    quantize_kernel<<<grid, block>>>(M, N, D_float, ldd, D_scales, scale_ldd, D, ldd);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) { cudaFree(D_float); return err; }
  }

  cudaFree(D_float);
  return cudaGetLastError();
}
