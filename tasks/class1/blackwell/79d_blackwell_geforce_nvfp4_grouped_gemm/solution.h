#pragma once
#include <cuda_runtime.h>
#include <cmath>
#include "cutlass/numeric_types.h"

// Naive FP4 grouped GEMM for testing the harness (two-pass: float accumulate, then quantize)
// Each group g: D_g = alpha * A_g * B_g + beta * D_g
// A_g row-major [Ms[g], Ks[g]], B_g col-major [Ks[g], Ns[g]], D_g row-major [Ms[g], Ns[g]]
// All in FP4 (float_e2m1_t), one element per byte.
// Scale group size = 32, output SF vector size = 16.

// Pass 1: accumulate into float buffer
__global__ void fp4_gemm_accum_kernel(
    int M, int N, int K, float alpha,
    cutlass::float_e2m1_t const *A, int lda,
    float const *A_scales, int scale_lda,
    cutlass::float_e2m1_t const *B, int ldb,
    float const *B_scales, int scale_ldb,
    float beta, float *D_float, int ldd) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  if (i < M && j < N) {
    float acc = 0;
    for (int k = 0; k < K; ++k) {
      int kg = k / 32;
      float a_val = float(A[i * lda + k]) * A_scales[i * scale_lda + kg];
      float b_val = float(B[k + j * ldb]) * B_scales[j * scale_ldb + kg];
      acc += a_val * b_val;
    }
    D_float[i * ldd + j] = alpha * acc;
  }
}

// Pass 2: quantize float buffer to FP4 with proper scale factors
__global__ void fp4_quantize_kernel(
    int M, int N,
    float const *D_float, int ldd,
    cutlass::float_e2m1_t *D_out, int ldd_out,
    float *D_scales, int scale_ldd,
    int sf_vec_size) {
  int i = blockIdx.x;  // row
  int sf_col = blockIdx.y;  // scale factor column
  if (i >= M) return;

  int j_start = sf_col * sf_vec_size;
  int j_end = j_start + sf_vec_size;
  if (j_end > N) j_end = N;

  // Find max abs in this group
  float max_abs = 0;
  for (int j = j_start + threadIdx.x; j < j_end; j += blockDim.x) {
    float val = fabsf(D_float[i * ldd + j]);
    if (val > max_abs) max_abs = val;
  }
  // Warp reduce
  for (int offset = 16; offset > 0; offset >>= 1) {
    float other = __shfl_down_sync(0xFFFFFFFF, max_abs, offset);
    if (other > max_abs) max_abs = other;
  }
  max_abs = __shfl_sync(0xFFFFFFFF, max_abs, 0);

  // Scale factor: max_abs / 1.0 (FP4 max representable is 1.0)
  float scale = (max_abs > 0) ? max_abs : 1.0f;

  // Write scale factor (one thread)
  if (threadIdx.x == 0) {
    D_scales[i * scale_ldd + sf_col] = scale;
  }

  // Quantize
  float inv_scale = 1.0f / scale;
  for (int j = j_start + threadIdx.x; j < j_end; j += blockDim.x) {
    float val = D_float[i * ldd + j] * inv_scale;
    D_out[i * ldd_out + j] = cutlass::float_e2m1_t(val);
  }
}

cudaError_t Nvfp4GroupedGemm(
    int num_groups,
    int const *Ms, int const *Ns, int const *Ks,
    float alpha, float beta,
    cutlass::float_e2m1_t const * const *A_ptrs, int const *ldas,
    float const * const *A_scale_ptrs, int const *scale_ldas,
    cutlass::float_e2m1_t const * const *B_ptrs, int const *ldbs,
    float const * const *B_scale_ptrs, int const *scale_ldbs,
    cutlass::float_e2m1_t * const *D_ptrs, int const *ldds,
    float * const *D_scale_ptrs, int const *scale_ldds) {
  // Copy arrays from device to host
  std::vector<int> h_Ms(num_groups), h_Ns(num_groups), h_Ks(num_groups);
  std::vector<int> h_ldas(num_groups), h_ldbs(num_groups), h_ldds(num_groups);
  std::vector<int> h_sldas(num_groups), h_sldbs(num_groups), h_sldds(num_groups);
  std::vector<cutlass::float_e2m1_t const*> h_Ap(num_groups);
  std::vector<cutlass::float_e2m1_t const*> h_Bp(num_groups);
  std::vector<cutlass::float_e2m1_t*> h_Dp(num_groups);
  std::vector<float const*> h_sAp(num_groups), h_sBp(num_groups);
  std::vector<float*> h_sDp(num_groups);

  cudaMemcpy(h_Ms.data(), Ms, sizeof(int)*num_groups, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_Ns.data(), Ns, sizeof(int)*num_groups, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_Ks.data(), Ks, sizeof(int)*num_groups, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_ldas.data(), ldas, sizeof(int)*num_groups, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_ldbs.data(), ldbs, sizeof(int)*num_groups, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_ldds.data(), ldds, sizeof(int)*num_groups, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_sldas.data(), scale_ldas, sizeof(int)*num_groups, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_sldbs.data(), scale_ldbs, sizeof(int)*num_groups, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_sldds.data(), scale_ldds, sizeof(int)*num_groups, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_Ap.data(), A_ptrs, sizeof(void*)*num_groups, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_Bp.data(), B_ptrs, sizeof(void*)*num_groups, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_Dp.data(), D_ptrs, sizeof(void*)*num_groups, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_sAp.data(), A_scale_ptrs, sizeof(void*)*num_groups, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_sBp.data(), B_scale_ptrs, sizeof(void*)*num_groups, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_sDp.data(), D_scale_ptrs, sizeof(void*)*num_groups, cudaMemcpyDeviceToHost);

  for (int g = 0; g < num_groups; ++g) {
    int M = h_Ms[g], N = h_Ns[g], K = h_Ks[g];
    int ldd = h_ldds[g];
    int n_sf = (N + 16 - 1) / 16;

    // Allocate temp float buffer
    float *d_float;
    cudaMalloc(&d_float, M * N * sizeof(float));

    // Pass 1: accumulate
    dim3 block1(16, 16);
    dim3 grid1((M + 15) / 16, (N + 15) / 16);
    fp4_gemm_accum_kernel<<<grid1, block1>>>(
        M, N, K, alpha,
        h_Ap[g], h_ldas[g], h_sAp[g], h_sldas[g],
        h_Bp[g], h_ldbs[g], h_sBp[g], h_sldbs[g],
        beta, d_float, N);

    // Pass 2: quantize
    dim3 grid2(M, n_sf);
    dim3 block2(32);
    fp4_quantize_kernel<<<grid2, block2>>>(
        M, N, d_float, N,
        h_Dp[g], ldd, h_sDp[g], h_sldds[g], 16);

    cudaFree(d_float);
  }
  return cudaGetLastError();
}
