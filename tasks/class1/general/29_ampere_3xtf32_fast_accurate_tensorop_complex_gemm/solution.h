#pragma once
#include <cuda_runtime.h>

// Naive complex GEMM kernel
// A is column-major, B/C/D are row-major.
// Complex stored as interleaved (re, im) float pairs.
// lda/ldb/ldc/ldd are in units of complex elements.
__global__ void naive_complex_gemm_kernel(
    int M, int N, int K,
    float alpha_re, float alpha_im,
    float const *A, int lda,
    float const *B, int ldb,
    float beta_re, float beta_im,
    float const *C, int ldc,
    float *D, int ldd) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < M && j < N) {
    float acc_re = 0, acc_im = 0;
    for (int k = 0; k < K; ++k) {
      // A column-major: element (i,k) at A[(i + k*lda)*2]
      int ai = (i + k * lda) * 2;
      float a_re = A[ai], a_im = A[ai + 1];
      // B row-major: element (k,j) at B[(k*ldb + j)*2]
      int bi = (k * ldb + j) * 2;
      float b_re = B[bi], b_im = B[bi + 1];
      // complex multiply-add
      acc_re += a_re * b_re - a_im * b_im;
      acc_im += a_re * b_im + a_im * b_re;
    }

    // alpha * acc
    float r_re = alpha_re * acc_re - alpha_im * acc_im;
    float r_im = alpha_re * acc_im + alpha_im * acc_re;

    // beta * C
    int ci = (i * ldc + j) * 2;
    float c_re = C[ci], c_im = C[ci + 1];
    float bc_re = beta_re * c_re - beta_im * c_im;
    float bc_im = beta_re * c_im + beta_im * c_re;

    // D = alpha*acc + beta*C
    int di = (i * ldd + j) * 2;
    D[di]     = r_re + bc_re;
    D[di + 1] = r_im + bc_im;
  }
}

cudaError_t ComplexGemm(
    int M, int N, int K,
    float alpha_real, float alpha_imag,
    float const *A, int lda,
    float const *B, int ldb,
    float beta_real, float beta_imag,
    float const *C, int ldc,
    float *D, int ldd) {

  dim3 block(16, 16);
  dim3 grid((M + 15) / 16, (N + 15) / 16);
  naive_complex_gemm_kernel<<<grid, block>>>(
    M, N, K, alpha_real, alpha_imag,
    A, lda, B, ldb, beta_real, beta_imag, C, ldc, D, ldd);
  return cudaGetLastError();
}
