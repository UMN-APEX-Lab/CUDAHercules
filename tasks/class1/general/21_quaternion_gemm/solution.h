#pragma once
#include <cuda_runtime.h>

// Naive quaternion GEMM kernel.
// A: row-major (M x K), B: column-major (K x N), C: row-major (M x N).
// Each quaternion = 4 consecutive floats in order (x, y, z, w).
// A: quaternion (i,k) at A[(i * lda + k) * 4], lda = K
// B: quaternion (k,j) at B[(k + j * ldb) * 4], ldb = K
// C: quaternion (i,j) at C[(i * ldc + j) * 4], ldc = N
// C = alpha * A * B + beta * C, where * is Hamilton product.
__global__ void naive_quaternion_gemm_kernel(
    int M, int N, int K, float alpha,
    float const *A, int lda,
    float const *B, int ldb,
    float beta,
    float *C, int ldc) {

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  if (i < M && j < N) {
    // Memory layout per quaternion: [0]=x, [1]=y, [2]=z, [3]=w
    float acc_x = 0, acc_y = 0, acc_z = 0, acc_w = 0;
    for (int kk = 0; kk < K; ++kk) {
      int a_off = (i * lda + kk) * 4;  // row-major
      int b_off = (kk + j * ldb) * 4;  // col-major
      float ax = A[a_off], ay = A[a_off+1], az = A[a_off+2], aw = A[a_off+3];
      float bx = B[b_off], by = B[b_off+1], bz = B[b_off+2], bw = B[b_off+3];
      // Hamilton product
      acc_x += aw*bx + bw*ax + ay*bz - az*by;
      acc_y += aw*by + bw*ay + az*bx - ax*bz;
      acc_z += aw*bz + bw*az + ax*by - ay*bx;
      acc_w += aw*bw - ax*bx - ay*by - az*bz;
    }
    int c_off = (i * ldc + j) * 4;  // row-major
    float cx = C[c_off], cy = C[c_off+1], cz = C[c_off+2], cw = C[c_off+3];
    C[c_off]   = alpha * acc_x + beta * cx;
    C[c_off+1] = alpha * acc_y + beta * cy;
    C[c_off+2] = alpha * acc_z + beta * cz;
    C[c_off+3] = alpha * acc_w + beta * cw;
  }
}

cudaError_t QuaternionGemm(
    int M, int N, int K, float alpha,
    float const *A, int lda,
    float const *B, int ldb,
    float beta,
    float *C, int ldc) {
  dim3 block(16, 16);
  dim3 grid((M + 15) / 16, (N + 15) / 16);
  naive_quaternion_gemm_kernel<<<grid, block>>>(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
  return cudaGetLastError();
}
