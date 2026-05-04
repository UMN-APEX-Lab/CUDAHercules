#pragma once
#include <cuda_runtime.h>
#include <cuda_fp8.h>

// Naive FP8 scaled GEMM: C = diag(A_scale) * (A * B^T) * diag(B_scale)
// A is [M,K] row-major (fp8), B is [N,K] (transposed, fp8), C is [M,N] row-major (float)
// A_scale is [M] (float), B_scale is [N] (float)
__global__ void naive_fp8_gemm_scaled_kernel(
    __nv_fp8_e4m3 *A, __nv_fp8_e4m3 *B, float *C,
    float *A_scale, float *B_scale, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; k++) {
            acc += float(A[row * K + k]) * float(B[col * K + k]); // B transposed: B[col, k]
        }
        C[row * N + col] = acc * A_scale[row] * B_scale[col];
    }
}

void Fp8GemmScaled(__nv_fp8_e4m3 *A, __nv_fp8_e4m3 *B, float *C, float *A_scale, float *B_scale, int M, int N, int K) {
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    naive_fp8_gemm_scaled_kernel<<<grid, block>>>(A, B, C, A_scale, B_scale, M, N, K);
}
