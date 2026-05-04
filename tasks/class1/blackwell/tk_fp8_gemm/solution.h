#pragma once
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>

// Naive FP8 GEMM: C = A * B^T
// A is [M,K] fp8, B is [N,K] fp8 (transposed), C is [M,N] half
__global__ void naive_fp8_gemm_kernel(__nv_fp8_e4m3 *A, __nv_fp8_e4m3 *B, half *C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; k++) {
            acc += float(A[row * K + k]) * float(B[col * K + k]);
        }
        C[row * N + col] = __float2half(acc);
    }
}

void Fp8Gemm(__nv_fp8_e4m3 *A, __nv_fp8_e4m3 *B, half *C, int M, int N, int K) {
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    naive_fp8_gemm_kernel<<<grid, block>>>(A, B, C, M, N, K);
}
