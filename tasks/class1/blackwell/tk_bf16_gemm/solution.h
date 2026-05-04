#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>

// Naive BF16 GEMM: C = A * B^T
// A is [M,K], B is [N,K] (transposed), C is [M,N]
__global__ void naive_bf16_gemm_kernel(__nv_bfloat16 *A, __nv_bfloat16 *B, __nv_bfloat16 *C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; k++) {
            acc += __bfloat162float(A[row * K + k]) * __bfloat162float(B[col * K + k]);
        }
        C[row * N + col] = __float2bfloat16(acc);
    }
}

void Bf16Gemm(__nv_bfloat16 *A, __nv_bfloat16 *B, __nv_bfloat16 *C, int M, int N, int K) {
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    naive_bf16_gemm_kernel<<<grid, block>>>(A, B, C, M, N, K);
}
