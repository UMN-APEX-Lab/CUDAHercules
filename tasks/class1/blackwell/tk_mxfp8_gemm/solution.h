#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

// Naive MXFP8 GEMM: C = A * B^T with block scaling
// This is a simplified naive implementation that ignores scale swizzling
__global__ void naive_mxfp8_gemm_kernel(
    __nv_fp8_e4m3 *A, __nv_fp8_e4m3 *B,
    __nv_fp8_e8m0 *A_scale, __nv_fp8_e8m0 *B_scale,
    __nv_bfloat16 *C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float acc = 0.0f;
        int K_blocks = K / 128;
        for (int k = 0; k < K; k++) {
            int k_block = k / 128;
            // Scale swizzle index computation
            int M_block = row / 128;
            int K_block_groups = K_blocks / 4;
            int K_block_group = k_block / 4;
            int row_in_32 = row % 32;
            int tile_in_block = (row / 32) % 4;
            int kb_in_block = k_block % 4;
            int a_scale_idx = (M_block * K_block_groups + K_block_group) * 512 + row_in_32 * 16 + tile_in_block * 4 + kb_in_block;

            int N_block = col / 128;
            int b_scale_idx = (N_block * K_block_groups + K_block_group) * 512 + (col % 32) * 16 + ((col / 32) % 4) * 4 + kb_in_block;

            float a_s = static_cast<float>(A_scale[a_scale_idx]);
            float b_s = static_cast<float>(B_scale[b_scale_idx]);
            float a = float(A[row * K + k]) * a_s;
            float b = float(B[col * K + k]) * b_s;
            acc += a * b;
        }
        C[row * N + col] = __float2bfloat16(acc);
    }
}

void Mxfp8Gemm(
    __nv_fp8_e4m3 *A, __nv_fp8_e4m3 *B,
    __nv_fp8_e8m0 *A_scale, __nv_fp8_e8m0 *B_scale,
    __nv_bfloat16 *C, int M, int N, int K) {
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    naive_mxfp8_gemm_kernel<<<grid, block>>>(A, B, A_scale, B_scale, C, M, N, K);
}
