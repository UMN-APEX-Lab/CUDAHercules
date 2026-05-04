#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>

// Naive NVFP4 GEMM: C = A * B^T with local + global scaling
__global__ void naive_nvfp4_gemm_kernel(
    __nv_fp4x2_e2m1 *A, __nv_fp4x2_e2m1 *B,
    __nv_fp8_e4m3 *A_scale, __nv_fp8_e4m3 *B_scale,
    float *A_scale_global, float *B_scale_global,
    __nv_bfloat16 *C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float acc = 0.0f;
        int K_blocks = K / 16;
        for (int k = 0; k < K; k += 2) {
            int a_idx = row * (K / 2) + k / 2;
            int b_idx = col * (K / 2) + k / 2;
            float2 a_vals = static_cast<float2>(A[a_idx]);
            float2 b_vals = static_cast<float2>(B[b_idx]);

            int k_block0 = k / 16;
            int k_block1 = (k + 1) / 16;

            // Scale swizzle index
            int M_block_a = row / 128;
            int K_block_groups = K_blocks / 4;
            int row_in_32 = row % 32;
            int tile_in_block_a = (row / 32) % 4;

            int a_si0 = (M_block_a * K_block_groups + k_block0/4) * 512 + row_in_32 * 16 + tile_in_block_a * 4 + k_block0 % 4;
            int a_si1 = (M_block_a * K_block_groups + k_block1/4) * 512 + row_in_32 * 16 + tile_in_block_a * 4 + k_block1 % 4;

            int N_block_b = col / 128;
            int col_in_32 = col % 32;
            int tile_in_block_b = (col / 32) % 4;

            int b_si0 = (N_block_b * K_block_groups + k_block0/4) * 512 + col_in_32 * 16 + tile_in_block_b * 4 + k_block0 % 4;
            int b_si1 = (N_block_b * K_block_groups + k_block1/4) * 512 + col_in_32 * 16 + tile_in_block_b * 4 + k_block1 % 4;

            float a_s0 = float(A_scale[a_si0]);
            float a_s1 = float(A_scale[a_si1]);
            float b_s0 = float(B_scale[b_si0]);
            float b_s1 = float(B_scale[b_si1]);

            acc += (a_vals.x * a_s0) * (b_vals.x * b_s0);
            acc += (a_vals.y * a_s1) * (b_vals.y * b_s1);
        }
        float global_scale = (*A_scale_global) * (*B_scale_global);
        C[row * N + col] = __float2bfloat16(acc * global_scale);
    }
}

void Nvfp4Gemm(
    __nv_fp4x2_e2m1 *A, __nv_fp4x2_e2m1 *B,
    __nv_fp8_e4m3 *A_scale, __nv_fp8_e4m3 *B_scale,
    float *A_scale_global, float *B_scale_global,
    __nv_bfloat16 *C, int M, int N, int K) {
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    naive_nvfp4_gemm_kernel<<<grid, block>>>(A, B, A_scale, B_scale, A_scale_global, B_scale_global, C, M, N, K);
}
