#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>

// Naive fused GEMM + L2-norm-squared kernel.
// Each thread computes one row of D and one element of sqr_sum.
__global__ void naive_hc_prenorm_kernel(
    const __nv_bfloat16* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ d,
    float* __restrict__ sqr_sum,
    int M, int N, int K) {

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;

    float sq = 0.0f;
    for (int n = 0; n < N; n++) {
        float acc = 0.0f;
        for (int ki = 0; ki < K; ki++) {
            float a_val = __bfloat162float(a[row * K + ki]);
            if (n == 0) sq += a_val * a_val;
            acc += a_val * b[n * K + ki];
        }
        d[row * N + n] = acc;
    }
    sqr_sum[row] = sq;
}

void solution_hc_prenorm_gemm(
    const __nv_bfloat16* a, const float* b,
    float* d, float* sqr_sum,
    int M, int N, int K,
    cudaStream_t stream) {

    int threads = 256;
    int blocks = (M + threads - 1) / threads;
    naive_hc_prenorm_kernel<<<blocks, threads, 0, stream>>>(
        a, b, d, sqr_sum, M, N, K);
}
