#include <cuda_runtime.h>
#include "sparse_formats.cuh"

struct SpMMContext {
    COOData coo;
    int nnz;
};

__device__ __forceinline__
void segmented_scan8_fullwarp(
    int lane, int row,
    float& v0, float& v1, float& v2, float& v3,
    float& v4, float& v5, float& v6, float& v7) {
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        const int prev_row = __shfl_up_sync(0xffffffffu, row, offset);
        const float t0 = __shfl_up_sync(0xffffffffu, v0, offset);
        const float t1 = __shfl_up_sync(0xffffffffu, v1, offset);
        const float t2 = __shfl_up_sync(0xffffffffu, v2, offset);
        const float t3 = __shfl_up_sync(0xffffffffu, v3, offset);
        const float t4 = __shfl_up_sync(0xffffffffu, v4, offset);
        const float t5 = __shfl_up_sync(0xffffffffu, v5, offset);
        const float t6 = __shfl_up_sync(0xffffffffu, v6, offset);
        const float t7 = __shfl_up_sync(0xffffffffu, v7, offset);

        if (lane >= offset && row == prev_row) {
            v0 += t0;
            v1 += t1;
            v2 += t2;
            v3 += t3;
            v4 += t4;
            v5 += t5;
            v6 += t6;
            v7 += t7;
        }
    }
}

__global__ __launch_bounds__(128)
void coo_spmm_k8_kernel(
    const int* __restrict__ row_ind,
    const int* __restrict__ col_ind,
    const float* __restrict__ values,
    const float* __restrict__ B,
    float* __restrict__ C,
    int nnz) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int base = blockIdx.x * 512 + warp * 128;

#pragma unroll
    for (int it = 0; it < 4; ++it) {
        const int p = base + it * 32 + lane;
        const bool valid = p < nnz;

        int row = -1;
        float v0 = 0.0f;
        float v1 = 0.0f;
        float v2 = 0.0f;
        float v3 = 0.0f;
        float v4 = 0.0f;
        float v5 = 0.0f;
        float v6 = 0.0f;
        float v7 = 0.0f;

        if (valid) {
            row = row_ind[p];
            const int col = col_ind[p];
            const float a = values[p];
            const float* bp = B + (size_t)col * 8;

            const float4 b0 = *reinterpret_cast<const float4*>(bp);
            const float4 b1 = *reinterpret_cast<const float4*>(bp + 4);

            v0 = a * b0.x;
            v1 = a * b0.y;
            v2 = a * b0.z;
            v3 = a * b0.w;
            v4 = a * b1.x;
            v5 = a * b1.y;
            v6 = a * b1.z;
            v7 = a * b1.w;
        }

        segmented_scan8_fullwarp(
            lane, row, v0, v1, v2, v3, v4, v5, v6, v7);

        const int next_row = __shfl_down_sync(0xffffffffu, row, 1);
        const bool segment_end = valid && (lane == 31 || row != next_row);

        if (segment_end) {
            float* out = C + (size_t)row * 8;
#if __CUDA_ARCH__ >= 900
            atomicAdd(reinterpret_cast<float4*>(out), make_float4(v0, v1, v2, v3));
            atomicAdd(reinterpret_cast<float4*>(out + 4), make_float4(v4, v5, v6, v7));
#else
            atomicAdd(out + 0, v0);
            atomicAdd(out + 1, v1);
            atomicAdd(out + 2, v2);
            atomicAdd(out + 3, v3);
            atomicAdd(out + 4, v4);
            atomicAdd(out + 5, v5);
            atomicAdd(out + 6, v6);
            atomicAdd(out + 7, v7);
#endif
        }
    }
}

__global__
void coo_spmm_generic_kernel(
    const int* __restrict__ row_ind,
    const int* __restrict__ col_ind,
    const float* __restrict__ values,
    const float* __restrict__ B,
    float* __restrict__ C,
    int nnz,
    int K) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= nnz) return;

    const int row = row_ind[p];
    const int col = col_ind[p];
    const float a = values[p];
    const float* bp = B + (size_t)col * K;
    float* cp = C + (size_t)row * K;

    for (int k = 0; k < K; ++k) {
        atomicAdd(cp + k, a * bp[k]);
    }
}

void* spmm_preprocess(
    const int* d_row_ptr,
    const int* d_col_ind,
    const float* d_values,
    int num_rows,
    int num_cols,
    int nnz,
    int K) {
    (void)K;
    SpMMContext* ctx = new SpMMContext;
    ctx->coo = csr_to_coo(
        d_row_ptr, d_col_ind, d_values,
        num_rows, num_cols, nnz);
    ctx->nnz = nnz;
    return ctx;
}

void spmm_compute(
    void* context,
    const int* d_row_ptr,
    const int* d_col_ind,
    const float* d_values,
    const float* d_B,
    float* d_C,
    int num_rows,
    int num_cols,
    int nnz,
    int K) {
    (void)d_row_ptr;
    (void)d_col_ind;
    (void)d_values;
    (void)num_cols;
    (void)nnz;

    if (num_rows <= 0 || K <= 0) return;

    cudaMemsetAsync(
        d_C, 0,
        (size_t)num_rows * (size_t)K * sizeof(float),
        0);

    SpMMContext* ctx = static_cast<SpMMContext*>(context);
    if (ctx == nullptr || ctx->nnz <= 0) return;

    if (K == 8) {
        const int blocks = (ctx->nnz + 511) >> 9;
        coo_spmm_k8_kernel<<<blocks, 128>>>(
            ctx->coo.d_row_ind,
            ctx->coo.d_col_ind,
            ctx->coo.d_values,
            d_B,
            d_C,
            ctx->nnz);
    } else {
        constexpr int threads = 256;
        const int blocks = (ctx->nnz + threads - 1) / threads;
        coo_spmm_generic_kernel<<<blocks, threads>>>(
            ctx->coo.d_row_ind,
            ctx->coo.d_col_ind,
            ctx->coo.d_values,
            d_B,
            d_C,
            ctx->nnz,
            K);
    }
}

void spmm_cleanup(void* context) {
    SpMMContext* ctx = static_cast<SpMMContext*>(context);
    if (ctx != nullptr) {
        free_coo(ctx->coo);
        delete ctx;
    }
}