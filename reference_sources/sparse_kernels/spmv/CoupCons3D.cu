#include <cuda_runtime.h>
#include <cstdlib>
#include "sparse_formats.cuh"

struct SpMVContext {
    SELLData sell;
    int num_rows;
    int num_cols;
};

__device__ __forceinline__ int load_streaming_i32(const int* p) {
    int v;
    asm volatile("ld.global.cs.s32 %0, [%1];" : "=r"(v) : "l"(p));
    return v;
}

__device__ __forceinline__ float load_streaming_f32(const float* p) {
    float v;
    asm volatile("ld.global.cs.f32 %0, [%1];" : "=f"(v) : "l"(p));
    return v;
}

__device__ __forceinline__ float load_x_readonly(
    const float* __restrict__ x,
    int col) {
    return __ldg(x + col);
}

__global__ __launch_bounds__(128, 12)
void sell_spmv_dual_slice_kernel(
    const int* __restrict__ col_ind,
    const float* __restrict__ values,
    const int* __restrict__ slice_ptr,
    const int* __restrict__ slice_width,
    const int* __restrict__ row_perm,
    const float* __restrict__ x,
    float* __restrict__ y,
    int num_rows,
    int num_slices) {
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int half = lane >> 4;
    const int local_row = lane & 15;

    const int slice = blockIdx.x * 8 + (warp << 1) + half;
    if (slice >= num_slices) return;

    const int base = slice_ptr[slice];
    const int width = slice_width[slice];
    float sum = 0.0f;

    int j = 0;
#pragma unroll 1
    for (; j + 3 < width; j += 4) {
        const int p0 = base + j * 16 + local_row;
        const int p1 = p0 + 16;
        const int p2 = p1 + 16;
        const int p3 = p2 + 16;

        const int c0 = load_streaming_i32(col_ind + p0);
        const int c1 = load_streaming_i32(col_ind + p1);
        const int c2 = load_streaming_i32(col_ind + p2);
        const int c3 = load_streaming_i32(col_ind + p3);

        const float a0 = load_streaming_f32(values + p0);
        const float a1 = load_streaming_f32(values + p1);
        const float a2 = load_streaming_f32(values + p2);
        const float a3 = load_streaming_f32(values + p3);

        const float x0 = (c0 >= 0) ? load_x_readonly(x, c0) : 0.0f;
        const float x1 = (c1 >= 0) ? load_x_readonly(x, c1) : 0.0f;
        const float x2 = (c2 >= 0) ? load_x_readonly(x, c2) : 0.0f;
        const float x3 = (c3 >= 0) ? load_x_readonly(x, c3) : 0.0f;

        sum = fmaf(a0, x0, sum);
        sum = fmaf(a1, x1, sum);
        sum = fmaf(a2, x2, sum);
        sum = fmaf(a3, x3, sum);
    }

#pragma unroll 1
    for (; j < width; ++j) {
        const int p = base + j * 16 + local_row;
        const int col = load_streaming_i32(col_ind + p);
        const float val = load_streaming_f32(values + p);
        const float xv = (col >= 0) ? load_x_readonly(x, col) : 0.0f;
        sum = fmaf(val, xv, sum);
    }

    const int packed_row = slice * 16 + local_row;
    const int original_row = row_perm[packed_row];
    if (original_row < num_rows) {
        y[original_row] = sum;
    }
}

void* spmv_preprocess(
    const int* d_row_ptr,
    const int* d_col_ind,
    const float* d_values,
    int num_rows,
    int num_cols,
    int nnz) {
    SpMVContext* ctx =
        static_cast<SpMVContext*>(std::malloc(sizeof(SpMVContext)));
    if (ctx == nullptr) return nullptr;

    ctx->num_rows = num_rows;
    ctx->num_cols = num_cols;
    ctx->sell = csr_to_sell(
        d_row_ptr,
        d_col_ind,
        d_values,
        num_rows,
        num_cols,
        nnz,
        16,
        16);
    return ctx;
}

void spmv_compute(
    void* opaque_ctx,
    const int* d_row_ptr,
    const int* d_col_ind,
    const float* d_values,
    const float* d_x,
    float* d_y,
    int num_rows,
    int num_cols,
    int nnz) {
    (void)d_row_ptr;
    (void)d_col_ind;
    (void)d_values;
    (void)num_cols;
    (void)nnz;

    SpMVContext* ctx = static_cast<SpMVContext*>(opaque_ctx);
    if (ctx == nullptr || num_rows <= 0 || ctx->sell.num_slices <= 0) return;

    const int blocks = (ctx->sell.num_slices + 7) >> 3;
    sell_spmv_dual_slice_kernel<<<blocks, 128>>>(
        ctx->sell.d_col_ind,
        ctx->sell.d_values,
        ctx->sell.d_slice_ptr,
        ctx->sell.d_slice_width,
        ctx->sell.d_row_perm,
        d_x,
        d_y,
        num_rows,
        ctx->sell.num_slices);
}

void spmv_cleanup(void* opaque_ctx) {
    SpMVContext* ctx = static_cast<SpMVContext*>(opaque_ctx);
    if (ctx == nullptr) return;
    free_sell(ctx->sell);
    std::free(ctx);
}