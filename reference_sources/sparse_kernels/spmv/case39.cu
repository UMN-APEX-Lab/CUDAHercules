#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <cstdint>
#include "sparse_formats.cuh"

struct SpmvContext {
    BSRData bsr;
    unsigned long long* d_descriptors;
    int num_descriptors;
    int num_rows;
    int num_cols;
};

static inline unsigned long long pack_descriptor(int block_row, int block_begin, int block_count)
{
    return (unsigned long long)(unsigned int)block_row |
           ((unsigned long long)(unsigned int)block_begin << 16) |
           ((unsigned long long)(unsigned int)block_count << 48);
}

__global__ __launch_bounds__(64, 24)
void bsr4_descriptor_spmv_kernel(
    const int* __restrict__ bsr_col_ind,
    const float* __restrict__ bsr_values,
    const unsigned long long* __restrict__ descriptors,
    const float* __restrict__ x,
    float* __restrict__ y,
    int num_descriptors,
    int num_rows,
    int num_cols)
{
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int descriptor_id = ((int)blockIdx.x << 1) + warp;

    if (descriptor_id >= num_descriptors) return;

    const unsigned long long packed = descriptors[descriptor_id];
    const int block_row = (int)(packed & 0xffffull);
    const int block_begin = (int)((packed >> 16) & 0xffffffffull);
    const int block_count = (int)(packed >> 48);

    float s0 = 0.0f;
    float s1 = 0.0f;
    float s2 = 0.0f;
    float s3 = 0.0f;

    if (lane < block_count) {
        const int block_index = block_begin + lane;
        const int block_col = bsr_col_ind[block_index];
        const int col = block_col << 2;

        float x0 = 0.0f;
        float x1 = 0.0f;
        float x2 = 0.0f;
        float x3 = 0.0f;

        if (col < num_cols) x0 = __ldg(x + col);
        if (col + 1 < num_cols) x1 = __ldg(x + col + 1);
        if (col + 2 < num_cols) x2 = __ldg(x + col + 2);
        if (col + 3 < num_cols) x3 = __ldg(x + col + 3);

        const float4* v = reinterpret_cast<const float4*>(
            bsr_values + ((size_t)block_index << 4));

        const float4 v0 = v[0];
        const float4 v1 = v[1];
        const float4 v2 = v[2];
        const float4 v3 = v[3];

        s0 = fmaf(v0.x, x0, s0);
        s0 = fmaf(v0.y, x1, s0);
        s0 = fmaf(v0.z, x2, s0);
        s0 = fmaf(v0.w, x3, s0);

        s1 = fmaf(v1.x, x0, s1);
        s1 = fmaf(v1.y, x1, s1);
        s1 = fmaf(v1.z, x2, s1);
        s1 = fmaf(v1.w, x3, s1);

        s2 = fmaf(v2.x, x0, s2);
        s2 = fmaf(v2.y, x1, s2);
        s2 = fmaf(v2.z, x2, s2);
        s2 = fmaf(v2.w, x3, s2);

        s3 = fmaf(v3.x, x0, s3);
        s3 = fmaf(v3.y, x1, s3);
        s3 = fmaf(v3.z, x2, s3);
        s3 = fmaf(v3.w, x3, s3);
    }

    if (lane < 16) {
        constexpr unsigned mask = 0x0000ffffu;

        s0 += __shfl_down_sync(mask, s0, 8);
        s1 += __shfl_down_sync(mask, s1, 8);
        s2 += __shfl_down_sync(mask, s2, 8);
        s3 += __shfl_down_sync(mask, s3, 8);

        s0 += __shfl_down_sync(mask, s0, 4);
        s1 += __shfl_down_sync(mask, s1, 4);
        s2 += __shfl_down_sync(mask, s2, 4);
        s3 += __shfl_down_sync(mask, s3, 4);

        s0 += __shfl_down_sync(mask, s0, 2);
        s1 += __shfl_down_sync(mask, s1, 2);
        s2 += __shfl_down_sync(mask, s2, 2);
        s3 += __shfl_down_sync(mask, s3, 2);

        s0 += __shfl_down_sync(mask, s0, 1);
        s1 += __shfl_down_sync(mask, s1, 1);
        s2 += __shfl_down_sync(mask, s2, 1);
        s3 += __shfl_down_sync(mask, s3, 1);

        if (lane == 0) {
            const int row = block_row << 2;
            if (row < num_rows) atomicAdd(y + row, s0);
            if (row + 1 < num_rows) atomicAdd(y + row + 1, s1);
            if (row + 2 < num_rows) atomicAdd(y + row + 2, s2);
            if (row + 3 < num_rows) atomicAdd(y + row + 3, s3);
        }
    }
}

void* spmv_preprocess(
    const int* d_row_ptr,
    const int* d_col_ind,
    const float* d_values,
    int num_rows,
    int num_cols,
    int nnz)
{
    SpmvContext* ctx = new SpmvContext();
    ctx->bsr = csr_to_bsr(
        d_row_ptr, d_col_ind, d_values,
        num_rows, num_cols, nnz, 4);

    ctx->d_descriptors = nullptr;
    ctx->num_descriptors = 0;
    ctx->num_rows = num_rows;
    ctx->num_cols = num_cols;

    const int mb = (num_rows + 3) >> 2;
    if (mb <= 0) return ctx;

    std::vector<int> h_row_ptr((size_t)mb + 1);
    cudaMemcpy(
        h_row_ptr.data(),
        ctx->bsr.d_bsr_row_ptr,
        ((size_t)mb + 1) * sizeof(int),
        cudaMemcpyDeviceToHost);

    constexpr int chunk_blocks = 16;
    std::vector<unsigned long long> host_descriptors;
    host_descriptors.reserve((size_t)mb + 4096);

    for (int block_row = 0; block_row < mb; ++block_row) {
        const int begin = h_row_ptr[block_row];
        const int end = h_row_ptr[block_row + 1];

        for (int p = begin; p < end; p += chunk_blocks) {
            const int count = std::min(chunk_blocks, end - p);
            host_descriptors.push_back(pack_descriptor(block_row, p, count));
        }
    }

    ctx->num_descriptors = (int)host_descriptors.size();

    if (ctx->num_descriptors > 0) {
        cudaMalloc(
            reinterpret_cast<void**>(&ctx->d_descriptors),
            (size_t)ctx->num_descriptors * sizeof(unsigned long long));

        cudaMemcpy(
            ctx->d_descriptors,
            host_descriptors.data(),
            (size_t)ctx->num_descriptors * sizeof(unsigned long long),
            cudaMemcpyHostToDevice);
    }

    return ctx;
}

void spmv_compute(
    void* opaque,
    const int* d_row_ptr,
    const int* d_col_ind,
    const float* d_values,
    const float* d_x,
    float* d_y,
    int num_rows,
    int num_cols,
    int nnz)
{
    (void)d_row_ptr;
    (void)d_col_ind;
    (void)d_values;
    (void)nnz;

    SpmvContext* ctx = static_cast<SpmvContext*>(opaque);
    if (!ctx || ctx->num_descriptors <= 0) return;

    const int blocks = (ctx->num_descriptors + 1) >> 1;

    bsr4_descriptor_spmv_kernel<<<blocks, 64>>>(
        ctx->bsr.d_bsr_col_ind,
        ctx->bsr.d_bsr_values,
        ctx->d_descriptors,
        d_x,
        d_y,
        ctx->num_descriptors,
        num_rows,
        num_cols);
}

void spmv_cleanup(void* opaque)
{
    SpmvContext* ctx = static_cast<SpmvContext*>(opaque);
    if (!ctx) return;

    if (ctx->d_descriptors) cudaFree(ctx->d_descriptors);
    free_bsr(ctx->bsr);
    delete ctx;
}