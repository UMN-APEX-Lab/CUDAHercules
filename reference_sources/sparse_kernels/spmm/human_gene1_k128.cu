#include <cuda_runtime.h>
#include <stddef.h>
#include "sparse_formats.cuh"

struct SpmmContext {
    NnzChunkData chunks;
    int chunk_size;
    int num_chunks;
};

__device__ __forceinline__ int find_row_for_nnz(
    const int* __restrict__ row_ptr,
    int num_rows,
    int nnz_index)
{
    int lo = 0;
    int hi = num_rows;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (row_ptr[mid + 1] <= nnz_index)
            lo = mid + 1;
        else
            hi = mid;
    }
    return lo < num_rows ? lo : num_rows - 1;
}

__global__ __launch_bounds__(128, 12)
void csr_chunk_spmm128_kernel(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_ind,
    const float* __restrict__ values,
    const int* __restrict__ chunk_first_row,
    const float* __restrict__ B,
    float* __restrict__ C,
    int num_rows,
    int nnz,
    int chunk_size)
{
    const int lane = (int)threadIdx.x & 31;
    const int warp = (int)threadIdx.x >> 5;
    const int chunk = ((int)blockIdx.x << 2) + warp;
    const int begin_nnz = chunk * chunk_size;

    if (begin_nnz >= nnz || num_rows <= 0)
        return;

    const int end_nnz = min(begin_nnz + chunk_size, nnz);
    int row = chunk_first_row ? chunk_first_row[chunk]
                              : find_row_for_nnz(row_ptr, num_rows, begin_nnz);
    int pos = begin_nnz;
    const unsigned mask = 0xffffffffu;

    while (pos < end_nnz && row < num_rows) {
        const int row_begin = row_ptr[row];
        const int row_end = row_ptr[row + 1];

        if (row_end <= pos) {
            ++row;
            continue;
        }

        const int fragment_begin = pos > row_begin ? pos : row_begin;
        const int fragment_end = row_end < end_nnz ? row_end : end_nnz;

        float acc0 = 0.0f;
        float acc1 = 0.0f;
        float acc2 = 0.0f;
        float acc3 = 0.0f;

        int p = fragment_begin;
        int cur_col = 0;
        float cur_val = 0.0f;

        if (p + lane < fragment_end) {
            cur_col = col_ind[p + lane];
            cur_val = values[p + lane];
        }

        while (p < fragment_end) {
            const int next_p = p + 32;
            int next_col = 0;
            float next_val = 0.0f;

            if (next_p + lane < fragment_end) {
                next_col = col_ind[next_p + lane];
                next_val = values[next_p + lane];
            }

            const int count = min(32, fragment_end - p);

#pragma unroll 1
            for (int q = 0; q < count; ++q) {
                const int col = __shfl_sync(mask, cur_col, q);
                const float val = __shfl_sync(mask, cur_val, q);
                const size_t base = ((size_t)col << 7) + lane;

                const float b0 = B[base];
                const float b1 = B[base + 32];
                const float b2 = B[base + 64];
                const float b3 = B[base + 96];

                acc0 = fmaf(val, b0, acc0);
                acc1 = fmaf(val, b1, acc1);
                acc2 = fmaf(val, b2, acc2);
                acc3 = fmaf(val, b3, acc3);
            }

            cur_col = next_col;
            cur_val = next_val;
            p = next_p;
        }

        const bool partial =
            (fragment_begin != row_begin) || (fragment_end != row_end);

        float* out = C + ((size_t)row << 7) + lane;

        if (partial) {
            atomicAdd(out, acc0);
            atomicAdd(out + 32, acc1);
            atomicAdd(out + 64, acc2);
            atomicAdd(out + 96, acc3);
        } else {
            out[0] = acc0;
            out[32] = acc1;
            out[64] = acc2;
            out[96] = acc3;
        }

        pos = fragment_end;
        if (fragment_end >= row_end)
            ++row;
    }
}

__global__ __launch_bounds__(128, 12)
void csr_fallback_spmm_kernel(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_ind,
    const float* __restrict__ values,
    const float* __restrict__ B,
    float* __restrict__ C,
    int num_rows,
    int K)
{
    const int warp = (int)threadIdx.x >> 5;
    const int lane = (int)threadIdx.x & 31;
    const int row = ((int)blockIdx.x << 2) + warp;

    if (row >= num_rows)
        return;

    const int begin = row_ptr[row];
    const int end = row_ptr[row + 1];

    for (int k = lane; k < K; k += 32) {
        float sum = 0.0f;
        for (int p = begin; p < end; ++p)
            sum = fmaf(values[p], B[(size_t)col_ind[p] * K + k], sum);
        C[(size_t)row * K + k] = sum;
    }
}

void* spmm_preprocess(
    const int* d_row_ptr,
    const int* d_col_ind,
    const float* d_values,
    int num_rows,
    int num_cols,
    int nnz,
    int K)
{
    (void)d_col_ind;
    (void)d_values;
    (void)num_cols;
    (void)K;

    SpmmContext* ctx = new SpmmContext;
    ctx->chunk_size = 1024;
    ctx->num_chunks = nnz > 0 ? (nnz + 1023) / 1024 : 0;

    if (ctx->num_chunks > 0) {
        ctx->chunks = balance_nnz_chunks(
            d_row_ptr, num_rows, nnz, ctx->num_chunks);
        ctx->chunk_size = ctx->chunks.nnz_per_chunk;
        ctx->num_chunks = ctx->chunks.num_chunks;
    }

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
    int K)
{
    (void)num_cols;

    if (num_rows <= 0 || K <= 0)
        return;

    if (K != 128 || context == nullptr) {
        dim3 block(128);
        dim3 grid((num_rows + 3) >> 2);
        csr_fallback_spmm_kernel<<<grid, block>>>(
            d_row_ptr, d_col_ind, d_values, d_B, d_C, num_rows, K);
        return;
    }

    cudaMemsetAsync(d_C, 0, (size_t)num_rows * 128 * sizeof(float), 0);

    if (nnz <= 0)
        return;

    SpmmContext* ctx = static_cast<SpmmContext*>(context);
    const int chunks = (nnz + ctx->chunk_size - 1) / ctx->chunk_size;
    const int blocks = (chunks + 3) >> 2;

    csr_chunk_spmm128_kernel<<<blocks, 128>>>(
        d_row_ptr,
        d_col_ind,
        d_values,
        ctx->chunks.d_chunk_first_row,
        d_B,
        d_C,
        num_rows,
        nnz,
        ctx->chunk_size);
}

void spmm_cleanup(void* context)
{
    SpmmContext* ctx = static_cast<SpmmContext*>(context);
    if (ctx != nullptr) {
        if (ctx->num_chunks > 0)
            free_nnz_chunks(ctx->chunks);
        delete ctx;
    }
}