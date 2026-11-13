#include <cuda_runtime.h>
#include "sparse_formats.cuh"

struct SpmvContext {
    NnzChunkData chunks;
    int num_rows;
    int nnz;
    bool empty;
};

__device__ __forceinline__ float warp_sum(float v) {
    v += __shfl_down_sync(0xffffffffu, v, 16);
    v += __shfl_down_sync(0xffffffffu, v, 8);
    v += __shfl_down_sync(0xffffffffu, v, 4);
    v += __shfl_down_sync(0xffffffffu, v, 2);
    v += __shfl_down_sync(0xffffffffu, v, 1);
    return v;
}

__device__ __forceinline__ int find_row_bounded(
    const int* __restrict__ row_ptr,
    int pos,
    int lo,
    int hi) {
    while (lo < hi) {
        const int mid = (lo + hi) >> 1;
        if (row_ptr[mid + 1] <= pos)
            lo = mid + 1;
        else
            hi = mid;
    }
    return lo;
}

__global__ __launch_bounds__(128, 12)
void nnz_chunk_micro_spmv_kernel(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_ind,
    const float* __restrict__ values,
    const float* __restrict__ x,
    float* __restrict__ y,
    const int* __restrict__ chunk_first_row,
    int num_chunks,
    int nnz_per_chunk,
    int num_rows,
    int nnz) {

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int parent = (int)blockIdx.x;

    if (parent >= num_chunks) return;

    const int parent_begin = parent * nnz_per_chunk;
    if (parent_begin >= nnz) return;

    int parent_end = parent_begin + nnz_per_chunk;
    if (parent_end > nnz) parent_end = nnz;

    const int span = parent_end - parent_begin;
    const int begin = parent_begin + ((span * warp) >> 2);
    const int end = parent_begin + ((span * (warp + 1)) >> 2);

    if (begin >= end) return;

    int first_row = 0;
    int last_row = 0;

    if (lane == 0) {
        int lo = chunk_first_row[parent];
        int hi = chunk_first_row[parent + 1] + 1;

        if (lo < 0) lo = 0;
        if (lo >= num_rows) lo = num_rows - 1;
        if (hi < lo + 1) hi = lo + 1;
        if (hi > num_rows) hi = num_rows;

        first_row = find_row_bounded(row_ptr, begin, lo, hi);
        last_row = find_row_bounded(row_ptr, end - 1, first_row, hi);
    }

    first_row = __shfl_sync(0xffffffffu, first_row, 0);
    last_row = __shfl_sync(0xffffffffu, last_row, 0);

    for (int row = first_row; row <= last_row; ++row) {
        int row_begin = 0;
        int row_end = 0;

        if (lane == 0) {
            row_begin = row_ptr[row];
            row_end = row_ptr[row + 1];
        }

        row_begin = __shfl_sync(0xffffffffu, row_begin, 0);
        row_end = __shfl_sync(0xffffffffu, row_end, 0);

        const int rb = row_begin > begin ? row_begin : begin;
        const int re = row_end < end ? row_end : end;

        if (rb < re) {
            float sum = 0.0f;
            int j = rb + lane;

            if (j < re) {
                int cur_col = col_ind[j];
                float cur_val = values[j];
                j += 32;

                while (j < re) {
                    const int next_col = col_ind[j];
                    const float next_val = values[j];
                    sum = fmaf(cur_val, __ldg(x + cur_col), sum);
                    cur_col = next_col;
                    cur_val = next_val;
                    j += 32;
                }

                sum = fmaf(cur_val, __ldg(x + cur_col), sum);
            }

            sum = warp_sum(sum);

            if (lane == 0) {
                if (rb == row_begin && re == row_end)
                    y[row] = sum;
                else
                    atomicAdd(y + row, sum);
            }
        }
    }
}

__global__ void clear_vector_kernel(float* y, int n) {
    const int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) y[i] = 0.0f;
}

void* spmv_preprocess(
    const int* d_row_ptr,
    const int* d_col_ind,
    const float* d_values,
    int num_rows,
    int num_cols,
    int nnz) {

    (void)d_col_ind;
    (void)d_values;
    (void)num_cols;

    SpmvContext* ctx = new SpmvContext;
    ctx->num_rows = num_rows;
    ctx->nnz = nnz;
    ctx->empty = (num_rows == 0 || nnz == 0);

    if (!ctx->empty) {
        ctx->chunks = balance_nnz_chunks(d_row_ptr, num_rows, nnz, 12048);
    } else {
        ctx->chunks.d_chunk_first_row = nullptr;
        ctx->chunks.num_chunks = 0;
        ctx->chunks.nnz_per_chunk = 0;
        ctx->chunks.num_rows = num_rows;
        ctx->chunks.nnz = nnz;
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
    int nnz) {

    (void)num_cols;

    SpmvContext* ctx = static_cast<SpmvContext*>(opaque);

    if (num_rows == 0) return;

    if (ctx == nullptr || ctx->empty || nnz == 0) {
        clear_vector_kernel<<<(num_rows + 255) / 256, 256>>>(d_y, num_rows);
        return;
    }

    const int blocks = ctx->chunks.num_chunks;
    if (blocks > 0) {
        nnz_chunk_micro_spmv_kernel<<<blocks, 128>>>(
            d_row_ptr,
            d_col_ind,
            d_values,
            d_x,
            d_y,
            ctx->chunks.d_chunk_first_row,
            ctx->chunks.num_chunks,
            ctx->chunks.nnz_per_chunk,
            num_rows,
            nnz);
    }
}

void spmv_cleanup(void* opaque) {
    SpmvContext* ctx = static_cast<SpmvContext*>(opaque);
    if (ctx == nullptr) return;

    if (!ctx->empty)
        free_nnz_chunks(ctx->chunks);

    delete ctx;
}