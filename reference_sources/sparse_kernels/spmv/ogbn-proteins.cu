#include <cuda_runtime.h>
#include "sparse_formats.cuh"

struct SpmvContext {
    NnzChunkData chunks;
    int num_chunks;
    int nnz_per_chunk;
};

__device__ __forceinline__ int find_row_for_nnz_bounded(
    const int* __restrict__ row_ptr,
    int lo,
    int hi,
    int pos)
{
    while (lo + 1 < hi) {
        int mid = lo + ((hi - lo) >> 1);
        if (row_ptr[mid] <= pos)
            lo = mid;
        else
            hi = mid;
    }
    return lo;
}

__device__ __forceinline__ float load_product(
    const int* __restrict__ col_ind,
    const float* __restrict__ values,
    const float* __restrict__ x,
    int p,
    int num_cols)
{
    int col = __ldg(col_ind + p);
    float v = __ldg(values + p);
    return ((unsigned)col < (unsigned)num_cols) ? v * __ldg(x + col) : 0.0f;
}

__global__ void spmv_zero_kernel(float* __restrict__ y, int n)
{
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n)
        y[i] = 0.0f;
}

__global__ __launch_bounds__(256, 6)
void spmv_nnz_chunk_prefetch_kernel(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_ind,
    const float* __restrict__ values,
    const float* __restrict__ x,
    float* __restrict__ y,
    const int* __restrict__ chunk_first_row,
    int num_rows,
    int num_cols,
    int nnz,
    int nnz_per_chunk)
{
    const int chunk = (int)blockIdx.x;
    const int chunk_begin = chunk * nnz_per_chunk;
    const int chunk_end_raw = chunk_begin + nnz_per_chunk;
    const int chunk_end = chunk_end_raw < nnz ? chunk_end_raw : nnz;

    if (chunk_begin >= chunk_end)
        return;

    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int warp_begin = chunk_begin + warp * 256;
    const int warp_end = (warp_begin + 256 < chunk_end) ? warp_begin + 256 : chunk_end;

    if (warp_begin >= warp_end)
        return;

    const unsigned full_mask = 0xffffffffu;
    const int first_row = chunk_first_row[chunk];
    int upper_row = chunk_first_row[chunk + 1] + 1;
    if (upper_row > num_rows)
        upper_row = num_rows;
    if (upper_row <= first_row)
        upper_row = first_row + 1;
    if (upper_row > num_rows)
        upper_row = num_rows;

    int p = warp_begin + lane;
    bool active = p < warp_end;
    int row = first_row;
    int row_end = 0;
    float sum = 0.0f;

    if (active) {
        row = find_row_for_nnz_bounded(row_ptr, first_row, upper_row, p);
        row_end = row_ptr[row + 1];
        sum = load_product(col_ind, values, x, p, num_cols);
    }

    #pragma unroll
    for (int iter = 0; iter < 8; ++iter) {
        if (active) {
            while (p >= row_end && row + 1 < num_rows) {
                ++row;
                row_end = row_ptr[row + 1];
            }
        }

        const int next_p = p + 32;
        const bool next_active = next_p < warp_end;
        float next_sum = 0.0f;

        if (next_active)
            next_sum = load_product(col_ind, values, x, next_p, num_cols);

        const int key = active ? row : -1;
        const unsigned group = __match_any_sync(full_mask, key);

        #pragma unroll
        for (int delta = 16; delta > 0; delta >>= 1) {
            const int src = lane + delta;
            const float other = __shfl_down_sync(group, sum, delta);
            if (src < 32 && (group & (1u << src)))
                sum += other;
        }

        if (active && (group & ((1u << lane) - 1u)) == 0u) {
            const int batch_begin = warp_begin + iter * 32;
            const int batch_end_raw = batch_begin + 32;
            const int batch_end = batch_end_raw < warp_end ? batch_end_raw : warp_end;
            const int rb = row_ptr[row];
            const int re = row_ptr[row + 1];

            if (rb >= batch_begin && re <= batch_end)
                y[row] = sum;
            else
                atomicAdd(y + row, sum);
        }

        p = next_p;
        active = next_active;
        sum = next_sum;
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
    (void)d_col_ind;
    (void)d_values;
    (void)num_cols;

    SpmvContext* ctx = new SpmvContext{};
    ctx->nnz_per_chunk = 2048;
    ctx->num_chunks = nnz > 0 ? (nnz + 2047) / 2048 : 0;

    if (ctx->num_chunks > 0) {
        ctx->chunks = balance_nnz_chunks(
            d_row_ptr, num_rows, nnz, ctx->num_chunks);
        ctx->num_chunks = ctx->chunks.num_chunks;
        ctx->nnz_per_chunk = ctx->chunks.nnz_per_chunk;
    }

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
    int nnz)
{
    if (num_rows <= 0)
        return;

    constexpr int threads = 256;
    const int zero_blocks = (num_rows + threads - 1) / threads;
    spmv_zero_kernel<<<zero_blocks, threads>>>(d_y, num_rows);

    SpmvContext* ctx = static_cast<SpmvContext*>(opaque_ctx);
    if (nnz <= 0 || ctx == nullptr || ctx->num_chunks <= 0)
        return;

    spmv_nnz_chunk_prefetch_kernel<<<ctx->num_chunks, threads>>>(
        d_row_ptr,
        d_col_ind,
        d_values,
        d_x,
        d_y,
        ctx->chunks.d_chunk_first_row,
        num_rows,
        num_cols,
        nnz,
        ctx->nnz_per_chunk);
}

void spmv_cleanup(void* opaque_ctx)
{
    SpmvContext* ctx = static_cast<SpmvContext*>(opaque_ctx);
    if (ctx == nullptr)
        return;

    if (ctx->num_chunks > 0)
        free_nnz_chunks(ctx->chunks);

    delete ctx;
}