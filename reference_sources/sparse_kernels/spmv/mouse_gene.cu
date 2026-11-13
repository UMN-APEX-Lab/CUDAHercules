#include <cuda_runtime.h>
#include "sparse_formats.cuh"

struct SpmvContext {
    NnzChunkData chunks;
    int num_chunks;
    int nnz_per_chunk;
    int num_rows;
    int num_cols;
    int nnz;
    bool valid_chunks;
};

__global__ void zero_vector_kernel(float* __restrict__ y, int n)
{
    const int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) y[i] = 0.0f;
}

__device__ __forceinline__ int find_row_for_nnz(
    const int* __restrict__ row_ptr,
    int num_rows,
    int pos,
    int hint)
{
    int lo = hint;
    if (lo < 0) lo = 0;
    if (lo > num_rows) lo = num_rows;

    int hi = num_rows + 1;
    while (lo < hi) {
        const int mid = lo + ((hi - lo) >> 1);
        if (row_ptr[mid] <= pos)
            lo = mid + 1;
        else
            hi = mid;
    }

    const int row = lo - 1;
    return row < 0 ? 0 : row;
}

__global__ __launch_bounds__(128, 12)
void nnz_chunk_spmv_kernel(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_ind,
    const float* __restrict__ values,
    const float* __restrict__ x,
    float* __restrict__ y,
    const int* __restrict__ chunk_first_row,
    int num_rows,
    int nnz,
    int nnz_per_chunk,
    int num_chunks)
{
    const int chunk = (int)blockIdx.x;
    if (chunk >= num_chunks) return;

    const int warp = (int)(threadIdx.x >> 5);
    const int lane = (int)(threadIdx.x & 31);

    const int chunk_begin = chunk * nnz_per_chunk;
    int chunk_end = chunk_begin + nnz_per_chunk;
    if (chunk_end > nnz) chunk_end = nnz;
    if (chunk_begin >= chunk_end) return;

    const int total = chunk_end - chunk_begin;
    const int warp_begin = chunk_begin + ((total * warp) >> 2);
    const int warp_end = chunk_begin + ((total * (warp + 1)) >> 2);
    if (warp_begin >= warp_end) return;

    int row = find_row_for_nnz(
        row_ptr, num_rows, warp_begin, chunk_first_row[chunk]);

    int cur = warp_begin;

    while (cur < warp_end && row < num_rows) {
        while (row < num_rows && row_ptr[row + 1] <= cur)
            ++row;

        if (row >= num_rows) break;

        const int row_begin = row_ptr[row];
        const int row_end = row_ptr[row + 1];
        const int seg_begin = cur > row_begin ? cur : row_begin;
        const int seg_end = warp_end < row_end ? warp_end : row_end;

        float sum = 0.0f;
        int p = seg_begin + lane;

        if (p < seg_end) {
            int col0 = col_ind[p];
            float val0 = values[p];
            p += 32;

            while (p < seg_end) {
                const int col1 = col_ind[p];
                const float val1 = values[p];

                const float x0 = __ldg(x + col0);
                const float x1 = __ldg(x + col1);

                sum = fmaf(val0, x0, sum);
                sum = fmaf(val1, x1, sum);

                p += 32;
                if (p >= seg_end) {
                    col0 = -1;
                    break;
                }

                col0 = col_ind[p];
                val0 = values[p];
                p += 32;
            }

            if (col0 >= 0)
                sum = fmaf(val0, __ldg(x + col0), sum);
        }

        sum += __shfl_down_sync(0xffffffff, sum, 16);
        sum += __shfl_down_sync(0xffffffff, sum, 8);
        sum += __shfl_down_sync(0xffffffff, sum, 4);
        sum += __shfl_down_sync(0xffffffff, sum, 2);
        sum += __shfl_down_sync(0xffffffff, sum, 1);

        if (lane == 0) {
            if (seg_begin == row_begin && seg_end == row_end)
                y[row] = sum;
            else
                atomicAdd(y + row, sum);
        }

        cur = seg_end;
        if (seg_end >= row_end)
            ++row;
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

    SpmvContext* ctx = new SpmvContext{};
    ctx->num_rows = num_rows;
    ctx->num_cols = num_cols;
    ctx->nnz = nnz;
    ctx->valid_chunks = false;

    if (num_rows > 0 && nnz > 0) {
        constexpr int kNnzPerChunk = 2048;
        int requested_chunks = (nnz + kNnzPerChunk - 1) / kNnzPerChunk;
        if (requested_chunks < 1) requested_chunks = 1;

        ctx->chunks = balance_nnz_chunks(
            d_row_ptr, num_rows, nnz, requested_chunks);

        ctx->num_chunks = ctx->chunks.num_chunks;
        ctx->nnz_per_chunk = ctx->chunks.nnz_per_chunk;
        ctx->valid_chunks = ctx->num_chunks > 0;
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
    (void)num_cols;

    SpmvContext* ctx = static_cast<SpmvContext*>(opaque_ctx);

    if (!ctx || !ctx->valid_chunks || nnz <= 0) {
        if (num_rows > 0) {
            const int blocks = (num_rows + 255) >> 8;
            zero_vector_kernel<<<blocks, 256>>>(d_y, num_rows);
        }
        return;
    }

    nnz_chunk_spmv_kernel<<<ctx->num_chunks, 128>>>(
        d_row_ptr,
        d_col_ind,
        d_values,
        d_x,
        d_y,
        ctx->chunks.d_chunk_first_row,
        num_rows,
        nnz,
        ctx->nnz_per_chunk,
        ctx->num_chunks);
}

void spmv_cleanup(void* opaque_ctx)
{
    SpmvContext* ctx = static_cast<SpmvContext*>(opaque_ctx);
    if (!ctx) return;

    if (ctx->valid_chunks)
        free_nnz_chunks(ctx->chunks);

    delete ctx;
}