#include <cuda_runtime.h>
#include <stdint.h>
#include "sparse_formats.cuh"

struct SpmvContext {
    NnzChunkData chunks;
    int num_chunks;
    int nnz_per_chunk;
    int num_rows;
    int nnz;
};

__device__ __forceinline__ float ro_load(const float* p) {
    return __ldg(p);
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
    unsigned mask = 0xffffffffu;
    v += __shfl_down_sync(mask, v, 16);
    v += __shfl_down_sync(mask, v, 8);
    v += __shfl_down_sync(mask, v, 4);
    v += __shfl_down_sync(mask, v, 2);
    v += __shfl_down_sync(mask, v, 1);
    return v;
}

__device__ __forceinline__ int find_row_for_nnz(
    const int* __restrict__ row_ptr,
    int pos,
    int lo,
    int hi) {

    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        int next = __ldg(row_ptr + mid + 1);
        if (next <= pos)
            lo = mid + 1;
        else
            hi = mid;
    }
    return lo;
}

__global__ __launch_bounds__(256, 6)
void nnz_chunk_pipelined_spmv_kernel(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_ind,
    const float* __restrict__ values,
    const float* __restrict__ x,
    float* __restrict__ y,
    const int* __restrict__ chunk_first_row,
    int nnz,
    int num_rows,
    int nnz_per_chunk) {

    const int chunk = (int)blockIdx.x;
    const int tid = (int)threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    const int64_t begin64 = (int64_t)chunk * (int64_t)nnz_per_chunk;
    if (begin64 >= (int64_t)nnz)
        return;

    const int chunk_begin = (int)begin64;
    const int chunk_end = (int)min((int64_t)nnz, begin64 + (int64_t)nnz_per_chunk);
    const int chunk_len = chunk_end - chunk_begin;

    const int seg_begin = chunk_begin + (chunk_len * warp) / 8;
    const int seg_end = chunk_begin + (chunk_len * (warp + 1)) / 8;

    if (seg_begin >= seg_end)
        return;

    int row_lo = __ldg(chunk_first_row + chunk);
    int row_hi = __ldg(chunk_first_row + chunk + 1) + 1;
    if (row_lo < 0)
        row_lo = 0;
    if (row_hi > num_rows)
        row_hi = num_rows;
    if (row_hi < row_lo)
        row_hi = row_lo;

    int row = find_row_for_nnz(row_ptr, seg_begin, row_lo, row_hi);

    while (row < row_hi && row < num_rows) {
        const int row_begin = __ldg(row_ptr + row);
        const int row_end = __ldg(row_ptr + row + 1);

        if (row_begin >= seg_end)
            break;

        const int lo = row_begin > seg_begin ? row_begin : seg_begin;
        const int hi = row_end < seg_end ? row_end : seg_end;

        float sum = 0.0f;

        for (int j = lo + lane; j < hi; j += 128) {
            const int j0 = j;
            const int j1 = j + 32;
            const int j2 = j + 64;
            const int j3 = j + 96;

            const bool v0 = j0 < hi;
            const bool v1 = j1 < hi;
            const bool v2 = j2 < hi;
            const bool v3 = j3 < hi;

            int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
            float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;

            if (v0) {
                c0 = __ldg(col_ind + j0);
                a0 = __ldg(values + j0);
            }
            if (v1) {
                c1 = __ldg(col_ind + j1);
                a1 = __ldg(values + j1);
            }
            if (v2) {
                c2 = __ldg(col_ind + j2);
                a2 = __ldg(values + j2);
            }
            if (v3) {
                c3 = __ldg(col_ind + j3);
                a3 = __ldg(values + j3);
            }

            float x0 = 0.0f, x1 = 0.0f, x2 = 0.0f, x3 = 0.0f;
            if (v0) x0 = ro_load(x + c0);
            if (v1) x1 = ro_load(x + c1);
            if (v2) x2 = ro_load(x + c2);
            if (v3) x3 = ro_load(x + c3);

            if (v0) sum = fmaf(a0, x0, sum);
            if (v1) sum = fmaf(a1, x1, sum);
            if (v2) sum = fmaf(a2, x2, sum);
            if (v3) sum = fmaf(a3, x3, sum);
        }

        sum = warp_reduce_sum(sum);

        if (lane == 0 && lo < hi) {
            if (lo == row_begin && hi == row_end)
                y[row] = sum;
            else
                atomicAdd(y + row, sum);
        }

        ++row;
    }
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
    ctx->num_chunks = 0;
    ctx->nnz_per_chunk = 0;
    ctx->num_rows = num_rows;
    ctx->nnz = nnz;
    ctx->chunks.d_chunk_first_row = nullptr;
    ctx->chunks.num_chunks = 0;
    ctx->chunks.nnz_per_chunk = 0;
    ctx->chunks.num_rows = num_rows;
    ctx->chunks.nnz = nnz;

    if (num_rows > 0 && nnz > 0) {
        ctx->chunks = balance_nnz_chunks(d_row_ptr, num_rows, nnz, 4096);
        ctx->num_chunks = ctx->chunks.num_chunks;
        ctx->nnz_per_chunk = ctx->chunks.nnz_per_chunk;
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
    if (ctx == nullptr ||
        num_rows <= 0 ||
        nnz <= 0 ||
        ctx->num_chunks <= 0 ||
        ctx->nnz_per_chunk <= 0 ||
        ctx->chunks.d_chunk_first_row == nullptr) {
        return;
    }

    nnz_chunk_pipelined_spmv_kernel<<<ctx->num_chunks, 256>>>(
        d_row_ptr,
        d_col_ind,
        d_values,
        d_x,
        d_y,
        ctx->chunks.d_chunk_first_row,
        nnz,
        num_rows,
        ctx->nnz_per_chunk);
}

void spmv_cleanup(void* opaque) {
    SpmvContext* ctx = static_cast<SpmvContext*>(opaque);
    if (ctx == nullptr)
        return;

    if (ctx->chunks.d_chunk_first_row != nullptr)
        free_nnz_chunks(ctx->chunks);

    delete ctx;
}