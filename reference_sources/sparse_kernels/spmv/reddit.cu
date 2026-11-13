#include <cuda_runtime.h>
#include "sparse_formats.cuh"

struct SpmvContext {
    NnzChunkData chunks;
    int num_chunks;
    int num_rows;
    int nnz;
    int nnz_per_chunk;
};

__device__ __forceinline__ float warp_reduce_sum(float v) {
    v += __shfl_down_sync(0xffffffff, v, 16);
    v += __shfl_down_sync(0xffffffff, v, 8);
    v += __shfl_down_sync(0xffffffff, v, 4);
    v += __shfl_down_sync(0xffffffff, v, 2);
    v += __shfl_down_sync(0xffffffff, v, 1);
    return v;
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
    int nnz_per_chunk) {

    const int tid = (int)threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int chunk = (int)blockIdx.x;

    const int chunk_begin = chunk * nnz_per_chunk;
    if (chunk_begin >= nnz) return;

    const int chunk_end = min(chunk_begin + nnz_per_chunk, nnz);
    const int seg_begin = min(chunk_begin + warp * 512, chunk_end);
    const int seg_end = min(seg_begin + 512, chunk_end);

    if (seg_begin >= seg_end) return;

    int row = chunk_first_row[chunk];
    if (row < 0) row = 0;
    if (row >= num_rows) row = num_rows - 1;

    while (row + 1 < num_rows && row_ptr[row + 1] <= seg_begin) {
        ++row;
    }

    while (row < num_rows) {
        const int row_begin = row_ptr[row];
        const int row_end = row_ptr[row + 1];

        if (row_begin >= seg_end) break;

        const int begin = row_begin > seg_begin ? row_begin : seg_begin;
        const int end = row_end < seg_end ? row_end : seg_end;

        float acc0 = 0.0f;
        float acc1 = 0.0f;
        float acc2 = 0.0f;
        float acc3 = 0.0f;

        int p = begin + lane;

        if (p + 96 < end) {
            int c0 = col_ind[p];
            int c1 = col_ind[p + 32];
            int c2 = col_ind[p + 64];
            int c3 = col_ind[p + 96];

            float v0 = values[p];
            float v1 = values[p + 32];
            float v2 = values[p + 64];
            float v3 = values[p + 96];

            float x0 = __ldg(x + c0);
            float x1 = __ldg(x + c1);
            float x2 = __ldg(x + c2);
            float x3 = __ldg(x + c3);

            for (;;) {
                const int next_p = p + 128;
                const bool have_next = next_p + 96 < end;

                if (!have_next) {
                    acc0 = fmaf(v0, x0, acc0);
                    acc1 = fmaf(v1, x1, acc1);
                    acc2 = fmaf(v2, x2, acc2);
                    acc3 = fmaf(v3, x3, acc3);
                    p = next_p;
                    break;
                }

                const int nc0 = col_ind[next_p];
                const int nc1 = col_ind[next_p + 32];
                const int nc2 = col_ind[next_p + 64];
                const int nc3 = col_ind[next_p + 96];

                const float nv0 = values[next_p];
                const float nv1 = values[next_p + 32];
                const float nv2 = values[next_p + 64];
                const float nv3 = values[next_p + 96];

                const float nx0 = __ldg(x + nc0);
                const float nx1 = __ldg(x + nc1);
                const float nx2 = __ldg(x + nc2);
                const float nx3 = __ldg(x + nc3);

                acc0 = fmaf(v0, x0, acc0);
                acc1 = fmaf(v1, x1, acc1);
                acc2 = fmaf(v2, x2, acc2);
                acc3 = fmaf(v3, x3, acc3);

                v0 = nv0;
                v1 = nv1;
                v2 = nv2;
                v3 = nv3;

                x0 = nx0;
                x1 = nx1;
                x2 = nx2;
                x3 = nx3;

                p = next_p;
            }
        }

        for (; p < end; p += 32) {
            const int col = col_ind[p];
            acc0 = fmaf(values[p], __ldg(x + col), acc0);
        }

        const float sum = warp_reduce_sum((acc0 + acc1) + (acc2 + acc3));

        if (lane == 0) {
            if (row_begin >= seg_begin && row_end <= seg_end) {
                y[row] = sum;
            } else {
                atomicAdd(y + row, sum);
            }
        }

        ++row;
    }
}

__global__ void zero_output_kernel(float* y, int n) {
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
    ctx->num_chunks = 0;
    ctx->num_rows = num_rows;
    ctx->nnz = nnz;
    ctx->nnz_per_chunk = 2048;

    if (num_rows <= 0 || nnz <= 0) return ctx;

    const int requested_chunks = (nnz + ctx->nnz_per_chunk - 1) / ctx->nnz_per_chunk;
    ctx->chunks = balance_nnz_chunks(
        d_row_ptr, num_rows, nnz, requested_chunks);

    ctx->num_chunks = requested_chunks;
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

    (void)num_cols;

    SpmvContext* ctx = static_cast<SpmvContext*>(opaque_ctx);
    if (ctx == nullptr || num_rows <= 0) return;

    if (nnz <= 0 || ctx->num_chunks <= 0) {
        constexpr int threads = 256;
        const int blocks = (num_rows + threads - 1) / threads;
        zero_output_kernel<<<blocks, threads>>>(d_y, num_rows);
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
        ctx->nnz_per_chunk);
}

void spmv_cleanup(void* opaque_ctx) {
    SpmvContext* ctx = static_cast<SpmvContext*>(opaque_ctx);
    if (ctx == nullptr) return;

    if (ctx->chunks.num_chunks > 0) {
        free_nnz_chunks(ctx->chunks);
    }

    delete ctx;
}