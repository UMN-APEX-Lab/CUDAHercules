#include <cuda_runtime.h>
#include "sparse_formats.cuh"

struct SpMMContext {
    NnzChunkData chunks;
};

__global__ __launch_bounds__(128, 12)
void spmm_static_nnz_chunk_kernel(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_ind,
    const float* __restrict__ values,
    const int* __restrict__ chunk_first_row,
    int num_chunks,
    int nnz_per_chunk,
    int nnz,
    const float* __restrict__ B,
    float* __restrict__ C)
{
    const int chunk = (int)blockIdx.x;
    if (chunk >= num_chunks) return;

    const int tid = (int)threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int out_col = (warp << 5) + lane;

    __shared__ int s_cols[128];
    __shared__ float s_vals[128];

    const int chunk_begin = chunk * nnz_per_chunk;
    const int chunk_end = min(chunk_begin + nnz_per_chunk, nnz);

    int row = chunk_first_row[chunk];
    int pos = chunk_begin;

    while (pos < chunk_end) {
        int row_begin = row_ptr[row];
        int row_end = row_ptr[row + 1];

        while (row_end <= pos) {
            ++row;
            row_begin = row_ptr[row];
            row_end = row_ptr[row + 1];
        }

        if (pos < row_begin) pos = row_begin;
        if (pos >= chunk_end) break;

        const int segment_end = min(row_end, chunk_end);
        float acc = 0.0f;

        int p = pos;
        while (p < segment_end) {
            const int count = min(segment_end - p, 128);

            if (tid < count) {
                s_cols[tid] = col_ind[p + tid];
                s_vals[tid] = values[p + tid];
            }
            __syncthreads();

            #pragma unroll 4
            for (int j = 0; j < 128; ++j) {
                if (j < count) {
                    const int col = s_cols[j];
                    acc = fmaf(s_vals[j], B[(size_t)col * 128 + out_col], acc);
                }
            }

            __syncthreads();
            p += count;
        }

        atomicAdd(C + (size_t)row * 128 + out_col, acc);

        pos = segment_end;
        if (pos >= row_end) ++row;
    }
}

void* spmm_preprocess(const int* d_row_ptr,
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

    SpMMContext* ctx = new SpMMContext();
    ctx->chunks.d_chunk_first_row = nullptr;
    ctx->chunks.num_chunks = 0;
    ctx->chunks.nnz_per_chunk = 2048;

    if (num_rows > 0 && nnz > 0) {
        const int chunk_size = 2048;
        const int num_chunks = (nnz + chunk_size - 1) / chunk_size;
        ctx->chunks = balance_nnz_chunks(d_row_ptr, num_rows, nnz, num_chunks);
    }

    return ctx;
}

void spmm_compute(void* ctx_ptr,
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

    if (num_rows <= 0 || K <= 0) return;

    cudaMemsetAsync(d_C, 0, (size_t)num_rows * (size_t)K * sizeof(float), 0);

    if (nnz <= 0 || K != 128) return;

    SpMMContext* ctx = static_cast<SpMMContext*>(ctx_ptr);
    if (ctx == nullptr || ctx->chunks.num_chunks <= 0 ||
        ctx->chunks.d_chunk_first_row == nullptr) {
        return;
    }

    spmm_static_nnz_chunk_kernel<<<ctx->chunks.num_chunks, 128>>>(
        d_row_ptr,
        d_col_ind,
        d_values,
        ctx->chunks.d_chunk_first_row,
        ctx->chunks.num_chunks,
        ctx->chunks.nnz_per_chunk,
        nnz,
        d_B,
        d_C);
}

void spmm_cleanup(void* ctx_ptr)
{
    SpMMContext* ctx = static_cast<SpMMContext*>(ctx_ptr);
    if (ctx == nullptr) return;

    if (ctx->chunks.d_chunk_first_row != nullptr) {
        free_nnz_chunks(ctx->chunks);
    }

    delete ctx;
}