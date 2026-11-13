#include <cuda_runtime.h>
#include <stdint.h>
#include <vector>
#include "sparse_formats.cuh"

struct SpmmContext {
    WindowedData windowed;
    int* d_light_rows;
    int light_count;
    int* d_chunk_rows;
    int* d_chunk_begin;
    int* d_chunk_end;
    int chunk_count;
    int num_rows;
    int nnz;
};

__global__ __launch_bounds__(256, 4)
void light_rows_spmm_kernel(
    const int* __restrict__ row_ids,
    int count,
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_ind,
    const float* __restrict__ values,
    const float* __restrict__ B,
    float* __restrict__ C)
{
    const int tid = (int)threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int idx = (int)blockIdx.x * 8 + warp;

    if (idx >= count) return;

    const int row = row_ids[idx];
    const int begin = row_ptr[row];
    const int end = row_ptr[row + 1];

    float x0 = 0.0f;
    float x1 = 0.0f;
    float x2 = 0.0f;
    float x3 = 0.0f;

    for (int p = begin; p < end; ++p) {
        const int col = __ldg(col_ind + p);
        const float a = __ldg(values + p);
        const float* bp = B + ((size_t)col << 7) + lane;

        x0 = fmaf(a, __ldg(bp), x0);
        x1 = fmaf(a, __ldg(bp + 32), x1);
        x2 = fmaf(a, __ldg(bp + 64), x2);
        x3 = fmaf(a, __ldg(bp + 96), x3);
    }

    float* cp = C + ((size_t)row << 7) + lane;
    cp[0] = x0;
    cp[32] = x1;
    cp[64] = x2;
    cp[96] = x3;
}

__global__ __launch_bounds__(128, 6)
void compact_dense_chunks_spmm_kernel(
    const int* __restrict__ chunk_rows,
    const int* __restrict__ chunk_begin,
    const int* __restrict__ chunk_end,
    int chunk_count,
    const int* __restrict__ col_ind,
    const float* __restrict__ values,
    const float* __restrict__ B,
    float* __restrict__ C)
{
    const int task = (int)blockIdx.x;
    if (task >= chunk_count) return;

    const int lane = (int)threadIdx.x;
    const int row = chunk_rows[task];
    const int begin = chunk_begin[task];
    const int end = chunk_end[task];

    float s = 0.0f;

    for (int p = begin; p < end; ++p) {
        const int col = __ldg(col_ind + p);
        const float a = __ldg(values + p);
        s = fmaf(a, __ldg(B + ((size_t)col << 7) + lane), s);
    }

    atomicAdd(C + ((size_t)row << 7) + lane, s);
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
    (void)K;

    SpmmContext* ctx = new SpmmContext;
    ctx->d_light_rows = nullptr;
    ctx->d_chunk_rows = nullptr;
    ctx->d_chunk_begin = nullptr;
    ctx->d_chunk_end = nullptr;
    ctx->light_count = 0;
    ctx->chunk_count = 0;
    ctx->num_rows = num_rows;
    ctx->nnz = nnz;

    ctx->windowed = csr_to_windowed(
        d_row_ptr, d_col_ind, d_values,
        num_rows, num_cols, nnz, 256, 4);

    if (num_rows <= 0)
        return ctx;

    std::vector<int> h_row_ptr((size_t)num_rows + 1);
    cudaMemcpy(h_row_ptr.data(), d_row_ptr,
               ((size_t)num_rows + 1) * sizeof(int),
               cudaMemcpyDeviceToHost);

    std::vector<int> h_light_rows;
    std::vector<int> h_chunk_rows;
    std::vector<int> h_chunk_begin;
    std::vector<int> h_chunk_end;

    h_light_rows.reserve(num_rows);

    const int chunk_nnz = 256;
    for (int r = 0; r < num_rows; ++r) {
        const int begin = h_row_ptr[r];
        const int end = h_row_ptr[r + 1];
        const int n = end - begin;

        if (n < chunk_nnz) {
            h_light_rows.push_back(r);
        } else {
            for (int p = begin; p < end; p += chunk_nnz) {
                int q = p + chunk_nnz;
                if (q > end) q = end;
                h_chunk_rows.push_back(r);
                h_chunk_begin.push_back(p);
                h_chunk_end.push_back(q);
            }
        }
    }

    ctx->light_count = (int)h_light_rows.size();
    ctx->chunk_count = (int)h_chunk_rows.size();

    if (ctx->light_count > 0) {
        cudaMalloc(&ctx->d_light_rows,
                   (size_t)ctx->light_count * sizeof(int));
        cudaMemcpy(ctx->d_light_rows, h_light_rows.data(),
                   (size_t)ctx->light_count * sizeof(int),
                   cudaMemcpyHostToDevice);
    }

    if (ctx->chunk_count > 0) {
        const size_t bytes = (size_t)ctx->chunk_count * sizeof(int);
        cudaMalloc(&ctx->d_chunk_rows, bytes);
        cudaMalloc(&ctx->d_chunk_begin, bytes);
        cudaMalloc(&ctx->d_chunk_end, bytes);

        cudaMemcpy(ctx->d_chunk_rows, h_chunk_rows.data(),
                   bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(ctx->d_chunk_begin, h_chunk_begin.data(),
                   bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(ctx->d_chunk_end, h_chunk_end.data(),
                   bytes, cudaMemcpyHostToDevice);
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
    (void)nnz;
    (void)K;

    SpmmContext* ctx = static_cast<SpmmContext*>(context);
    if (!ctx || num_rows <= 0)
        return;

    if (ctx->nnz == 0) {
        cudaMemsetAsync(d_C, 0, (size_t)num_rows * 128 * sizeof(float), 0);
        return;
    }

    if (ctx->chunk_count > 0) {
        cudaMemsetAsync(d_C, 0, (size_t)num_rows * 128 * sizeof(float), 0);

        if (ctx->light_count > 0) {
            const int blocks = (ctx->light_count + 7) >> 3;
            light_rows_spmm_kernel<<<blocks, 256>>>(
                ctx->d_light_rows,
                ctx->light_count,
                d_row_ptr,
                d_col_ind,
                d_values,
                d_B,
                d_C);
        }

        compact_dense_chunks_spmm_kernel<<<ctx->chunk_count, 128>>>(
            ctx->d_chunk_rows,
            ctx->d_chunk_begin,
            ctx->d_chunk_end,
            ctx->chunk_count,
            d_col_ind,
            d_values,
            d_B,
            d_C);
    } else if (ctx->light_count > 0) {
        const int blocks = (ctx->light_count + 7) >> 3;
        light_rows_spmm_kernel<<<blocks, 256>>>(
            ctx->d_light_rows,
            ctx->light_count,
            d_row_ptr,
            d_col_ind,
            d_values,
            d_B,
            d_C);
    }
}

void spmm_cleanup(void* context)
{
    SpmmContext* ctx = static_cast<SpmmContext*>(context);
    if (!ctx)
        return;

    if (ctx->d_light_rows)
        cudaFree(ctx->d_light_rows);
    if (ctx->d_chunk_rows)
        cudaFree(ctx->d_chunk_rows);
    if (ctx->d_chunk_begin)
        cudaFree(ctx->d_chunk_begin);
    if (ctx->d_chunk_end)
        cudaFree(ctx->d_chunk_end);

    free_windowed(ctx->windowed);
    delete ctx;
}