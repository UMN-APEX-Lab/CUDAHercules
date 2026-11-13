#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

struct SpmmContext {
    int* d_first_rows;
    int num_chunks;
    int chunk_nnz;
};

__device__ __forceinline__ void load_a_entry(
    int lane,
    int p,
    const int* __restrict__ col_ind,
    const float* __restrict__ values,
    int& col,
    float& val)
{
    int c = 0;
    float v = 0.0f;
    if (lane == 0) {
        c = __ldg(col_ind + p);
        v = __ldg(values + p);
    }
    col = __shfl_sync(0xffffffffu, c, 0);
    val = __shfl_sync(0xffffffffu, v, 0);
}

__global__ __launch_bounds__(128, 12)
void csr_nnz_chunk_spmm32_kernel(
    const int* __restrict__ first_rows,
    int num_chunks,
    int chunk_nnz,
    int nnz,
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_ind,
    const float* __restrict__ values,
    const float* __restrict__ B,
    float* __restrict__ C)
{
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int chunk = blockIdx.x * 4 + warp;

    if (chunk >= num_chunks) return;

    const int begin = chunk * chunk_nnz;
    const int chunk_end = min(begin + chunk_nnz, nnz);

    int row = 0;
    if (lane == 0) row = __ldg(first_rows + chunk);
    row = __shfl_sync(0xffffffffu, row, 0);

    int p = begin;

    while (p < chunk_end) {
        int row_end = 0;
        if (lane == 0) row_end = __ldg(row_ptr + row + 1);
        row_end = __shfl_sync(0xffffffffu, row_end, 0);

        if (row_end <= p) {
            ++row;
            continue;
        }

        const int end = row_end < chunk_end ? row_end : chunk_end;

        float sum0 = 0.0f;
        float sum1 = 0.0f;

        while (p + 3 < end) {
            int c0, c1, c2, c3;
            float a0, a1, a2, a3;

            load_a_entry(lane, p,     col_ind, values, c0, a0);
            load_a_entry(lane, p + 1, col_ind, values, c1, a1);
            load_a_entry(lane, p + 2, col_ind, values, c2, a2);
            load_a_entry(lane, p + 3, col_ind, values, c3, a3);

            const float b0 = __ldg(B + ((size_t)c0 << 5) + lane);
            const float b1 = __ldg(B + ((size_t)c1 << 5) + lane);
            const float b2 = __ldg(B + ((size_t)c2 << 5) + lane);
            const float b3 = __ldg(B + ((size_t)c3 << 5) + lane);

            sum0 = __fmaf_rn(a0, b0, sum0);
            sum1 = __fmaf_rn(a1, b1, sum1);
            sum0 = __fmaf_rn(a2, b2, sum0);
            sum1 = __fmaf_rn(a3, b3, sum1);

            p += 4;
        }

        while (p < end) {
            int col;
            float val;
            load_a_entry(lane, p, col_ind, values, col, val);
            const float b = __ldg(B + ((size_t)col << 5) + lane);
            sum0 = __fmaf_rn(val, b, sum0);
            ++p;
        }

        atomicAdd(C + ((size_t)row << 5) + lane, sum0 + sum1);

        if (p == row_end) ++row;
    }
}

void* spmm_preprocess(
    const int* d_row_ptr,
    const int*,
    const float*,
    int num_rows,
    int,
    int nnz,
    int)
{
    SpmmContext* ctx = new SpmmContext();
    ctx->d_first_rows = nullptr;
    ctx->num_chunks = 0;
    ctx->chunk_nnz = 512;

    if (num_rows <= 0 || nnz <= 0) return ctx;

    std::vector<int> h_row_ptr((size_t)num_rows + 1);
    cudaMemcpy(
        h_row_ptr.data(),
        d_row_ptr,
        ((size_t)num_rows + 1) * sizeof(int),
        cudaMemcpyDeviceToHost);

    const int num_chunks = (nnz + ctx->chunk_nnz - 1) / ctx->chunk_nnz;
    std::vector<int> h_first_rows((size_t)num_chunks);

    int row = 0;
    for (int chunk = 0; chunk < num_chunks; ++chunk) {
        const int begin = chunk * ctx->chunk_nnz;
        while (row + 1 < num_rows && h_row_ptr[row + 1] <= begin) {
            ++row;
        }
        h_first_rows[chunk] = row;
    }

    cudaMalloc(
        reinterpret_cast<void**>(&ctx->d_first_rows),
        (size_t)num_chunks * sizeof(int));

    cudaMemcpy(
        ctx->d_first_rows,
        h_first_rows.data(),
        (size_t)num_chunks * sizeof(int),
        cudaMemcpyHostToDevice);

    ctx->num_chunks = num_chunks;
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
    int,
    int nnz,
    int)
{
    if (num_rows <= 0) return;

    cudaMemsetAsync(
        d_C,
        0,
        (size_t)num_rows * 32 * sizeof(float),
        0);

    SpmmContext* ctx = static_cast<SpmmContext*>(context);
    if (ctx != nullptr && ctx->num_chunks > 0 && nnz > 0) {
        const int blocks = (ctx->num_chunks + 3) >> 2;
        csr_nnz_chunk_spmm32_kernel<<<blocks, 128>>>(
            ctx->d_first_rows,
            ctx->num_chunks,
            ctx->chunk_nnz,
            nnz,
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
    if (ctx == nullptr) return;

    if (ctx->d_first_rows != nullptr) {
        cudaFree(ctx->d_first_rows);
    }
    delete ctx;
}