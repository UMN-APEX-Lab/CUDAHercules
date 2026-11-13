#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include "sparse_formats.cuh"

struct SpmmContext {
    COOData coo;
    int nnz;
    int num_chunks;
    int* d_zero_rows;
    int zero_count;
};

__global__ void collect_empty_rows_kernel(
    const int* __restrict__ row_ptr,
    int num_rows,
    int* __restrict__ rows,
    int* __restrict__ count)
{
    int row = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (row < num_rows && row_ptr[row] == row_ptr[row + 1]) {
        int pos = atomicAdd(count, 1);
        rows[pos] = row;
    }
}

__global__ void gather_boundary_rows_kernel(
    const int* __restrict__ rows,
    int nnz,
    int num_chunks,
    int* __restrict__ boundary_rows)
{
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int total = num_chunks << 1;
    if (i < total) {
        int chunk = i >> 1;
        int p = (i & 1) ? min((chunk + 1) << 8, nnz) - 1 : (chunk << 8);
        boundary_rows[i] = (p >= 0 && p < nnz) ? rows[p] : -1;
    }
}

__global__ void zero_selected_rows_kernel(
    const int* __restrict__ rows,
    int count,
    float* __restrict__ C)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)count * 256LL;
    long long stride = (long long)blockDim.x * gridDim.x;

    for (; idx < total; idx += stride) {
        int row = rows[(int)(idx >> 8)];
        C[(long long)row * 256LL + (idx & 255LL)] = 0.0f;
    }
}

__device__ __forceinline__ void flush8(
    float* __restrict__ C,
    int row,
    int lane,
    bool atomic_mode,
    float s0, float s1, float s2, float s3,
    float s4, float s5, float s6, float s7)
{
    float* out = C + (long long)row * 256LL + lane;
    if (atomic_mode) {
        atomicAdd(out,       s0);
        atomicAdd(out + 32,  s1);
        atomicAdd(out + 64,  s2);
        atomicAdd(out + 96,  s3);
        atomicAdd(out + 128, s4);
        atomicAdd(out + 160, s5);
        atomicAdd(out + 192, s6);
        atomicAdd(out + 224, s7);
    } else {
        out[0]   = s0;
        out[32]  = s1;
        out[64]  = s2;
        out[96]  = s3;
        out[128] = s4;
        out[160] = s5;
        out[192] = s6;
        out[224] = s7;
    }
}

__global__ __launch_bounds__(128, 8)
void coo_wide_chunk_kernel(
    const int* __restrict__ rows,
    const int* __restrict__ cols,
    const float* __restrict__ vals,
    const float* __restrict__ B,
    float* __restrict__ C,
    int nnz,
    int num_chunks)
{
    int tid = (int)threadIdx.x;
    int warp = tid >> 5;
    int lane = tid & 31;
    int chunk = (int)blockIdx.x * 4 + warp;

    if (chunk >= num_chunks) return;

    int begin = chunk << 8;
    int end = min(begin + 256, nnz);
    if (begin >= end) return;

    int row0 = 0;
    int prev_row = -1;
    if (lane == 0) {
        row0 = rows[begin];
        if (begin > 0) prev_row = rows[begin - 1];
    }
    row0 = __shfl_sync(0xffffffff, row0, 0);
    prev_row = __shfl_sync(0xffffffff, prev_row, 0);

    int meta_row = 0;
    int meta_col = 0;
    float meta_val = 0.0f;
    if (lane == 0) {
        meta_row = rows[begin];
        meta_col = cols[begin];
        meta_val = vals[begin];
    }
    meta_row = __shfl_sync(0xffffffff, meta_row, 0);
    meta_col = __shfl_sync(0xffffffff, meta_col, 0);
    meta_val = __shfl_sync(0xffffffff, meta_val, 0);

    int current_row = row0;
    bool partial = (begin > 0 && prev_row == row0);

    float s0 = 0.0f;
    float s1 = 0.0f;
    float s2 = 0.0f;
    float s3 = 0.0f;
    float s4 = 0.0f;
    float s5 = 0.0f;
    float s6 = 0.0f;
    float s7 = 0.0f;

    for (int p = begin; p < end; ++p) {
        int row = meta_row;
        int col = meta_col;
        float av = meta_val;

        if (row != current_row) {
            flush8(C, current_row, lane, partial,
                   s0, s1, s2, s3, s4, s5, s6, s7);
            current_row = row;
            partial = false;
            s0 = 0.0f;
            s1 = 0.0f;
            s2 = 0.0f;
            s3 = 0.0f;
            s4 = 0.0f;
            s5 = 0.0f;
            s6 = 0.0f;
            s7 = 0.0f;
        }

        int next_row = 0;
        int next_col = 0;
        float next_val = 0.0f;
        if (p + 1 < end && lane == 0) {
            next_row = rows[p + 1];
            next_col = cols[p + 1];
            next_val = vals[p + 1];
        }

        const float* bp = B + (long long)col * 256LL + lane;

        float b0 = __ldg(bp);
        float b1 = __ldg(bp + 32);
        float b2 = __ldg(bp + 64);
        float b3 = __ldg(bp + 96);
        float b4 = __ldg(bp + 128);
        float b5 = __ldg(bp + 160);
        float b6 = __ldg(bp + 192);
        float b7 = __ldg(bp + 224);

        s0 = fmaf(av, b0, s0);
        s1 = fmaf(av, b1, s1);
        s2 = fmaf(av, b2, s2);
        s3 = fmaf(av, b3, s3);
        s4 = fmaf(av, b4, s4);
        s5 = fmaf(av, b5, s5);
        s6 = fmaf(av, b6, s6);
        s7 = fmaf(av, b7, s7);

        if (p + 1 < end) {
            meta_row = __shfl_sync(0xffffffff, next_row, 0);
            meta_col = __shfl_sync(0xffffffff, next_col, 0);
            meta_val = __shfl_sync(0xffffffff, next_val, 0);
        }
    }

    int following_row = -1;
    if (lane == 0 && end < nnz) following_row = rows[end];
    following_row = __shfl_sync(0xffffffff, following_row, 0);

    bool final_partial = partial || (end < nnz && following_row == current_row);
    flush8(C, current_row, lane, final_partial,
           s0, s1, s2, s3, s4, s5, s6, s7);
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
    SpmmContext* ctx = new SpmmContext();
    ctx->coo = csr_to_coo(d_row_ptr, d_col_ind, d_values,
                          num_rows, num_cols, nnz);
    ctx->nnz = nnz;
    ctx->num_chunks = (nnz + 255) >> 8;
    ctx->d_zero_rows = nullptr;
    ctx->zero_count = 0;

    std::vector<int> host_rows;

    if (num_rows > 0) {
        int* d_empty_rows = nullptr;
        int* d_count = nullptr;
        cudaMalloc(&d_empty_rows, (size_t)num_rows * sizeof(int));
        cudaMalloc(&d_count, sizeof(int));
        cudaMemset(d_count, 0, sizeof(int));

        int blocks = (num_rows + 255) >> 8;
        collect_empty_rows_kernel<<<blocks, 256>>>(
            d_row_ptr, num_rows, d_empty_rows, d_count);

        int empty_count = 0;
        cudaMemcpy(&empty_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);

        if (empty_count > 0) {
            host_rows.resize((size_t)empty_count);
            cudaMemcpy(host_rows.data(), d_empty_rows,
                       (size_t)empty_count * sizeof(int),
                       cudaMemcpyDeviceToHost);
        }

        cudaFree(d_count);
        cudaFree(d_empty_rows);
    }

    if (ctx->num_chunks > 0) {
        int boundary_count = ctx->num_chunks << 1;
        int* d_boundary_rows = nullptr;
        cudaMalloc(&d_boundary_rows, (size_t)boundary_count * sizeof(int));

        int blocks = (boundary_count + 255) >> 8;
        gather_boundary_rows_kernel<<<blocks, 256>>>(
            ctx->coo.d_row_ind, nnz, ctx->num_chunks, d_boundary_rows);

        size_t old_size = host_rows.size();
        host_rows.resize(old_size + (size_t)boundary_count);
        cudaMemcpy(host_rows.data() + old_size, d_boundary_rows,
                   (size_t)boundary_count * sizeof(int),
                   cudaMemcpyDeviceToHost);
        cudaFree(d_boundary_rows);
    }

    host_rows.erase(
        std::remove_if(host_rows.begin(), host_rows.end(),
                       [](int r) { return r < 0; }),
        host_rows.end());
    std::sort(host_rows.begin(), host_rows.end());
    host_rows.erase(std::unique(host_rows.begin(), host_rows.end()),
                    host_rows.end());

    ctx->zero_count = (int)host_rows.size();
    if (ctx->zero_count > 0) {
        cudaMalloc(&ctx->d_zero_rows,
                   (size_t)ctx->zero_count * sizeof(int));
        cudaMemcpy(ctx->d_zero_rows, host_rows.data(),
                   (size_t)ctx->zero_count * sizeof(int),
                   cudaMemcpyHostToDevice);
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
    (void)d_row_ptr;
    (void)d_col_ind;
    (void)d_values;
    (void)num_rows;
    (void)num_cols;
    (void)K;

    SpmmContext* ctx = static_cast<SpmmContext*>(context);
    if (!ctx) return;

    if (ctx->zero_count > 0) {
        long long total = (long long)ctx->zero_count * 256LL;
        int blocks = (int)((total + 255) >> 8);
        if (blocks > 65535) blocks = 65535;
        zero_selected_rows_kernel<<<blocks, 256>>>(
            ctx->d_zero_rows, ctx->zero_count, d_C);
    }

    if (nnz <= 0 || ctx->num_chunks <= 0) return;

    int blocks = (ctx->num_chunks + 3) >> 2;
    coo_wide_chunk_kernel<<<blocks, 128>>>(
        ctx->coo.d_row_ind,
        ctx->coo.d_col_ind,
        ctx->coo.d_values,
        d_B,
        d_C,
        nnz,
        ctx->num_chunks);
}

void spmm_cleanup(void* context)
{
    SpmmContext* ctx = static_cast<SpmmContext*>(context);
    if (!ctx) return;
    if (ctx->d_zero_rows) cudaFree(ctx->d_zero_rows);
    free_coo(ctx->coo);
    delete ctx;
}