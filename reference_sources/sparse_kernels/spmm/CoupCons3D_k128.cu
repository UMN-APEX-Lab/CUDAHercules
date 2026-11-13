#include <cuda_runtime.h>
#include <cstdlib>
#include "sparse_formats.cuh"

struct SpMMContext {
    SELLData sell;
    int* d_row_lengths;
    int num_rows;
    int num_slices;
};

__global__ void build_sell_row_lengths_kernel(
    const int* __restrict__ row_ptr,
    const int* __restrict__ row_perm,
    int* __restrict__ row_lengths,
    int num_rows)
{
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < num_rows) {
        int row = row_perm[i];
        row_lengths[i] = row_ptr[row + 1] - row_ptr[row];
    }
}

__global__ __launch_bounds__(256, 4)
void sell_spmm_128_full_slice_kernel(
    const int* __restrict__ slice_ptr,
    const int* __restrict__ slice_width,
    const int* __restrict__ row_perm,
    const int* __restrict__ row_lengths,
    const int* __restrict__ col_ind,
    const float* __restrict__ values,
    const float* __restrict__ B,
    float* __restrict__ C,
    int num_rows)
{
    constexpr int SELL_C = 32;
    constexpr int MICRO_NNZ = 4;

    __shared__ int sh_cols[2][SELL_C * MICRO_NNZ];
    __shared__ float sh_vals[2][SELL_C * MICRO_NNZ];
    __shared__ int sh_base;
    __shared__ int sh_width;

    const int tid = (int)threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int row_group = lane >> 3;
    const int subgroup_lane = lane & 7;

    const int slice = (int)blockIdx.x;
    const int local_row = warp * 4 + row_group;
    const int sell_row = slice * SELL_C + local_row;
    const bool active = sell_row < num_rows;

    if (tid == 0) {
        sh_base = __ldg(slice_ptr + slice);
        sh_width = __ldg(slice_width + slice);
    }
    __syncthreads();

    const int base = sh_base;
    const int width = sh_width;
    const int microtiles = (width + MICRO_NNZ - 1) >> 2;

    int out_row = 0;
    int row_len = 0;
    if (active && subgroup_lane == 0) {
        out_row = __ldg(row_perm + sell_row);
        row_len = __ldg(row_lengths + sell_row);
    }

    const unsigned subgroup_mask = 0xFFu << (row_group * 8);
    const int subgroup_source = row_group * 8;
    out_row = __shfl_sync(subgroup_mask, out_row, subgroup_source);
    row_len = __shfl_sync(subgroup_mask, row_len, subgroup_source);

    float acc[16];
    #pragma unroll
    for (int t = 0; t < 16; ++t) {
        acc[t] = 0.0f;
    }

    if (microtiles > 0) {
        if (tid < SELL_C * MICRO_NNZ) {
            const int j = tid >> 5;
            const int r = tid & 31;
            if (j < width) {
                const int pos = base + j * SELL_C + r;
                sh_cols[0][tid] = __ldg(col_ind + pos);
                sh_vals[0][tid] = __ldg(values + pos);
            } else {
                sh_cols[0][tid] = 0;
                sh_vals[0][tid] = 0.0f;
            }
        }
        __syncthreads();

        for (int mt = 0; mt < microtiles; ++mt) {
            const int current = mt & 1;
            const int next = current ^ 1;
            const int jbase = mt << 2;

            if (active) {
                #pragma unroll
                for (int jj = 0; jj < MICRO_NNZ; ++jj) {
                    const int j = jbase + jj;
                    if (j < row_len) {
                        const int spos = jj * SELL_C + local_row;
                        const int col = sh_cols[current][spos];
                        const float a = sh_vals[current][spos];
                        const float* b = B + ((size_t)col << 7);

                        float bv[16];
                        #pragma unroll
                        for (int t = 0; t < 16; ++t) {
                            bv[t] = __ldg(b + subgroup_lane + (t << 3));
                        }

                        #pragma unroll
                        for (int t = 0; t < 16; ++t) {
                            acc[t] = fmaf(a, bv[t], acc[t]);
                        }
                    }
                }
            }

            if (mt + 1 < microtiles) {
                const int next_jbase = (mt + 1) << 2;
                if (tid < SELL_C * MICRO_NNZ) {
                    const int j = tid >> 5;
                    const int r = tid & 31;
                    const int pos_j = next_jbase + j;
                    if (pos_j < width) {
                        const int pos = base + pos_j * SELL_C + r;
                        sh_cols[next][tid] = __ldg(col_ind + pos);
                        sh_vals[next][tid] = __ldg(values + pos);
                    } else {
                        sh_cols[next][tid] = 0;
                        sh_vals[next][tid] = 0.0f;
                    }
                }
                __syncthreads();
            }
        }
    }

    if (active) {
        float* c = C + ((size_t)out_row << 7);
        #pragma unroll
        for (int t = 0; t < 16; ++t) {
            c[subgroup_lane + (t << 3)] = acc[t];
        }
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
    (void)K;

    SpMMContext* ctx = new SpMMContext();
    ctx->num_rows = num_rows;
    ctx->num_slices = (num_rows + 31) >> 5;
    ctx->d_row_lengths = nullptr;

    ctx->sell = csr_to_sell(
        d_row_ptr,
        d_col_ind,
        d_values,
        num_rows,
        num_cols,
        nnz,
        32,
        32);

    if (num_rows > 0) {
        cudaMalloc(&ctx->d_row_lengths, sizeof(int) * (size_t)num_rows);
        constexpr int threads = 256;
        const int blocks = (num_rows + threads - 1) / threads;
        build_sell_row_lengths_kernel<<<blocks, threads>>>(
            d_row_ptr,
            ctx->sell.d_row_perm,
            ctx->d_row_lengths,
            num_rows);
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
    (void)num_cols;
    (void)nnz;
    (void)K;

    SpMMContext* ctx = static_cast<SpMMContext*>(context);
    if (ctx == nullptr || num_rows <= 0 || ctx->num_slices <= 0) {
        return;
    }

    sell_spmm_128_full_slice_kernel<<<ctx->num_slices, 256>>>(
        ctx->sell.d_slice_ptr,
        ctx->sell.d_slice_width,
        ctx->sell.d_row_perm,
        ctx->d_row_lengths,
        ctx->sell.d_col_ind,
        ctx->sell.d_values,
        d_B,
        d_C,
        num_rows);
}

void spmm_cleanup(void* context)
{
    SpMMContext* ctx = static_cast<SpMMContext*>(context);
    if (ctx == nullptr) {
        return;
    }

    if (ctx->d_row_lengths != nullptr) {
        cudaFree(ctx->d_row_lengths);
    }

    free_sell(ctx->sell);
    delete ctx;
}