#include <cuda_runtime.h>
#include <cstdint>
#include "sparse_formats.cuh"

struct SpMMContext {
    ELLData ell;
    int num_rows;
    int num_cols;
    int width;
};

__device__ __forceinline__
void load_b8(const float* __restrict__ B, int col, int num_cols, int k0,
             float4& b0, float4& b1)
{
    if ((unsigned)col < (unsigned)num_cols) {
        const float* p = B + (size_t)col * 256ull + (size_t)k0;
        b0 = *reinterpret_cast<const float4*>(p);
        b1 = *reinterpret_cast<const float4*>(p + 4);
    } else {
        b0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        b1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
}

__global__ __launch_bounds__(128, 12)
void ell_spmm_k256_contiguous8_shared35_prefetch(
    const int* __restrict__ ell_col,
    const float* __restrict__ ell_val,
    const float* __restrict__ B,
    float* __restrict__ C,
    int num_rows,
    int num_cols)
{
    __shared__ int s_col[4][35];
    __shared__ float s_val[4][35];

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int base_row = (int)blockIdx.x * 4;
    const int row = base_row + warp;

    for (int item = tid; item < 140; item += 128) {
        const int slot = item >> 2;
        const int local_row = item & 3;
        const int global_row = base_row + local_row;

        int col = -1;
        float val = 0.0f;

        if (global_row < num_rows) {
            const size_t pos = (size_t)slot * (size_t)num_rows + (size_t)global_row;
            col = __ldg(ell_col + pos);
            val = __ldg(ell_val + pos);
        }

        s_col[local_row][slot] = col;
        s_val[local_row][slot] = val;
    }

    __syncthreads();

    if (row >= num_rows) return;

    const int k0 = lane << 3;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    float acc4 = 0.0f;
    float acc5 = 0.0f;
    float acc6 = 0.0f;
    float acc7 = 0.0f;

    int cur_col = s_col[warp][0];
    float cur_a = s_val[warp][0];

    float4 cur_b0;
    float4 cur_b1;
    load_b8(B, cur_col, num_cols, k0, cur_b0, cur_b1);

    #pragma unroll 1
    for (int slot = 0; slot < 34; ++slot) {
        const int next_col = s_col[warp][slot + 1];
        const float next_a = s_val[warp][slot + 1];

        float4 next_b0;
        float4 next_b1;
        load_b8(B, next_col, num_cols, k0, next_b0, next_b1);

        acc0 = fmaf(cur_a, cur_b0.x, acc0);
        acc1 = fmaf(cur_a, cur_b0.y, acc1);
        acc2 = fmaf(cur_a, cur_b0.z, acc2);
        acc3 = fmaf(cur_a, cur_b0.w, acc3);
        acc4 = fmaf(cur_a, cur_b1.x, acc4);
        acc5 = fmaf(cur_a, cur_b1.y, acc5);
        acc6 = fmaf(cur_a, cur_b1.z, acc6);
        acc7 = fmaf(cur_a, cur_b1.w, acc7);

        cur_col = next_col;
        cur_a = next_a;
        cur_b0 = next_b0;
        cur_b1 = next_b1;
    }

    acc0 = fmaf(cur_a, cur_b0.x, acc0);
    acc1 = fmaf(cur_a, cur_b0.y, acc1);
    acc2 = fmaf(cur_a, cur_b0.z, acc2);
    acc3 = fmaf(cur_a, cur_b0.w, acc3);
    acc4 = fmaf(cur_a, cur_b1.x, acc4);
    acc5 = fmaf(cur_a, cur_b1.y, acc5);
    acc6 = fmaf(cur_a, cur_b1.z, acc6);
    acc7 = fmaf(cur_a, cur_b1.w, acc7);

    float* cp = C + (size_t)row * 256ull + (size_t)k0;
    *reinterpret_cast<float4*>(cp) = make_float4(acc0, acc1, acc2, acc3);
    *reinterpret_cast<float4*>(cp + 4) = make_float4(acc4, acc5, acc6, acc7);
}

__global__ __launch_bounds__(128, 12)
void ell_spmm_k256_generic(
    const int* __restrict__ ell_col,
    const float* __restrict__ ell_val,
    const float* __restrict__ B,
    float* __restrict__ C,
    int num_rows,
    int num_cols,
    int width)
{
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int row = (int)blockIdx.x * 4 + warp;

    if (row >= num_rows) return;

    const int k0 = lane << 3;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    float acc4 = 0.0f;
    float acc5 = 0.0f;
    float acc6 = 0.0f;
    float acc7 = 0.0f;

    #pragma unroll 1
    for (int slot = 0; slot < width; ++slot) {
        int col = -1;
        float val = 0.0f;

        if (lane == 0) {
            const size_t pos = (size_t)slot * (size_t)num_rows + (size_t)row;
            col = __ldg(ell_col + pos);
            val = __ldg(ell_val + pos);
        }

        col = __shfl_sync(0xffffffffu, col, 0);
        val = __shfl_sync(0xffffffffu, val, 0);

        if ((unsigned)col < (unsigned)num_cols) {
            const float* bp = B + (size_t)col * 256ull + (size_t)k0;
            const float4 b0 = *reinterpret_cast<const float4*>(bp);
            const float4 b1 = *reinterpret_cast<const float4*>(bp + 4);

            acc0 = fmaf(val, b0.x, acc0);
            acc1 = fmaf(val, b0.y, acc1);
            acc2 = fmaf(val, b0.z, acc2);
            acc3 = fmaf(val, b0.w, acc3);
            acc4 = fmaf(val, b1.x, acc4);
            acc5 = fmaf(val, b1.y, acc5);
            acc6 = fmaf(val, b1.z, acc6);
            acc7 = fmaf(val, b1.w, acc7);
        }
    }

    float* cp = C + (size_t)row * 256ull + (size_t)k0;
    *reinterpret_cast<float4*>(cp) = make_float4(acc0, acc1, acc2, acc3);
    *reinterpret_cast<float4*>(cp + 4) = make_float4(acc4, acc5, acc6, acc7);
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

    SpMMContext* ctx = new SpMMContext;
    ctx->ell = csr_to_ell(
        d_row_ptr, d_col_ind, d_values,
        num_rows, num_cols, nnz);
    ctx->num_rows = num_rows;
    ctx->num_cols = num_cols;
    ctx->width = ctx->ell.max_nnz_per_row;
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
    (void)nnz;
    (void)K;

    SpMMContext* ctx = static_cast<SpMMContext*>(context);
    if (ctx == nullptr || num_rows <= 0) return;

    const dim3 block(128);
    const dim3 grid((unsigned)((num_rows + 3) >> 2));

    if (ctx->width == 35) {
        ell_spmm_k256_contiguous8_shared35_prefetch<<<grid, block>>>(
            ctx->ell.d_col_ind,
            ctx->ell.d_values,
            d_B,
            d_C,
            num_rows,
            num_cols);
    } else {
        ell_spmm_k256_generic<<<grid, block>>>(
            ctx->ell.d_col_ind,
            ctx->ell.d_values,
            d_B,
            d_C,
            num_rows,
            num_cols,
            ctx->width);
    }
}

void spmm_cleanup(void* context)
{
    SpMMContext* ctx = static_cast<SpMMContext*>(context);
    if (ctx != nullptr) {
        free_ell(ctx->ell);
        delete ctx;
    }
}