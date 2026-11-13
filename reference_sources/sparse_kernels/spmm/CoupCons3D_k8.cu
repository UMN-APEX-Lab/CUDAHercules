#include <cuda_runtime.h>
#include <stdint.h>
#include "sparse_formats.cuh"

struct SpMMContext {
    BSRData bsr;
    int num_rows;
    int num_cols;
};

__global__ __launch_bounds__(128, 12)
void bsr4_spmm_k8_aligned_kernel(
    const int* __restrict__ bsr_row_ptr,
    const int* __restrict__ bsr_col_ind,
    const float* __restrict__ bsr_values,
    const float* __restrict__ B,
    float* __restrict__ C)
{
    const int warp = (int)threadIdx.x >> 5;
    const int lane = (int)threadIdx.x & 31;
    const int br = ((int)blockIdx.x << 2) + warp;
    const int local_row = lane >> 3;
    const int k = lane & 7;
    const int row = (br << 2) + local_row;

    const int begin = __ldg(bsr_row_ptr + br);
    const int end = __ldg(bsr_row_ptr + br + 1);

    float sum = 0.0f;

    for (int p = begin; p < end; ++p) {
        const int bc = __ldg(bsr_col_ind + p);
        const float* av = bsr_values + ((size_t)p << 4) + (local_row << 2);
        const float* bp = B + ((size_t)bc << 5) + k;

        const float a0 = __ldg(av);
        const float a1 = __ldg(av + 1);
        const float a2 = __ldg(av + 2);
        const float a3 = __ldg(av + 3);

        const float b0 = __ldg(bp);
        const float b1 = __ldg(bp + 8);
        const float b2 = __ldg(bp + 16);
        const float b3 = __ldg(bp + 24);

        sum = fmaf(a0, b0, sum);
        sum = fmaf(a1, b1, sum);
        sum = fmaf(a2, b2, sum);
        sum = fmaf(a3, b3, sum);
    }

    C[((size_t)row << 3) + k] = sum;
}

__global__ __launch_bounds__(128, 12)
void bsr4_spmm_k8_tail_kernel(
    const int* __restrict__ bsr_row_ptr,
    const int* __restrict__ bsr_col_ind,
    const float* __restrict__ bsr_values,
    const float* __restrict__ B,
    float* __restrict__ C,
    int num_rows,
    int mb)
{
    const int warp = (int)threadIdx.x >> 5;
    const int lane = (int)threadIdx.x & 31;
    const int br = ((int)blockIdx.x << 2) + warp;

    if (br >= mb) return;

    const int local_row = lane >> 3;
    const int k = lane & 7;
    const int row = (br << 2) + local_row;

    const int begin = __ldg(bsr_row_ptr + br);
    const int end = __ldg(bsr_row_ptr + br + 1);

    float sum = 0.0f;

    for (int p = begin; p < end; ++p) {
        const int bc = __ldg(bsr_col_ind + p);
        const float* av = bsr_values + ((size_t)p << 4) + (local_row << 2);
        const float* bp = B + ((size_t)bc << 5) + k;

        const float a0 = __ldg(av);
        const float a1 = __ldg(av + 1);
        const float a2 = __ldg(av + 2);
        const float a3 = __ldg(av + 3);

        const float b0 = __ldg(bp);
        const float b1 = __ldg(bp + 8);
        const float b2 = __ldg(bp + 16);
        const float b3 = __ldg(bp + 24);

        sum = fmaf(a0, b0, sum);
        sum = fmaf(a1, b1, sum);
        sum = fmaf(a2, b2, sum);
        sum = fmaf(a3, b3, sum);
    }

    if (row < num_rows) {
        C[((size_t)row << 3) + k] = sum;
    }
}

void* spmm_preprocess(const int* d_row_ptr, const int* d_col_ind,
                      const float* d_values,
                      int num_rows, int num_cols, int nnz, int K)
{
    (void)K;

    SpMMContext* ctx = new SpMMContext;
    ctx->num_rows = num_rows;
    ctx->num_cols = num_cols;
    ctx->bsr = csr_to_bsr(d_row_ptr, d_col_ind, d_values,
                          num_rows, num_cols, nnz, 4);
    return ctx;
}

void spmm_compute(void* ctx_ptr,
                  const int* d_row_ptr, const int* d_col_ind,
                  const float* d_values,
                  const float* d_B, float* d_C,
                  int num_rows, int num_cols, int nnz, int K)
{
    (void)d_row_ptr;
    (void)d_col_ind;
    (void)d_values;
    (void)num_cols;
    (void)nnz;
    (void)K;

    SpMMContext* ctx = static_cast<SpMMContext*>(ctx_ptr);
    if (ctx == nullptr || ctx->bsr.mb <= 0 || num_rows <= 0) return;

    const int mb = ctx->bsr.mb;
    const int blocks = (mb + 3) >> 2;

    if ((mb & 3) == 0 && (num_rows & 3) == 0) {
        bsr4_spmm_k8_aligned_kernel<<<mb >> 2, 128>>>(
            ctx->bsr.d_bsr_row_ptr,
            ctx->bsr.d_bsr_col_ind,
            ctx->bsr.d_bsr_values,
            d_B,
            d_C);
    } else {
        bsr4_spmm_k8_tail_kernel<<<blocks, 128>>>(
            ctx->bsr.d_bsr_row_ptr,
            ctx->bsr.d_bsr_col_ind,
            ctx->bsr.d_bsr_values,
            d_B,
            d_C,
            num_rows,
            mb);
    }
}

void spmm_cleanup(void* ctx_ptr)
{
    SpMMContext* ctx = static_cast<SpMMContext*>(ctx_ptr);
    if (ctx != nullptr) {
        free_bsr(ctx->bsr);
        delete ctx;
    }
}