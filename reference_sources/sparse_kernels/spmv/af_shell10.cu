#include <cuda_runtime.h>
#include "sparse_formats.cuh"

struct SpMVContext {
    BSRData bsr;
    int num_rows;
    int num_cols;
    int mb;
};

__device__ __forceinline__
float bsr4_load_x(const float* __restrict__ x, int block_col, int lane_row,
                  int num_cols) {
    const int col = (block_col << 2) + lane_row;
    return (col < num_cols) ? __ldg(x + col) : 0.0f;
}

__global__ __launch_bounds__(128, 12)
void bsr4_subwarp_spmv_kernel(
    const int* __restrict__ bsr_row_ptr,
    const int* __restrict__ bsr_col_ind,
    const float* __restrict__ bsr_values,
    const float* __restrict__ x,
    float* __restrict__ y,
    int mb,
    int num_rows,
    int num_cols) {

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int br = (int)blockIdx.x * 4 + warp;

    if (br >= mb) return;

    const int subgroup = lane >> 2;
    const int r = lane & 3;
    const int subgroup_base = subgroup << 2;
    const unsigned subgroup_mask = 0xFu << subgroup_base;

    const int begin = bsr_row_ptr[br];
    const int end = bsr_row_ptr[br + 1];

    float acc = 0.0f;

    for (int p = begin + subgroup; p < end; p += 8) {
        int col = 0;
        if (r == 0) {
            col = bsr_col_ind[p];
        }
        col = __shfl_sync(subgroup_mask, col, subgroup_base);

        const float xv = bsr4_load_x(x, col, r, num_cols);
        const float x0 = __shfl_sync(subgroup_mask, xv, subgroup_base);
        const float x1 = __shfl_sync(subgroup_mask, xv, subgroup_base + 1);
        const float x2 = __shfl_sync(subgroup_mask, xv, subgroup_base + 2);
        const float x3 = __shfl_sync(subgroup_mask, xv, subgroup_base + 3);

        const float4 v = *reinterpret_cast<const float4*>(
            bsr_values + ((size_t)p << 4) + (r << 2));

        acc = fmaf(v.x, x0, acc);
        acc = fmaf(v.y, x1, acc);
        acc = fmaf(v.z, x2, acc);
        acc = fmaf(v.w, x3, acc);
    }

    const unsigned warp_mask = 0xffffffffu;
    const float acc4 = __shfl_sync(warp_mask, acc, r + 4);
    const float acc8 = __shfl_sync(warp_mask, acc, r + 8);
    const float acc12 = __shfl_sync(warp_mask, acc, r + 12);
    const float acc16 = __shfl_sync(warp_mask, acc, r + 16);
    const float acc20 = __shfl_sync(warp_mask, acc, r + 20);
    const float acc24 = __shfl_sync(warp_mask, acc, r + 24);
    const float acc28 = __shfl_sync(warp_mask, acc, r + 28);

    if (subgroup == 0) {
        const float sum = acc + acc4 + acc8 + acc12 +
                          acc16 + acc20 + acc24 + acc28;
        const int row = (br << 2) + r;
        if (row < num_rows) {
            y[row] = sum;
        }
    }
}

void* spmv_preprocess(const int* d_row_ptr, const int* d_col_ind,
                      const float* d_values,
                      int num_rows, int num_cols, int nnz) {
    SpMVContext* ctx = new SpMVContext;
    ctx->num_rows = num_rows;
    ctx->num_cols = num_cols;
    ctx->mb = (num_rows + 3) >> 2;
    ctx->bsr = csr_to_bsr(d_row_ptr, d_col_ind, d_values,
                          num_rows, num_cols, nnz, 4);
    return ctx;
}

void spmv_compute(void* opaque_ctx,
                  const int* d_row_ptr, const int* d_col_ind,
                  const float* d_values,
                  const float* d_x, float* d_y,
                  int num_rows, int num_cols, int nnz) {
    (void)d_row_ptr;
    (void)d_col_ind;
    (void)d_values;
    (void)nnz;

    SpMVContext* ctx = static_cast<SpMVContext*>(opaque_ctx);
    if (ctx == nullptr || ctx->mb == 0) return;

    const int grid = (ctx->mb + 3) >> 2;

    bsr4_subwarp_spmv_kernel<<<grid, 128>>>(
        ctx->bsr.d_bsr_row_ptr,
        ctx->bsr.d_bsr_col_ind,
        ctx->bsr.d_bsr_values,
        d_x,
        d_y,
        ctx->mb,
        num_rows,
        num_cols);
}

void spmv_cleanup(void* opaque_ctx) {
    SpMVContext* ctx = static_cast<SpMVContext*>(opaque_ctx);
    if (ctx != nullptr) {
        free_bsr(ctx->bsr);
        delete ctx;
    }
}