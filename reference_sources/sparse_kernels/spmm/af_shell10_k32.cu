#include <cuda_runtime.h>
#include "sparse_formats.cuh"

struct SpmmContext {
    ELLData ell;
    int width;
};

__device__ __forceinline__ void load_ell_meta(
    const int* __restrict__ ell_col,
    const float* __restrict__ ell_val,
    int pos,
    int row,
    int num_rows,
    bool leader,
    bool valid_row,
    int leader_lane,
    int& col,
    float& val)
{
    int c = -1;
    float v = 0.0f;

    if (leader && valid_row) {
        const int idx = pos * num_rows + row;
        c = ell_col[idx];
        v = ell_val[idx];
    }

    col = __shfl_sync(0xffffffffu, c, leader_lane);
    val = __shfl_sync(0xffffffffu, v, leader_lane);
}

__global__ __launch_bounds__(128, 12)
void ell_spmm_k32_subwarp_pipe3_fixed35(
    const int* __restrict__ ell_col,
    const float* __restrict__ ell_val,
    const float* __restrict__ d_B,
    float* __restrict__ d_C,
    int num_rows)
{
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int subgroup = lane >> 2;
    const int sublane = lane & 3;

    const int row = blockIdx.x * 32 + warp * 8 + subgroup;
    const int kbase = sublane << 3;

    const bool valid_row = row < num_rows;
    const bool leader = (sublane == 0);
    const int leader_lane = lane & ~3;

    float a0 = 0.0f;
    float a1 = 0.0f;
    float a2 = 0.0f;
    float a3 = 0.0f;
    float a4 = 0.0f;
    float a5 = 0.0f;
    float a6 = 0.0f;
    float a7 = 0.0f;

    int c0, c1, c2;
    float v0, v1, v2;

    load_ell_meta(ell_col, ell_val, 0, row, num_rows,
                  leader, valid_row, leader_lane, c0, v0);
    load_ell_meta(ell_col, ell_val, 1, row, num_rows,
                  leader, valid_row, leader_lane, c1, v1);
    load_ell_meta(ell_col, ell_val, 2, row, num_rows,
                  leader, valid_row, leader_lane, c2, v2);

#pragma unroll 1
    for (int p = 0; p < 35; ++p) {
        int cn = -1;
        float vn = 0.0f;

        if (p + 3 < 35) {
            load_ell_meta(ell_col, ell_val, p + 3, row, num_rows,
                          leader, valid_row, leader_lane, cn, vn);
        }

        if (c0 >= 0) {
            const float* bp = d_B + (size_t)c0 * 32 + kbase;

            const float4 b0 = __ldg(reinterpret_cast<const float4*>(bp));
            const float4 b1 = __ldg(reinterpret_cast<const float4*>(bp + 4));

            a0 = fmaf(v0, b0.x, a0);
            a1 = fmaf(v0, b0.y, a1);
            a2 = fmaf(v0, b0.z, a2);
            a3 = fmaf(v0, b0.w, a3);
            a4 = fmaf(v0, b1.x, a4);
            a5 = fmaf(v0, b1.y, a5);
            a6 = fmaf(v0, b1.z, a6);
            a7 = fmaf(v0, b1.w, a7);
        }

        c0 = c1;
        v0 = v1;
        c1 = c2;
        v1 = v2;
        c2 = cn;
        v2 = vn;
    }

    if (valid_row) {
        float* cp = d_C + (size_t)row * 32 + kbase;
        *reinterpret_cast<float4*>(cp) = make_float4(a0, a1, a2, a3);
        *reinterpret_cast<float4*>(cp + 4) = make_float4(a4, a5, a6, a7);
    }
}

__global__ __launch_bounds__(128, 12)
void ell_spmm_k32_subwarp_generic(
    const int* __restrict__ ell_col,
    const float* __restrict__ ell_val,
    const float* __restrict__ d_B,
    float* __restrict__ d_C,
    int num_rows,
    int width)
{
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int subgroup = lane >> 2;
    const int sublane = lane & 3;

    const int row = blockIdx.x * 32 + warp * 8 + subgroup;
    const int kbase = sublane << 3;

    const bool valid_row = row < num_rows;
    const bool leader = (sublane == 0);
    const int leader_lane = lane & ~3;

    float a0 = 0.0f;
    float a1 = 0.0f;
    float a2 = 0.0f;
    float a3 = 0.0f;
    float a4 = 0.0f;
    float a5 = 0.0f;
    float a6 = 0.0f;
    float a7 = 0.0f;

#pragma unroll 1
    for (int p = 0; p < width; ++p) {
        int col;
        float val;

        load_ell_meta(ell_col, ell_val, p, row, num_rows,
                      leader, valid_row, leader_lane, col, val);

        if (col >= 0) {
            const float* bp = d_B + (size_t)col * 32 + kbase;

            const float4 b0 = __ldg(reinterpret_cast<const float4*>(bp));
            const float4 b1 = __ldg(reinterpret_cast<const float4*>(bp + 4));

            a0 = fmaf(val, b0.x, a0);
            a1 = fmaf(val, b0.y, a1);
            a2 = fmaf(val, b0.z, a2);
            a3 = fmaf(val, b0.w, a3);
            a4 = fmaf(val, b1.x, a4);
            a5 = fmaf(val, b1.y, a5);
            a6 = fmaf(val, b1.z, a6);
            a7 = fmaf(val, b1.w, a7);
        }
    }

    if (valid_row) {
        float* cp = d_C + (size_t)row * 32 + kbase;
        *reinterpret_cast<float4*>(cp) = make_float4(a0, a1, a2, a3);
        *reinterpret_cast<float4*>(cp + 4) = make_float4(a4, a5, a6, a7);
    }
}

__global__ void zero_output_k32(float* d_C, int num_rows)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = num_rows * 32;

    if (idx < total) {
        d_C[idx] = 0.0f;
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

    SpmmContext* ctx = new SpmmContext;
    ctx->ell = csr_to_ell(d_row_ptr, d_col_ind, d_values,
                          num_rows, num_cols, nnz);
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
    (void)num_cols;
    (void)nnz;
    (void)K;

    if (num_rows <= 0) {
        return;
    }

    SpmmContext* ctx = static_cast<SpmmContext*>(context);

    if (ctx == nullptr || ctx->width <= 0) {
        const int total = num_rows * 32;
        zero_output_k32<<<(total + 255) / 256, 256>>>(d_C, num_rows);
        return;
    }

    const dim3 block(128);
    const dim3 grid((num_rows + 31) >> 5);

    if (ctx->width == 35) {
        ell_spmm_k32_subwarp_pipe3_fixed35<<<grid, block>>>(
            ctx->ell.d_col_ind,
            ctx->ell.d_values,
            d_B,
            d_C,
            num_rows);
    } else {
        ell_spmm_k32_subwarp_generic<<<grid, block>>>(
            ctx->ell.d_col_ind,
            ctx->ell.d_values,
            d_B,
            d_C,
            num_rows,
            ctx->width);
    }
}

void spmm_cleanup(void* context)
{
    SpmmContext* ctx = static_cast<SpmmContext*>(context);

    if (ctx != nullptr) {
        free_ell(ctx->ell);
        delete ctx;
    }
}