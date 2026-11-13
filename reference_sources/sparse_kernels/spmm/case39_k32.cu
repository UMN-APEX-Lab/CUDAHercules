#include <cuda_runtime.h>
#include "sparse_formats.cuh"

struct SpMMContext {
    COOData coo;
    int nnz;
};

static constexpr int K_FIXED = 32;
static constexpr int WARPS_PER_BLOCK = 8;
static constexpr int NNZ_PER_WARP = 32;

__global__ void coo_spmm_staged_k32_kernel(
    const int* __restrict__ row_ind,
    const int* __restrict__ col_ind,
    const float* __restrict__ values,
    const float* __restrict__ B,
    float* __restrict__ C,
    int nnz)
{
    __shared__ int s_rows[WARPS_PER_BLOCK][NNZ_PER_WARP];
    __shared__ int s_cols[WARPS_PER_BLOCK][NNZ_PER_WARP];
    __shared__ float s_vals[WARPS_PER_BLOCK][NNZ_PER_WARP];

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int chunk = blockIdx.x * WARPS_PER_BLOCK + warp;
    const int begin = chunk * NNZ_PER_WARP;

    if (begin >= nnz) return;

    const int remaining = nnz - begin;
    const int count = remaining < NNZ_PER_WARP ? remaining : NNZ_PER_WARP;
    const int idx = begin + lane;

    if (lane < count) {
        s_rows[warp][lane] = __ldg(row_ind + idx);
        s_cols[warp][lane] = __ldg(col_ind + idx);
        s_vals[warp][lane] = __ldg(values + idx);
    }

    __syncwarp();

    int left_row = -1;
    int right_row = -1;

    if (lane == 0) {
        if (begin > 0) left_row = __ldg(row_ind + begin - 1);
        if (begin + count < nnz) right_row = __ldg(row_ind + begin + count);
    }

    left_row = __shfl_sync(0xffffffffu, left_row, 0);
    right_row = __shfl_sync(0xffffffffu, right_row, 0);

    int current_row = s_rows[warp][0];
    bool shared_left = (begin > 0 && current_row == left_row);
    float accum = 0.0f;

#pragma unroll
    for (int offset = 0; offset < NNZ_PER_WARP; ++offset) {
        if (offset >= count) break;

        const int row = s_rows[warp][offset];
        const int col = s_cols[warp][offset];
        const float val = s_vals[warp][offset];

        if (row != current_row) {
            float* out = C + static_cast<size_t>(current_row) * K_FIXED + lane;
            if (shared_left) {
                atomicAdd(out, accum);
            } else {
                *out = accum;
            }
            accum = 0.0f;
            current_row = row;
            shared_left = false;
        }

        const float b = __ldg(B + static_cast<size_t>(col) * K_FIXED + lane);
        accum = fmaf(val, b, accum);
    }

    float* out = C + static_cast<size_t>(current_row) * K_FIXED + lane;
    const bool shared_right =
        (begin + count < nnz && current_row == right_row);

    if (shared_left || shared_right) {
        atomicAdd(out, accum);
    } else {
        *out = accum;
    }
}

__global__ void empty_spmm_kernel() {}

void* spmm_preprocess(const int* d_row_ptr,
                      const int* d_col_ind,
                      const float* d_values,
                      int num_rows,
                      int num_cols,
                      int nnz,
                      int K)
{
    (void)K;

    SpMMContext* ctx = new SpMMContext;
    ctx->coo = csr_to_coo(d_row_ptr, d_col_ind, d_values,
                          num_rows, num_cols, nnz);
    ctx->nnz = nnz;
    return ctx;
}

void spmm_compute(void* context,
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

    cudaMemsetAsync(
        d_C,
        0,
        static_cast<size_t>(num_rows) * K_FIXED * sizeof(float),
        0);

    if (ctx == nullptr || ctx->nnz <= 0) {
        empty_spmm_kernel<<<1, 1>>>();
        return;
    }

    const int chunks =
        (ctx->nnz + NNZ_PER_WARP - 1) / NNZ_PER_WARP;
    const int blocks =
        (chunks + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

    coo_spmm_staged_k32_kernel<<<blocks, WARPS_PER_BLOCK * 32>>>(
        ctx->coo.d_row_ind,
        ctx->coo.d_col_ind,
        ctx->coo.d_values,
        d_B,
        d_C,
        ctx->nnz);
}

void spmm_cleanup(void* context)
{
    SpMMContext* ctx = static_cast<SpMMContext*>(context);
    if (ctx != nullptr) {
        free_coo(ctx->coo);
        delete ctx;
    }
}