#include <cuda_runtime.h>
#include "sparse_formats.cuh"

struct SpMMContext {
    COOData coo;
    const int* d_row_ptr;
    int nnz;
    int num_rows;
};

static constexpr int K_FIXED = 256;
static constexpr int CHUNK_NNZ = 4096;
static constexpr int THREADS = 128;

__global__ void spmm_coo_k64_chunk_kernel(
    const int* __restrict__ row_ind,
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_ind,
    const float* __restrict__ values,
    const float* __restrict__ B,
    float* __restrict__ C,
    int nnz,
    int num_rows)
{
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int k0 = (warp << 6) + lane;
    const int k1 = k0 + 32;

    const int begin = (int)blockIdx.x * CHUNK_NNZ;
    if (begin >= nnz) return;

    const int end = min(begin + CHUNK_NNZ, nnz);

    int row = 0;
    if (lane == 0) row = row_ind[begin];
    row = __shfl_sync(0xffffffff, row, 0);

    int p = begin;

    while (p < end) {
        int row_end = 0;
        if (lane == 0) {
            row_end = row_ptr[row + 1];
            if (row_end > end) row_end = end;
        }
        row_end = __shfl_sync(0xffffffff, row_end, 0);

        float acc0 = 0.0f;
        float acc1 = 0.0f;
        int i = p;

        for (; i + 3 < row_end; i += 4) {
            int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
            float v0 = 0.0f, v1 = 0.0f, v2 = 0.0f, v3 = 0.0f;

            if (lane == 0) {
                c0 = col_ind[i];
                v0 = values[i];
                c1 = col_ind[i + 1];
                v1 = values[i + 1];
                c2 = col_ind[i + 2];
                v2 = values[i + 2];
                c3 = col_ind[i + 3];
                v3 = values[i + 3];
            }

            c0 = __shfl_sync(0xffffffff, c0, 0);
            v0 = __shfl_sync(0xffffffff, v0, 0);
            c1 = __shfl_sync(0xffffffff, c1, 0);
            v1 = __shfl_sync(0xffffffff, v1, 0);
            c2 = __shfl_sync(0xffffffff, c2, 0);
            v2 = __shfl_sync(0xffffffff, v2, 0);
            c3 = __shfl_sync(0xffffffff, c3, 0);
            v3 = __shfl_sync(0xffffffff, v3, 0);

            const size_t b0 = ((size_t)c0 << 8);
            const size_t b1 = ((size_t)c1 << 8);
            const size_t b2 = ((size_t)c2 << 8);
            const size_t b3 = ((size_t)c3 << 8);

            acc0 = fmaf(v0, __ldg(B + b0 + k0), acc0);
            acc1 = fmaf(v0, __ldg(B + b0 + k1), acc1);
            acc0 = fmaf(v1, __ldg(B + b1 + k0), acc0);
            acc1 = fmaf(v1, __ldg(B + b1 + k1), acc1);
            acc0 = fmaf(v2, __ldg(B + b2 + k0), acc0);
            acc1 = fmaf(v2, __ldg(B + b2 + k1), acc1);
            acc0 = fmaf(v3, __ldg(B + b3 + k0), acc0);
            acc1 = fmaf(v3, __ldg(B + b3 + k1), acc1);
        }

        for (; i < row_end; ++i) {
            int col = 0;
            float val = 0.0f;
            if (lane == 0) {
                col = col_ind[i];
                val = values[i];
            }
            col = __shfl_sync(0xffffffff, col, 0);
            val = __shfl_sync(0xffffffff, val, 0);

            const size_t boff = ((size_t)col << 8);
            acc0 = fmaf(val, __ldg(B + boff + k0), acc0);
            acc1 = fmaf(val, __ldg(B + boff + k1), acc1);
        }

        int fragmented = 0;
        if (lane == 0) {
            const bool left = (p == begin) && (begin != 0) &&
                              (row_ptr[row] < begin);
            const bool right = (row_end == end) && (end != nnz) &&
                               (row_ptr[row + 1] > end);
            fragmented = (left || right) ? 1 : 0;
        }
        fragmented = __shfl_sync(0xffffffff, fragmented, 0);

        float* out = C + ((size_t)row << 8);
        if (fragmented) {
            atomicAdd(out + k0, acc0);
            atomicAdd(out + k1, acc1);
        } else {
            out[k0] = acc0;
            out[k1] = acc1;
        }

        p = row_end;
        if (p >= end) break;

        if (lane == 0) {
            ++row;
            while (row < num_rows && row_ptr[row + 1] <= p) ++row;
        }
        row = __shfl_sync(0xffffffff, row, 0);
    }
}

__global__ void spmm_empty_kernel() {}

void* spmm_preprocess(const int* d_row_ptr, const int* d_col_ind,
                      const float* d_values,
                      int num_rows, int num_cols, int nnz, int K)
{
    (void)K;
    SpMMContext* ctx = new SpMMContext;
    ctx->coo = csr_to_coo(d_row_ptr, d_col_ind, d_values,
                          num_rows, num_cols, nnz);
    ctx->d_row_ptr = d_row_ptr;
    ctx->nnz = nnz;
    ctx->num_rows = num_rows;
    return ctx;
}

void spmm_compute(void* context,
                  const int* d_row_ptr, const int* d_col_ind,
                  const float* d_values,
                  const float* d_B, float* d_C,
                  int num_rows, int num_cols, int nnz, int K)
{
    (void)d_col_ind;
    (void)d_values;
    (void)num_cols;
    (void)nnz;
    (void)K;

    SpMMContext* ctx = static_cast<SpMMContext*>(context);

    if (num_rows > 0) {
        cudaMemsetAsync(d_C, 0, (size_t)num_rows * K_FIXED * sizeof(float), 0);
    }

    if (ctx == nullptr || ctx->nnz == 0 || num_rows == 0) {
        spmm_empty_kernel<<<1, 1>>>();
        return;
    }

    const int blocks = (ctx->nnz + CHUNK_NNZ - 1) / CHUNK_NNZ;
    const int* row_ptr = d_row_ptr ? d_row_ptr : ctx->d_row_ptr;

    spmm_coo_k64_chunk_kernel<<<blocks, THREADS>>>(
        ctx->coo.d_row_ind,
        row_ptr,
        ctx->coo.d_col_ind,
        ctx->coo.d_values,
        d_B,
        d_C,
        ctx->nnz,
        ctx->num_rows);
}

void spmm_cleanup(void* context)
{
    SpMMContext* ctx = static_cast<SpMMContext*>(context);
    if (ctx == nullptr) return;
    free_coo(ctx->coo);
    delete ctx;
}