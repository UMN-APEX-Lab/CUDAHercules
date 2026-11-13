#include <cuda_runtime.h>
#include <stddef.h>

__global__ __launch_bounds__(128, 12)
void csr_spmm_k128_kernel(const int* __restrict__ row_ptr,
                          const int* __restrict__ col_ind,
                          const float* __restrict__ values,
                          const float* __restrict__ B,
                          float* __restrict__ C,
                          int num_rows,
                          int num_cols,
                          int nnz)
{
    const int lane = (int)(threadIdx.x & 31);
    const int warp = (int)(threadIdx.x >> 5);
    const int row = (int)((unsigned int)blockIdx.x * 4u + (unsigned int)warp);

    if (row >= num_rows) {
        return;
    }

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;

    if (nnz > 0 && num_cols > 0 &&
        row_ptr != nullptr && col_ind != nullptr &&
        values != nullptr && B != nullptr) {
        int begin = 0;
        int end = 0;

        if (lane == 0) {
            begin = row_ptr[row];
            end = row_ptr[row + 1];

            if (begin < 0) begin = 0;
            if (begin > nnz) begin = nnz;
            if (end < begin) end = begin;
            if (end > nnz) end = nnz;
        }

        begin = __shfl_sync(0xffffffffu, begin, 0);
        end = __shfl_sync(0xffffffffu, end, 0);

        if (begin < end) {
            int col = 0;
            float val = 0.0f;

            if (lane == 0) {
                col = col_ind[begin];
                val = values[begin];
            }

            col = __shfl_sync(0xffffffffu, col, 0);
            val = __shfl_sync(0xffffffffu, val, 0);

            for (int p = begin; p < end; ++p) {
                int next_col = 0;
                float next_val = 0.0f;
                const bool has_next = (p + 1 < end);

                if (has_next && lane == 0) {
                    next_col = col_ind[p + 1];
                    next_val = values[p + 1];
                }

                if ((unsigned int)col < (unsigned int)num_cols) {
                    const size_t bbase =
                        (size_t)(unsigned int)col * (size_t)128 + (size_t)lane;
                    const float* bptr = B + bbase;

                    const float b0 = __ldg(bptr);
                    const float b1 = __ldg(bptr + 32);
                    const float b2 = __ldg(bptr + 64);
                    const float b3 = __ldg(bptr + 96);

                    acc0 = fmaf(val, b0, acc0);
                    acc1 = fmaf(val, b1, acc1);
                    acc2 = fmaf(val, b2, acc2);
                    acc3 = fmaf(val, b3, acc3);
                }

                if (has_next) {
                    col = __shfl_sync(0xffffffffu, next_col, 0);
                    val = __shfl_sync(0xffffffffu, next_val, 0);
                }
            }
        }
    }

    if (C != nullptr) {
        const size_t cbase =
            (size_t)(unsigned int)row * (size_t)128 + (size_t)lane;
        float* cptr = C + cbase;

        cptr[0] = acc0;
        cptr[32] = acc1;
        cptr[64] = acc2;
        cptr[96] = acc3;
    }
}

void* spmm_preprocess(const int* d_row_ptr,
                      const int* d_col_ind,
                      const float* d_values,
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
    (void)nnz;
    (void)K;
    return nullptr;
}

void spmm_compute(void* ctx,
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
    (void)ctx;

    if (K != 128 || num_rows <= 0 || d_C == nullptr) {
        return;
    }

    if (nnz < 0) {
        nnz = 0;
    }

    if (nnz > 0 &&
        (d_row_ptr == nullptr || d_col_ind == nullptr ||
         d_values == nullptr || d_B == nullptr || num_cols <= 0)) {
        return;
    }

    const unsigned int blocks = ((unsigned int)num_rows + 3u) >> 2;

    csr_spmm_k128_kernel<<<blocks, 128>>>(
        d_row_ptr,
        d_col_ind,
        d_values,
        d_B,
        d_C,
        num_rows,
        num_cols,
        nnz);
}

void spmm_cleanup(void* ctx)
{
    (void)ctx;
}