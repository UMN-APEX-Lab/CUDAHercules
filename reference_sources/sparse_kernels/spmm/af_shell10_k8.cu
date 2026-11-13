#include <cuda_runtime.h>

__global__ __launch_bounds__(128)
void spmm_csr_k8_subwarp_kernel(const int* __restrict__ row_ptr,
                                 const int* __restrict__ col_ind,
                                 const float* __restrict__ values,
                                 const float* __restrict__ B,
                                 float* __restrict__ C,
                                 int num_rows) {
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int k = tid & 7;
    const int subgroup_lane_base = lane & ~7;
    const unsigned mask = 0xffu << subgroup_lane_base;

    const int row = (int)blockIdx.x * 16 + (tid >> 3);
    if (row >= num_rows) return;

    const int begin = __ldg(row_ptr + row);
    const int end = __ldg(row_ptr + row + 1);

    float sum0 = 0.0f;
    float sum1 = 0.0f;

    int p = begin;

    for (; p + 1 < end; p += 2) {
        int col0 = 0;
        int col1 = 0;
        float a0 = 0.0f;
        float a1 = 0.0f;

        if (k == 0) {
            col0 = __ldg(col_ind + p);
            a0 = __ldg(values + p);
            col1 = __ldg(col_ind + p + 1);
            a1 = __ldg(values + p + 1);
        }

        col0 = __shfl_sync(mask, col0, 0, 8);
        a0 = __shfl_sync(mask, a0, 0, 8);
        col1 = __shfl_sync(mask, col1, 0, 8);
        a1 = __shfl_sync(mask, a1, 0, 8);

        const float b0 = __ldg(B + ((size_t)col0 << 3) + k);
        const float b1 = __ldg(B + ((size_t)col1 << 3) + k);

        sum0 = fmaf(a0, b0, sum0);
        sum1 = fmaf(a1, b1, sum1);
    }

    if (p < end) {
        int col = 0;
        float a = 0.0f;

        if (k == 0) {
            col = __ldg(col_ind + p);
            a = __ldg(values + p);
        }

        col = __shfl_sync(mask, col, 0, 8);
        a = __shfl_sync(mask, a, 0, 8);

        const float b = __ldg(B + ((size_t)col << 3) + k);
        sum0 = fmaf(a, b, sum0);
    }

    C[((size_t)row << 3) + k] = sum0 + sum1;
}

void* spmm_preprocess(const int* d_row_ptr, const int* d_col_ind,
                      const float* d_values,
                      int num_rows, int num_cols, int nnz, int K) {
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
                  const int* d_row_ptr, const int* d_col_ind,
                  const float* d_values,
                  const float* d_B, float* d_C,
                  int num_rows, int num_cols, int nnz, int K) {
    (void)ctx;
    (void)num_cols;
    (void)nnz;

    if (num_rows <= 0 || K != 8) return;

    const dim3 block(128);
    const dim3 grid((num_rows + 15) >> 4);

    spmm_csr_k8_subwarp_kernel<<<grid, block>>>(
        d_row_ptr, d_col_ind, d_values, d_B, d_C, num_rows);
}

void spmm_cleanup(void* ctx) {
    (void)ctx;
}