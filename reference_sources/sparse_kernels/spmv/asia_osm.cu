#include <cuda_runtime.h>

__device__ __forceinline__ int ro_load_i(const int* p) {
    return __ldg(p);
}

__device__ __forceinline__ float ro_load_f(const float* p) {
    return __ldg(p);
}

__global__ __launch_bounds__(128, 12)
void csr_spmv_thread_rows_kernel(const int* __restrict__ row_ptr,
                                  const int* __restrict__ col_ind,
                                  const float* __restrict__ values,
                                  const float* __restrict__ x,
                                  float* __restrict__ y,
                                  int num_rows) {
    int row = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int stride = (int)(gridDim.x * blockDim.x);

    for (; row < num_rows; row += stride) {
        const int begin = ro_load_i(row_ptr + row);
        const int end = ro_load_i(row_ptr + row + 1);
        const int n = end - begin;
        float sum = 0.0f;

        if (n == 1) {
            const int c0 = ro_load_i(col_ind + begin);
            const float v0 = ro_load_f(values + begin);
            sum = v0 * ro_load_f(x + c0);
        } else if (n == 2) {
            const int c0 = ro_load_i(col_ind + begin);
            const float v0 = ro_load_f(values + begin);
            const int c1 = ro_load_i(col_ind + begin + 1);
            const float v1 = ro_load_f(values + begin + 1);
            const float x0 = ro_load_f(x + c0);
            const float x1 = ro_load_f(x + c1);
            sum = fmaf(v0, x0, v1 * x1);
        } else if (n == 3) {
            const int c0 = ro_load_i(col_ind + begin);
            const float v0 = ro_load_f(values + begin);
            const int c1 = ro_load_i(col_ind + begin + 1);
            const float v1 = ro_load_f(values + begin + 1);
            const int c2 = ro_load_i(col_ind + begin + 2);
            const float v2 = ro_load_f(values + begin + 2);
            const float x0 = ro_load_f(x + c0);
            const float x1 = ro_load_f(x + c1);
            const float x2 = ro_load_f(x + c2);
            sum = fmaf(v0, x0, fmaf(v1, x1, v2 * x2));
        } else if (n == 4) {
            const int c0 = ro_load_i(col_ind + begin);
            const float v0 = ro_load_f(values + begin);
            const int c1 = ro_load_i(col_ind + begin + 1);
            const float v1 = ro_load_f(values + begin + 1);
            const int c2 = ro_load_i(col_ind + begin + 2);
            const float v2 = ro_load_f(values + begin + 2);
            const int c3 = ro_load_i(col_ind + begin + 3);
            const float v3 = ro_load_f(values + begin + 3);
            const float x0 = ro_load_f(x + c0);
            const float x1 = ro_load_f(x + c1);
            const float x2 = ro_load_f(x + c2);
            const float x3 = ro_load_f(x + c3);
            sum = fmaf(v0, x0, fmaf(v1, x1, fmaf(v2, x2, v3 * x3)));
        } else {
            int p = begin;
            if (p < end) {
                int c = ro_load_i(col_ind + p);
                float v = ro_load_f(values + p);
                ++p;
                while (p < end) {
                    const int next_c = ro_load_i(col_ind + p);
                    const float next_v = ro_load_f(values + p);
                    sum = fmaf(v, ro_load_f(x + c), sum);
                    c = next_c;
                    v = next_v;
                    ++p;
                }
                sum = fmaf(v, ro_load_f(x + c), sum);
            }
        }

        y[row] = sum;
    }
}

void* spmv_preprocess(const int* d_row_ptr, const int* d_col_ind,
                      const float* d_values,
                      int num_rows, int num_cols, int nnz) {
    (void)d_row_ptr;
    (void)d_col_ind;
    (void)d_values;
    (void)num_rows;
    (void)num_cols;
    (void)nnz;
    return nullptr;
}

void spmv_compute(void* ctx,
                  const int* d_row_ptr, const int* d_col_ind,
                  const float* d_values,
                  const float* d_x, float* d_y,
                  int num_rows, int num_cols, int nnz) {
    (void)ctx;
    (void)num_cols;
    (void)nnz;

    if (num_rows <= 0) return;

    const int threads = 128;
    const int blocks = (num_rows + threads - 1) / threads;
    csr_spmv_thread_rows_kernel<<<blocks, threads>>>(
        d_row_ptr, d_col_ind, d_values, d_x, d_y, num_rows);
}

void spmv_cleanup(void* ctx) {
    (void)ctx;
}