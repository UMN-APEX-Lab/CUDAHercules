#include <cuda_runtime.h>

__device__ __forceinline__ void accumulate_b8(
    const float* __restrict__ B,
    int col,
    float a,
    float& c0, float& c1, float& c2, float& c3,
    float& c4, float& c5, float& c6, float& c7)
{
    const size_t offset = ((size_t)col) << 3;
    const float4 b0 = *reinterpret_cast<const float4*>(B + offset);
    const float4 b1 = *reinterpret_cast<const float4*>(B + offset + 4);

    c0 = fmaf(a, b0.x, c0);
    c1 = fmaf(a, b0.y, c1);
    c2 = fmaf(a, b0.z, c2);
    c3 = fmaf(a, b0.w, c3);
    c4 = fmaf(a, b1.x, c4);
    c5 = fmaf(a, b1.y, c5);
    c6 = fmaf(a, b1.z, c6);
    c7 = fmaf(a, b1.w, c7);
}

__global__ __launch_bounds__(256, 6)
void csr_spmm_k8_thread_rows(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_ind,
    const float* __restrict__ values,
    const float* __restrict__ B,
    float* __restrict__ C,
    int num_rows)
{
    const int row = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= num_rows) return;

    const int begin = __ldg(row_ptr + row);
    const int end   = __ldg(row_ptr + row + 1);

    float c0 = 0.0f;
    float c1 = 0.0f;
    float c2 = 0.0f;
    float c3 = 0.0f;
    float c4 = 0.0f;
    float c5 = 0.0f;
    float c6 = 0.0f;
    float c7 = 0.0f;

    if (begin < end) {
        int cur_col = __ldg(col_ind + begin);
        float cur_val = __ldg(values + begin);

        int p = begin;
        while (p + 1 < end) {
            const int next_col = __ldg(col_ind + p + 1);
            const float next_val = __ldg(values + p + 1);

            accumulate_b8(
                B, cur_col, cur_val,
                c0, c1, c2, c3, c4, c5, c6, c7);

            cur_col = next_col;
            cur_val = next_val;
            ++p;
        }

        accumulate_b8(
            B, cur_col, cur_val,
            c0, c1, c2, c3, c4, c5, c6, c7);
    }

    const size_t out = ((size_t)row) << 3;
    *reinterpret_cast<float4*>(C + out) =
        make_float4(c0, c1, c2, c3);
    *reinterpret_cast<float4*>(C + out + 4) =
        make_float4(c4, c5, c6, c7);
}

void* spmm_preprocess(const int* d_row_ptr, const int* d_col_ind,
                      const float* d_values,
                      int num_rows, int num_cols, int nnz, int K)
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
                  const int* d_row_ptr, const int* d_col_ind,
                  const float* d_values,
                  const float* d_B, float* d_C,
                  int num_rows, int num_cols, int nnz, int K)
{
    (void)ctx;
    (void)num_cols;
    (void)nnz;

    if (num_rows <= 0 || K != 8) return;

    const int threads = 256;
    const int blocks = (num_rows + threads - 1) / threads;

    csr_spmm_k8_thread_rows<<<blocks, threads>>>(
        d_row_ptr, d_col_ind, d_values, d_B, d_C, num_rows);
}

void spmm_cleanup(void* ctx)
{
    (void)ctx;
}