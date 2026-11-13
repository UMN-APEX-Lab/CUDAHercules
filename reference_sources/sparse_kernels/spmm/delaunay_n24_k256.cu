#include <cuda_runtime.h>

__global__ __launch_bounds__(256, 5)
void csr_spmm_k256_kernel(const int* __restrict__ row_ptr,
                           const int* __restrict__ col_ind,
                           const float* __restrict__ values,
                           const float* __restrict__ B,
                           float* __restrict__ C,
                           int num_rows)
{
    constexpr int WARPS_PER_BLOCK = 8;
    constexpr int MAX_ROW_NNZ = 26;
    constexpr int K = 256;

    __shared__ int s_cols[WARPS_PER_BLOCK * MAX_ROW_NNZ];
    __shared__ float s_vals[WARPS_PER_BLOCK * MAX_ROW_NNZ];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int row = blockIdx.x * WARPS_PER_BLOCK + warp;
    const unsigned mask = 0xffffffffu;

    if (row >= num_rows) return;

    int begin = 0;
    int len = 0;
    if (lane == 0) {
        begin = __ldg(row_ptr + row);
        const int end = __ldg(row_ptr + row + 1);
        len = end - begin;
    }

    begin = __shfl_sync(mask, begin, 0);
    len = __shfl_sync(mask, len, 0);

    float* const cptr = C + (size_t)row * K;

    if (len <= MAX_ROW_NNZ) {
        const int base = warp * MAX_ROW_NNZ;

        if (lane < len) {
            s_cols[base + lane] = __ldg(col_ind + begin + lane);
            s_vals[base + lane] = __ldg(values + begin + lane);
        }

        __syncwarp(mask);

        #pragma unroll
        for (int kt = 0; kt < 4; ++kt) {
            const int k0 = (kt << 6) + lane;
            float sum0 = 0.0f;
            float sum1 = 0.0f;

            #pragma unroll 1
            for (int p = 0; p < len; ++p) {
                const int col = s_cols[base + p];
                const float a = s_vals[base + p];
                const float* const bptr = B + (size_t)col * K + k0;
                sum0 = fmaf(a, __ldg(bptr), sum0);
                sum1 = fmaf(a, __ldg(bptr + 32), sum1);
            }

            cptr[k0] = sum0;
            cptr[k0 + 32] = sum1;
        }
    } else {
        float acc0[4];
        float acc1[4];

        #pragma unroll
        for (int t = 0; t < 4; ++t) {
            acc0[t] = 0.0f;
            acc1[t] = 0.0f;
        }

        for (int chunk = 0; chunk < len; chunk += 32) {
            int local_col = 0;
            float local_val = 0.0f;
            if (chunk + lane < len) {
                local_col = __ldg(col_ind + begin + chunk + lane);
                local_val = __ldg(values + begin + chunk + lane);
            }

            const int remaining = len - chunk;
            const int count = remaining < 32 ? remaining : 32;

            #pragma unroll 1
            for (int q = 0; q < count; ++q) {
                const int col = __shfl_sync(mask, local_col, q);
                const float a = __shfl_sync(mask, local_val, q);
                const float* const bptr = B + (size_t)col * K + lane;

                #pragma unroll
                for (int t = 0; t < 4; ++t) {
                    acc0[t] = fmaf(a, __ldg(bptr + (t << 6)), acc0[t]);
                    acc1[t] = fmaf(a, __ldg(bptr + (t << 6) + 32), acc1[t]);
                }
            }
        }

        #pragma unroll
        for (int t = 0; t < 4; ++t) {
            cptr[(t << 6) + lane] = acc0[t];
            cptr[(t << 6) + lane + 32] = acc1[t];
        }
    }
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

    if (num_rows <= 0 || K != 256) return;

    const dim3 block(256);
    const dim3 grid((num_rows + 7) >> 3);

    csr_spmm_k256_kernel<<<grid, block>>>(
        d_row_ptr, d_col_ind, d_values, d_B, d_C, num_rows);
}

void spmm_cleanup(void* ctx)
{
    (void)ctx;
}