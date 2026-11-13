#include <cuda_runtime.h>
#include <cstddef>

template <int N>
__device__ __forceinline__
float csr_accum_fixed(const int* __restrict__ col_ind,
                      const float* __restrict__ values,
                      const float* __restrict__ B,
                      int begin, int lane)
{
    float sum = 0.0f;

    #pragma unroll
    for (int i = 0; i < N; ++i) {
        int col = 0;
        float val = 0.0f;

        if (lane == 0) {
            col = __ldg(col_ind + begin + i);
            val = __ldg(values + begin + i);
        }

        col = __shfl_sync(0xffffffffu, col, 0);
        val = __shfl_sync(0xffffffffu, val, 0);

        const float b = __ldg(B + (static_cast<size_t>(col) << 5) + lane);
        sum = fmaf(val, b, sum);
    }

    return sum;
}

__global__ __launch_bounds__(128, 12)
void csr_spmm_k32_kernel(const int* __restrict__ row_ptr,
                         const int* __restrict__ col_ind,
                         const float* __restrict__ values,
                         const float* __restrict__ B,
                         float* __restrict__ C,
                         int num_rows)
{
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int row = static_cast<int>(blockIdx.x) * 4 + warp;

    if (row >= num_rows) return;

    int begin = 0;
    int end = 0;

    if (lane == 0) {
        begin = __ldg(row_ptr + row);
        end = __ldg(row_ptr + row + 1);
    }

    begin = __shfl_sync(0xffffffffu, begin, 0);
    end = __shfl_sync(0xffffffffu, end, 0);

    const int len = end - begin;
    float sum;

    switch (len) {
        case 0:  sum = 0.0f; break;
        case 1:  sum = csr_accum_fixed<1>(col_ind, values, B, begin, lane); break;
        case 2:  sum = csr_accum_fixed<2>(col_ind, values, B, begin, lane); break;
        case 3:  sum = csr_accum_fixed<3>(col_ind, values, B, begin, lane); break;
        case 4:  sum = csr_accum_fixed<4>(col_ind, values, B, begin, lane); break;
        case 5:  sum = csr_accum_fixed<5>(col_ind, values, B, begin, lane); break;
        case 6:  sum = csr_accum_fixed<6>(col_ind, values, B, begin, lane); break;
        case 7:  sum = csr_accum_fixed<7>(col_ind, values, B, begin, lane); break;
        case 8:  sum = csr_accum_fixed<8>(col_ind, values, B, begin, lane); break;
        case 9:  sum = csr_accum_fixed<9>(col_ind, values, B, begin, lane); break;
        case 10: sum = csr_accum_fixed<10>(col_ind, values, B, begin, lane); break;
        default: {
            sum = 0.0f;
            for (int p = begin; p < end; ++p) {
                int col = 0;
                float val = 0.0f;

                if (lane == 0) {
                    col = __ldg(col_ind + p);
                    val = __ldg(values + p);
                }

                col = __shfl_sync(0xffffffffu, col, 0);
                val = __shfl_sync(0xffffffffu, val, 0);

                const float b = __ldg(B + (static_cast<size_t>(col) << 5) + lane);
                sum = fmaf(val, b, sum);
            }
            break;
        }
    }

    C[(static_cast<size_t>(row) << 5) + lane] = sum;
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

    if (num_rows <= 0 || K != 32) return;

    const dim3 block(128);
    const dim3 grid((num_rows + 3) >> 2);

    csr_spmm_k32_kernel<<<grid, block>>>(
        d_row_ptr, d_col_ind, d_values, d_B, d_C, num_rows);
}

void spmm_cleanup(void* ctx)
{
    (void)ctx;
}