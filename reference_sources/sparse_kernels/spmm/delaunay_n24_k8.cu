#include <cuda_runtime.h>
#include <stddef.h>

__device__ __forceinline__ unsigned subgroup_mask_8(int group)
{
    return 0xffu << (group << 3);
}

__global__ __launch_bounds__(128, 12)
void csr_spmm_k8_subgroup_kernel(const int* __restrict__ row_ptr,
                                  const int* __restrict__ col_ind,
                                  const float* __restrict__ values,
                                  const float* __restrict__ B,
                                  float* __restrict__ C,
                                  int num_rows)
{
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int group = lane >> 3;
    const int subgroup_lane = lane & 7;
    const int row = ((int)blockIdx.x << 4) + (warp << 2) + group;
    const unsigned mask = subgroup_mask_8(group);

    if (row >= num_rows) return;

    int begin = 0;
    int end = 0;

    if (subgroup_lane == 0) {
        begin = __ldg(row_ptr + row);
        end = __ldg(row_ptr + row + 1);
    }

    begin = __shfl_sync(mask, begin, 0, 8);
    end = __shfl_sync(mask, end, 0, 8);

    float s0 = 0.0f;
    float s1 = 0.0f;
    float s2 = 0.0f;
    float s3 = 0.0f;
    float s4 = 0.0f;
    float s5 = 0.0f;
    float s6 = 0.0f;
    float s7 = 0.0f;

    for (int p = begin + subgroup_lane; p < end; p += 8) {
        const int col = __ldg(col_ind + p);
        const float a = __ldg(values + p);
        const float* __restrict__ bp = B + ((size_t)col << 3);

        s0 = fmaf(a, __ldg(bp + 0), s0);
        s1 = fmaf(a, __ldg(bp + 1), s1);
        s2 = fmaf(a, __ldg(bp + 2), s2);
        s3 = fmaf(a, __ldg(bp + 3), s3);
        s4 = fmaf(a, __ldg(bp + 4), s4);
        s5 = fmaf(a, __ldg(bp + 5), s5);
        s6 = fmaf(a, __ldg(bp + 6), s6);
        s7 = fmaf(a, __ldg(bp + 7), s7);
    }

    s0 += __shfl_down_sync(mask, s0, 4, 8);
    s1 += __shfl_down_sync(mask, s1, 4, 8);
    s2 += __shfl_down_sync(mask, s2, 4, 8);
    s3 += __shfl_down_sync(mask, s3, 4, 8);
    s4 += __shfl_down_sync(mask, s4, 4, 8);
    s5 += __shfl_down_sync(mask, s5, 4, 8);
    s6 += __shfl_down_sync(mask, s6, 4, 8);
    s7 += __shfl_down_sync(mask, s7, 4, 8);

    s0 += __shfl_down_sync(mask, s0, 2, 8);
    s1 += __shfl_down_sync(mask, s1, 2, 8);
    s2 += __shfl_down_sync(mask, s2, 2, 8);
    s3 += __shfl_down_sync(mask, s3, 2, 8);
    s4 += __shfl_down_sync(mask, s4, 2, 8);
    s5 += __shfl_down_sync(mask, s5, 2, 8);
    s6 += __shfl_down_sync(mask, s6, 2, 8);
    s7 += __shfl_down_sync(mask, s7, 2, 8);

    s0 += __shfl_down_sync(mask, s0, 1, 8);
    s1 += __shfl_down_sync(mask, s1, 1, 8);
    s2 += __shfl_down_sync(mask, s2, 1, 8);
    s3 += __shfl_down_sync(mask, s3, 1, 8);
    s4 += __shfl_down_sync(mask, s4, 1, 8);
    s5 += __shfl_down_sync(mask, s5, 1, 8);
    s6 += __shfl_down_sync(mask, s6, 1, 8);
    s7 += __shfl_down_sync(mask, s7, 1, 8);

    if (subgroup_lane == 0) {
        float* cp = C + ((size_t)row << 3);
        reinterpret_cast<float4*>(cp)[0] = make_float4(s0, s1, s2, s3);
        reinterpret_cast<float4*>(cp)[1] = make_float4(s4, s5, s6, s7);
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
    (void)num_cols;
    (void)nnz;

    if (num_rows <= 0 || K <= 0) return;

    const int blocks = (num_rows + 15) >> 4;

    csr_spmm_k8_subgroup_kernel<<<blocks, 128>>>(
        d_row_ptr, d_col_ind, d_values, d_B, d_C, num_rows);
}

void spmm_cleanup(void* ctx)
{
    (void)ctx;
}