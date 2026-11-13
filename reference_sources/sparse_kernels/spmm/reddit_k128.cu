#include <cuda_runtime.h>

struct SpmmContext {
    int num_segments;
};

__device__ __forceinline__ int find_row_for_nnz(
    const int* __restrict__ row_ptr,
    int num_rows,
    int pos)
{
    int lo = 0;
    int hi = num_rows + 1;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        int v = __ldg(row_ptr + mid);
        if (v <= pos) lo = mid + 1;
        else hi = mid;
    }
    return lo - 1;
}

__global__ __launch_bounds__(128, 12)
void csr_segment_k128_compact_kernel(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_ind,
    const float* __restrict__ values,
    const float* __restrict__ B,
    float* __restrict__ C,
    int num_rows,
    int nnz)
{
    constexpr int SEGMENT_NNZ = 1024;

    __shared__ int s_col[SEGMENT_NNZ];
    __shared__ float s_val[SEGMENT_NNZ];
    __shared__ int s_row;
    __shared__ int s_row_begin;
    __shared__ int s_fragment_end;
    __shared__ int s_row_end;

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int k = (warp << 5) + lane;

    const int begin = (int)blockIdx.x * SEGMENT_NNZ;
    if (begin >= nnz) return;

    const int end = min(begin + SEGMENT_NNZ, nnz);
    const int count = end - begin;

    for (int i = tid; i < count; i += 128) {
        s_col[i] = __ldg(col_ind + begin + i);
        s_val[i] = __ldg(values + begin + i);
    }
    __syncthreads();

    if (tid == 0) {
        s_row = find_row_for_nnz(row_ptr, num_rows, begin);
    }
    __syncthreads();

    int row = s_row;
    int p = begin;

    while (p < end && row < num_rows) {
        if (tid == 0) {
            int rb = __ldg(row_ptr + row);
            int re = __ldg(row_ptr + row + 1);
            s_row_begin = rb;
            s_row_end = re;
            s_fragment_end = min(re, end);
        }
        __syncthreads();

        const int fragment_end = s_fragment_end;
        const int local_begin = p - begin;
        const int local_end = fragment_end - begin;

        float acc0 = 0.0f;
        float acc1 = 0.0f;
        int i = local_begin;

        for (; i + 1 < local_end; i += 2) {
            const int c0 = s_col[i];
            const int c1 = s_col[i + 1];
            const float v0 = s_val[i];
            const float v1 = s_val[i + 1];

            const float b0 = __ldg(B + ((size_t)c0 << 7) + k);
            const float b1 = __ldg(B + ((size_t)c1 << 7) + k);

            acc0 = fmaf(v0, b0, acc0);
            acc1 = fmaf(v1, b1, acc1);
        }

        if (i < local_end) {
            const int c = s_col[i];
            acc0 = fmaf(s_val[i], __ldg(B + ((size_t)c << 7) + k), acc0);
        }

        const float result = acc0 + acc1;
        float* out = C + ((size_t)row << 7) + k;

        if (p == s_row_begin && fragment_end == s_row_end) {
            *out = result;
        } else {
            atomicAdd(out, result);
        }

        p = fragment_end;
        ++row;
        __syncthreads();
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
    (void)d_row_ptr;
    (void)d_col_ind;
    (void)d_values;
    (void)num_rows;
    (void)num_cols;
    (void)K;

    SpmmContext* ctx = new SpmmContext;
    ctx->num_segments = (nnz + 1023) >> 10;
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
    (void)num_cols;

    if (num_rows <= 0 || K != 128) return;

    cudaMemsetAsync(d_C, 0, (size_t)num_rows * 128 * sizeof(float), 0);

    if (nnz <= 0) return;

    SpmmContext* ctx = static_cast<SpmmContext*>(context);
    const int segments = ctx ? ctx->num_segments : ((nnz + 1023) >> 10);

    csr_segment_k128_compact_kernel<<<segments, 128>>>(
        d_row_ptr,
        d_col_ind,
        d_values,
        d_B,
        d_C,
        num_rows,
        nnz);
}

void spmm_cleanup(void* context)
{
    delete static_cast<SpmmContext*>(context);
}