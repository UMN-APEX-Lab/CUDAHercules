#include <cuda_runtime.h>

static constexpr int K_FIXED = 8;
static constexpr int THREADS_PER_BLOCK = 128;
static constexpr int NNZ_PER_CHUNK = 2387;
static constexpr int NNZ_PER_WARP = 597;

__device__ __forceinline__ int csr_row_from_nnz(
    const int* __restrict__ row_ptr, int num_rows, int nz)
{
    int lo = 0;
    int hi = num_rows;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        int next = __ldg(row_ptr + mid + 1);
        if (next <= nz) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

__device__ __forceinline__ float warp_sum(float v)
{
    v += __shfl_down_sync(0xffffffffu, v, 16);
    v += __shfl_down_sync(0xffffffffu, v, 8);
    v += __shfl_down_sync(0xffffffffu, v, 4);
    v += __shfl_down_sync(0xffffffffu, v, 2);
    v += __shfl_down_sync(0xffffffffu, v, 1);
    return v;
}

__device__ __forceinline__ void load_b8(
    const float* __restrict__ B,
    int col,
    float4& b01,
    float4& b45)
{
    const float* bp = B + ((size_t)col << 3);
    b01 = *reinterpret_cast<const float4*>(bp);
    b45 = *reinterpret_cast<const float4*>(bp + 4);
}

__global__ void zero_c_kernel(float* __restrict__ C, int count)
{
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int stride = (int)(blockDim.x * gridDim.x);
    for (int i = idx; i < count; i += stride) {
        C[i] = 0.0f;
    }
}

__global__ __launch_bounds__(THREADS_PER_BLOCK)
void csr_nnz_chunk_spmm_k8_prefetch_kernel(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_ind,
    const float* __restrict__ values,
    const float* __restrict__ B,
    float* __restrict__ C,
    int num_rows,
    int nnz)
{
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    const int chunk_begin = (int)blockIdx.x * NNZ_PER_CHUNK;
    if (chunk_begin >= nnz) return;

    const int chunk_end = min(chunk_begin + NNZ_PER_CHUNK, nnz);
    const int begin = min(chunk_begin + warp * NNZ_PER_WARP, chunk_end);
    const int end = min(begin + NNZ_PER_WARP, chunk_end);
    if (begin >= end) return;

    const int first_row = csr_row_from_nnz(row_ptr, num_rows, begin);
    const int last_row = csr_row_from_nnz(row_ptr, num_rows, end - 1);

    for (int row = first_row; row <= last_row; ++row) {
        const int row_begin = __ldg(row_ptr + row);
        const int row_end = __ldg(row_ptr + row + 1);
        const int seg_begin = max(row_begin, begin);
        const int seg_end = min(row_end, end);

        float s0 = 0.0f;
        float s1 = 0.0f;
        float s2 = 0.0f;
        float s3 = 0.0f;
        float s4 = 0.0f;
        float s5 = 0.0f;
        float s6 = 0.0f;
        float s7 = 0.0f;

        int p = seg_begin + lane;
        if (p < seg_end) {
            float a = __ldg(values + p);
            int col = __ldg(col_ind + p);
            float4 b01, b45;
            load_b8(B, col, b01, b45);

            while (true) {
                const int next_p = p + 32;
                const bool has_next = next_p < seg_end;

                float next_a = 0.0f;
                int next_col = 0;
                float4 next_b01;
                float4 next_b45;

                if (has_next) {
                    next_a = __ldg(values + next_p);
                    next_col = __ldg(col_ind + next_p);
                    load_b8(B, next_col, next_b01, next_b45);
                }

                s0 = fmaf(a, b01.x, s0);
                s1 = fmaf(a, b01.y, s1);
                s2 = fmaf(a, b01.z, s2);
                s3 = fmaf(a, b01.w, s3);
                s4 = fmaf(a, b45.x, s4);
                s5 = fmaf(a, b45.y, s5);
                s6 = fmaf(a, b45.z, s6);
                s7 = fmaf(a, b45.w, s7);

                if (!has_next) break;

                p = next_p;
                a = next_a;
                col = next_col;
                b01 = next_b01;
                b45 = next_b45;
            }
        }

        s0 = warp_sum(s0);
        s1 = warp_sum(s1);
        s2 = warp_sum(s2);
        s3 = warp_sum(s3);
        s4 = warp_sum(s4);
        s5 = warp_sum(s5);
        s6 = warp_sum(s6);
        s7 = warp_sum(s7);

        if (lane == 0) {
            float* out = C + ((size_t)row << 3);
            const bool complete_row =
                (seg_begin == row_begin) && (seg_end == row_end);

            if (complete_row) {
                reinterpret_cast<float4*>(out)[0] = make_float4(s0, s1, s2, s3);
                reinterpret_cast<float4*>(out)[1] = make_float4(s4, s5, s6, s7);
            } else {
                atomicAdd(out + 0, s0);
                atomicAdd(out + 1, s1);
                atomicAdd(out + 2, s2);
                atomicAdd(out + 3, s3);
                atomicAdd(out + 4, s4);
                atomicAdd(out + 5, s5);
                atomicAdd(out + 6, s6);
                atomicAdd(out + 7, s7);
            }
        }
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
    (void)nnz;
    (void)K;
    return nullptr;
}

void spmm_compute(
    void* ctx,
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
    (void)K;

    if (num_rows <= 0) return;

    const int output_count = num_rows * K_FIXED;
    int zero_blocks = (output_count + 255) >> 8;
    zero_blocks = min(zero_blocks, 65535);
    if (zero_blocks > 0) {
        zero_c_kernel<<<zero_blocks, 256>>>(d_C, output_count);
    }

    if (nnz <= 0) return;

    const int chunks = (nnz + NNZ_PER_CHUNK - 1) / NNZ_PER_CHUNK;
    csr_nnz_chunk_spmm_k8_prefetch_kernel<<<chunks, THREADS_PER_BLOCK>>>(
        d_row_ptr, d_col_ind, d_values, d_B, d_C, num_rows, nnz);
}

void spmm_cleanup(void* ctx)
{
    (void)ctx;
}