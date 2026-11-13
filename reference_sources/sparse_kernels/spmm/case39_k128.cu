#include <cuda_runtime.h>
#include <stdint.h>

struct SpmmContext {
    int* d_warp_start_rows;
    int num_warps;
};

__device__ __forceinline__ int find_row_for_nnz(
    const int* __restrict__ row_ptr,
    int num_rows,
    int p)
{
    int lo = 0;
    int hi = num_rows;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (row_ptr[mid + 1] <= p)
            lo = mid + 1;
        else
            hi = mid;
    }
    return lo;
}

__global__ void build_warp_start_rows(
    const int* __restrict__ row_ptr,
    int* __restrict__ warp_rows,
    int num_rows,
    int nnz)
{
    int warp_id = (int)blockIdx.x * blockDim.x + threadIdx.x;
    int p = warp_id << 5;
    if (p >= nnz)
        return;

    int lo = 0;
    int hi = num_rows;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (row_ptr[mid + 1] <= p)
            lo = mid + 1;
        else
            hi = mid;
    }
    warp_rows[warp_id] = lo;
}

__global__ __launch_bounds__(128, 12)
void csr_spmm_k128_nnz_chunks_cached_rows(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_ind,
    const float* __restrict__ values,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int* __restrict__ warp_start_rows,
    int num_rows,
    int nnz)
{
    constexpr int CHUNK_NNZ = 128;
    constexpr int WARP_NNZ = 32;

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int warp_id = ((int)blockIdx.x << 2) + warp;

    const int begin = ((int)blockIdx.x * CHUNK_NNZ) + warp * WARP_NNZ;
    const int end = min(begin + WARP_NNZ, nnz);

    if (begin >= end)
        return;

    int row = warp_start_rows[warp_id];
    int p = begin;

    while (p < end && row < num_rows) {
        int row_begin;
        int row_end;

        if (lane == 0) {
            row_begin = row_ptr[row];
            row_end = row_ptr[row + 1];
        }

        row_begin = __shfl_sync(0xffffffff, row_begin, 0);
        row_end = __shfl_sync(0xffffffff, row_end, 0);

        const int seg_end = min(end, row_end);

        float c0 = 0.0f;
        float c1 = 0.0f;
        float c2 = 0.0f;
        float c3 = 0.0f;

        for (int q = p; q < seg_end; ++q) {
            int col;
            float a;

            if (lane == 0) {
                col = col_ind[q];
                a = values[q];
            }

            col = __shfl_sync(0xffffffff, col, 0);
            a = __shfl_sync(0xffffffff, a, 0);

            const float* b = B + ((size_t)col << 7) + lane;

            float b0 = b[0];
            float b1 = b[32];
            float b2 = b[64];
            float b3 = b[96];

            c0 = fmaf(a, b0, c0);
            c1 = fmaf(a, b1, c1);
            c2 = fmaf(a, b2, c2);
            c3 = fmaf(a, b3, c3);
        }

        if (seg_end > p) {
            float* out = C + ((size_t)row << 7) + lane;

            if (p == row_begin && seg_end == row_end) {
                out[0] = c0;
                out[32] = c1;
                out[64] = c2;
                out[96] = c3;
            } else {
                atomicAdd(out, c0);
                atomicAdd(out + 32, c1);
                atomicAdd(out + 64, c2);
                atomicAdd(out + 96, c3);
            }
        }

        p = seg_end;
        ++row;
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
    (void)d_col_ind;
    (void)d_values;
    (void)num_cols;
    (void)K;

    SpmmContext* ctx = new SpmmContext();
    ctx->d_warp_start_rows = nullptr;
    ctx->num_warps = (nnz + 31) >> 5;

    if (nnz > 0 && num_rows > 0) {
        cudaMalloc(&ctx->d_warp_start_rows,
                   (size_t)ctx->num_warps * sizeof(int));

        const int threads = 256;
        const int blocks = (ctx->num_warps + threads - 1) / threads;
        build_warp_start_rows<<<blocks, threads>>>(
            d_row_ptr,
            ctx->d_warp_start_rows,
            num_rows,
            nnz);
    }

    return ctx;
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
    (void)num_cols;
    (void)K;

    if (num_rows <= 0)
        return;

    cudaMemsetAsync(
        d_C,
        0,
        (size_t)num_rows * 128 * sizeof(float),
        0);

    if (nnz <= 0)
        return;

    SpmmContext* state = (SpmmContext*)ctx;
    if (state == nullptr || state->d_warp_start_rows == nullptr)
        return;

    const dim3 block(128);
    const dim3 grid((nnz + 127) >> 7);

    csr_spmm_k128_nnz_chunks_cached_rows<<<grid, block>>>(
        d_row_ptr,
        d_col_ind,
        d_values,
        d_B,
        d_C,
        state->d_warp_start_rows,
        num_rows,
        nnz);
}

void spmm_cleanup(void* ctx)
{
    SpmmContext* state = (SpmmContext*)ctx;
    if (state != nullptr) {
        if (state->d_warp_start_rows != nullptr)
            cudaFree(state->d_warp_start_rows);
        delete state;
    }
}