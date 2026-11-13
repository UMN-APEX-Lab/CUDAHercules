#include <cuda_runtime.h>
#include <stdint.h>
#include <vector>

struct SpmmContext {
    int* d_short_rows;
    int* d_long_rows;
    int* d_seg_begin;
    int* d_seg_end;
    int* d_row_seg_offset;
    int* d_row_seg_count;
    float* d_partials;
    int short_count;
    int long_count;
    int seg_count;
};

__device__ __forceinline__ void accum8(
    float a, int col, const float* __restrict__ B,
    float& x0, float& x1, float& x2, float& x3,
    float& x4, float& x5, float& x6, float& x7)
{
    const float4* bp = reinterpret_cast<const float4*>(B + ((size_t)col << 3));
    float4 b0 = bp[0];
    float4 b1 = bp[1];

    x0 = fmaf(a, b0.x, x0);
    x1 = fmaf(a, b0.y, x1);
    x2 = fmaf(a, b0.z, x2);
    x3 = fmaf(a, b0.w, x3);
    x4 = fmaf(a, b1.x, x4);
    x5 = fmaf(a, b1.y, x5);
    x6 = fmaf(a, b1.z, x6);
    x7 = fmaf(a, b1.w, x7);
}

__global__ __launch_bounds__(128, 10)
void short_rows_kernel(
    const int* __restrict__ rows,
    int count,
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_ind,
    const float* __restrict__ values,
    const float* __restrict__ B,
    float* __restrict__ C)
{
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int item = blockIdx.x * 4 + warp;

    if (item >= count) return;

    int row = rows[item];
    int begin = row_ptr[row];
    int end = row_ptr[row + 1];

    float x0 = 0.0f, x1 = 0.0f, x2 = 0.0f, x3 = 0.0f;
    float x4 = 0.0f, x5 = 0.0f, x6 = 0.0f, x7 = 0.0f;

    for (int p = begin + lane; p < end; p += 32) {
        int col = col_ind[p];
        float a = values[p];
        accum8(a, col, B, x0, x1, x2, x3, x4, x5, x6, x7);
    }

    unsigned mask = 0xffffffffu;
    for (int d = 16; d > 0; d >>= 1) {
        x0 += __shfl_down_sync(mask, x0, d);
        x1 += __shfl_down_sync(mask, x1, d);
        x2 += __shfl_down_sync(mask, x2, d);
        x3 += __shfl_down_sync(mask, x3, d);
        x4 += __shfl_down_sync(mask, x4, d);
        x5 += __shfl_down_sync(mask, x5, d);
        x6 += __shfl_down_sync(mask, x6, d);
        x7 += __shfl_down_sync(mask, x7, d);
    }

    if (lane == 0) {
        float4* out = reinterpret_cast<float4*>(C + ((size_t)row << 3));
        out[0] = make_float4(x0, x1, x2, x3);
        out[1] = make_float4(x4, x5, x6, x7);
    }
}

__global__ __launch_bounds__(128, 10)
void segment_kernel(
    const int* __restrict__ seg_begin,
    const int* __restrict__ seg_end,
    int seg_count,
    const int* __restrict__ col_ind,
    const float* __restrict__ values,
    const float* __restrict__ B,
    float* __restrict__ partials)
{
    int seg = blockIdx.x;
    if (seg >= seg_count) return;

    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;

    int begin = seg_begin[seg];
    int end = seg_end[seg];

    float x0 = 0.0f, x1 = 0.0f, x2 = 0.0f, x3 = 0.0f;
    float x4 = 0.0f, x5 = 0.0f, x6 = 0.0f, x7 = 0.0f;

    int p = begin + tid;

    if (p < end) {
        int next_col = col_ind[p];
        float next_a = values[p];

        for (;;) {
            int col = next_col;
            float a = next_a;
            int np = p + 128;

            if (np < end) {
                next_col = col_ind[np];
                next_a = values[np];
            }

            accum8(a, col, B, x0, x1, x2, x3, x4, x5, x6, x7);

            if (np >= end) break;
            p = np;
        }
    }

    unsigned mask = 0xffffffffu;
    for (int d = 16; d > 0; d >>= 1) {
        x0 += __shfl_down_sync(mask, x0, d);
        x1 += __shfl_down_sync(mask, x1, d);
        x2 += __shfl_down_sync(mask, x2, d);
        x3 += __shfl_down_sync(mask, x3, d);
        x4 += __shfl_down_sync(mask, x4, d);
        x5 += __shfl_down_sync(mask, x5, d);
        x6 += __shfl_down_sync(mask, x6, d);
        x7 += __shfl_down_sync(mask, x7, d);
    }

    __shared__ float warp_sums[4][8];

    if (lane == 0) {
        warp_sums[warp][0] = x0;
        warp_sums[warp][1] = x1;
        warp_sums[warp][2] = x2;
        warp_sums[warp][3] = x3;
        warp_sums[warp][4] = x4;
        warp_sums[warp][5] = x5;
        warp_sums[warp][6] = x6;
        warp_sums[warp][7] = x7;
    }

    __syncthreads();

    if (warp == 0 && lane < 8) {
        float sum =
            warp_sums[0][lane] +
            warp_sums[1][lane] +
            warp_sums[2][lane] +
            warp_sums[3][lane];
        partials[((size_t)seg << 3) + lane] = sum;
    }
}

__global__ __launch_bounds__(128, 10)
void finalize_rows_kernel(
    const int* __restrict__ rows,
    const int* __restrict__ offsets,
    const int* __restrict__ counts,
    int row_count,
    const float* __restrict__ partials,
    float* __restrict__ C)
{
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int item = blockIdx.x * 4 + warp;

    if (item >= row_count) return;

    int row = rows[item];
    int offset = offsets[item];
    int count = counts[item];

    if (lane < 8) {
        float sum = 0.0f;
        for (int i = 0; i < count; ++i) {
            sum += partials[((size_t)(offset + i) << 3) + lane];
        }
        C[((size_t)row << 3) + lane] = sum;
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
    (void)nnz;
    (void)K;

    SpmmContext* ctx = new SpmmContext();
    ctx->d_short_rows = nullptr;
    ctx->d_long_rows = nullptr;
    ctx->d_seg_begin = nullptr;
    ctx->d_seg_end = nullptr;
    ctx->d_row_seg_offset = nullptr;
    ctx->d_row_seg_count = nullptr;
    ctx->d_partials = nullptr;
    ctx->short_count = 0;
    ctx->long_count = 0;
    ctx->seg_count = 0;

    if (num_rows <= 0) return ctx;

    std::vector<int> h_ptr((size_t)num_rows + 1);
    cudaMemcpy(
        h_ptr.data(),
        d_row_ptr,
        ((size_t)num_rows + 1) * sizeof(int),
        cudaMemcpyDeviceToHost);

    std::vector<int> h_short;
    std::vector<int> h_long;
    std::vector<int> h_seg_begin;
    std::vector<int> h_seg_end;
    std::vector<int> h_offsets;
    std::vector<int> h_counts;

    h_short.reserve((size_t)num_rows / 3);
    h_long.reserve((size_t)num_rows * 2 / 3);
    h_seg_begin.reserve((size_t)h_ptr[num_rows] / 1024 + num_rows);
    h_seg_end.reserve((size_t)h_ptr[num_rows] / 1024 + num_rows);
    h_offsets.reserve((size_t)num_rows);
    h_counts.reserve((size_t)num_rows);

    constexpr int segment_nnz = 1024;

    for (int r = 0; r < num_rows; ++r) {
        int begin = h_ptr[r];
        int end = h_ptr[r + 1];
        int len = end - begin;

        if (len < 32) {
            h_short.push_back(r);
            continue;
        }

        int offset = (int)h_seg_begin.size();
        int nseg = (len + segment_nnz - 1) / segment_nnz;

        h_long.push_back(r);
        h_offsets.push_back(offset);
        h_counts.push_back(nseg);

        for (int p = begin; p < end; p += segment_nnz) {
            int q = p + segment_nnz;
            if (q > end) q = end;
            h_seg_begin.push_back(p);
            h_seg_end.push_back(q);
        }
    }

    ctx->short_count = (int)h_short.size();
    ctx->long_count = (int)h_long.size();
    ctx->seg_count = (int)h_seg_begin.size();

    if (ctx->short_count > 0) {
        cudaMalloc(&ctx->d_short_rows, (size_t)ctx->short_count * sizeof(int));
        cudaMemcpy(
            ctx->d_short_rows,
            h_short.data(),
            (size_t)ctx->short_count * sizeof(int),
            cudaMemcpyHostToDevice);
    }

    if (ctx->long_count > 0) {
        cudaMalloc(&ctx->d_long_rows, (size_t)ctx->long_count * sizeof(int));
        cudaMalloc(&ctx->d_row_seg_offset, (size_t)ctx->long_count * sizeof(int));
        cudaMalloc(&ctx->d_row_seg_count, (size_t)ctx->long_count * sizeof(int));

        cudaMemcpy(
            ctx->d_long_rows,
            h_long.data(),
            (size_t)ctx->long_count * sizeof(int),
            cudaMemcpyHostToDevice);
        cudaMemcpy(
            ctx->d_row_seg_offset,
            h_offsets.data(),
            (size_t)ctx->long_count * sizeof(int),
            cudaMemcpyHostToDevice);
        cudaMemcpy(
            ctx->d_row_seg_count,
            h_counts.data(),
            (size_t)ctx->long_count * sizeof(int),
            cudaMemcpyHostToDevice);
    }

    if (ctx->seg_count > 0) {
        cudaMalloc(&ctx->d_seg_begin, (size_t)ctx->seg_count * sizeof(int));
        cudaMalloc(&ctx->d_seg_end, (size_t)ctx->seg_count * sizeof(int));
        cudaMalloc(&ctx->d_partials, (size_t)ctx->seg_count * 8 * sizeof(float));

        cudaMemcpy(
            ctx->d_seg_begin,
            h_seg_begin.data(),
            (size_t)ctx->seg_count * sizeof(int),
            cudaMemcpyHostToDevice);
        cudaMemcpy(
            ctx->d_seg_end,
            h_seg_end.data(),
            (size_t)ctx->seg_count * sizeof(int),
            cudaMemcpyHostToDevice);
    }

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
    (void)num_rows;
    (void)num_cols;
    (void)nnz;
    (void)K;

    SpmmContext* ctx = reinterpret_cast<SpmmContext*>(context);
    if (!ctx) return;

    if (ctx->short_count > 0) {
        int blocks = (ctx->short_count + 3) >> 2;
        short_rows_kernel<<<blocks, 128>>>(
            ctx->d_short_rows,
            ctx->short_count,
            d_row_ptr,
            d_col_ind,
            d_values,
            d_B,
            d_C);
    }

    if (ctx->seg_count > 0) {
        segment_kernel<<<ctx->seg_count, 128>>>(
            ctx->d_seg_begin,
            ctx->d_seg_end,
            ctx->seg_count,
            d_col_ind,
            d_values,
            d_B,
            ctx->d_partials);
    }

    if (ctx->long_count > 0) {
        int blocks = (ctx->long_count + 3) >> 2;
        finalize_rows_kernel<<<blocks, 128>>>(
            ctx->d_long_rows,
            ctx->d_row_seg_offset,
            ctx->d_row_seg_count,
            ctx->long_count,
            ctx->d_partials,
            d_C);
    }
}

void spmm_cleanup(void* context)
{
    SpmmContext* ctx = reinterpret_cast<SpmmContext*>(context);
    if (!ctx) return;

    if (ctx->d_short_rows) cudaFree(ctx->d_short_rows);
    if (ctx->d_long_rows) cudaFree(ctx->d_long_rows);
    if (ctx->d_seg_begin) cudaFree(ctx->d_seg_begin);
    if (ctx->d_seg_end) cudaFree(ctx->d_seg_end);
    if (ctx->d_row_seg_offset) cudaFree(ctx->d_row_seg_offset);
    if (ctx->d_row_seg_count) cudaFree(ctx->d_row_seg_count);
    if (ctx->d_partials) cudaFree(ctx->d_partials);

    delete ctx;
}