#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <cstddef>

struct SegmentDesc {
    int row;
    int begin;
    int end;
};

struct SpMMContext {
    int* d_regular_rows;
    SegmentDesc* d_segments;
    int* d_dense_rows;
    int* d_dense_seg_offsets;
    float* d_partials;
    int regular_count;
    int segment_count;
    int dense_count;
};

__global__ __launch_bounds__(256, 5)
void regular_rows_k256_kernel(
    const int* __restrict__ regular_rows,
    int regular_count,
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_ind,
    const float* __restrict__ values,
    const float* __restrict__ B,
    float* __restrict__ C)
{
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int work = (int)blockIdx.x * 8 + warp;

    if (work >= regular_count) return;

    const int row = regular_rows[work];
    const int begin = row_ptr[row];
    const int end = row_ptr[row + 1];

    float s0 = 0.0f;
    float s1 = 0.0f;
    float s2 = 0.0f;
    float s3 = 0.0f;
    float s4 = 0.0f;
    float s5 = 0.0f;
    float s6 = 0.0f;
    float s7 = 0.0f;

    for (int p = begin; p < end; ++p) {
        int col = 0;
        float av = 0.0f;

        if (lane == 0) {
            col = col_ind[p];
            av = values[p];
        }

        col = __shfl_sync(0xffffffffu, col, 0);
        av = __shfl_sync(0xffffffffu, av, 0);

        const float* bp = B + (size_t)col * 256 + lane;
        s0 = fmaf(av, __ldg(bp), s0);
        s1 = fmaf(av, __ldg(bp + 32), s1);
        s2 = fmaf(av, __ldg(bp + 64), s2);
        s3 = fmaf(av, __ldg(bp + 96), s3);
        s4 = fmaf(av, __ldg(bp + 128), s4);
        s5 = fmaf(av, __ldg(bp + 160), s5);
        s6 = fmaf(av, __ldg(bp + 192), s6);
        s7 = fmaf(av, __ldg(bp + 224), s7);
    }

    float* cp = C + (size_t)row * 256 + lane;
    cp[0] = s0;
    cp[32] = s1;
    cp[64] = s2;
    cp[96] = s3;
    cp[128] = s4;
    cp[160] = s5;
    cp[192] = s6;
    cp[224] = s7;
}

__global__ __launch_bounds__(256, 5)
void dense_segment_k256_kernel(
    const SegmentDesc* __restrict__ segments,
    int segment_count,
    const int* __restrict__ col_ind,
    const float* __restrict__ values,
    const float* __restrict__ B,
    float* __restrict__ partials)
{
    const int seg_id = (int)blockIdx.x;
    if (seg_id >= segment_count) return;

    const int tid = (int)threadIdx.x;
    const SegmentDesc seg = segments[seg_id];
    const int count = seg.end - seg.begin;

    __shared__ int s_cols[384];
    __shared__ float s_vals[384];

    for (int i = tid; i < count; i += 256) {
        s_cols[i] = col_ind[seg.begin + i];
        s_vals[i] = values[seg.begin + i];
    }

    __syncthreads();

    float sum = 0.0f;

    #pragma unroll 4
    for (int base = 0; base < 384; base += 4) {
        if (base < count) {
            const int c0 = s_cols[base];
            const float a0 = s_vals[base];
            sum = fmaf(a0, __ldg(B + (size_t)c0 * 256 + tid), sum);
        }
        if (base + 1 < count) {
            const int c1 = s_cols[base + 1];
            const float a1 = s_vals[base + 1];
            sum = fmaf(a1, __ldg(B + (size_t)c1 * 256 + tid), sum);
        }
        if (base + 2 < count) {
            const int c2 = s_cols[base + 2];
            const float a2 = s_vals[base + 2];
            sum = fmaf(a2, __ldg(B + (size_t)c2 * 256 + tid), sum);
        }
        if (base + 3 < count) {
            const int c3 = s_cols[base + 3];
            const float a3 = s_vals[base + 3];
            sum = fmaf(a3, __ldg(B + (size_t)c3 * 256 + tid), sum);
        }
    }

    partials[(size_t)seg_id * 256 + tid] = sum;
}

__global__ __launch_bounds__(256, 6)
void dense_reduce_k256_kernel(
    const int* __restrict__ dense_rows,
    const int* __restrict__ dense_seg_offsets,
    int dense_count,
    const float* __restrict__ partials,
    float* __restrict__ C)
{
    const int dense_id = (int)blockIdx.x;
    if (dense_id >= dense_count) return;

    const int tid = (int)threadIdx.x;
    const int first = dense_seg_offsets[dense_id];
    const int last = dense_seg_offsets[dense_id + 1];

    float sum = 0.0f;
    for (int s = first; s < last; ++s) {
        sum += partials[(size_t)s * 256 + tid];
    }

    const int row = dense_rows[dense_id];
    C[(size_t)row * 256 + tid] = sum;
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

    SpMMContext* ctx = new SpMMContext();
    ctx->d_regular_rows = nullptr;
    ctx->d_segments = nullptr;
    ctx->d_dense_rows = nullptr;
    ctx->d_dense_seg_offsets = nullptr;
    ctx->d_partials = nullptr;
    ctx->regular_count = 0;
    ctx->segment_count = 0;
    ctx->dense_count = 0;

    if (num_rows <= 0) return ctx;

    std::vector<int> h_row_ptr((size_t)num_rows + 1);
    cudaMemcpy(
        h_row_ptr.data(),
        d_row_ptr,
        ((size_t)num_rows + 1) * sizeof(int),
        cudaMemcpyDeviceToHost);

    std::vector<int> h_regular_rows;
    std::vector<int> h_dense_rows;
    std::vector<int> h_dense_offsets;
    std::vector<SegmentDesc> h_segments;

    h_regular_rows.reserve(num_rows);
    h_dense_rows.reserve(num_rows / 32 + 1);
    h_dense_offsets.reserve(num_rows / 32 + 2);
    h_dense_offsets.push_back(0);

    constexpr int dense_threshold = 2461;
    constexpr int segment_nnz = 384;

    for (int row = 0; row < num_rows; ++row) {
        const int begin = h_row_ptr[row];
        const int end = h_row_ptr[row + 1];
        const int length = end - begin;

        if (length < dense_threshold) {
            h_regular_rows.push_back(row);
        } else {
            h_dense_rows.push_back(row);

            for (int p = begin; p < end; p += segment_nnz) {
                SegmentDesc d;
                d.row = row;
                d.begin = p;
                d.end = std::min(p + segment_nnz, end);
                h_segments.push_back(d);
            }

            h_dense_offsets.push_back((int)h_segments.size());
        }
    }

    ctx->regular_count = (int)h_regular_rows.size();
    ctx->dense_count = (int)h_dense_rows.size();
    ctx->segment_count = (int)h_segments.size();

    if (ctx->regular_count > 0) {
        cudaMalloc(
            &ctx->d_regular_rows,
            (size_t)ctx->regular_count * sizeof(int));
        cudaMemcpy(
            ctx->d_regular_rows,
            h_regular_rows.data(),
            (size_t)ctx->regular_count * sizeof(int),
            cudaMemcpyHostToDevice);
    }

    if (ctx->dense_count > 0) {
        cudaMalloc(
            &ctx->d_dense_rows,
            (size_t)ctx->dense_count * sizeof(int));
        cudaMalloc(
            &ctx->d_dense_seg_offsets,
            ((size_t)ctx->dense_count + 1) * sizeof(int));

        cudaMemcpy(
            ctx->d_dense_rows,
            h_dense_rows.data(),
            (size_t)ctx->dense_count * sizeof(int),
            cudaMemcpyHostToDevice);

        cudaMemcpy(
            ctx->d_dense_seg_offsets,
            h_dense_offsets.data(),
            ((size_t)ctx->dense_count + 1) * sizeof(int),
            cudaMemcpyHostToDevice);
    }

    if (ctx->segment_count > 0) {
        cudaMalloc(
            &ctx->d_segments,
            (size_t)ctx->segment_count * sizeof(SegmentDesc));

        cudaMalloc(
            &ctx->d_partials,
            (size_t)ctx->segment_count * 256 * sizeof(float));

        cudaMemcpy(
            ctx->d_segments,
            h_segments.data(),
            (size_t)ctx->segment_count * sizeof(SegmentDesc),
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

    SpMMContext* ctx = static_cast<SpMMContext*>(context);
    if (!ctx || K != 256) return;

    if (ctx->regular_count > 0) {
        const int grid = (ctx->regular_count + 7) / 8;

        regular_rows_k256_kernel<<<grid, 256>>>(
            ctx->d_regular_rows,
            ctx->regular_count,
            d_row_ptr,
            d_col_ind,
            d_values,
            d_B,
            d_C);
    }

    if (ctx->segment_count > 0) {
        dense_segment_k256_kernel<<<ctx->segment_count, 256>>>(
            ctx->d_segments,
            ctx->segment_count,
            d_col_ind,
            d_values,
            d_B,
            ctx->d_partials);
    }

    if (ctx->dense_count > 0) {
        dense_reduce_k256_kernel<<<ctx->dense_count, 256>>>(
            ctx->d_dense_rows,
            ctx->d_dense_seg_offsets,
            ctx->dense_count,
            ctx->d_partials,
            d_C);
    }
}

void spmm_cleanup(void* context)
{
    SpMMContext* ctx = static_cast<SpMMContext*>(context);
    if (!ctx) return;

    if (ctx->d_regular_rows) cudaFree(ctx->d_regular_rows);
    if (ctx->d_segments) cudaFree(ctx->d_segments);
    if (ctx->d_dense_rows) cudaFree(ctx->d_dense_rows);
    if (ctx->d_dense_seg_offsets) cudaFree(ctx->d_dense_seg_offsets);
    if (ctx->d_partials) cudaFree(ctx->d_partials);

    delete ctx;
}