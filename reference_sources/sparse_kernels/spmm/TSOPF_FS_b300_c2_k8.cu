#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <cstddef>

struct SpanTask {
    int row;
    int begin;
    int end;
};

struct SpMMContext {
    int* d_short_rows = nullptr;
    SpanTask* d_span_tasks = nullptr;
    int short_count = 0;
    int span_count = 0;
};

static constexpr int SHORT_LIMIT = 32;
static constexpr int SPAN_NNZ = 256;

__device__ __forceinline__ void load_b8(
    const float* __restrict__ B, int col,
    float& b0, float& b1, float& b2, float& b3,
    float& b4, float& b5, float& b6, float& b7) {
    const float4* p = reinterpret_cast<const float4*>(
        B + ((size_t)col << 3));
    float4 x0 = __ldg(p);
    float4 x1 = __ldg(p + 1);
    b0 = x0.x;
    b1 = x0.y;
    b2 = x0.z;
    b3 = x0.w;
    b4 = x1.x;
    b5 = x1.y;
    b6 = x1.z;
    b7 = x1.w;
}

__device__ __forceinline__ void reduce8_warp(
    float& s0, float& s1, float& s2, float& s3,
    float& s4, float& s5, float& s6, float& s7) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        s0 += __shfl_down_sync(0xffffffff, s0, offset);
        s1 += __shfl_down_sync(0xffffffff, s1, offset);
        s2 += __shfl_down_sync(0xffffffff, s2, offset);
        s3 += __shfl_down_sync(0xffffffff, s3, offset);
        s4 += __shfl_down_sync(0xffffffff, s4, offset);
        s5 += __shfl_down_sync(0xffffffff, s5, offset);
        s6 += __shfl_down_sync(0xffffffff, s6, offset);
        s7 += __shfl_down_sync(0xffffffff, s7, offset);
    }
}

__global__ void short_rows_kernel(
    const int* __restrict__ rows,
    int count,
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_ind,
    const float* __restrict__ values,
    const float* __restrict__ B,
    float* __restrict__ C) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int task = (blockIdx.x << 3) + warp;

    if (task >= count) return;

    int row = __ldg(rows + task);
    int begin = __ldg(row_ptr + row);
    int end = __ldg(row_ptr + row + 1);

    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    float s4 = 0.0f, s5 = 0.0f, s6 = 0.0f, s7 = 0.0f;

    for (int p = begin + lane; p < end; p += 32) {
        float a = __ldg(values + p);
        int col = __ldg(col_ind + p);

        float b0, b1, b2, b3, b4, b5, b6, b7;
        load_b8(B, col, b0, b1, b2, b3, b4, b5, b6, b7);

        s0 = fmaf(a, b0, s0);
        s1 = fmaf(a, b1, s1);
        s2 = fmaf(a, b2, s2);
        s3 = fmaf(a, b3, s3);
        s4 = fmaf(a, b4, s4);
        s5 = fmaf(a, b5, s5);
        s6 = fmaf(a, b6, s6);
        s7 = fmaf(a, b7, s7);
    }

    reduce8_warp(s0, s1, s2, s3, s4, s5, s6, s7);

    if (lane == 0) {
        float* out = C + ((size_t)row << 3);
        reinterpret_cast<float4*>(out)[0] = make_float4(s0, s1, s2, s3);
        reinterpret_cast<float4*>(out)[1] = make_float4(s4, s5, s6, s7);
    }
}

__global__ void span_rows_kernel(
    const SpanTask* __restrict__ tasks,
    int count,
    const int* __restrict__ col_ind,
    const float* __restrict__ values,
    const float* __restrict__ B,
    float* __restrict__ C) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int task_id = (blockIdx.x << 3) + warp;

    if (task_id >= count) return;

    SpanTask task = tasks[task_id];
    int row = task.row;

    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    float s4 = 0.0f, s5 = 0.0f, s6 = 0.0f, s7 = 0.0f;

    for (int p = task.begin + lane; p < task.end; p += 32) {
        float a = __ldg(values + p);
        int col = __ldg(col_ind + p);

        float b0, b1, b2, b3, b4, b5, b6, b7;
        load_b8(B, col, b0, b1, b2, b3, b4, b5, b6, b7);

        s0 = fmaf(a, b0, s0);
        s1 = fmaf(a, b1, s1);
        s2 = fmaf(a, b2, s2);
        s3 = fmaf(a, b3, s3);
        s4 = fmaf(a, b4, s4);
        s5 = fmaf(a, b5, s5);
        s6 = fmaf(a, b6, s6);
        s7 = fmaf(a, b7, s7);
    }

    reduce8_warp(s0, s1, s2, s3, s4, s5, s6, s7);

    if (lane == 0) {
        float* out = C + ((size_t)row << 3);
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

void* spmm_preprocess(
    const int* d_row_ptr,
    const int*,
    const float*,
    int num_rows,
    int,
    int,
    int) {
    SpMMContext* ctx = new SpMMContext();
    if (num_rows <= 0) return ctx;

    std::vector<int> h_row_ptr((size_t)num_rows + 1);
    cudaMemcpy(
        h_row_ptr.data(),
        d_row_ptr,
        ((size_t)num_rows + 1) * sizeof(int),
        cudaMemcpyDeviceToHost);

    std::vector<int> short_rows;
    std::vector<SpanTask> span_tasks;
    short_rows.reserve((size_t)num_rows / 2);

    for (int row = 0; row < num_rows; ++row) {
        int begin = h_row_ptr[row];
        int end = h_row_ptr[row + 1];
        int length = end - begin;

        if (length < SHORT_LIMIT) {
            short_rows.push_back(row);
        } else {
            for (int p = begin; p < end; p += SPAN_NNZ) {
                SpanTask task;
                task.row = row;
                task.begin = p;
                task.end = std::min(p + SPAN_NNZ, end);
                span_tasks.push_back(task);
            }
        }
    }

    ctx->short_count = static_cast<int>(short_rows.size());
    ctx->span_count = static_cast<int>(span_tasks.size());

    if (ctx->short_count > 0) {
        cudaMalloc(
            reinterpret_cast<void**>(&ctx->d_short_rows),
            (size_t)ctx->short_count * sizeof(int));
        cudaMemcpy(
            ctx->d_short_rows,
            short_rows.data(),
            (size_t)ctx->short_count * sizeof(int),
            cudaMemcpyHostToDevice);
    }

    if (ctx->span_count > 0) {
        cudaMalloc(
            reinterpret_cast<void**>(&ctx->d_span_tasks),
            (size_t)ctx->span_count * sizeof(SpanTask));
        cudaMemcpy(
            ctx->d_span_tasks,
            span_tasks.data(),
            (size_t)ctx->span_count * sizeof(SpanTask),
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
    int,
    int,
    int) {
    SpMMContext* ctx = static_cast<SpMMContext*>(context);
    if (!ctx || num_rows <= 0) return;

    cudaMemsetAsync(d_C, 0, (size_t)num_rows * 8 * sizeof(float), 0);

    if (ctx->short_count > 0) {
        int blocks = (ctx->short_count + 7) >> 3;
        short_rows_kernel<<<blocks, 256>>>(
            ctx->d_short_rows,
            ctx->short_count,
            d_row_ptr,
            d_col_ind,
            d_values,
            d_B,
            d_C);
    }

    if (ctx->span_count > 0) {
        int blocks = (ctx->span_count + 7) >> 3;
        span_rows_kernel<<<blocks, 256>>>(
            ctx->d_span_tasks,
            ctx->span_count,
            d_col_ind,
            d_values,
            d_B,
            d_C);
    }
}

void spmm_cleanup(void* context) {
    SpMMContext* ctx = static_cast<SpMMContext*>(context);
    if (!ctx) return;

    if (ctx->d_short_rows) cudaFree(ctx->d_short_rows);
    if (ctx->d_span_tasks) cudaFree(ctx->d_span_tasks);

    delete ctx;
}