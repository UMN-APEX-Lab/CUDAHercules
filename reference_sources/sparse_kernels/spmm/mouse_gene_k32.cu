#include <cuda_runtime.h>
#include <stdint.h>
#include "sparse_formats.cuh"

struct SegmentDesc {
    int begin;
    int end;
    int row;
    int split;
};

struct SpmmContext {
    COOData coo;
    SegmentDesc* d_segments;
    int num_segments;
    int capacity;
};

__global__ void build_segments_kernel(const int* __restrict__ row_ptr,
                                      int num_rows,
                                      SegmentDesc* __restrict__ segments,
                                      int* __restrict__ counter) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows) return;

    const int begin = row_ptr[row];
    const int end = row_ptr[row + 1];
    const int length = end - begin;
    if (length <= 0) return;

    const int count = (length + 255) >> 8;
    const int base = atomicAdd(counter, count);
    const int split = count > 1;

    #pragma unroll 1
    for (int s = 0; s < count; ++s) {
        const int seg_begin = begin + (s << 8);
        const int seg_end = (seg_begin + 256 < end) ? (seg_begin + 256) : end;
        segments[base + s] = SegmentDesc{seg_begin, seg_end, row, split};
    }
}

__global__ void zero_output_kernel(float4* __restrict__ output, int count) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        output[idx] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
}

#define SPMM_FMA_STEP(I)                                                       \
    do {                                                                       \
        const int col_##I = __shfl_sync(mask, local_col, I);                  \
        const float val_##I = __shfl_sync(mask, local_value, I);              \
        const float b_##I = __ldg(B + (size_t)col_##I * 32 + lane);           \
        sum = fmaf(val_##I, b_##I, sum);                                      \
    } while (0)

__global__ __launch_bounds__(128, 12)
void coo_segment_spmm_kernel(const SegmentDesc* __restrict__ segments,
                             int num_segments,
                             const int* __restrict__ col_ind,
                             const float* __restrict__ values,
                             const float* __restrict__ B,
                             float* __restrict__ C) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int seg_id = blockIdx.x * 4 + warp;

    if (seg_id >= num_segments) return;

    const SegmentDesc desc = segments[seg_id];
    constexpr unsigned mask = 0xffffffffu;

    float sum = 0.0f;
    int p = desc.begin;
    const int end = desc.end;

    while (p + 32 <= end) {
        const int local_col = __ldg(col_ind + p + lane);
        const float local_value = __ldg(values + p + lane);

        SPMM_FMA_STEP(0);
        SPMM_FMA_STEP(1);
        SPMM_FMA_STEP(2);
        SPMM_FMA_STEP(3);
        SPMM_FMA_STEP(4);
        SPMM_FMA_STEP(5);
        SPMM_FMA_STEP(6);
        SPMM_FMA_STEP(7);
        SPMM_FMA_STEP(8);
        SPMM_FMA_STEP(9);
        SPMM_FMA_STEP(10);
        SPMM_FMA_STEP(11);
        SPMM_FMA_STEP(12);
        SPMM_FMA_STEP(13);
        SPMM_FMA_STEP(14);
        SPMM_FMA_STEP(15);
        SPMM_FMA_STEP(16);
        SPMM_FMA_STEP(17);
        SPMM_FMA_STEP(18);
        SPMM_FMA_STEP(19);
        SPMM_FMA_STEP(20);
        SPMM_FMA_STEP(21);
        SPMM_FMA_STEP(22);
        SPMM_FMA_STEP(23);
        SPMM_FMA_STEP(24);
        SPMM_FMA_STEP(25);
        SPMM_FMA_STEP(26);
        SPMM_FMA_STEP(27);
        SPMM_FMA_STEP(28);
        SPMM_FMA_STEP(29);
        SPMM_FMA_STEP(30);
        SPMM_FMA_STEP(31);

        p += 32;
    }

    if (p < end) {
        const int valid = end - p;
        int local_col = 0;
        float local_value = 0.0f;

        if (lane < valid) {
            local_col = __ldg(col_ind + p + lane);
            local_value = __ldg(values + p + lane);
        }

        #pragma unroll 1
        for (int i = 0; i < valid; ++i) {
            const int col = __shfl_sync(mask, local_col, i);
            const float value = __shfl_sync(mask, local_value, i);
            const float b = __ldg(B + (size_t)col * 32 + lane);
            sum = fmaf(value, b, sum);
        }
    }

    float* output = C + (size_t)desc.row * 32 + lane;
    if (desc.split) {
        atomicAdd(output, sum);
    } else {
        *output = sum;
    }
}

#undef SPMM_FMA_STEP

void* spmm_preprocess(const int* d_row_ptr, const int* d_col_ind,
                      const float* d_values,
                      int num_rows, int num_cols, int nnz, int K) {
    SpmmContext* ctx = new SpmmContext;
    ctx->coo.d_row_ind = nullptr;
    ctx->coo.d_col_ind = nullptr;
    ctx->coo.d_values = nullptr;
    ctx->d_segments = nullptr;
    ctx->num_segments = 0;
    ctx->capacity = 0;

    if (num_rows <= 0 || nnz <= 0) {
        return ctx;
    }

    ctx->coo = csr_to_coo(d_row_ptr, d_col_ind, d_values,
                          num_rows, num_cols, nnz);

    ctx->capacity = num_rows + ((nnz + 255) >> 8);
    cudaMalloc(&ctx->d_segments,
               (size_t)ctx->capacity * sizeof(SegmentDesc));

    int* d_counter = nullptr;
    cudaMalloc(&d_counter, sizeof(int));
    cudaMemset(d_counter, 0, sizeof(int));

    constexpr int threads = 256;
    const int blocks = (num_rows + threads - 1) / threads;
    build_segments_kernel<<<blocks, threads>>>(
        d_row_ptr, num_rows, ctx->d_segments, d_counter);

    cudaMemcpy(&ctx->num_segments, d_counter, sizeof(int),
               cudaMemcpyDeviceToHost);
    cudaFree(d_counter);

    return ctx;
}

void spmm_compute(void* context,
                  const int* d_row_ptr, const int* d_col_ind,
                  const float* d_values,
                  const float* d_B, float* d_C,
                  int num_rows, int num_cols, int nnz, int K) {
    SpmmContext* ctx = reinterpret_cast<SpmmContext*>(context);

    if (num_rows <= 0) return;

    constexpr int zero_threads = 256;
    const int float4_count = num_rows * 8;
    const int zero_blocks = (float4_count + zero_threads - 1) / zero_threads;
    zero_output_kernel<<<zero_blocks, zero_threads>>>(
        reinterpret_cast<float4*>(d_C), float4_count);

    if (ctx == nullptr || ctx->num_segments <= 0) return;

    const int blocks = (ctx->num_segments + 3) >> 2;
    coo_segment_spmm_kernel<<<blocks, 128>>>(
        ctx->d_segments,
        ctx->num_segments,
        ctx->coo.d_col_ind,
        ctx->coo.d_values,
        d_B,
        d_C);
}

void spmm_cleanup(void* context) {
    SpmmContext* ctx = reinterpret_cast<SpmmContext*>(context);
    if (ctx == nullptr) return;

    if (ctx->d_segments != nullptr) {
        cudaFree(ctx->d_segments);
    }

    if (ctx->coo.d_col_ind != nullptr) {
        free_coo(ctx->coo);
    }

    delete ctx;
}