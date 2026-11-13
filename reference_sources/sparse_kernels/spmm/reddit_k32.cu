#include <cuda_runtime.h>

struct SpmmContext {
    int* d_normal_rows;
    int* d_dense_rows;
    int normal_count;
    int dense_count;
    int normal_tasks;
};

__global__ void classify_rows_kernel(const int* __restrict__ row_ptr,
                                     int num_rows,
                                     int threshold,
                                     int* __restrict__ normal_rows,
                                     int* __restrict__ dense_rows,
                                     int* __restrict__ counts) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows) return;

    int length = row_ptr[row + 1] - row_ptr[row];
    if (length > threshold) {
        int pos = atomicAdd(&counts[1], 1);
        dense_rows[pos] = row;
    } else {
        int pos = atomicAdd(&counts[0], 1);
        normal_rows[pos] = row;
    }
}

__global__ void csr_spmm_k32_kernel(const int* __restrict__ row_ptr,
                                    const int* __restrict__ col_ind,
                                    const float* __restrict__ values,
                                    const float* __restrict__ B,
                                    float* __restrict__ C,
                                    const int* __restrict__ normal_rows,
                                    const int* __restrict__ dense_rows,
                                    int normal_count,
                                    int normal_tasks) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int task = blockIdx.x;

    if (task < normal_tasks) {
        int index = task * 4 + warp;
        if (index >= normal_count) return;

        int row = normal_rows[index];
        int begin = row_ptr[row];
        int end = row_ptr[row + 1];
        float sum = 0.0f;

        for (int p = begin; p < end; ++p) {
            int col = __ldg(col_ind + p);
            float a = __ldg(values + p);
            float b = __ldg(B + ((long long)col << 5) + lane);
            sum = fmaf(a, b, sum);
        }
        C[((long long)row << 5) + lane] = sum;
    } else {
        int dense_index = task - normal_tasks;
        int row = dense_rows[dense_index];
        int begin = row_ptr[row];
        int end = row_ptr[row + 1];
        int n = end - begin;
        int local_begin = begin + (n * warp) / 4;
        int local_end = begin + (n * (warp + 1)) / 4;

        float sum = 0.0f;
        for (int p = local_begin; p < local_end; ++p) {
            int col = __ldg(col_ind + p);
            float a = __ldg(values + p);
            float b = __ldg(B + ((long long)col << 5) + lane);
            sum = fmaf(a, b, sum);
        }

        __shared__ float partial[4][32];
        partial[warp][lane] = sum;
        __syncthreads();

        if (warp == 0) {
            float total = partial[0][lane] + partial[1][lane] +
                          partial[2][lane] + partial[3][lane];
            C[((long long)row << 5) + lane] = total;
        }
    }
}

void* spmm_preprocess(const int* d_row_ptr, const int* d_col_ind,
                      const float* d_values,
                      int num_rows, int num_cols, int nnz, int K) {
    (void)d_col_ind;
    (void)d_values;
    (void)num_cols;
    (void)nnz;
    (void)K;

    SpmmContext* ctx = new SpmmContext;
    ctx->d_normal_rows = nullptr;
    ctx->d_dense_rows = nullptr;
    ctx->normal_count = 0;
    ctx->dense_count = 0;
    ctx->normal_tasks = 0;

    if (num_rows <= 0) return ctx;

    cudaMalloc(&ctx->d_normal_rows, (size_t)num_rows * sizeof(int));
    cudaMalloc(&ctx->d_dense_rows, (size_t)num_rows * sizeof(int));

    int* d_counts = nullptr;
    cudaMalloc(&d_counts, 2 * sizeof(int));
    cudaMemset(d_counts, 0, 2 * sizeof(int));

    int threads = 256;
    int blocks = (num_rows + threads - 1) / threads;
    classify_rows_kernel<<<blocks, threads>>>(d_row_ptr, num_rows, 2891,
                                              ctx->d_normal_rows,
                                              ctx->d_dense_rows, d_counts);

    int counts[2] = {0, 0};
    cudaMemcpy(counts, d_counts, 2 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_counts);

    ctx->normal_count = counts[0];
    ctx->dense_count = counts[1];
    ctx->normal_tasks = (ctx->normal_count + 3) / 4;
    return ctx;
}

void spmm_compute(void* context,
                  const int* d_row_ptr, const int* d_col_ind,
                  const float* d_values,
                  const float* d_B, float* d_C,
                  int num_rows, int num_cols, int nnz, int K) {
    (void)num_rows;
    (void)num_cols;
    (void)nnz;

    SpmmContext* ctx = static_cast<SpmmContext*>(context);
    if (K != 32 || ctx == nullptr) return;

    int total_tasks = ctx->normal_tasks + ctx->dense_count;
    if (total_tasks <= 0) return;

    csr_spmm_k32_kernel<<<total_tasks, 128>>>(
        d_row_ptr, d_col_ind, d_values, d_B, d_C,
        ctx->d_normal_rows, ctx->d_dense_rows,
        ctx->normal_count, ctx->normal_tasks);
}

void spmm_cleanup(void* context) {
    SpmmContext* ctx = static_cast<SpmmContext*>(context);
    if (ctx == nullptr) return;
    if (ctx->d_normal_rows) cudaFree(ctx->d_normal_rows);
    if (ctx->d_dense_rows) cudaFree(ctx->d_dense_rows);
    delete ctx;
}