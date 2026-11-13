#include <cuda_runtime.h>
#include <vector>
#include "sparse_formats.cuh"

struct TaskDesc {
    int begin;
    int end;
    int row;
    int partial;
};

struct SpMMContext {
    COOData coo;
    TaskDesc* d_tasks;
    int num_tasks;
    int* d_partial_rows;
    int num_partial_rows;
};

__global__ __launch_bounds__(128, 8)
void clear_partial_rows_kernel(const int* __restrict__ rows,
                               int count,
                               float* __restrict__ C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = count * 256;
    if (idx < total) {
        int r = rows[idx >> 8];
        C[r * 256 + (idx & 255)] = 0.0f;
    }
}

__global__ __launch_bounds__(128, 8)
void coo_spmm_k256_kernel(const TaskDesc* __restrict__ tasks,
                           int num_tasks,
                           const int* __restrict__ col_ind,
                           const float* __restrict__ values,
                           const float* __restrict__ B,
                           float* __restrict__ C) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int task_id = (int)blockIdx.x * 4 + warp;

    if (task_id >= num_tasks) return;

    int begin = 0, end = 0, row = 0, partial = 0;
    if (lane == 0) {
        TaskDesc t = tasks[task_id];
        begin = t.begin;
        end = t.end;
        row = t.row;
        partial = t.partial;
    }

    begin = __shfl_sync(0xffffffff, begin, 0);
    end = __shfl_sync(0xffffffff, end, 0);
    row = __shfl_sync(0xffffffff, row, 0);
    partial = __shfl_sync(0xffffffff, partial, 0);

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
        col = __shfl_sync(0xffffffff, col, 0);
        av = __shfl_sync(0xffffffff, av, 0);

        const float* bp = B + ((size_t)col << 8) + lane;
        s0 = fmaf(av, bp[0], s0);
        s1 = fmaf(av, bp[32], s1);
        s2 = fmaf(av, bp[64], s2);
        s3 = fmaf(av, bp[96], s3);
        s4 = fmaf(av, bp[128], s4);
        s5 = fmaf(av, bp[160], s5);
        s6 = fmaf(av, bp[192], s6);
        s7 = fmaf(av, bp[224], s7);
    }

    float* cp = C + ((size_t)row << 8) + lane;
    if (partial) {
        atomicAdd(cp + 0, s0);
        atomicAdd(cp + 32, s1);
        atomicAdd(cp + 64, s2);
        atomicAdd(cp + 96, s3);
        atomicAdd(cp + 128, s4);
        atomicAdd(cp + 160, s5);
        atomicAdd(cp + 192, s6);
        atomicAdd(cp + 224, s7);
    } else {
        cp[0] = s0;
        cp[32] = s1;
        cp[64] = s2;
        cp[96] = s3;
        cp[128] = s4;
        cp[160] = s5;
        cp[192] = s6;
        cp[224] = s7;
    }
}

void* spmm_preprocess(const int* d_row_ptr, const int* d_col_ind,
                      const float* d_values,
                      int num_rows, int num_cols, int nnz, int K) {
    SpMMContext* ctx = new SpMMContext();
    ctx->d_tasks = nullptr;
    ctx->num_tasks = 0;
    ctx->d_partial_rows = nullptr;
    ctx->num_partial_rows = 0;

    ctx->coo = csr_to_coo(d_row_ptr, d_col_ind, d_values,
                          num_rows, num_cols, nnz);

    std::vector<int> h_row_ptr((size_t)num_rows + 1);
    cudaMemcpy(h_row_ptr.data(), d_row_ptr,
               ((size_t)num_rows + 1) * sizeof(int),
               cudaMemcpyDeviceToHost);

    std::vector<TaskDesc> h_tasks;
    std::vector<int> h_partial_rows;
    h_tasks.reserve((size_t)num_rows + 1024);

    const int segment_nnz = 256;
    for (int r = 0; r < num_rows; ++r) {
        int begin = h_row_ptr[r];
        int end = h_row_ptr[r + 1];
        int len = end - begin;

        if (len > segment_nnz) {
            h_partial_rows.push_back(r);
            for (int p = begin; p < end; p += segment_nnz) {
                TaskDesc t;
                t.begin = p;
                t.end = (p + segment_nnz < end) ? (p + segment_nnz) : end;
                t.row = r;
                t.partial = 1;
                h_tasks.push_back(t);
            }
        } else {
            TaskDesc t;
            t.begin = begin;
            t.end = end;
            t.row = r;
            t.partial = 0;
            h_tasks.push_back(t);
        }
    }

    ctx->num_tasks = (int)h_tasks.size();
    ctx->num_partial_rows = (int)h_partial_rows.size();

    if (ctx->num_tasks > 0) {
        cudaMalloc(&ctx->d_tasks, (size_t)ctx->num_tasks * sizeof(TaskDesc));
        cudaMemcpy(ctx->d_tasks, h_tasks.data(),
                   (size_t)ctx->num_tasks * sizeof(TaskDesc),
                   cudaMemcpyHostToDevice);
    }

    if (ctx->num_partial_rows > 0) {
        cudaMalloc(&ctx->d_partial_rows,
                   (size_t)ctx->num_partial_rows * sizeof(int));
        cudaMemcpy(ctx->d_partial_rows, h_partial_rows.data(),
                   (size_t)ctx->num_partial_rows * sizeof(int),
                   cudaMemcpyHostToDevice);
    }

    return ctx;
}

void spmm_compute(void* context,
                  const int* d_row_ptr, const int* d_col_ind,
                  const float* d_values,
                  const float* d_B, float* d_C,
                  int num_rows, int num_cols, int nnz, int K) {
    SpMMContext* ctx = static_cast<SpMMContext*>(context);
    if (ctx == nullptr || ctx->num_tasks == 0) return;

    if (ctx->num_partial_rows > 0) {
        int total = ctx->num_partial_rows * 256;
        int blocks = (total + 127) >> 7;
        clear_partial_rows_kernel<<<blocks, 128>>>(ctx->d_partial_rows,
                                                    ctx->num_partial_rows,
                                                    d_C);
    }

    int blocks = (ctx->num_tasks + 3) >> 2;
    coo_spmm_k256_kernel<<<blocks, 128>>>(ctx->d_tasks,
                                           ctx->num_tasks,
                                           ctx->coo.d_col_ind,
                                           ctx->coo.d_values,
                                           d_B,
                                           d_C);
}

void spmm_cleanup(void* context) {
    SpMMContext* ctx = static_cast<SpMMContext*>(context);
    if (ctx == nullptr) return;
    if (ctx->d_tasks) cudaFree(ctx->d_tasks);
    if (ctx->d_partial_rows) cudaFree(ctx->d_partial_rows);
    free_coo(ctx->coo);
    delete ctx;
}