#include <cuda_runtime.h>
#include <stdint.h>

struct DenseChunk {
    int row;
    int begin;
    int end;
};

struct SpmmContext {
    int* d_short_rows;
    int* d_regular_rows;
    int* d_dense_rows;
    DenseChunk* d_dense_chunks;
    int short_count;
    int regular_count;
    int dense_count;
    int dense_chunk_count;
};

__global__ void spmm_short_kernel(const int* __restrict__ row_ptr,
                                  const int* __restrict__ col_ind,
                                  const float* __restrict__ values,
                                  const int* __restrict__ rows,
                                  int count,
                                  const float* __restrict__ B,
                                  float* __restrict__ C) {
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int item = blockIdx.x * 4 + warp;
    if (item >= count) return;

    int row = rows[item];
    int begin = row_ptr[row];
    int end = row_ptr[row + 1];

    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    float s4 = 0.0f, s5 = 0.0f, s6 = 0.0f, s7 = 0.0f;

    for (int p = begin; p < end; ++p) {
        int col = 0;
        float a = 0.0f;
        if (lane == 0) {
            col = col_ind[p];
            a = values[p];
        }
        col = __shfl_sync(0xffffffff, col, 0);
        a = __shfl_sync(0xffffffff, a, 0);

        const float* bp = B + ((size_t)col << 8) + lane;
        s0 = fmaf(a, bp[0], s0);
        s1 = fmaf(a, bp[32], s1);
        s2 = fmaf(a, bp[64], s2);
        s3 = fmaf(a, bp[96], s3);
        s4 = fmaf(a, bp[128], s4);
        s5 = fmaf(a, bp[160], s5);
        s6 = fmaf(a, bp[192], s6);
        s7 = fmaf(a, bp[224], s7);
    }

    float* cp = C + ((size_t)row << 8) + lane;
    cp[0] = s0;
    cp[32] = s1;
    cp[64] = s2;
    cp[96] = s3;
    cp[128] = s4;
    cp[160] = s5;
    cp[192] = s6;
    cp[224] = s7;
}

__global__ void spmm_regular_kernel(const int* __restrict__ row_ptr,
                                    const int* __restrict__ col_ind,
                                    const float* __restrict__ values,
                                    const int* __restrict__ rows,
                                    int count,
                                    const float* __restrict__ B,
                                    float* __restrict__ C) {
    int item = blockIdx.x;
    if (item >= count) return;

    int tid = threadIdx.x;
    int row = rows[item];
    int begin = row_ptr[row];
    int end = row_ptr[row + 1];

    __shared__ int s_col[128];
    __shared__ float s_val[128];

    float x00 = 0.0f, x01 = 0.0f, x02 = 0.0f, x03 = 0.0f;
    float x10 = 0.0f, x11 = 0.0f, x12 = 0.0f, x13 = 0.0f;

    for (int base = begin; base < end; base += 128) {
        int p = base + tid;
        if (p < end) {
            s_col[tid] = col_ind[p];
            s_val[tid] = values[p];
        }
        __syncthreads();

        int n = end - base;
        if (n > 128) n = 128;
        int full = n & ~3;

        for (int j = 0; j < full; j += 4) {
            float a0 = s_val[j];
            float a1 = s_val[j + 1];
            float a2 = s_val[j + 2];
            float a3 = s_val[j + 3];

            const float* b0 = B + ((size_t)s_col[j] << 8) + tid;
            const float* b1 = B + ((size_t)s_col[j + 1] << 8) + tid;
            const float* b2 = B + ((size_t)s_col[j + 2] << 8) + tid;
            const float* b3 = B + ((size_t)s_col[j + 3] << 8) + tid;

            x00 = fmaf(a0, b0[0], x00);
            x01 = fmaf(a1, b1[0], x01);
            x02 = fmaf(a2, b2[0], x02);
            x03 = fmaf(a3, b3[0], x03);

            x10 = fmaf(a0, b0[128], x10);
            x11 = fmaf(a1, b1[128], x11);
            x12 = fmaf(a2, b2[128], x12);
            x13 = fmaf(a3, b3[128], x13);
        }

        for (int j = full; j < n; ++j) {
            float a = s_val[j];
            const float* bp = B + ((size_t)s_col[j] << 8) + tid;
            x00 = fmaf(a, bp[0], x00);
            x10 = fmaf(a, bp[128], x10);
        }
        __syncthreads();
    }

    float* cp = C + ((size_t)row << 8) + tid;
    cp[0] = (x00 + x01) + (x02 + x03);
    cp[128] = (x10 + x11) + (x12 + x13);
}

__global__ void clear_dense_rows_kernel(const int* __restrict__ rows,
                                        int count,
                                        float* __restrict__ C) {
    int item = blockIdx.x;
    int k = threadIdx.x;
    if (item < count && k < 256) {
        int row = rows[item];
        C[((size_t)row << 8) + k] = 0.0f;
    }
}

__global__ void spmm_dense_kernel(const int* __restrict__ col_ind,
                                  const float* __restrict__ values,
                                  const DenseChunk* __restrict__ chunks,
                                  int chunk_count,
                                  const float* __restrict__ B,
                                  float* __restrict__ C) {
    int item = blockIdx.x;
    if (item >= chunk_count) return;

    int tid = threadIdx.x;
    DenseChunk desc = chunks[item];

    __shared__ int s_col[128];
    __shared__ float s_val[128];

    float x00 = 0.0f, x01 = 0.0f, x02 = 0.0f, x03 = 0.0f;
    float x10 = 0.0f, x11 = 0.0f, x12 = 0.0f, x13 = 0.0f;

    for (int base = desc.begin; base < desc.end; base += 128) {
        int p = base + tid;
        if (p < desc.end) {
            s_col[tid] = col_ind[p];
            s_val[tid] = values[p];
        }
        __syncthreads();

        int n = desc.end - base;
        if (n > 128) n = 128;
        int full = n & ~3;

        for (int j = 0; j < full; j += 4) {
            float a0 = s_val[j];
            float a1 = s_val[j + 1];
            float a2 = s_val[j + 2];
            float a3 = s_val[j + 3];

            const float* b0 = B + ((size_t)s_col[j] << 8) + tid;
            const float* b1 = B + ((size_t)s_col[j + 1] << 8) + tid;
            const float* b2 = B + ((size_t)s_col[j + 2] << 8) + tid;
            const float* b3 = B + ((size_t)s_col[j + 3] << 8) + tid;

            x00 = fmaf(a0, b0[0], x00);
            x01 = fmaf(a1, b1[0], x01);
            x02 = fmaf(a2, b2[0], x02);
            x03 = fmaf(a3, b3[0], x03);

            x10 = fmaf(a0, b0[128], x10);
            x11 = fmaf(a1, b1[128], x11);
            x12 = fmaf(a2, b2[128], x12);
            x13 = fmaf(a3, b3[128], x13);
        }

        for (int j = full; j < n; ++j) {
            float a = s_val[j];
            const float* bp = B + ((size_t)s_col[j] << 8) + tid;
            x00 = fmaf(a, bp[0], x00);
            x10 = fmaf(a, bp[128], x10);
        }
        __syncthreads();
    }

    float* cp = C + ((size_t)desc.row << 8) + tid;
    atomicAdd(cp, (x00 + x01) + (x02 + x03));
    atomicAdd(cp + 128, (x10 + x11) + (x12 + x13));
}

void* spmm_preprocess(const int* d_row_ptr, const int* d_col_ind,
                      const float* d_values,
                      int num_rows, int num_cols, int nnz, int K) {
    (void)d_col_ind;
    (void)d_values;
    (void)num_cols;
    (void)nnz;
    (void)K;

    SpmmContext* ctx = new SpmmContext{};
    if (num_rows <= 0) return ctx;

    int* h_ptr = new int[(size_t)num_rows + 1];
    cudaMemcpy(h_ptr, d_row_ptr, ((size_t)num_rows + 1) * sizeof(int),
               cudaMemcpyDeviceToHost);

    int* h_short = new int[num_rows];
    int* h_regular = new int[num_rows];
    int* h_dense = new int[num_rows];

    const int short_limit = 32;
    const int dense_limit = 3212;
    const int chunk_nnz = 1024;

    int max_chunks = 0;
    for (int r = 0; r < num_rows; ++r) {
        int len = h_ptr[r + 1] - h_ptr[r];
        if (len > dense_limit) max_chunks += (len + chunk_nnz - 1) / chunk_nnz;
    }

    DenseChunk* h_chunks = max_chunks ? new DenseChunk[max_chunks] : nullptr;

    for (int r = 0; r < num_rows; ++r) {
        int begin = h_ptr[r];
        int end = h_ptr[r + 1];
        int len = end - begin;

        if (len < short_limit) {
            h_short[ctx->short_count++] = r;
        } else if (len <= dense_limit) {
            h_regular[ctx->regular_count++] = r;
        } else {
            h_dense[ctx->dense_count++] = r;
            for (int p = begin; p < end; p += chunk_nnz) {
                DenseChunk d;
                d.row = r;
                d.begin = p;
                d.end = (p + chunk_nnz < end) ? p + chunk_nnz : end;
                h_chunks[ctx->dense_chunk_count++] = d;
            }
        }
    }

    if (ctx->short_count) {
        cudaMalloc(&ctx->d_short_rows, (size_t)ctx->short_count * sizeof(int));
        cudaMemcpy(ctx->d_short_rows, h_short,
                   (size_t)ctx->short_count * sizeof(int),
                   cudaMemcpyHostToDevice);
    }

    if (ctx->regular_count) {
        cudaMalloc(&ctx->d_regular_rows, (size_t)ctx->regular_count * sizeof(int));
        cudaMemcpy(ctx->d_regular_rows, h_regular,
                   (size_t)ctx->regular_count * sizeof(int),
                   cudaMemcpyHostToDevice);
    }

    if (ctx->dense_count) {
        cudaMalloc(&ctx->d_dense_rows, (size_t)ctx->dense_count * sizeof(int));
        cudaMemcpy(ctx->d_dense_rows, h_dense,
                   (size_t)ctx->dense_count * sizeof(int),
                   cudaMemcpyHostToDevice);
    }

    if (ctx->dense_chunk_count) {
        cudaMalloc(&ctx->d_dense_chunks,
                   (size_t)ctx->dense_chunk_count * sizeof(DenseChunk));
        cudaMemcpy(ctx->d_dense_chunks, h_chunks,
                   (size_t)ctx->dense_chunk_count * sizeof(DenseChunk),
                   cudaMemcpyHostToDevice);
    }

    delete[] h_ptr;
    delete[] h_short;
    delete[] h_regular;
    delete[] h_dense;
    delete[] h_chunks;

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
    if (!ctx || K != 256) return;

    if (ctx->short_count) {
        spmm_short_kernel<<<(ctx->short_count + 3) / 4, 128>>>(
            d_row_ptr, d_col_ind, d_values,
            ctx->d_short_rows, ctx->short_count, d_B, d_C);
    }

    if (ctx->regular_count) {
        spmm_regular_kernel<<<ctx->regular_count, 128>>>(
            d_row_ptr, d_col_ind, d_values,
            ctx->d_regular_rows, ctx->regular_count, d_B, d_C);
    }

    if (ctx->dense_count) {
        clear_dense_rows_kernel<<<ctx->dense_count, 256>>>(
            ctx->d_dense_rows, ctx->dense_count, d_C);

        if (ctx->dense_chunk_count) {
            spmm_dense_kernel<<<ctx->dense_chunk_count, 128>>>(
                d_col_ind, d_values, ctx->d_dense_chunks,
                ctx->dense_chunk_count, d_B, d_C);
        }
    }
}

void spmm_cleanup(void* context) {
    SpmmContext* ctx = static_cast<SpmmContext*>(context);
    if (!ctx) return;

    if (ctx->d_short_rows) cudaFree(ctx->d_short_rows);
    if (ctx->d_regular_rows) cudaFree(ctx->d_regular_rows);
    if (ctx->d_dense_rows) cudaFree(ctx->d_dense_rows);
    if (ctx->d_dense_chunks) cudaFree(ctx->d_dense_chunks);

    delete ctx;
}