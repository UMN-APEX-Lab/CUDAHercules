#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <stddef.h>
#include <stdint.h>

struct SpGemmContext {
    int rows, cols, nnz;
    const int* b_rp;
    const int* b_ci;
    const float* b_va;

    int* bt_rp;
    int* bt_ci;
    float* bt_va;

    int* tiny_rows;
    int* small_rows;
    int* overflow_rows;
    int tiny_count;
    int small_count;
    int overflow_count;

    int* counts;
    int* c_rp;
    int* c_ci;
    float* c_va;
    size_t capacity;

    void* scan_temp;
    size_t scan_bytes;
};

__global__ static void write_int(int* p, int v) {
    if (blockIdx.x == 0 && threadIdx.x == 0) *p = v;
}

__global__ static void make_transpose_counts(const int* ci, int nnz, int* counts) {
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (tid < nnz) atomicAdd(counts + ci[tid], 1);
}

__global__ static void scatter_transpose(
    const int* rp, const int* ci, const float* va, int rows,
    int* offsets, int* out_ci, float* out_va)
{
    int r = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (r >= rows) return;
    for (int p = rp[r]; p < rp[r + 1]; ++p) {
        int c = ci[p];
        int dst = atomicAdd(offsets + c, 1);
        out_ci[dst] = r;
        out_va[dst] = va[p];
    }
}

__global__ static void classify_rows(
    const int* arp, const int* aci, const int* brp, int rows,
    int* tiny, int* small, int* overflow,
    int* tiny_count, int* small_count, int* overflow_count,
    unsigned long long* total_ub)
{
    int r = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (r >= rows) return;

    unsigned int ub = 0;
    for (int p = arp[r]; p < arp[r + 1]; ++p) {
        int k = aci[p];
        ub += (unsigned int)(brp[k + 1] - brp[k]);
    }

    atomicAdd(total_ub, (unsigned long long)ub);

    if (ub <= 8) {
        int pos = atomicAdd(tiny_count, 1);
        tiny[pos] = r;
    } else if (ub <= 36) {
        int pos = atomicAdd(small_count, 1);
        small[pos] = r;
    } else {
        int pos = atomicAdd(overflow_count, 1);
        overflow[pos] = r;
    }
}

template<int CAP>
__global__ static void symbolic_register(
    const int* rows_list, int n,
    const int* arp, const int* aci,
    const int* brp, const int* bci,
    int* counts)
{
    int q = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (q >= n) return;

    int row = rows_list[q];
    int keys[CAP];
    int count = 0;

    for (int p = arp[row]; p < arp[row + 1]; ++p) {
        int k = aci[p];
        int begin = brp[k];
        int end = brp[k + 1];

        for (int x = begin; x < end; ++x) {
            int col = bci[x];
            int found = 0;
#pragma unroll
            for (int t = 0; t < CAP; ++t) {
                if (t < count && keys[t] == col) found = 1;
            }
            if (!found) {
                if (count < CAP) keys[count++] = col;
            }
        }
    }
    counts[row] = count;
}

template<int CAP>
__global__ static void numeric_register(
    const int* rows_list, int n,
    const int* arp, const int* aci, const float* ava,
    const int* brp, const int* bci, const float* bva,
    const int* crp, int* cci, float* cva)
{
    int q = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (q >= n) return;

    int row = rows_list[q];
    int keys[CAP];
    float vals[CAP];
    int count = 0;

    for (int p = arp[row]; p < arp[row + 1]; ++p) {
        int k = aci[p];
        float av = ava[p];
        int begin = brp[k];
        int end = brp[k + 1];

        for (int x = begin; x < end; ++x) {
            int col = bci[x];
            float prod = av * bva[x];
            int hit = -1;
#pragma unroll
            for (int t = 0; t < CAP; ++t) {
                if (t < count && keys[t] == col) hit = t;
            }
            if (hit >= 0) {
                vals[hit] += prod;
            } else if (count < CAP) {
                keys[count] = col;
                vals[count] = prod;
                ++count;
            }
        }
    }

    int out = crp[row];
#pragma unroll
    for (int t = 0; t < CAP; ++t) {
        if (t < count) {
            cci[out + t] = keys[t];
            cva[out + t] = vals[t];
        }
    }
}

__global__ static void symbolic_overflow(
    const int* list, int n,
    const int* arp, const int* aci,
    const int* brp, const int* bci,
    int* counts)
{
    int q = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (q >= n) return;

    int row = list[q];
    int a0 = arp[row];
    int a1 = arp[row + 1];
    int unique = 0;

    for (int p = a0; p < a1; ++p) {
        int k = aci[p];
        for (int x = brp[k]; x < brp[k + 1]; ++x) {
            int col = bci[x];
            int seen = 0;
            for (int pp = a0; pp <= p && !seen; ++pp) {
                int kk = aci[pp];
                int stop = (pp == p) ? x : brp[kk + 1];
                for (int xx = brp[kk]; xx < stop; ++xx) {
                    if (bci[xx] == col) {
                        seen = 1;
                        break;
                    }
                }
            }
            if (!seen) ++unique;
        }
    }
    counts[row] = unique;
}

__global__ static void numeric_overflow(
    const int* list, int n,
    const int* arp, const int* aci, const float* ava,
    const int* brp, const int* bci, const float* bva,
    const int* crp, int* cci, float* cva)
{
    int q = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (q >= n) return;

    int row = list[q];
    int a0 = arp[row];
    int a1 = arp[row + 1];
    int out = crp[row];
    int written = 0;

    for (int p = a0; p < a1; ++p) {
        int k = aci[p];
        for (int x = brp[k]; x < brp[k + 1]; ++x) {
            int col = bci[x];
            int seen = 0;

            for (int pp = a0; pp <= p && !seen; ++pp) {
                int kk = aci[pp];
                int stop = (pp == p) ? x : brp[kk + 1];
                for (int xx = brp[kk]; xx < stop; ++xx) {
                    if (bci[xx] == col) {
                        seen = 1;
                        break;
                    }
                }
            }

            if (!seen) {
                float sum = 0.0f;
                for (int pp = a0; pp < a1; ++pp) {
                    int kk = aci[pp];
                    float av = ava[pp];
                    for (int xx = brp[kk]; xx < brp[kk + 1]; ++xx) {
                        if (bci[xx] == col) sum += av * bva[xx];
                    }
                }
                cci[out + written] = col;
                cva[out + written] = sum;
                ++written;
            }
        }
    }
}

void* spgemm_preprocess(
    const int* d_row_ptr, const int* d_col_ind,
    const float* d_values,
    int num_rows, int num_cols, int nnz)
{
    SpGemmContext* ctx = new SpGemmContext();
    *ctx = {};

    ctx->rows = num_rows;
    ctx->cols = num_cols;
    ctx->nnz = nnz;

    if (num_rows == num_cols) {
        ctx->b_rp = d_row_ptr;
        ctx->b_ci = d_col_ind;
        ctx->b_va = d_values;
    } else {
        cudaMalloc(&ctx->bt_rp, (size_t)(num_cols + 1) * sizeof(int));

        if (nnz > 0) {
            cudaMalloc(&ctx->bt_ci, (size_t)nnz * sizeof(int));
            cudaMalloc(&ctx->bt_va, (size_t)nnz * sizeof(float));
        }

        if (num_cols > 0) {
            int* col_counts = nullptr;
            int* offsets = nullptr;
            cudaMalloc(&col_counts, (size_t)num_cols * sizeof(int));
            cudaMalloc(&offsets, (size_t)num_cols * sizeof(int));
            cudaMemset(col_counts, 0, (size_t)num_cols * sizeof(int));

            const int threads = 256;
            if (nnz > 0) {
                make_transpose_counts<<<(nnz + threads - 1) / threads, threads>>>(
                    d_col_ind, nnz, col_counts);
            }

            size_t scan_bytes = 0;
            cub::DeviceScan::ExclusiveSum(
                nullptr, scan_bytes, col_counts, ctx->bt_rp, num_cols);

            void* scan_temp = nullptr;
            if (scan_bytes) cudaMalloc(&scan_temp, scan_bytes);

            cub::DeviceScan::ExclusiveSum(
                scan_temp, scan_bytes, col_counts, ctx->bt_rp, num_cols);

            write_int<<<1, 1>>>(ctx->bt_rp + num_cols, nnz);

            cudaMemcpy(offsets, ctx->bt_rp,
                       (size_t)num_cols * sizeof(int),
                       cudaMemcpyDeviceToDevice);

            if (nnz > 0) {
                scatter_transpose<<<(num_rows + threads - 1) / threads, threads>>>(
                    d_row_ptr, d_col_ind, d_values, num_rows,
                    offsets, ctx->bt_ci, ctx->bt_va);
            }

            if (scan_temp) cudaFree(scan_temp);
            cudaFree(col_counts);
            cudaFree(offsets);
        } else {
            write_int<<<1, 1>>>(ctx->bt_rp, 0);
        }

        ctx->b_rp = ctx->bt_rp;
        ctx->b_ci = ctx->bt_ci;
        ctx->b_va = ctx->bt_va;
    }

    cudaMalloc(&ctx->c_rp, (size_t)(num_rows + 1) * sizeof(int));

    if (num_rows == 0) {
        write_int<<<1, 1>>>(ctx->c_rp, 0);
        return ctx;
    }

    cudaMalloc(&ctx->tiny_rows, (size_t)num_rows * sizeof(int));
    cudaMalloc(&ctx->small_rows, (size_t)num_rows * sizeof(int));
    cudaMalloc(&ctx->overflow_rows, (size_t)num_rows * sizeof(int));
    cudaMalloc(&ctx->counts, (size_t)num_rows * sizeof(int));

    int* d_tiny_count = nullptr;
    int* d_small_count = nullptr;
    int* d_overflow_count = nullptr;
    unsigned long long* d_total = nullptr;

    cudaMalloc(&d_tiny_count, sizeof(int));
    cudaMalloc(&d_small_count, sizeof(int));
    cudaMalloc(&d_overflow_count, sizeof(int));
    cudaMalloc(&d_total, sizeof(unsigned long long));

    cudaMemset(d_tiny_count, 0, sizeof(int));
    cudaMemset(d_small_count, 0, sizeof(int));
    cudaMemset(d_overflow_count, 0, sizeof(int));
    cudaMemset(d_total, 0, sizeof(unsigned long long));

    const int threads = 256;
    classify_rows<<<(num_rows + threads - 1) / threads, threads>>>(
        d_row_ptr, d_col_ind, ctx->b_rp, num_rows,
        ctx->tiny_rows, ctx->small_rows, ctx->overflow_rows,
        d_tiny_count, d_small_count, d_overflow_count, d_total);

    unsigned long long total_ub = 0;
    cudaMemcpy(&ctx->tiny_count, d_tiny_count, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&ctx->small_count, d_small_count, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&ctx->overflow_count, d_overflow_count, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&total_ub, d_total, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    cudaFree(d_tiny_count);
    cudaFree(d_small_count);
    cudaFree(d_overflow_count);
    cudaFree(d_total);

    ctx->capacity = (size_t)total_ub;
    if (ctx->capacity > 0) {
        cudaMalloc(&ctx->c_ci, ctx->capacity * sizeof(int));
        cudaMalloc(&ctx->c_va, ctx->capacity * sizeof(float));
    }

    cub::DeviceScan::ExclusiveSum(
        nullptr, ctx->scan_bytes, ctx->counts, ctx->c_rp, num_rows);
    if (ctx->scan_bytes) cudaMalloc(&ctx->scan_temp, ctx->scan_bytes);

    return ctx;
}

void spgemm_compute(
    void* opaque,
    const int* d_row_ptr, const int* d_col_ind,
    const float* d_values,
    int** d_c_row_ptr, int** d_c_col_ind, float** d_c_values,
    int* c_nnz,
    int num_rows, int num_cols, int nnz)
{
    (void)num_rows;
    (void)num_cols;
    (void)nnz;

    SpGemmContext* ctx = (SpGemmContext*)opaque;

    if (ctx->rows == 0) {
        write_int<<<1, 1>>>(ctx->c_rp, 0);
        *d_c_row_ptr = ctx->c_rp;
        *d_c_col_ind = ctx->c_ci;
        *d_c_values = ctx->c_va;
        *c_nnz = 0;
        return;
    }

    if (ctx->tiny_count > 0) {
        const int threads = 256;
        symbolic_register<8><<<(ctx->tiny_count + threads - 1) / threads, threads>>>(
            ctx->tiny_rows, ctx->tiny_count,
            d_row_ptr, d_col_ind, ctx->b_rp, ctx->b_ci, ctx->counts);
    }

    if (ctx->small_count > 0) {
        const int threads = 128;
        symbolic_register<36><<<(ctx->small_count + threads - 1) / threads, threads>>>(
            ctx->small_rows, ctx->small_count,
            d_row_ptr, d_col_ind, ctx->b_rp, ctx->b_ci, ctx->counts);
    }

    if (ctx->overflow_count > 0) {
        const int threads = 128;
        symbolic_overflow<<<(ctx->overflow_count + threads - 1) / threads, threads>>>(
            ctx->overflow_rows, ctx->overflow_count,
            d_row_ptr, d_col_ind, ctx->b_rp, ctx->b_ci, ctx->counts);
    }

    cub::DeviceScan::ExclusiveSum(
        ctx->scan_temp, ctx->scan_bytes, ctx->counts, ctx->c_rp, ctx->rows);

    int prefix = 0;
    int last_count = 0;
    cudaMemcpy(&prefix, ctx->c_rp + ctx->rows - 1, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&last_count, ctx->counts + ctx->rows - 1, sizeof(int), cudaMemcpyDeviceToHost);

    int total = prefix + last_count;
    cudaMemcpy(ctx->c_rp + ctx->rows, &total, sizeof(int), cudaMemcpyHostToDevice);

    if (ctx->tiny_count > 0) {
        const int threads = 256;
        numeric_register<8><<<(ctx->tiny_count + threads - 1) / threads, threads>>>(
            ctx->tiny_rows, ctx->tiny_count,
            d_row_ptr, d_col_ind, d_values,
            ctx->b_rp, ctx->b_ci, ctx->b_va,
            ctx->c_rp, ctx->c_ci, ctx->c_va);
    }

    if (ctx->small_count > 0) {
        const int threads = 128;
        numeric_register<36><<<(ctx->small_count + threads - 1) / threads, threads>>>(
            ctx->small_rows, ctx->small_count,
            d_row_ptr, d_col_ind, d_values,
            ctx->b_rp, ctx->b_ci, ctx->b_va,
            ctx->c_rp, ctx->c_ci, ctx->c_va);
    }

    if (ctx->overflow_count > 0) {
        const int threads = 128;
        numeric_overflow<<<(ctx->overflow_count + threads - 1) / threads, threads>>>(
            ctx->overflow_rows, ctx->overflow_count,
            d_row_ptr, d_col_ind, d_values,
            ctx->b_rp, ctx->b_ci, ctx->b_va,
            ctx->c_rp, ctx->c_ci, ctx->c_va);
    }

    *d_c_row_ptr = ctx->c_rp;
    *d_c_col_ind = ctx->c_ci;
    *d_c_values = ctx->c_va;
    *c_nnz = total;
}

void spgemm_cleanup(void* opaque)
{
    SpGemmContext* ctx = (SpGemmContext*)opaque;
    if (!ctx) return;

    cudaFree(ctx->tiny_rows);
    cudaFree(ctx->small_rows);
    cudaFree(ctx->overflow_rows);
    cudaFree(ctx->counts);

    cudaFree(ctx->c_rp);
    cudaFree(ctx->c_ci);
    cudaFree(ctx->c_va);
    cudaFree(ctx->scan_temp);

    cudaFree(ctx->bt_rp);
    cudaFree(ctx->bt_ci);
    cudaFree(ctx->bt_va);

    delete ctx;
}