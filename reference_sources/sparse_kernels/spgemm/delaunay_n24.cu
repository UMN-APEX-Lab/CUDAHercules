#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <stdint.h>

struct SpGemmContext {
    int rows, cols, nnz;
    bool rectangular;

    int* bt_row_ptr;
    int* bt_col_ind;
    float* bt_values;

    uint64_t* a_packed;
    uint64_t* b_packed;

    int* upper_bound;
    int* provisional_ptr;
    int* row_nnz;
    int* c_row_ptr;

    int* small_rows;
    int* large_rows;
    int* small_count;
    int* large_count;
    int small_n;
    int large_n;

    int* provisional_col;
    float* provisional_val;
    int* c_col_ind;
    float* c_values;
    int capacity;

    void* scan_temp;
    size_t scan_temp_bytes;
};

static constexpr int THREADS = 256;

__device__ __forceinline__ unsigned hash64(unsigned x) {
    x *= 2654435761u;
    x ^= x >> 16;
    return x & 63u;
}

__device__ __forceinline__ unsigned hash512(unsigned x) {
    x *= 2654435761u;
    x ^= x >> 16;
    return x & 511u;
}

__device__ __forceinline__ int packed_col(uint64_t x) {
    return (int)(x >> 32);
}

__device__ __forceinline__ float packed_val(uint64_t x) {
    return __uint_as_float((unsigned)x);
}

__global__ void pack_entries_kernel(const int* col, const float* val,
                                    uint64_t* packed, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        packed[i] = ((uint64_t)(uint32_t)col[i] << 32) |
                    (uint64_t)__float_as_uint(val[i]);
    }
}

__global__ void transpose_count_kernel(const int* col, int nnz, int* counts) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nnz) atomicAdd(counts + col[i], 1);
}

__global__ void set_last_kernel(int* p, int n, int value) {
    if (blockIdx.x == 0 && threadIdx.x == 0) p[n] = value;
}

__global__ void copy_int_kernel(const int* a, int* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) b[i] = a[i];
}

__global__ void transpose_fill_kernel(const int* rp, const int* ci,
                                      const float* va, int rows,
                                      int* cursor, int* out_ci, float* out_va) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    for (int p = rp[r]; p < rp[r + 1]; ++p) {
        int c = ci[p];
        int dst = atomicAdd(cursor + c, 1);
        out_ci[dst] = r;
        out_va[dst] = va[p];
    }
}

__global__ void upper_bound_kernel(const int* arp, const int* aci,
                                   const int* brp, int rows, int* ub) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    int total = 0;
    for (int p = arp[r]; p < arp[r + 1]; ++p) {
        int k = aci[p];
        total += brp[k + 1] - brp[k];
    }
    ub[r] = total;
}

__global__ void bin_rows_kernel(const int* ub, int rows,
                                int* small, int* large,
                                int* small_count, int* large_count) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    if (ub[r] <= 64) {
        int pos = atomicAdd(small_count, 1);
        small[pos] = r;
    } else {
        int pos = atomicAdd(large_count, 1);
        large[pos] = r;
    }
}

__global__ void small_numeric_kernel(
    const int* row_ids, int nrows,
    const int* arp, const uint64_t* apacked,
    const int* brp, const uint64_t* bpacked,
    const int* provisional_ptr,
    int* provisional_col, float* provisional_val,
    int* row_nnz) {
    __shared__ int keys[8][64];
    __shared__ float vals[8][64];
    __shared__ int counts[8];

    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int item = blockIdx.x * 8 + warp;

    for (int s = lane; s < 64; s += 32) {
        keys[warp][s] = -1;
        vals[warp][s] = 0.0f;
    }
    if (lane == 0) counts[warp] = 0;
    __syncwarp();

    if (item >= nrows) return;

    int row = row_ids[item];
    int ab = arp[row];
    int ae = arp[row + 1];

    for (int ap = ab; ap < ae; ++ap) {
        uint64_t ax = 0;
        if (lane == 0) ax = apacked[ap];
        int k = __shfl_sync(0xffffffff, packed_col(ax), 0);
        float av = __shfl_sync(0xffffffff, packed_val(ax), 0);
        int bb = __shfl_sync(0xffffffff, lane == 0 ? brp[k] : 0, 0);
        int be = __shfl_sync(0xffffffff, lane == 0 ? brp[k + 1] : 0, 0);

        for (int p = bb + lane; p < be; p += 32) {
            uint64_t bx = bpacked[p];
            int col = packed_col(bx);
            float prod = av * packed_val(bx);
            unsigned slot = hash64((unsigned)col);

            #pragma unroll 1
            for (int probe = 0; probe < 64; ++probe) {
                int old = atomicCAS(&keys[warp][slot], -1, col);
                if (old == -1 || old == col) {
                    atomicAdd(&vals[warp][slot], prod);
                    break;
                }
                slot = (slot + 1) & 63;
            }
        }
        __syncwarp();
    }

    int base = provisional_ptr[row];
    for (int s = lane; s < 64; s += 32) {
        int key = keys[warp][s];
        if (key != -1) {
            int out = atomicAdd(&counts[warp], 1);
            provisional_col[base + out] = key;
            provisional_val[base + out] = vals[warp][s];
        }
    }
    __syncwarp();
    if (lane == 0) row_nnz[row] = counts[warp];
}

__global__ void large_numeric_kernel(
    const int* row_ids, int nrows,
    const int* arp, const uint64_t* apacked,
    const int* brp, const uint64_t* bpacked,
    const int* provisional_ptr,
    int* provisional_col, float* provisional_val,
    int* row_nnz) {
    __shared__ int keys[4][512];
    __shared__ float vals[4][512];
    __shared__ int counts[4];

    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int item = blockIdx.x * 4 + warp;

    for (int s = lane; s < 512; s += 32) {
        keys[warp][s] = -1;
        vals[warp][s] = 0.0f;
    }
    if (lane == 0) counts[warp] = 0;
    __syncwarp();

    if (item >= nrows) return;

    int row = row_ids[item];
    int ab = arp[row];
    int ae = arp[row + 1];

    for (int ap = ab; ap < ae; ++ap) {
        uint64_t ax = 0;
        if (lane == 0) ax = apacked[ap];
        int k = __shfl_sync(0xffffffff, packed_col(ax), 0);
        float av = __shfl_sync(0xffffffff, packed_val(ax), 0);
        int bb = __shfl_sync(0xffffffff, lane == 0 ? brp[k] : 0, 0);
        int be = __shfl_sync(0xffffffff, lane == 0 ? brp[k + 1] : 0, 0);

        for (int p = bb + lane; p < be; p += 32) {
            uint64_t bx = bpacked[p];
            int col = packed_col(bx);
            float prod = av * packed_val(bx);
            unsigned slot = hash512((unsigned)col);

            #pragma unroll 1
            for (int probe = 0; probe < 512; ++probe) {
                int old = atomicCAS(&keys[warp][slot], -1, col);
                if (old == -1 || old == col) {
                    atomicAdd(&vals[warp][slot], prod);
                    break;
                }
                slot = (slot + 1) & 511;
            }
        }
        __syncwarp();
    }

    int base = provisional_ptr[row];
    for (int s = lane; s < 512; s += 32) {
        int key = keys[warp][s];
        if (key != -1) {
            int out = atomicAdd(&counts[warp], 1);
            provisional_col[base + out] = key;
            provisional_val[base + out] = vals[warp][s];
        }
    }
    __syncwarp();
    if (lane == 0) row_nnz[row] = counts[warp];
}

__global__ void compact_kernel(
    const int* row_ids, int nrows,
    const int* row_nnz, const int* provisional_ptr,
    const int* c_row_ptr,
    const int* provisional_col, const float* provisional_val,
    int* c_col, float* c_val) {
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int item = blockIdx.x * 8 + warp;
    if (item >= nrows) return;

    int row = row_ids[item];
    int count = row_nnz[row];
    int src = provisional_ptr[row];
    int dst = c_row_ptr[row];

    for (int j = lane; j < count; j += 32) {
        c_col[dst + j] = provisional_col[src + j];
        c_val[dst + j] = provisional_val[src + j];
    }
}

void* spgemm_preprocess(const int* d_row_ptr, const int* d_col_ind,
                        const float* d_values,
                        int num_rows, int num_cols, int nnz) {
    SpGemmContext* ctx = new SpGemmContext{};
    ctx->rows = num_rows;
    ctx->cols = num_cols;
    ctx->nnz = nnz;
    ctx->rectangular = (num_rows != num_cols);

    if (nnz > 0) {
        cudaMalloc(&ctx->a_packed, (size_t)nnz * sizeof(uint64_t));
        pack_entries_kernel<<<(nnz + 255) / 256, 256>>>(
            d_col_ind, d_values, ctx->a_packed, nnz);
    }

    const int* brp = d_row_ptr;

    if (ctx->rectangular) {
        cudaMalloc(&ctx->bt_row_ptr, (size_t)(num_cols + 1) * sizeof(int));
        cudaMemset(ctx->bt_row_ptr, 0, (size_t)(num_cols + 1) * sizeof(int));

        if (nnz > 0) {
            cudaMalloc(&ctx->bt_col_ind, (size_t)nnz * sizeof(int));
            cudaMalloc(&ctx->bt_values, (size_t)nnz * sizeof(float));

            transpose_count_kernel<<<(nnz + 255) / 256, 256>>>(
                d_col_ind, nnz, ctx->bt_row_ptr);
        }

        size_t temp_bytes = 0;
        cub::DeviceScan::ExclusiveSum(nullptr, temp_bytes,
                                      ctx->bt_row_ptr, ctx->bt_row_ptr, num_cols);
        void* temp = nullptr;
        if (temp_bytes) cudaMalloc(&temp, temp_bytes);
        if (num_cols) {
            cub::DeviceScan::ExclusiveSum(temp, temp_bytes,
                                          ctx->bt_row_ptr, ctx->bt_row_ptr, num_cols);
        }
        set_last_kernel<<<1, 1>>>(ctx->bt_row_ptr, num_cols, nnz);

        if (nnz > 0) {
            int* cursor = nullptr;
            cudaMalloc(&cursor, (size_t)num_cols * sizeof(int));
            copy_int_kernel<<<(num_cols + 255) / 256, 256>>>(
                ctx->bt_row_ptr, cursor, num_cols);
            transpose_fill_kernel<<<(num_rows + 255) / 256, 256>>>(
                d_row_ptr, d_col_ind, d_values, num_rows,
                cursor, ctx->bt_col_ind, ctx->bt_values);
            cudaFree(cursor);

            cudaMalloc(&ctx->b_packed, (size_t)nnz * sizeof(uint64_t));
            pack_entries_kernel<<<(nnz + 255) / 256, 256>>>(
                ctx->bt_col_ind, ctx->bt_values, ctx->b_packed, nnz);
        }
        if (temp) cudaFree(temp);
        brp = ctx->bt_row_ptr;
    } else {
        ctx->b_packed = ctx->a_packed;
    }

    cudaMalloc(&ctx->upper_bound, (size_t)num_rows * sizeof(int));
    cudaMalloc(&ctx->provisional_ptr, (size_t)(num_rows + 1) * sizeof(int));
    cudaMalloc(&ctx->row_nnz, (size_t)num_rows * sizeof(int));
    cudaMalloc(&ctx->c_row_ptr, (size_t)(num_rows + 1) * sizeof(int));
    cudaMalloc(&ctx->small_rows, (size_t)num_rows * sizeof(int));
    cudaMalloc(&ctx->large_rows, (size_t)num_rows * sizeof(int));
    cudaMalloc(&ctx->small_count, sizeof(int));
    cudaMalloc(&ctx->large_count, sizeof(int));
    cudaMemset(ctx->small_count, 0, sizeof(int));
    cudaMemset(ctx->large_count, 0, sizeof(int));

    if (num_rows > 0) {
        upper_bound_kernel<<<(num_rows + 255) / 256, 256>>>(
            d_row_ptr, d_col_ind, brp, num_rows, ctx->upper_bound);

        bin_rows_kernel<<<(num_rows + 255) / 256, 256>>>(
            ctx->upper_bound, num_rows,
            ctx->small_rows, ctx->large_rows,
            ctx->small_count, ctx->large_count);

        size_t temp_bytes = 0;
        cub::DeviceScan::ExclusiveSum(nullptr, temp_bytes,
                                      ctx->upper_bound, ctx->provisional_ptr,
                                      num_rows);
        void* temp = nullptr;
        if (temp_bytes) cudaMalloc(&temp, temp_bytes);
        cub::DeviceScan::ExclusiveSum(temp, temp_bytes,
                                      ctx->upper_bound, ctx->provisional_ptr,
                                      num_rows);

        int final_offset = 0;
        int final_ub = 0;
        cudaMemcpy(&final_offset, ctx->provisional_ptr + num_rows - 1,
                   sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&final_ub, ctx->upper_bound + num_rows - 1,
                   sizeof(int), cudaMemcpyDeviceToHost);

        long long total64 = (long long)final_offset + final_ub;
        if (total64 > 0 && total64 <= 2147483647LL) {
            ctx->capacity = (int)total64;
            cudaMalloc(&ctx->provisional_col, (size_t)ctx->capacity * sizeof(int));
            cudaMalloc(&ctx->provisional_val, (size_t)ctx->capacity * sizeof(float));
            cudaMalloc(&ctx->c_col_ind, (size_t)ctx->capacity * sizeof(int));
            cudaMalloc(&ctx->c_values, (size_t)ctx->capacity * sizeof(float));
        }

        set_last_kernel<<<1, 1>>>(ctx->provisional_ptr, num_rows, (int)total64);

        cudaMemcpy(&ctx->small_n, ctx->small_count, sizeof(int),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(&ctx->large_n, ctx->large_count, sizeof(int),
                   cudaMemcpyDeviceToHost);

        if (temp) cudaFree(temp);
    }

    cub::DeviceScan::ExclusiveSum(nullptr, ctx->scan_temp_bytes,
                                  ctx->row_nnz, ctx->c_row_ptr, num_rows);
    if (ctx->scan_temp_bytes) cudaMalloc(&ctx->scan_temp, ctx->scan_temp_bytes);

    return ctx;
}

void spgemm_compute(void* opaque,
                    const int* d_row_ptr, const int* d_col_ind,
                    const float* d_values,
                    int** d_c_row_ptr, int** d_c_col_ind, float** d_c_values,
                    int* c_nnz,
                    int num_rows, int num_cols, int nnz) {
    (void)d_col_ind;
    (void)d_values;
    (void)num_cols;
    (void)nnz;

    SpGemmContext* ctx = (SpGemmContext*)opaque;

    if (num_rows == 0) {
        *d_c_row_ptr = ctx->c_row_ptr;
        *d_c_col_ind = ctx->c_col_ind;
        *d_c_values = ctx->c_values;
        *c_nnz = 0;
        return;
    }

    const int* brp = ctx->rectangular ? ctx->bt_row_ptr : d_row_ptr;

    if (ctx->small_n > 0) {
        small_numeric_kernel<<<(ctx->small_n + 7) / 8, THREADS>>>(
            ctx->small_rows, ctx->small_n,
            d_row_ptr, ctx->a_packed,
            brp, ctx->b_packed,
            ctx->provisional_ptr,
            ctx->provisional_col, ctx->provisional_val,
            ctx->row_nnz);
    }

    if (ctx->large_n > 0) {
        large_numeric_kernel<<<(ctx->large_n + 3) / 4, 128>>>(
            ctx->large_rows, ctx->large_n,
            d_row_ptr, ctx->a_packed,
            brp, ctx->b_packed,
            ctx->provisional_ptr,
            ctx->provisional_col, ctx->provisional_val,
            ctx->row_nnz);
    }

    cub::DeviceScan::ExclusiveSum(ctx->scan_temp, ctx->scan_temp_bytes,
                                  ctx->row_nnz, ctx->c_row_ptr, num_rows);

    int last_offset = 0;
    int last_count = 0;
    cudaMemcpy(&last_offset, ctx->c_row_ptr + num_rows - 1,
               sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&last_count, ctx->row_nnz + num_rows - 1,
               sizeof(int), cudaMemcpyDeviceToHost);

    int total = last_offset + last_count;
    set_last_kernel<<<1, 1>>>(ctx->c_row_ptr, num_rows, total);

    if (ctx->small_n > 0) {
        compact_kernel<<<(ctx->small_n + 7) / 8, THREADS>>>(
            ctx->small_rows, ctx->small_n,
            ctx->row_nnz, ctx->provisional_ptr, ctx->c_row_ptr,
            ctx->provisional_col, ctx->provisional_val,
            ctx->c_col_ind, ctx->c_values);
    }

    if (ctx->large_n > 0) {
        compact_kernel<<<(ctx->large_n + 7) / 8, THREADS>>>(
            ctx->large_rows, ctx->large_n,
            ctx->row_nnz, ctx->provisional_ptr, ctx->c_row_ptr,
            ctx->provisional_col, ctx->provisional_val,
            ctx->c_col_ind, ctx->c_values);
    }

    *d_c_row_ptr = ctx->c_row_ptr;
    *d_c_col_ind = ctx->c_col_ind;
    *d_c_values = ctx->c_values;
    *c_nnz = total;
}

void spgemm_cleanup(void* opaque) {
    SpGemmContext* ctx = (SpGemmContext*)opaque;
    if (!ctx) return;

    if (ctx->bt_row_ptr) cudaFree(ctx->bt_row_ptr);
    if (ctx->bt_col_ind) cudaFree(ctx->bt_col_ind);
    if (ctx->bt_values) cudaFree(ctx->bt_values);

    if (ctx->a_packed) cudaFree(ctx->a_packed);
    if (ctx->rectangular && ctx->b_packed) cudaFree(ctx->b_packed);

    if (ctx->upper_bound) cudaFree(ctx->upper_bound);
    if (ctx->provisional_ptr) cudaFree(ctx->provisional_ptr);
    if (ctx->row_nnz) cudaFree(ctx->row_nnz);
    if (ctx->c_row_ptr) cudaFree(ctx->c_row_ptr);

    if (ctx->small_rows) cudaFree(ctx->small_rows);
    if (ctx->large_rows) cudaFree(ctx->large_rows);
    if (ctx->small_count) cudaFree(ctx->small_count);
    if (ctx->large_count) cudaFree(ctx->large_count);

    if (ctx->provisional_col) cudaFree(ctx->provisional_col);
    if (ctx->provisional_val) cudaFree(ctx->provisional_val);
    if (ctx->c_col_ind) cudaFree(ctx->c_col_ind);
    if (ctx->c_values) cudaFree(ctx->c_values);

    if (ctx->scan_temp) cudaFree(ctx->scan_temp);

    delete ctx;
}