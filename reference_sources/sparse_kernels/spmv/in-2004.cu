#include <cuda_runtime.h>
#include <stdint.h>
#include "sparse_formats.cuh"

struct SpmvContext {
    Tiled2DData tiled;
    int* d_cursor;
    int num_rows;
    int num_cols;
};

__device__ __forceinline__ float ro_load(const float* p) {
    return __ldg(p);
}

__global__ __launch_bounds__(256, 6)
void tiled2d_persistent_spmv_kernel(
    const int* __restrict__ blk_row,
    const int* __restrict__ blk_col,
    const int* __restrict__ blk_nnz,
    const int* __restrict__ blk_type,
    const int64_t* __restrict__ blk_vp,
    const char* __restrict__ data,
    int num_slots,
    int* __restrict__ cursor,
    const float* __restrict__ x,
    float* __restrict__ y,
    int num_rows,
    int num_cols)
{
    __shared__ int s_row[64];
    __shared__ int s_col[64];
    __shared__ int s_nnz[64];
    __shared__ int s_type[64];
    __shared__ int64_t s_vp[64];
    __shared__ int s_base;
    __shared__ int s_next;
    __shared__ int s_count;

    const int tid = threadIdx.x;
    const int lane = tid & 31;

    while (true) {
        if (tid == 0) {
            s_base = atomicAdd(cursor, 64);
            s_next = 0;
            int remain = num_slots - s_base;
            s_count = remain > 64 ? 64 : (remain > 0 ? remain : 0);
        }
        __syncthreads();

        if (s_count == 0)
            break;

        if (tid < s_count) {
            const int slot = s_base + tid;
            s_row[tid] = blk_row[slot];
            s_col[tid] = blk_col[slot];
            s_nnz[tid] = blk_nnz[slot];
            s_type[tid] = blk_type[slot];
            s_vp[tid] = blk_vp[slot];
        }
        __syncthreads();

        while (true) {
            int local_task = 0;
            if (lane == 0)
                local_task = atomicAdd(&s_next, 1);
            local_task = __shfl_sync(0xffffffffu, local_task, 0);

            if (local_task >= s_count)
                break;

            const int type = s_type[local_task];
            if (type == TILED2D_EMPTY)
                continue;

            const int row0 = s_row[local_task] << 4;
            const int col0 = s_col[local_task] << 4;
            const int bn = s_nnz[local_task];
            const char* p = data + s_vp[local_task];

            if (type == TILED2D_DENSE) {
                const float* dv = reinterpret_cast<const float*>(p);
                const int r = lane & 15;
                const int cbase = (lane < 16) ? 0 : 8;

                float sum = 0.0f;
                #pragma unroll
                for (int c = cbase; c < cbase + 8; ++c) {
                    const int col = col0 + c;
                    const float xv = col < num_cols ? ro_load(x + col) : 0.0f;
                    sum += dv[r * 16 + c] * xv;
                }
                sum += __shfl_xor_sync(0xffffffffu, sum, 16);

                if (lane < 16) {
                    const int row = row0 + r;
                    if (row < num_rows && sum != 0.0f)
                        atomicAdd(y + row, sum);
                }
            } else if (type == TILED2D_CSR) {
                const unsigned char* rp8 =
                    reinterpret_cast<const unsigned char*>(p);
                const unsigned char* cols = rp8 + 17;
                const float* dv = reinterpret_cast<const float*>(
                    p + ((17 + bn + 3) & ~3));

                if (lane < 16) {
                    const int begin = static_cast<int>(rp8[lane]);
                    const int end = static_cast<int>(rp8[lane + 1]);
                    float sum = 0.0f;

                    for (int i = begin; i < end; ++i) {
                        const int col = col0 + static_cast<int>(cols[i]);
                        sum += dv[i] * ro_load(x + col);
                    }

                    const int row = row0 + lane;
                    if (row < num_rows && sum != 0.0f)
                        atomicAdd(y + row, sum);
                }
            } else {
                const unsigned char* coords =
                    reinterpret_cast<const unsigned char*>(p);
                const float* dv = reinterpret_cast<const float*>(
                    p + ((bn + 3) & ~3));

                if (lane < bn) {
                    const unsigned char coord = coords[lane];
                    const int row = row0 + static_cast<int>(coord & 15);
                    const int col = col0 + static_cast<int>(coord >> 4);
                    const float value = dv[lane] * ro_load(x + col);
                    if (row < num_rows && value != 0.0f)
                        atomicAdd(y + row, value);
                }
            }
        }

        __syncthreads();
    }
}

void* spmv_preprocess(const int* d_row_ptr, const int* d_col_ind,
                      const float* d_values,
                      int num_rows, int num_cols, int nnz)
{
    SpmvContext* ctx = new SpmvContext;
    ctx->tiled = csr_to_tiled2d(
        d_row_ptr, d_col_ind, d_values,
        num_rows, num_cols, nnz, true);
    ctx->num_rows = num_rows;
    ctx->num_cols = num_cols;
    ctx->d_cursor = nullptr;
    cudaMalloc(&ctx->d_cursor, sizeof(int));
    return ctx;
}

void spmv_compute(void* opaque_ctx,
                  const int* d_row_ptr, const int* d_col_ind,
                  const float* d_values,
                  const float* d_x, float* d_y,
                  int num_rows, int num_cols, int nnz)
{
    (void)d_row_ptr;
    (void)d_col_ind;
    (void)d_values;
    (void)nnz;

    SpmvContext* ctx = reinterpret_cast<SpmvContext*>(opaque_ctx);
    if (ctx == nullptr || ctx->tiled.num_slots <= 0)
        return;

    cudaMemsetAsync(ctx->d_cursor, 0, sizeof(int));

    const int persistent_blocks = 1128;
    tiled2d_persistent_spmv_kernel<<<persistent_blocks, 256>>>(
        ctx->tiled.d_blk_row,
        ctx->tiled.d_blk_col,
        ctx->tiled.d_blk_nnz,
        ctx->tiled.d_blk_type,
        ctx->tiled.d_blk_vp,
        ctx->tiled.d_data,
        ctx->tiled.num_slots,
        ctx->d_cursor,
        d_x,
        d_y,
        num_rows,
        num_cols);
}

void spmv_cleanup(void* opaque_ctx)
{
    SpmvContext* ctx = reinterpret_cast<SpmvContext*>(opaque_ctx);
    if (ctx != nullptr) {
        if (ctx->d_cursor != nullptr)
            cudaFree(ctx->d_cursor);
        free_tiled2d(ctx->tiled);
        delete ctx;
    }
}