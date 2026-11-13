#include <cuda_runtime.h>
#include <stdint.h>
#include <stdlib.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include "sparse_formats.cuh"

struct SpmmContext {
    Tiled2DData tiled;
    int* d_sorted_slots;
    int* d_row_tile_ptr;
    int num_block_rows;
    int num_slots;
    int num_rows;
};

__global__ static void init_slots_kernel(
    int* __restrict__ slots,
    int* __restrict__ keys,
    const int* __restrict__ blk_row,
    int n)
{
    const int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) {
        slots[i] = i;
        keys[i] = blk_row[i];
    }
}

__global__ static void build_row_ptr_kernel(
    const int* __restrict__ keys,
    int* __restrict__ row_ptr,
    int nrows,
    int nslots)
{
    const int r = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (r <= nrows) {
        int lo = 0;
        int hi = nslots;
        while (lo < hi) {
            const int mid = (lo + hi) >> 1;
            if (keys[mid] < r) lo = mid + 1;
            else hi = mid;
        }
        row_ptr[r] = lo;
    }
}

#define ACCUMULATE_B(V, BPTR) do {                                            \
    const float4 q0 = *reinterpret_cast<const float4*>((BPTR) + lane4);      \
    const float4 q1 = *reinterpret_cast<const float4*>((BPTR) + 32 + lane4); \
    const float4 q2 = *reinterpret_cast<const float4*>((BPTR) + 64 + lane4); \
    const float4 q3 = *reinterpret_cast<const float4*>((BPTR) + 96 + lane4); \
    a0  = fmaf((V), q0.x, a0);                                                \
    a1  = fmaf((V), q0.y, a1);                                                \
    a2  = fmaf((V), q0.z, a2);                                                \
    a3  = fmaf((V), q0.w, a3);                                                \
    a4  = fmaf((V), q1.x, a4);                                                \
    a5  = fmaf((V), q1.y, a5);                                                \
    a6  = fmaf((V), q1.z, a6);                                                \
    a7  = fmaf((V), q1.w, a7);                                                \
    a8  = fmaf((V), q2.x, a8);                                                \
    a9  = fmaf((V), q2.y, a9);                                                \
    a10 = fmaf((V), q2.z, a10);                                               \
    a11 = fmaf((V), q2.w, a11);                                               \
    a12 = fmaf((V), q3.x, a12);                                               \
    a13 = fmaf((V), q3.y, a13);                                               \
    a14 = fmaf((V), q3.z, a14);                                               \
    a15 = fmaf((V), q3.w, a15);                                               \
} while (0)

__global__ __launch_bounds__(128, 8)
static void tiled2d_spmm_kernel(
    const int* __restrict__ sorted_slots,
    const int* __restrict__ row_tile_ptr,
    const int* __restrict__ blk_col,
    const int* __restrict__ blk_nnz,
    const int* __restrict__ blk_type,
    const int64_t* __restrict__ blk_vp,
    const char* __restrict__ data,
    const float* __restrict__ B,
    float* __restrict__ C,
    int num_rows)
{
    const int tid = (int)threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int subgroup = lane >> 3;
    const int lane4 = (lane & 7) << 2;
    const int local_row = (warp << 2) + subgroup;
    const int block_row = (int)blockIdx.x;
    const int out_row = (block_row << 4) + local_row;

    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    float a4 = 0.0f, a5 = 0.0f, a6 = 0.0f, a7 = 0.0f;
    float a8 = 0.0f, a9 = 0.0f, a10 = 0.0f, a11 = 0.0f;
    float a12 = 0.0f, a13 = 0.0f, a14 = 0.0f, a15 = 0.0f;

    const int begin = row_tile_ptr[block_row];
    const int end = row_tile_ptr[block_row + 1];

    for (int qi = begin; qi < end; ++qi) {
        const int slot = sorted_slots[qi];
        const int count = blk_nnz[slot];
        const int type = blk_type[slot];
        const int bcol = blk_col[slot];
        const char* src = data + blk_vp[slot];

        if (type == TILED2D_DENSE) {
            const float* vals = reinterpret_cast<const float*>(src);
#pragma unroll
            for (int c = 0; c < 16; ++c) {
                const float v = vals[(local_row << 4) + c];
                const float* bp = B + (((bcol << 4) + c) << 7);
                ACCUMULATE_B(v, bp);
            }
        } else if (type == TILED2D_CSR) {
            const unsigned char* rp = reinterpret_cast<const unsigned char*>(src);
            const unsigned char* cols = rp + 17;
            const int values_offset = ((17 + count + 3) & ~3);
            const float* vals = reinterpret_cast<const float*>(src + values_offset);

            const int first = (int)rp[local_row];
            const int last = (int)rp[local_row + 1];
            for (int e = first; e < last; ++e) {
                const float v = vals[e];
                const int c = (int)cols[e];
                const float* bp = B + (((bcol << 4) + c) << 7);
                ACCUMULATE_B(v, bp);
            }
        } else if (type == TILED2D_COO) {
            const unsigned char* coords = reinterpret_cast<const unsigned char*>(src);
            const int values_offset = ((count + 3) & ~3);
            const float* vals = reinterpret_cast<const float*>(src + values_offset);

            for (int e = 0; e < count; ++e) {
                const unsigned char coord = coords[e];
                if ((coord & 15) == local_row) {
                    const float v = vals[e];
                    const int c = (int)(coord >> 4);
                    const float* bp = B + (((bcol << 4) + c) << 7);
                    ACCUMULATE_B(v, bp);
                }
            }
        }
    }

    if (out_row < num_rows) {
        float* out = C + ((size_t)out_row << 7) + lane4;
        *reinterpret_cast<float4*>(out) = make_float4(a0, a1, a2, a3);
        *reinterpret_cast<float4*>(out + 32) = make_float4(a4, a5, a6, a7);
        *reinterpret_cast<float4*>(out + 64) = make_float4(a8, a9, a10, a11);
        *reinterpret_cast<float4*>(out + 96) = make_float4(a12, a13, a14, a15);
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
    SpmmContext* ctx = new SpmmContext();
    ctx->tiled = csr_to_tiled2d(
        d_row_ptr, d_col_ind, d_values,
        num_rows, num_cols, nnz, true);

    ctx->num_rows = num_rows;
    ctx->num_slots = ctx->tiled.num_slots;
    ctx->num_block_rows = (num_rows + 15) >> 4;
    ctx->d_sorted_slots = nullptr;
    ctx->d_row_tile_ptr = nullptr;

    cudaMalloc(
        reinterpret_cast<void**>(&ctx->d_row_tile_ptr),
        (size_t)(ctx->num_block_rows + 1) * sizeof(int));

    if (ctx->num_slots <= 0) {
        cudaMemset(
            ctx->d_row_tile_ptr, 0,
            (size_t)(ctx->num_block_rows + 1) * sizeof(int));
        return ctx;
    }

    int* d_keys = nullptr;
    cudaMalloc(
        reinterpret_cast<void**>(&ctx->d_sorted_slots),
        (size_t)ctx->num_slots * sizeof(int));
    cudaMalloc(
        reinterpret_cast<void**>(&d_keys),
        (size_t)ctx->num_slots * sizeof(int));

    constexpr int threads = 256;
    const int blocks = (ctx->num_slots + threads - 1) / threads;

    init_slots_kernel<<<blocks, threads>>>(
        ctx->d_sorted_slots,
        d_keys,
        ctx->tiled.d_blk_row,
        ctx->num_slots);

    thrust::device_ptr<int> keys = thrust::device_pointer_cast(d_keys);
    thrust::device_ptr<int> slots =
        thrust::device_pointer_cast(ctx->d_sorted_slots);
    thrust::sort_by_key(thrust::device, keys, keys + ctx->num_slots, slots);

    const int rp_blocks =
        (ctx->num_block_rows + 1 + threads - 1) / threads;
    build_row_ptr_kernel<<<rp_blocks, threads>>>(
        d_keys,
        ctx->d_row_tile_ptr,
        ctx->num_block_rows,
        ctx->num_slots);

    cudaFree(d_keys);
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
    SpmmContext* ctx = reinterpret_cast<SpmmContext*>(context);
    if (ctx == nullptr || num_rows <= 0) return;

    tiled2d_spmm_kernel<<<ctx->num_block_rows, 128>>>(
        ctx->d_sorted_slots,
        ctx->d_row_tile_ptr,
        ctx->tiled.d_blk_col,
        ctx->tiled.d_blk_nnz,
        ctx->tiled.d_blk_type,
        ctx->tiled.d_blk_vp,
        ctx->tiled.d_data,
        d_B,
        d_C,
        num_rows);
}

void spmm_cleanup(void* context)
{
    SpmmContext* ctx = reinterpret_cast<SpmmContext*>(context);
    if (ctx == nullptr) return;

    if (ctx->d_sorted_slots) cudaFree(ctx->d_sorted_slots);
    if (ctx->d_row_tile_ptr) cudaFree(ctx->d_row_tile_ptr);
    free_tiled2d(ctx->tiled);
    delete ctx;
}