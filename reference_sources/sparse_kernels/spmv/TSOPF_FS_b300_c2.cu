#include <cuda_runtime.h>
#include <stdint.h>
#include "sparse_formats.cuh"

struct SpmvContext {
    Tiled2DData tiled;
};

static constexpr int CTA_THREADS = 256;
static constexpr int WARPS_PER_CTA = 8;
static constexpr unsigned FULL_MASK = 0xffffffffu;

__global__ __launch_bounds__(CTA_THREADS, 6)
void tiled2d_spmv_kernel(
    const int* __restrict__ blk_row,
    const int* __restrict__ blk_col,
    const int* __restrict__ blk_nnz,
    const int* __restrict__ blk_type,
    const int64_t* __restrict__ blk_vp,
    const char* __restrict__ data,
    const float* __restrict__ x,
    float* __restrict__ y,
    int num_rows,
    int num_cols,
    int num_slots)
{
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int slot = blockIdx.x * WARPS_PER_CTA + warp;

    __shared__ float sx[WARPS_PER_CTA][16];
    __shared__ float coo_values[WARPS_PER_CTA][32];

    if (slot >= num_slots) return;

    const int type = blk_type[slot];
    if (type == TILED2D_EMPTY) return;

    const int row0 = blk_row[slot] << 4;
    const int col0 = blk_col[slot] << 4;
    const int bn = blk_nnz[slot];
    const char* p = data + blk_vp[slot];

    if (lane < 16) {
        const int col = col0 + lane;
        sx[warp][lane] = (col < num_cols) ? __ldg(x + col) : 0.0f;
    }
    __syncwarp(FULL_MASK);

    if (type == TILED2D_DENSE) {
        if (lane < 16) {
            const float* dv = reinterpret_cast<const float*>(p);
            const float* v = dv + lane * 16;
            const float* xv = sx[warp];

            float sum = 0.0f;
            sum = fmaf(v[0],  xv[0],  sum);
            sum = fmaf(v[1],  xv[1],  sum);
            sum = fmaf(v[2],  xv[2],  sum);
            sum = fmaf(v[3],  xv[3],  sum);
            sum = fmaf(v[4],  xv[4],  sum);
            sum = fmaf(v[5],  xv[5],  sum);
            sum = fmaf(v[6],  xv[6],  sum);
            sum = fmaf(v[7],  xv[7],  sum);
            sum = fmaf(v[8],  xv[8],  sum);
            sum = fmaf(v[9],  xv[9],  sum);
            sum = fmaf(v[10], xv[10], sum);
            sum = fmaf(v[11], xv[11], sum);
            sum = fmaf(v[12], xv[12], sum);
            sum = fmaf(v[13], xv[13], sum);
            sum = fmaf(v[14], xv[14], sum);
            sum = fmaf(v[15], xv[15], sum);

            const int out_row = row0 + lane;
            if (out_row < num_rows) {
                atomicAdd(y + out_row, sum);
            }
        }
    } else if (type == TILED2D_CSR) {
        if (lane < 16) {
            const unsigned char* rp8 =
                reinterpret_cast<const unsigned char*>(p);
            const unsigned char* cols = rp8 + 17;
            const float* dv = reinterpret_cast<const float*>(
                p + ((17 + bn + 3) & ~3));

            const int begin = rp8[lane];
            const int end = rp8[lane + 1];
            const float* xv = sx[warp];
            float sum = 0.0f;

            for (int i = begin; i < end; ++i) {
                sum = fmaf(dv[i], xv[cols[i]], sum);
            }

            const int out_row = row0 + lane;
            if (out_row < num_rows && begin != end) {
                atomicAdd(y + out_row, sum);
            }
        }
    } else {
        const unsigned char* coords =
            reinterpret_cast<const unsigned char*>(p);
        const float* dv = reinterpret_cast<const float*>(
            p + ((bn + 3) & ~3));

        const bool valid = lane < bn;
        int row_key = 16 + lane;
        float value = 0.0f;

        if (valid) {
            const unsigned char q = coords[lane];
            const int r = q & 15;
            const int c = q >> 4;
            row_key = r;
            value = dv[lane] * sx[warp][c];
        }

        coo_values[warp][lane] = value;
        __syncwarp(FULL_MASK);

        const unsigned group_mask = __match_any_sync(FULL_MASK, row_key);

        if (valid && lane == (__ffs(group_mask) - 1)) {
            float total = 0.0f;
            unsigned remaining = group_mask;

            while (remaining != 0u) {
                const int source_lane = __ffs(remaining) - 1;
                total += coo_values[warp][source_lane];
                remaining &= remaining - 1u;
            }

            const int out_row = row0 + row_key;
            if (out_row < num_rows) {
                atomicAdd(y + out_row, total);
            }
        }
    }
}

void* spmv_preprocess(
    const int* d_row_ptr,
    const int* d_col_ind,
    const float* d_values,
    int num_rows,
    int num_cols,
    int nnz)
{
    SpmvContext* ctx = new SpmvContext;
    ctx->tiled = csr_to_tiled2d(
        d_row_ptr,
        d_col_ind,
        d_values,
        num_rows,
        num_cols,
        nnz,
        true);
    return ctx;
}

void spmv_compute(
    void* ptr,
    const int* d_row_ptr,
    const int* d_col_ind,
    const float* d_values,
    const float* d_x,
    float* d_y,
    int num_rows,
    int num_cols,
    int nnz)
{
    (void)d_row_ptr;
    (void)d_col_ind;
    (void)d_values;
    (void)nnz;

    SpmvContext* ctx = reinterpret_cast<SpmvContext*>(ptr);
    if (ctx == nullptr) return;

    const Tiled2DData& t = ctx->tiled;
    int blocks = t.num_tbs;
    if (blocks < 1) blocks = 1;

    tiled2d_spmv_kernel<<<blocks, CTA_THREADS>>>(
        t.d_blk_row,
        t.d_blk_col,
        t.d_blk_nnz,
        t.d_blk_type,
        t.d_blk_vp,
        t.d_data,
        d_x,
        d_y,
        num_rows,
        num_cols,
        t.num_slots);
}

void spmv_cleanup(void* ptr)
{
    SpmvContext* ctx = reinterpret_cast<SpmvContext*>(ptr);
    if (ctx != nullptr) {
        free_tiled2d(ctx->tiled);
        delete ctx;
    }
}