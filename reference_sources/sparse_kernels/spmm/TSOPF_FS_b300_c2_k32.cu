#include <cuda_runtime.h>
#include <stdint.h>
#include "sparse_formats.cuh"

struct SpMMContext {
    Tiled2DData tiled;
    int num_groups;
    int num_rows;
    int num_cols;
};

__global__ void tiled2d_spmm_kernel(
    const int* __restrict__ blk_row,
    const int* __restrict__ blk_col,
    const int* __restrict__ blk_nnz,
    const int* __restrict__ blk_type,
    const int64_t* __restrict__ blk_vp,
    const char* __restrict__ data,
    const float* __restrict__ B,
    float* __restrict__ C,
    int num_slots,
    int num_rows,
    int num_cols)
{
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int slot = blockIdx.x * 8 + warp;

    if (slot >= num_slots) return;

    const int type = blk_type[slot];
    if (type == TILED2D_EMPTY) return;

    const int br = blk_row[slot];
    const int bc = blk_col[slot];
    const int bn = blk_nnz[slot];

    if (br < 0 || bc < 0) return;

    const char* p = data + blk_vp[slot];
    const int gc_base = bc << 4;
    float acc[16];

    #pragma unroll
    for (int r = 0; r < 16; ++r) acc[r] = 0.0f;

    if (type == TILED2D_DENSE) {
        const float* vals = reinterpret_cast<const float*>(p);

        int gc0 = gc_base;
        float bcur = (gc0 < num_cols) ? B[gc0 * 32 + lane] : 0.0f;

        #pragma unroll
        for (int c = 0; c < 16; ++c) {
            float bnext = 0.0f;
            if (c + 1 < 16) {
                const int gcn = gc_base + c + 1;
                if (gcn < num_cols) bnext = B[gcn * 32 + lane];
            }

            #pragma unroll
            for (int r = 0; r < 16; ++r) {
                acc[r] = fmaf(vals[r * 16 + c], bcur, acc[r]);
            }
            bcur = bnext;
        }
    } else if (type == TILED2D_CSR) {
        const unsigned char* rp = reinterpret_cast<const unsigned char*>(p);
        const unsigned char* cols = rp + 17;
        const int value_offset = (17 + bn + 3) & ~3;
        const float* vals = reinterpret_cast<const float*>(p + value_offset);

        #pragma unroll
        for (int r = 0; r < 16; ++r) {
            const int first = (int)rp[r];
            const int last = (int)rp[r + 1];

            if (first < last) {
                int jcur = first;
                int gccur = gc_base + (int)cols[jcur];
                float vcur = vals[jcur];
                float bcur = (gccur < num_cols) ? B[gccur * 32 + lane] : 0.0f;

                for (; jcur + 1 < last; ++jcur) {
                    const int jnext = jcur + 1;
                    const int gcnext = gc_base + (int)cols[jnext];
                    const float vnext = vals[jnext];
                    const float bnext = (gcnext < num_cols)
                        ? B[gcnext * 32 + lane] : 0.0f;

                    acc[r] = fmaf(vcur, bcur, acc[r]);

                    vcur = vnext;
                    bcur = bnext;
                }

                acc[r] = fmaf(vcur, bcur, acc[r]);
            }
        }
    } else {
        const unsigned char* coords = reinterpret_cast<const unsigned char*>(p);
        const int value_offset = (bn + 3) & ~3;
        const float* vals = reinterpret_cast<const float*>(p + value_offset);

        if (bn > 0) {
            unsigned char rccur = coords[0];
            float vcur = vals[0];
            int gccur = gc_base + (int)(rccur >> 4);
            float bcur = (gccur < num_cols) ? B[gccur * 32 + lane] : 0.0f;

            for (int j = 0; j + 1 < bn; ++j) {
                const int jnext = j + 1;
                const unsigned char rcnext = coords[jnext];
                const float vnext = vals[jnext];
                const int gcnext = gc_base + (int)(rcnext >> 4);
                const float bnext = (gcnext < num_cols)
                    ? B[gcnext * 32 + lane] : 0.0f;

                acc[(int)(rccur & 15)] = fmaf(vcur, bcur, acc[(int)(rccur & 15)]);

                rccur = rcnext;
                vcur = vnext;
                bcur = bnext;
            }

            acc[(int)(rccur & 15)] = fmaf(vcur, bcur, acc[(int)(rccur & 15)]);
        }
    }

    const int row_base = br << 4;

    #pragma unroll
    for (int r = 0; r < 16; ++r) {
        const int row = row_base + r;
        if (row < num_rows) {
            atomicAdd(C + row * 32 + lane, acc[r]);
        }
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
    SpMMContext* ctx = new SpMMContext();

    ctx->tiled = csr_to_tiled2d(
        d_row_ptr,
        d_col_ind,
        d_values,
        num_rows,
        num_cols,
        nnz,
        true);

    ctx->num_groups = ctx->tiled.num_tbs;
    if (ctx->num_groups <= 0 && ctx->tiled.num_slots > 0) {
        ctx->num_groups = (ctx->tiled.num_slots + 7) >> 3;
    }

    ctx->num_rows = num_rows;
    ctx->num_cols = num_cols;
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
    SpMMContext* ctx = reinterpret_cast<SpMMContext*>(context);

    if (num_rows <= 0) return;

    cudaMemsetAsync(
        d_C,
        0,
        (size_t)num_rows * 32 * sizeof(float),
        0);

    if (!ctx || ctx->tiled.num_slots <= 0 || ctx->num_groups <= 0) return;

    tiled2d_spmm_kernel<<<ctx->num_groups, 256>>>(
        ctx->tiled.d_blk_row,
        ctx->tiled.d_blk_col,
        ctx->tiled.d_blk_nnz,
        ctx->tiled.d_blk_type,
        ctx->tiled.d_blk_vp,
        ctx->tiled.d_data,
        d_B,
        d_C,
        ctx->tiled.num_slots,
        num_rows,
        num_cols);
}

void spmm_cleanup(void* context)
{
    SpMMContext* ctx = reinterpret_cast<SpMMContext*>(context);
    if (!ctx) return;

    free_tiled2d(ctx->tiled);
    delete ctx;
}