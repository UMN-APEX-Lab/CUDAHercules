#include <cuda_runtime.h>
#include "sparse_formats.cuh"

struct SpmmContext {
    COOData coo;
    int nnz;
};

__global__ __launch_bounds__(256)
void coo_pipelined_spmm_k32(
    const int* __restrict__ row_ind,
    const int* __restrict__ col_ind,
    const float* __restrict__ values,
    const float* __restrict__ B,
    float* __restrict__ C,
    int nnz)
{
    constexpr int SEG_NNZ = 2048;
    constexpr unsigned MASK = 0xffffffffu;

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int chunk = (int)blockIdx.x * 8 + warp;
    const int begin = chunk * SEG_NNZ;
    const int end = min(begin + SEG_NNZ, nnz);

    if (begin >= end) return;

    int first_row = -1;
    int current_row = -1;
    float accum = 0.0f;
    bool started = false;

#define CONSUME_ENTRY(R, CIDX, AV, BV)                                      \
    do {                                                                     \
        const int _r = (R);                                                  \
        if (!started) {                                                      \
            started = true;                                                  \
            first_row = _r;                                                  \
            current_row = _r;                                                \
        } else if (_r != current_row) {                                     \
            float* _out = C + ((size_t)current_row << 5) + lane;            \
            if (current_row == first_row) atomicAdd(_out, accum);           \
            else *_out = accum;                                             \
            current_row = _r;                                               \
            accum = 0.0f;                                                   \
        }                                                                    \
        accum = fmaf((AV), (BV), accum);                                    \
    } while (0)

    for (int batch = begin; batch < end; batch += 32) {
        const int p = batch + lane;
        int local_r = 0;
        int local_c = 0;
        float local_a = 0.0f;

        if (p < end) {
            local_r = row_ind[p];
            local_c = col_ind[p];
            local_a = values[p];
        }

        const int count = min(32, end - batch);

#pragma unroll
        for (int g = 0; g < 32; g += 4) {
            const bool v0 = (g + 0) < count;
            const bool v1 = (g + 1) < count;
            const bool v2 = (g + 2) < count;
            const bool v3 = (g + 3) < count;

            const int r0 = __shfl_sync(MASK, local_r, g + 0);
            const int c0 = __shfl_sync(MASK, local_c, g + 0);
            const float a0 = __shfl_sync(MASK, local_a, g + 0);

            const int r1 = __shfl_sync(MASK, local_r, g + 1);
            const int c1 = __shfl_sync(MASK, local_c, g + 1);
            const float a1 = __shfl_sync(MASK, local_a, g + 1);

            const int r2 = __shfl_sync(MASK, local_r, g + 2);
            const int c2 = __shfl_sync(MASK, local_c, g + 2);
            const float a2 = __shfl_sync(MASK, local_a, g + 2);

            const int r3 = __shfl_sync(MASK, local_r, g + 3);
            const int c3 = __shfl_sync(MASK, local_c, g + 3);
            const float a3 = __shfl_sync(MASK, local_a, g + 3);

            float b0 = 0.0f;
            float b1 = 0.0f;
            float b2 = 0.0f;
            float b3 = 0.0f;

            if (v0) b0 = __ldg(B + ((size_t)c0 << 5) + lane);
            if (v1) b1 = __ldg(B + ((size_t)c1 << 5) + lane);
            if (v2) b2 = __ldg(B + ((size_t)c2 << 5) + lane);
            if (v3) b3 = __ldg(B + ((size_t)c3 << 5) + lane);

            if (v0) CONSUME_ENTRY(r0, c0, a0, b0);
            if (v1) CONSUME_ENTRY(r1, c1, a1, b1);
            if (v2) CONSUME_ENTRY(r2, c2, a2, b2);
            if (v3) CONSUME_ENTRY(r3, c3, a3, b3);
        }
    }

#undef CONSUME_ENTRY

    if (started) {
        atomicAdd(C + ((size_t)current_row << 5) + lane, accum);
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
    (void)K;

    SpmmContext* ctx = new SpmmContext;
    ctx->coo = csr_to_coo(
        d_row_ptr,
        d_col_ind,
        d_values,
        num_rows,
        num_cols,
        nnz);
    ctx->nnz = nnz;
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
    (void)d_row_ptr;
    (void)d_col_ind;
    (void)d_values;
    (void)num_cols;
    (void)nnz;
    (void)K;

    if (num_rows <= 0) return;

    cudaMemsetAsync(
        d_C,
        0,
        (size_t)num_rows * 32 * sizeof(float),
        0);

    SpmmContext* ctx = static_cast<SpmmContext*>(context);
    if (ctx == nullptr || ctx->nnz <= 0) return;

    constexpr int CTA_NNZ = 8 * 2048;
    const int blocks = (ctx->nnz + CTA_NNZ - 1) / CTA_NNZ;

    coo_pipelined_spmm_k32<<<blocks, 256>>>(
        ctx->coo.d_row_ind,
        ctx->coo.d_col_ind,
        ctx->coo.d_values,
        d_B,
        d_C,
        ctx->nnz);
}

void spmm_cleanup(void* context)
{
    SpmmContext* ctx = static_cast<SpmmContext*>(context);
    if (ctx != nullptr) {
        free_coo(ctx->coo);
        delete ctx;
    }
}