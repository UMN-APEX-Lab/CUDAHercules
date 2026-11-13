#include <cuda_runtime.h>
#include "sparse_formats.cuh"

struct SpMMContext {
    COOData coo;
    int nnz;
};

__global__ void zero_c_kernel(float* __restrict__ c, size_t count) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)blockDim.x * gridDim.x;
    for (; idx < count; idx += stride) {
        c[idx] = 0.0f;
    }
}

__global__ __launch_bounds__(128, 12)
void coo_spmm_k32_kernel(const int* __restrict__ row_ind,
                         const int* __restrict__ col_ind,
                         const float* __restrict__ values,
                         const float* __restrict__ b,
                         float* __restrict__ c,
                         int nnz) {
    constexpr int CHUNK_NNZ = 256;
    constexpr unsigned FULL_MASK = 0xffffffffu;

    const int lane = threadIdx.x & 31;
    const int warp_in_block = threadIdx.x >> 5;
    const int warp_id = blockIdx.x * 4 + warp_in_block;
    const int begin = warp_id * CHUNK_NNZ;

    if (begin >= nnz) return;

    const int end = (begin + CHUNK_NNZ < nnz) ? (begin + CHUNK_NNZ) : nnz;

    int pref_row = 0;
    int pref_col = 0;
    float pref_val = 0.0f;

    if (lane == 0) {
        pref_row = row_ind[begin];
        pref_col = col_ind[begin];
        pref_val = values[begin];
    }

    int current_row = 0;
    float accum = 0.0f;
    bool have_row = false;

    for (int i = begin; i < end; ++i) {
        const int row_local = pref_row;
        const int col_local = pref_col;
        const float val_local = pref_val;

        if (lane == 0 && i + 1 < end) {
            const int next_i = i + 1;
            pref_row = row_ind[next_i];
            pref_col = col_ind[next_i];
            pref_val = values[next_i];
        }

        const int row = __shfl_sync(FULL_MASK, row_local, 0);
        const int col = __shfl_sync(FULL_MASK, col_local, 0);
        const float val = __shfl_sync(FULL_MASK, val_local, 0);

        if (!have_row) {
            current_row = row;
            have_row = true;
        } else if (row != current_row) {
            atomicAdd(&c[(size_t)current_row * 32 + lane], accum);
            accum = 0.0f;
            current_row = row;
        }

        accum = fmaf(val, b[(size_t)col * 32 + lane], accum);
    }

    if (have_row) {
        atomicAdd(&c[(size_t)current_row * 32 + lane], accum);
    }
}

void* spmm_preprocess(const int* d_row_ptr, const int* d_col_ind,
                      const float* d_values,
                      int num_rows, int num_cols, int nnz, int K) {
    (void)K;

    SpMMContext* ctx = new SpMMContext;
    ctx->coo = csr_to_coo(d_row_ptr, d_col_ind, d_values,
                          num_rows, num_cols, nnz);
    ctx->nnz = nnz;
    return ctx;
}

void spmm_compute(void* context,
                  const int* d_row_ptr, const int* d_col_ind,
                  const float* d_values,
                  const float* d_B, float* d_C,
                  int num_rows, int num_cols, int nnz, int K) {
    (void)d_row_ptr;
    (void)d_col_ind;
    (void)d_values;
    (void)num_cols;
    (void)nnz;
    (void)K;

    SpMMContext* ctx = static_cast<SpMMContext*>(context);

    const size_t c_count = (size_t)num_rows * 32;
    if (c_count != 0) {
        int zero_blocks = (int)((c_count + 255) / 256);
        if (zero_blocks > 65535) zero_blocks = 65535;
        zero_c_kernel<<<zero_blocks, 256>>>(d_C, c_count);
    }

    if (ctx != nullptr && ctx->nnz > 0) {
        const int chunks = (ctx->nnz + 255) / 256;
        const int blocks = (chunks + 3) / 4;

        coo_spmm_k32_kernel<<<blocks, 128>>>(
            ctx->coo.d_row_ind,
            ctx->coo.d_col_ind,
            ctx->coo.d_values,
            d_B,
            d_C,
            ctx->nnz);
    }
}

void spmm_cleanup(void* context) {
    SpMMContext* ctx = static_cast<SpMMContext*>(context);
    if (ctx != nullptr) {
        free_coo(ctx->coo);
        delete ctx;
    }
}