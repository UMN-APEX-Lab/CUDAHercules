#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

__device__ __forceinline__
void accumulate_k256(const float* __restrict__ B, int col, float val,
                     int lane,
                     float& a0, float& a1, float& a2, float& a3,
                     float& a4, float& a5, float& a6, float& a7) {
    const size_t base = static_cast<size_t>(static_cast<unsigned int>(col)) << 8;
    a0 = fmaf(val, __ldg(B + base + lane),       a0);
    a1 = fmaf(val, __ldg(B + base + lane + 32),  a1);
    a2 = fmaf(val, __ldg(B + base + lane + 64),  a2);
    a3 = fmaf(val, __ldg(B + base + lane + 96),  a3);
    a4 = fmaf(val, __ldg(B + base + lane + 128), a4);
    a5 = fmaf(val, __ldg(B + base + lane + 160), a5);
    a6 = fmaf(val, __ldg(B + base + lane + 192), a6);
    a7 = fmaf(val, __ldg(B + base + lane + 224), a7);
}

__global__ __launch_bounds__(256, 5)
void csr_spmm_k256_kernel(const int* __restrict__ row_ptr,
                           const int* __restrict__ col_ind,
                           const float* __restrict__ values,
                           const float* __restrict__ B,
                           float* __restrict__ C,
                           int num_rows,
                           int num_cols) {
    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int row = static_cast<int>(blockIdx.x) * 8 + warp;

    if (row >= num_rows) return;

    int begin = 0;
    int end = 0;
    if (lane == 0) {
        begin = row_ptr[row];
        end = row_ptr[row + 1];
    }
    begin = __shfl_sync(0xffffffffu, begin, 0);
    end = __shfl_sync(0xffffffffu, end, 0);

    float a0 = 0.0f;
    float a1 = 0.0f;
    float a2 = 0.0f;
    float a3 = 0.0f;
    float a4 = 0.0f;
    float a5 = 0.0f;
    float a6 = 0.0f;
    float a7 = 0.0f;

    int col = 0;
    float val = 0.0f;
    if (lane == 0 && begin < end) {
        col = col_ind[begin];
        val = values[begin];
    }
    col = __shfl_sync(0xffffffffu, col, 0);
    val = __shfl_sync(0xffffffffu, val, 0);

    for (int p = begin; p < end; ++p) {
        int next_col = 0;
        float next_val = 0.0f;

        if (lane == 0 && p + 1 < end) {
            next_col = col_ind[p + 1];
            next_val = values[p + 1];
        }

        if (static_cast<unsigned int>(col) <
            static_cast<unsigned int>(num_cols)) {
            accumulate_k256(B, col, val, lane,
                             a0, a1, a2, a3, a4, a5, a6, a7);
        }

        col = __shfl_sync(0xffffffffu, next_col, 0);
        val = __shfl_sync(0xffffffffu, next_val, 0);
    }

    const size_t cbase =
        static_cast<size_t>(static_cast<unsigned int>(row)) << 8;
    C[cbase + lane]       = a0;
    C[cbase + lane + 32]  = a1;
    C[cbase + lane + 64]  = a2;
    C[cbase + lane + 96]  = a3;
    C[cbase + lane + 128] = a4;
    C[cbase + lane + 160] = a5;
    C[cbase + lane + 192] = a6;
    C[cbase + lane + 224] = a7;
}

__global__ __launch_bounds__(128, 8)
void csr_spmm_fallback_kernel(const int* __restrict__ row_ptr,
                               const int* __restrict__ col_ind,
                               const float* __restrict__ values,
                               const float* __restrict__ B,
                               float* __restrict__ C,
                               int num_rows,
                               int num_cols,
                               int K) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int row = static_cast<int>(blockIdx.x) * 4 + warp;
    const int k = static_cast<int>(blockIdx.y) * 32 + lane;
    const bool valid_k = k < K;

    if (row >= num_rows) return;

    int begin = 0;
    int end = 0;
    if (lane == 0) {
        begin = row_ptr[row];
        end = row_ptr[row + 1];
    }
    begin = __shfl_sync(0xffffffffu, begin, 0);
    end = __shfl_sync(0xffffffffu, end, 0);

    float acc = 0.0f;
    for (int p = begin; p < end; ++p) {
        int col = 0;
        float val = 0.0f;
        if (lane == 0) {
            col = col_ind[p];
            val = values[p];
        }
        col = __shfl_sync(0xffffffffu, col, 0);
        val = __shfl_sync(0xffffffffu, val, 0);

        if (valid_k &&
            static_cast<unsigned int>(col) <
                static_cast<unsigned int>(num_cols)) {
            const size_t bidx =
                static_cast<size_t>(static_cast<unsigned int>(col)) *
                    static_cast<size_t>(K) +
                static_cast<size_t>(k);
            acc = fmaf(val, __ldg(B + bidx), acc);
        }
    }

    if (valid_k) {
        const size_t cidx =
            static_cast<size_t>(static_cast<unsigned int>(row)) *
                static_cast<size_t>(K) +
            static_cast<size_t>(k);
        C[cidx] = acc;
    }
}

void* spmm_preprocess(const int* d_row_ptr,
                      const int* d_col_ind,
                      const float* d_values,
                      int num_rows,
                      int num_cols,
                      int nnz,
                      int K) {
    (void)d_row_ptr;
    (void)d_col_ind;
    (void)d_values;
    (void)num_rows;
    (void)num_cols;
    (void)nnz;
    (void)K;
    return nullptr;
}

void spmm_compute(void* ctx,
                  const int* d_row_ptr,
                  const int* d_col_ind,
                  const float* d_values,
                  const float* d_B,
                  float* d_C,
                  int num_rows,
                  int num_cols,
                  int nnz,
                  int K) {
    (void)ctx;
    (void)nnz;

    if (num_rows <= 0 || K <= 0) return;

    if (K == 256) {
        const unsigned int blocks = static_cast<unsigned int>(
            (static_cast<unsigned long long>(num_rows) + 7ull) / 8ull);
        csr_spmm_k256_kernel<<<dim3(blocks, 1u, 1u), dim3(256u, 1u, 1u)>>>(
            d_row_ptr, d_col_ind, d_values, d_B, d_C, num_rows, num_cols);
    } else {
        const unsigned int row_blocks = static_cast<unsigned int>(
            (static_cast<unsigned long long>(num_rows) + 3ull) / 4ull);
        const unsigned int k_blocks = static_cast<unsigned int>(
            (static_cast<unsigned long long>(K) + 31ull) / 32ull);
        csr_spmm_fallback_kernel<<<dim3(row_blocks, k_blocks, 1u),
                                    dim3(128u, 1u, 1u)>>>(
            d_row_ptr, d_col_ind, d_values, d_B, d_C,
            num_rows, num_cols, K);
    }
}

void spmm_cleanup(void* ctx) {
    (void)ctx;
}