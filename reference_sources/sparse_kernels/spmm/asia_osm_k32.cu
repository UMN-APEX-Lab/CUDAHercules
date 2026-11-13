#include <cuda_runtime.h>
#include <stdint.h>
#include <stddef.h>

struct WindowContext {
    unsigned long long* d_packed;
    int num_rows;
    int num_cols;
    int nnz;
};

__global__ void pack_windowed_entries_kernel(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_ind,
    const float* __restrict__ values,
    unsigned long long* __restrict__ packed,
    int num_rows)
{
    int row = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= num_rows) return;

    int begin = row_ptr[row];
    int end = row_ptr[row + 1];

    for (int p = begin; p < end; ++p) {
        unsigned int col = (unsigned int)col_ind[p];
        unsigned int val = __float_as_uint(values[p]);
        packed[p] = ((unsigned long long)col << 32) | (unsigned long long)val;
    }
}

__device__ __forceinline__ void prefetch_b_row_l2(const float* ptr)
{
    asm volatile("prefetch.global.L2 [%0];" :: "l"(ptr));
}

#define FMA4(C0, C1, C2, C3, V, B) \
    C0 = fmaf(V, B.x, C0);          \
    C1 = fmaf(V, B.y, C1);          \
    C2 = fmaf(V, B.z, C2);          \
    C3 = fmaf(V, B.w, C3)

__global__ void windowed_interleaved_spmm32_kernel(
    const int* __restrict__ row_ptr,
    const unsigned long long* __restrict__ packed,
    const float* __restrict__ B,
    float* __restrict__ C,
    int num_rows)
{
    int row = (int)(blockIdx.x * 256 + threadIdx.x);
    if (row >= num_rows) return;

    float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;
    float c4 = 0.0f, c5 = 0.0f, c6 = 0.0f, c7 = 0.0f;
    float c8 = 0.0f, c9 = 0.0f, c10 = 0.0f, c11 = 0.0f;
    float c12 = 0.0f, c13 = 0.0f, c14 = 0.0f, c15 = 0.0f;
    float c16 = 0.0f, c17 = 0.0f, c18 = 0.0f, c19 = 0.0f;
    float c20 = 0.0f, c21 = 0.0f, c22 = 0.0f, c23 = 0.0f;
    float c24 = 0.0f, c25 = 0.0f, c26 = 0.0f, c27 = 0.0f;
    float c28 = 0.0f, c29 = 0.0f, c30 = 0.0f, c31 = 0.0f;

    const int begin = row_ptr[row];
    const int end = row_ptr[row + 1];

    if (begin < end) {
        unsigned long long entry = __ldg(packed + begin);
        int p = begin;

        while (true) {
            const unsigned int col = (unsigned int)(entry >> 32);
            const float value = __uint_as_float((unsigned int)entry);

            const bool has_next = (p + 1 < end);
            unsigned long long next_entry = 0;

            if (has_next) {
                next_entry = __ldg(packed + p + 1);
                const unsigned int next_col = (unsigned int)(next_entry >> 32);
                prefetch_b_row_l2(B + (size_t)next_col * 32u);
            }

            const float4* bp = reinterpret_cast<const float4*>(
                B + (size_t)col * 32u);

            const float4 b0 = bp[0];
            FMA4(c0, c1, c2, c3, value, b0);

            const float4 b1 = bp[1];
            FMA4(c4, c5, c6, c7, value, b1);

            const float4 b2 = bp[2];
            FMA4(c8, c9, c10, c11, value, b2);

            const float4 b3 = bp[3];
            FMA4(c12, c13, c14, c15, value, b3);

            const float4 b4 = bp[4];
            FMA4(c16, c17, c18, c19, value, b4);

            const float4 b5 = bp[5];
            FMA4(c20, c21, c22, c23, value, b5);

            const float4 b6 = bp[6];
            FMA4(c24, c25, c26, c27, value, b6);

            const float4 b7 = bp[7];
            FMA4(c28, c29, c30, c31, value, b7);

            if (!has_next) break;
            entry = next_entry;
            ++p;
        }
    }

    float4* cp = reinterpret_cast<float4*>(C + (size_t)row * 32u);
    cp[0] = make_float4(c0, c1, c2, c3);
    cp[1] = make_float4(c4, c5, c6, c7);
    cp[2] = make_float4(c8, c9, c10, c11);
    cp[3] = make_float4(c12, c13, c14, c15);
    cp[4] = make_float4(c16, c17, c18, c19);
    cp[5] = make_float4(c20, c21, c22, c23);
    cp[6] = make_float4(c24, c25, c26, c27);
    cp[7] = make_float4(c28, c29, c30, c31);
}

#undef FMA4

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

    WindowContext* ctx = new WindowContext;
    ctx->d_packed = nullptr;
    ctx->num_rows = num_rows;
    ctx->num_cols = num_cols;
    ctx->nnz = nnz;

    if (nnz > 0) {
        cudaMalloc((void**)&ctx->d_packed,
                   (size_t)nnz * sizeof(unsigned long long));

        const int threads = 256;
        const int blocks = (num_rows + threads - 1) / threads;
        if (blocks > 0) {
            pack_windowed_entries_kernel<<<blocks, threads>>>(
                d_row_ptr, d_col_ind, d_values, ctx->d_packed, num_rows);
        }
    }

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
    (void)d_col_ind;
    (void)d_values;
    (void)num_cols;
    (void)nnz;
    (void)K;

    WindowContext* ctx = (WindowContext*)context;
    if (ctx == nullptr || num_rows <= 0) return;

    const int blocks = (num_rows + 255) / 256;
    windowed_interleaved_spmm32_kernel<<<blocks, 256>>>(
        d_row_ptr, ctx->d_packed, d_B, d_C, num_rows);
}

void spmm_cleanup(void* context)
{
    WindowContext* ctx = (WindowContext*)context;
    if (ctx == nullptr) return;

    if (ctx->d_packed != nullptr) cudaFree(ctx->d_packed);
    delete ctx;
}