#include <cuda_runtime.h>
#include "sparse_formats.cuh"

struct SpMMContext {
    COOData coo;
    int nnz;
    int num_rows;
};

__global__ void zero_c_k8_kernel(float4* __restrict__ c, int count4) {
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx < count4) {
        c[idx] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
}

__device__ __forceinline__ void atomic_add_k8(
    float* __restrict__ c, int row,
    float x0, float x1, float x2, float x3,
    float x4, float x5, float x6, float x7)
{
    float* out = c + ((size_t)row << 3);
    atomicAdd(out + 0, x0);
    atomicAdd(out + 1, x1);
    atomicAdd(out + 2, x2);
    atomicAdd(out + 3, x3);
    atomicAdd(out + 4, x4);
    atomicAdd(out + 5, x5);
    atomicAdd(out + 6, x6);
    atomicAdd(out + 7, x7);
}

__global__ __launch_bounds__(128) void coo_spmm_k8_kernel(
    const int* __restrict__ row_ind,
    const int* __restrict__ col_ind,
    const float* __restrict__ values,
    const float* __restrict__ b,
    float* __restrict__ c,
    int nnz)
{
    constexpr int CHUNK_NNZ = 2048;
    constexpr int WARP_NNZ = 512;
    constexpr unsigned FULL_MASK = 0xffffffffu;

    __shared__ float4 carry0[4];
    __shared__ float4 carry1[4];
    __shared__ int carry_row[4];

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int chunk_begin = (int)blockIdx.x * CHUNK_NNZ;
    const int warp_begin = chunk_begin + warp * WARP_NNZ;
    const int warp_end = min(warp_begin + WARP_NNZ, nnz);
    const float4* __restrict__ b4 = reinterpret_cast<const float4*>(b);

    if (lane == 0) {
        carry_row[warp] = -1;
        carry0[warp] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        carry1[warp] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    __syncwarp();

    int e = warp_begin + lane;
    bool active = e < warp_end;
    int row = active ? row_ind[e] : -1;
    int col = active ? col_ind[e] : 0;
    float av = active ? values[e] : 0.0f;
    float4 q0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 q1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    if (active) {
        q0 = b4[((size_t)col << 1)];
        q1 = b4[((size_t)col << 1) + 1];
    }

    for (int base = warp_begin; base < warp_end; base += 32) {
        const int next_e = base + 32 + lane;
        const bool next_active = next_e < warp_end;

        int next_row = -1;
        int next_col = 0;
        float next_av = 0.0f;
        float4 next_q0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        float4 next_q1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

        if (next_active) {
            next_row = row_ind[next_e];
            next_col = col_ind[next_e];
            next_av = values[next_e];
            next_q0 = b4[((size_t)next_col << 1)];
            next_q1 = b4[((size_t)next_col << 1) + 1];
        }

        float x0 = av * q0.x;
        float x1 = av * q0.y;
        float x2 = av * q0.z;
        float x3 = av * q0.w;
        float x4 = av * q1.x;
        float x5 = av * q1.y;
        float x6 = av * q1.z;
        float x7 = av * q1.w;

        const int first_row = __shfl_sync(FULL_MASK, row, 0);
        const unsigned same_mask = __ballot_sync(FULL_MASK, active && row == first_row);
        const bool uniform_full = (same_mask == FULL_MASK);

        bool continues = false;
        if (lane == 0 && next_active) {
            continues = (next_row == row);
        }
        continues = __shfl_sync(FULL_MASK, (int)continues, 0) != 0;

        if (uniform_full) {
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                x0 += __shfl_down_sync(FULL_MASK, x0, offset);
                x1 += __shfl_down_sync(FULL_MASK, x1, offset);
                x2 += __shfl_down_sync(FULL_MASK, x2, offset);
                x3 += __shfl_down_sync(FULL_MASK, x3, offset);
                x4 += __shfl_down_sync(FULL_MASK, x4, offset);
                x5 += __shfl_down_sync(FULL_MASK, x5, offset);
                x6 += __shfl_down_sync(FULL_MASK, x6, offset);
                x7 += __shfl_down_sync(FULL_MASK, x7, offset);
            }

            if (lane == 0) {
                if (carry_row[warp] == row) {
                    float4 a = carry0[warp];
                    float4 d = carry1[warp];
                    x0 += a.x;
                    x1 += a.y;
                    x2 += a.z;
                    x3 += a.w;
                    x4 += d.x;
                    x5 += d.y;
                    x6 += d.z;
                    x7 += d.w;
                }

                if (continues) {
                    carry0[warp] = make_float4(x0, x1, x2, x3);
                    carry1[warp] = make_float4(x4, x5, x6, x7);
                    carry_row[warp] = row;
                } else {
                    atomic_add_k8(c, row, x0, x1, x2, x3, x4, x5, x6, x7);
                }
            }
        } else {
#pragma unroll
            for (int offset = 1; offset < 32; offset <<= 1) {
                int prev_row = __shfl_up_sync(FULL_MASK, row, offset);
                float p0 = __shfl_up_sync(FULL_MASK, x0, offset);
                float p1 = __shfl_up_sync(FULL_MASK, x1, offset);
                float p2 = __shfl_up_sync(FULL_MASK, x2, offset);
                float p3 = __shfl_up_sync(FULL_MASK, x3, offset);
                float p4 = __shfl_up_sync(FULL_MASK, x4, offset);
                float p5 = __shfl_up_sync(FULL_MASK, x5, offset);
                float p6 = __shfl_up_sync(FULL_MASK, x6, offset);
                float p7 = __shfl_up_sync(FULL_MASK, x7, offset);

                if (lane >= offset && row >= 0 && row == prev_row) {
                    x0 += p0;
                    x1 += p1;
                    x2 += p2;
                    x3 += p3;
                    x4 += p4;
                    x5 += p5;
                    x6 += p6;
                    x7 += p7;
                }
            }

            int following_row = __shfl_down_sync(FULL_MASK, row, 1);
            bool segment_end = active && (lane == 31 || following_row != row);

            if (segment_end) {
                if (row == first_row && carry_row[warp] == row) {
                    float4 a = carry0[warp];
                    float4 d = carry1[warp];
                    x0 += a.x;
                    x1 += a.y;
                    x2 += a.z;
                    x3 += a.w;
                    x4 += d.x;
                    x5 += d.y;
                    x6 += d.z;
                    x7 += d.w;
                }

                if (lane == 31 && continues) {
                    carry0[warp] = make_float4(x0, x1, x2, x3);
                    carry1[warp] = make_float4(x4, x5, x6, x7);
                    carry_row[warp] = row;
                } else {
                    atomic_add_k8(c, row, x0, x1, x2, x3, x4, x5, x6, x7);
                }
            }
        }

        active = next_active;
        row = next_row;
        col = next_col;
        av = next_av;
        q0 = next_q0;
        q1 = next_q1;
    }
}

void* spmm_preprocess(const int* d_row_ptr, const int* d_col_ind,
                      const float* d_values,
                      int num_rows, int num_cols, int nnz, int K)
{
    SpMMContext* ctx = new SpMMContext{};
    ctx->nnz = nnz;
    ctx->num_rows = num_rows;

    if (nnz > 0) {
        ctx->coo = csr_to_coo(d_row_ptr, d_col_ind, d_values,
                              num_rows, num_cols, nnz);
    }

    return ctx;
}

void spmm_compute(void* context,
                  const int* d_row_ptr, const int* d_col_ind,
                  const float* d_values,
                  const float* d_B, float* d_C,
                  int num_rows, int num_cols, int nnz, int K)
{
    SpMMContext* ctx = static_cast<SpMMContext*>(context);

    if (num_rows > 0) {
        int count4 = num_rows * 2;
        constexpr int threads = 256;
        int blocks = (count4 + threads - 1) / threads;
        zero_c_k8_kernel<<<blocks, threads>>>(
            reinterpret_cast<float4*>(d_C), count4);
    }

    if (ctx != nullptr && ctx->nnz > 0) {
        constexpr int CHUNK_NNZ = 2048;
        int blocks = (ctx->nnz + CHUNK_NNZ - 1) / CHUNK_NNZ;
        coo_spmm_k8_kernel<<<blocks, 128>>>(
            ctx->coo.d_row_ind,
            ctx->coo.d_col_ind,
            ctx->coo.d_values,
            d_B,
            d_C,
            ctx->nnz);
    }
}

void spmm_cleanup(void* context)
{
    SpMMContext* ctx = static_cast<SpMMContext*>(context);
    if (ctx != nullptr) {
        if (ctx->nnz > 0) {
            free_coo(ctx->coo);
        }
        delete ctx;
    }
}