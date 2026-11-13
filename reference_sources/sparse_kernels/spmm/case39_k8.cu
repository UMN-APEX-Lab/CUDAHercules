#include <cuda_runtime.h>
#include <stdint.h>
#include "sparse_formats.cuh"

struct SpMMContext {
    Tiled2DData tiled;
    int num_rows;
    int num_cols;
};

__global__ void zero_c_kernel(float* __restrict__ c, int n4, int rem)
{
    float4* c4 = reinterpret_cast<float4*>(c);
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int stride = (int)(blockDim.x * gridDim.x);

    for (; i < n4; i += stride) {
        c4[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    if (rem && blockIdx.x == 0 && threadIdx.x < rem) {
        c[n4 * 4 + threadIdx.x] = 0.0f;
    }
}

__global__ void tiled2d_spmm_k8_balanced4_kernel(
    const int* __restrict__ blk_row,
    const int* __restrict__ blk_col,
    const int* __restrict__ blk_nnz,
    const int* __restrict__ blk_type,
    const int64_t* __restrict__ blk_vp,
    const char* __restrict__ data,
    int num_slots,
    const float* __restrict__ B,
    float* __restrict__ C,
    int num_rows,
    int num_cols)
{
    __shared__ float bpanel[4][128];

    const int tid = (int)threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int group = lane >> 3;
    const int k = lane & 7;

    const unsigned group_mask = 0xffu << (group << 3);
    const int source_lane = group << 3;

    const int slot = (int)blockIdx.x * 4 + warp;
    if (slot >= num_slots) return;

    const int type = blk_type[slot];
    if (type == TILED2D_EMPTY) return;

    const int br = blk_row[slot];
    const int bc = blk_col[slot];
    const int bn = blk_nnz[slot];

    float* const my_bpanel = bpanel[warp];

    #pragma unroll
    for (int x = lane; x < 128; x += 32) {
        const int local_col = x >> 3;
        const int kk = x & 7;
        const int global_col = (bc << 4) + local_col;
        my_bpanel[x] = global_col < num_cols ? B[global_col * 8 + kk] : 0.0f;
    }

    __syncwarp();

    const int base_local_row = group;
    const int global_base_row = (br << 4) + base_local_row;

    float s0 = 0.0f;
    float s1 = 0.0f;
    float s2 = 0.0f;
    float s3 = 0.0f;

    const char* const p = data + blk_vp[slot];

    if (type == TILED2D_DENSE) {
        const float* const vals = reinterpret_cast<const float*>(p);

        #pragma unroll
        for (int lc = 0; lc < 16; ++lc) {
            float a = 0.0f;

            if (lane < 16) {
                a = vals[lane * 16 + lc];
            }

            const float a0 = __shfl_sync(0xffffffffu, a, group);
            const float a1 = __shfl_sync(0xffffffffu, a, group + 4);
            const float a2 = __shfl_sync(0xffffffffu, a, group + 8);
            const float a3 = __shfl_sync(0xffffffffu, a, group + 12);

            const float bv = my_bpanel[lc * 8 + k];
            s0 = fmaf(a0, bv, s0);
            s1 = fmaf(a1, bv, s1);
            s2 = fmaf(a2, bv, s2);
            s3 = fmaf(a3, bv, s3);
        }
    } else if (type == TILED2D_CSR) {
        const unsigned char* const rp = reinterpret_cast<const unsigned char*>(p);
        const unsigned char* const cols = rp + 17;
        const float* const vals = reinterpret_cast<const float*>(
            p + ((17 + bn + 3) & ~3));

        #pragma unroll
        for (int rr = 0; rr < 4; ++rr) {
            const int lr = base_local_row + (rr << 2);
            const int gr = (br << 4) + lr;

            if (gr < num_rows) {
                const int begin = (int)rp[lr];
                const int end = (int)rp[lr + 1];

                for (int q = begin; q < end; ++q) {
                    int lc = 0;
                    float av = 0.0f;

                    if (k == 0) {
                        lc = (int)cols[q];
                        av = vals[q];
                    }

                    lc = __shfl_sync(group_mask, lc, source_lane);
                    av = __shfl_sync(group_mask, av, source_lane);

                    const float prod = av * my_bpanel[lc * 8 + k];

                    if (rr == 0) s0 += prod;
                    else if (rr == 1) s1 += prod;
                    else if (rr == 2) s2 += prod;
                    else s3 += prod;
                }
            }
        }
    } else {
        const unsigned char* const coords = reinterpret_cast<const unsigned char*>(p);
        const float* const vals = reinterpret_cast<const float*>(
            p + ((bn + 3) & ~3));

        for (int q = 0; q < bn; ++q) {
            int coord = 0;
            float av = 0.0f;

            if (lane == 0) {
                coord = (int)coords[q];
                av = vals[q];
            }

            coord = __shfl_sync(0xffffffffu, coord, 0);
            av = __shfl_sync(0xffffffffu, av, 0);

            const int lr = coord & 15;
            const int lc = (coord >> 4) & 15;
            const float prod = av * my_bpanel[lc * 8 + k];

            if (lr == base_local_row) s0 += prod;
            else if (lr == base_local_row + 4) s1 += prod;
            else if (lr == base_local_row + 8) s2 += prod;
            else if (lr == base_local_row + 12) s3 += prod;
        }
    }

    if (global_base_row < num_rows && s0 != 0.0f)
        atomicAdd(&C[global_base_row * 8 + k], s0);
    if (global_base_row + 4 < num_rows && s1 != 0.0f)
        atomicAdd(&C[(global_base_row + 4) * 8 + k], s1);
    if (global_base_row + 8 < num_rows && s2 != 0.0f)
        atomicAdd(&C[(global_base_row + 8) * 8 + k], s2);
    if (global_base_row + 12 < num_rows && s3 != 0.0f)
        atomicAdd(&C[(global_base_row + 12) * 8 + k], s3);
}

void* spmm_preprocess(const int* d_row_ptr, const int* d_col_ind,
                      const float* d_values,
                      int num_rows, int num_cols, int nnz, int K)
{
    (void)K;

    SpMMContext* ctx = new SpMMContext;
    ctx->tiled = csr_to_tiled2d(
        d_row_ptr, d_col_ind, d_values,
        num_rows, num_cols, nnz, true);
    ctx->num_rows = num_rows;
    ctx->num_cols = num_cols;
    return ctx;
}

void spmm_compute(void* context,
                  const int* d_row_ptr, const int* d_col_ind,
                  const float* d_values,
                  const float* d_B, float* d_C,
                  int num_rows, int num_cols, int nnz, int K)
{
    (void)d_row_ptr;
    (void)d_col_ind;
    (void)d_values;
    (void)nnz;
    (void)K;

    SpMMContext* ctx = reinterpret_cast<SpMMContext*>(context);
    const int total_c = num_rows * 8;

    if (total_c > 0) {
        const int n4 = total_c >> 2;
        const int rem = total_c & 3;
        int zero_blocks = (n4 + 255) >> 8;
        if (zero_blocks < 188) zero_blocks = 188;
        if (zero_blocks > 376) zero_blocks = 376;

        zero_c_kernel<<<zero_blocks, 256>>>(d_C, n4, rem);
    }

    if (ctx != nullptr && ctx->tiled.num_slots > 0 && num_rows > 0) {
        const int blocks = (ctx->tiled.num_slots + 3) >> 2;

        tiled2d_spmm_k8_balanced4_kernel<<<blocks, 128>>>(
            ctx->tiled.d_blk_row,
            ctx->tiled.d_blk_col,
            ctx->tiled.d_blk_nnz,
            ctx->tiled.d_blk_type,
            ctx->tiled.d_blk_vp,
            ctx->tiled.d_data,
            ctx->tiled.num_slots,
            d_B,
            d_C,
            num_rows,
            num_cols);
    }
}

void spmm_cleanup(void* context)
{
    SpMMContext* ctx = reinterpret_cast<SpMMContext*>(context);
    if (ctx != nullptr) {
        free_tiled2d(ctx->tiled);
        delete ctx;
    }
}