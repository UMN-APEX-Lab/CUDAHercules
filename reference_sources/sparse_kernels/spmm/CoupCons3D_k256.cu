#include <cuda_runtime.h>
#include "sparse_formats.cuh"

struct SpmmContext {
    ELLData ell;
    int num_rows;
    int width;
};

#define FMA4(a, b, c) \
    do { \
        (c).x = fmaf((a), (b).x, (c).x); \
        (c).y = fmaf((a), (b).y, (c).y); \
        (c).z = fmaf((a), (b).z, (c).z); \
        (c).w = fmaf((a), (b).w, (c).w); \
    } while (0)

template<int WIDTH>
__global__ __launch_bounds__(128, 8)
void ell_spmm_subgroup4_k256(
    const int* __restrict__ ell_col,
    const float* __restrict__ ell_val,
    const float* __restrict__ B,
    float* __restrict__ C,
    int num_rows)
{
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int subgroup = lane >> 3;
    const int slane = lane & 7;
    const int src_lane = subgroup << 3;
    const unsigned mask = 0xffu << src_lane;

    const int row = ((int)blockIdx.x << 4) + (warp << 2) + subgroup;
    const bool valid = row < num_rows;
    const int safe_row = valid ? row : 0;

    const int kbase = (slane << 3);

    float4 c00 = make_float4(0.f, 0.f, 0.f, 0.f);
    float4 c01 = make_float4(0.f, 0.f, 0.f, 0.f);
    float4 c10 = make_float4(0.f, 0.f, 0.f, 0.f);
    float4 c11 = make_float4(0.f, 0.f, 0.f, 0.f);
    float4 c20 = make_float4(0.f, 0.f, 0.f, 0.f);
    float4 c21 = make_float4(0.f, 0.f, 0.f, 0.f);
    float4 c30 = make_float4(0.f, 0.f, 0.f, 0.f);
    float4 c31 = make_float4(0.f, 0.f, 0.f, 0.f);

    int lead_col = -1;
    float lead_val = 0.0f;
    if (slane == 0 && valid) {
        lead_col = __ldg(ell_col + safe_row);
        lead_val = __ldg(ell_val + safe_row);
    }

    int cur_col = __shfl_sync(mask, lead_col, src_lane);
    float cur_val = __shfl_sync(mask, lead_val, src_lane);

    #pragma unroll 1
    for (int slot = 0; slot < WIDTH; ++slot) {
        int next_col_local = -1;
        float next_val_local = 0.0f;

        if (slane == 0 && valid && slot + 1 < WIDTH) {
            const size_t pos = (size_t)(slot + 1) * (size_t)num_rows + (size_t)safe_row;
            next_col_local = __ldg(ell_col + pos);
            next_val_local = __ldg(ell_val + pos);
        }

        if (cur_col >= 0) {
            const float* bp = B + ((size_t)cur_col << 8) + kbase;

            float4 b00 = __ldg(reinterpret_cast<const float4*>(bp));
            float4 b01 = __ldg(reinterpret_cast<const float4*>(bp + 4));
            FMA4(cur_val, b00, c00);
            FMA4(cur_val, b01, c01);

            float4 b10 = __ldg(reinterpret_cast<const float4*>(bp + 64));
            float4 b11 = __ldg(reinterpret_cast<const float4*>(bp + 68));
            FMA4(cur_val, b10, c10);
            FMA4(cur_val, b11, c11);

            float4 b20 = __ldg(reinterpret_cast<const float4*>(bp + 128));
            float4 b21 = __ldg(reinterpret_cast<const float4*>(bp + 132));
            FMA4(cur_val, b20, c20);
            FMA4(cur_val, b21, c21);

            float4 b30 = __ldg(reinterpret_cast<const float4*>(bp + 192));
            float4 b31 = __ldg(reinterpret_cast<const float4*>(bp + 196));
            FMA4(cur_val, b30, c30);
            FMA4(cur_val, b31, c31);
        }

        cur_col = __shfl_sync(mask, next_col_local, src_lane);
        cur_val = __shfl_sync(mask, next_val_local, src_lane);
    }

    if (valid) {
        float* out = C + ((size_t)row << 8) + kbase;
        reinterpret_cast<float4*>(out)[0] = c00;
        reinterpret_cast<float4*>(out + 4)[0] = c01;
        reinterpret_cast<float4*>(out + 64)[0] = c10;
        reinterpret_cast<float4*>(out + 68)[0] = c11;
        reinterpret_cast<float4*>(out + 128)[0] = c20;
        reinterpret_cast<float4*>(out + 132)[0] = c21;
        reinterpret_cast<float4*>(out + 192)[0] = c30;
        reinterpret_cast<float4*>(out + 196)[0] = c31;
    }
}

__global__ __launch_bounds__(128, 8)
void ell_spmm_subgroup4_k256_generic(
    const int* __restrict__ ell_col,
    const float* __restrict__ ell_val,
    const float* __restrict__ B,
    float* __restrict__ C,
    int num_rows,
    int width)
{
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int subgroup = lane >> 3;
    const int slane = lane & 7;
    const int src_lane = subgroup << 3;
    const unsigned mask = 0xffu << src_lane;

    const int row = ((int)blockIdx.x << 4) + (warp << 2) + subgroup;
    const bool valid = row < num_rows;
    const int safe_row = valid ? row : 0;
    const int kbase = slane << 3;

    float4 c00 = make_float4(0.f, 0.f, 0.f, 0.f);
    float4 c01 = make_float4(0.f, 0.f, 0.f, 0.f);
    float4 c10 = make_float4(0.f, 0.f, 0.f, 0.f);
    float4 c11 = make_float4(0.f, 0.f, 0.f, 0.f);
    float4 c20 = make_float4(0.f, 0.f, 0.f, 0.f);
    float4 c21 = make_float4(0.f, 0.f, 0.f, 0.f);
    float4 c30 = make_float4(0.f, 0.f, 0.f, 0.f);
    float4 c31 = make_float4(0.f, 0.f, 0.f, 0.f);

    int lead_col = -1;
    float lead_val = 0.0f;
    if (slane == 0 && valid && width > 0) {
        lead_col = __ldg(ell_col + safe_row);
        lead_val = __ldg(ell_val + safe_row);
    }

    int cur_col = __shfl_sync(mask, lead_col, src_lane);
    float cur_val = __shfl_sync(mask, lead_val, src_lane);

    for (int slot = 0; slot < width; ++slot) {
        int next_col_local = -1;
        float next_val_local = 0.0f;

        if (slane == 0 && valid && slot + 1 < width) {
            const size_t pos = (size_t)(slot + 1) * (size_t)num_rows + (size_t)safe_row;
            next_col_local = __ldg(ell_col + pos);
            next_val_local = __ldg(ell_val + pos);
        }

        if (cur_col >= 0) {
            const float* bp = B + ((size_t)cur_col << 8) + kbase;

            float4 b00 = __ldg(reinterpret_cast<const float4*>(bp));
            float4 b01 = __ldg(reinterpret_cast<const float4*>(bp + 4));
            FMA4(cur_val, b00, c00);
            FMA4(cur_val, b01, c01);

            float4 b10 = __ldg(reinterpret_cast<const float4*>(bp + 64));
            float4 b11 = __ldg(reinterpret_cast<const float4*>(bp + 68));
            FMA4(cur_val, b10, c10);
            FMA4(cur_val, b11, c11);

            float4 b20 = __ldg(reinterpret_cast<const float4*>(bp + 128));
            float4 b21 = __ldg(reinterpret_cast<const float4*>(bp + 132));
            FMA4(cur_val, b20, c20);
            FMA4(cur_val, b21, c21);

            float4 b30 = __ldg(reinterpret_cast<const float4*>(bp + 192));
            float4 b31 = __ldg(reinterpret_cast<const float4*>(bp + 196));
            FMA4(cur_val, b30, c30);
            FMA4(cur_val, b31, c31);
        }

        cur_col = __shfl_sync(mask, next_col_local, src_lane);
        cur_val = __shfl_sync(mask, next_val_local, src_lane);
    }

    if (valid) {
        float* out = C + ((size_t)row << 8) + kbase;
        reinterpret_cast<float4*>(out)[0] = c00;
        reinterpret_cast<float4*>(out + 4)[0] = c01;
        reinterpret_cast<float4*>(out + 64)[0] = c10;
        reinterpret_cast<float4*>(out + 68)[0] = c11;
        reinterpret_cast<float4*>(out + 128)[0] = c20;
        reinterpret_cast<float4*>(out + 132)[0] = c21;
        reinterpret_cast<float4*>(out + 192)[0] = c30;
        reinterpret_cast<float4*>(out + 196)[0] = c31;
    }
}

void* spmm_preprocess(const int* d_row_ptr, const int* d_col_ind,
                      const float* d_values,
                      int num_rows, int num_cols, int nnz, int K)
{
    (void)K;
    SpmmContext* ctx = new SpmmContext;
    ctx->ell = csr_to_ell(d_row_ptr, d_col_ind, d_values,
                          num_rows, num_cols, nnz);
    ctx->num_rows = num_rows;
    ctx->width = ctx->ell.max_nnz_per_row;
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
    (void)num_rows;
    (void)num_cols;
    (void)nnz;
    (void)K;

    SpmmContext* ctx = static_cast<SpmmContext*>(context);
    if (ctx == nullptr || ctx->num_rows <= 0) return;

    const int blocks = (ctx->num_rows + 15) >> 4;

    if (ctx->width == 76) {
        ell_spmm_subgroup4_k256<76><<<blocks, 128>>>(
            ctx->ell.d_col_ind, ctx->ell.d_values,
            d_B, d_C, ctx->num_rows);
    } else {
        ell_spmm_subgroup4_k256_generic<<<blocks, 128>>>(
            ctx->ell.d_col_ind, ctx->ell.d_values,
            d_B, d_C, ctx->num_rows, ctx->width);
    }
}

void spmm_cleanup(void* context)
{
    SpmmContext* ctx = static_cast<SpmmContext*>(context);
    if (ctx != nullptr) {
        free_ell(ctx->ell);
        delete ctx;
    }
}