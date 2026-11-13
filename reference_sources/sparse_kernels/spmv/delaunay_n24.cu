#include <cuda_runtime.h>
#include <stdint.h>

struct SpmvContext {
    const float* last_x;
    int last_cols;
};

__global__ __launch_bounds__(128, 12)
void csr_spmv_subwarp4_ilp_kernel(const int* __restrict__ row_ptr,
                                   const int* __restrict__ col_ind,
                                   const float* __restrict__ values,
                                   const float* __restrict__ x,
                                   float* __restrict__ y,
                                   int num_rows)
{
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int group = lane >> 2;
    const int sublane = lane & 3;
    const int row = (int)blockIdx.x * 32 + warp * 8 + group;
    const unsigned mask = 0xffffffffu;

    int start = 0;
    int end = 0;

    if (sublane == 0 && row < num_rows) {
        start = row_ptr[row];
        end = row_ptr[row + 1];
    }

    start = __shfl_sync(mask, start, 0, 4);
    end = __shfl_sync(mask, end, 0, 4);

    float sum0 = 0.0f;
    float sum1 = 0.0f;

    if (row < num_rows) {
        int j = start + sublane;

        while (j < end) {
            const int c0 = col_ind[j];
            const float a0 = values[j];
            const int j1 = j + 4;

            if (j1 < end) {
                const int c1 = col_ind[j1];
                const float a1 = values[j1];

                const float x0 = __ldg(x + c0);
                const float x1 = __ldg(x + c1);

                sum0 = fmaf(a0, x0, sum0);
                sum1 = fmaf(a1, x1, sum1);
            } else {
                sum0 = fmaf(a0, __ldg(x + c0), sum0);
            }

            j += 8;
        }
    }

    float sum = sum0 + sum1;
    sum += __shfl_down_sync(mask, sum, 2, 4);
    sum += __shfl_down_sync(mask, sum, 1, 4);

    if (sublane == 0 && row < num_rows) {
        y[row] = sum;
    }
}

void* spmv_preprocess(const int* d_row_ptr, const int* d_col_ind,
                      const float* d_values,
                      int num_rows, int num_cols, int nnz)
{
    (void)d_row_ptr;
    (void)d_col_ind;
    (void)d_values;
    (void)num_rows;
    (void)nnz;

    size_t reserve = (size_t)num_cols * sizeof(float);
    const size_t max_reserve = 64ull * 1024ull * 1024ull;
    if (reserve > max_reserve) reserve = max_reserve;
    if (reserve != 0) {
        cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, reserve);
    }

    SpmvContext* ctx = new SpmvContext;
    ctx->last_x = nullptr;
    ctx->last_cols = 0;
    return ctx;
}

void spmv_compute(void* opaque,
                  const int* d_row_ptr, const int* d_col_ind,
                  const float* d_values,
                  const float* d_x, float* d_y,
                  int num_rows, int num_cols, int nnz)
{
    (void)nnz;

    if (num_rows <= 0) return;

    SpmvContext* ctx = static_cast<SpmvContext*>(opaque);
    if (ctx && (ctx->last_x != d_x || ctx->last_cols != num_cols)) {
        cudaStreamAttrValue attr = {};
        size_t bytes = (size_t)num_cols * sizeof(float);
        const size_t max_window = 64ull * 1024ull * 1024ull;
        if (bytes > max_window) bytes = max_window;

        if (bytes != 0) {
            attr.accessPolicyWindow.base_ptr = (void*)d_x;
            attr.accessPolicyWindow.num_bytes = bytes;
            attr.accessPolicyWindow.hitRatio = 1.0f;
            attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
            attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
            cudaStreamSetAttribute(0, cudaStreamAttributeAccessPolicyWindow, &attr);
        }

        ctx->last_x = d_x;
        ctx->last_cols = num_cols;
    }

    constexpr int threads = 128;
    constexpr int rows_per_block = 32;
    const int blocks = (num_rows + rows_per_block - 1) / rows_per_block;

    csr_spmv_subwarp4_ilp_kernel<<<blocks, threads>>>(
        d_row_ptr, d_col_ind, d_values, d_x, d_y, num_rows);
}

void spmv_cleanup(void* opaque)
{
    SpmvContext* ctx = static_cast<SpmvContext*>(opaque);
    delete ctx;
}