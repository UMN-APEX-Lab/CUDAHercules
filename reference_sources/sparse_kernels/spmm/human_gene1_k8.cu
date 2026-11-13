#include <cuda_runtime.h>

static constexpr int kChunkNnz = 1024;
static constexpr int kWarpsPerBlock = 8;
static constexpr int kPersistentBlocks = 1128;

struct SpmmContext {
    unsigned int* d_work_counter;
};

__device__ __forceinline__ int find_row_for_nnz(
    const int* __restrict__ row_ptr,
    int num_rows,
    int p) {
    int lo = 0;
    int hi = num_rows;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (row_ptr[mid + 1] <= p) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

__device__ __forceinline__ void reduce8(
    float& s0, float& s1, float& s2, float& s3,
    float& s4, float& s5, float& s6, float& s7) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        s0 += __shfl_down_sync(0xffffffff, s0, offset);
        s1 += __shfl_down_sync(0xffffffff, s1, offset);
        s2 += __shfl_down_sync(0xffffffff, s2, offset);
        s3 += __shfl_down_sync(0xffffffff, s3, offset);
        s4 += __shfl_down_sync(0xffffffff, s4, offset);
        s5 += __shfl_down_sync(0xffffffff, s5, offset);
        s6 += __shfl_down_sync(0xffffffff, s6, offset);
        s7 += __shfl_down_sync(0xffffffff, s7, offset);
    }
}

__global__ __launch_bounds__(256, 5)
void csr_persistent_chunks_k8_kernel(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_ind,
    const float* __restrict__ values,
    const float* __restrict__ B,
    float* __restrict__ C,
    int num_rows,
    int nnz,
    unsigned int* __restrict__ work_counter,
    int num_chunks) {

    const int lane = threadIdx.x & 31;

    while (true) {
        unsigned int chunk = 0;
        if (lane == 0) {
            chunk = atomicAdd(work_counter, 1u);
        }
        chunk = __shfl_sync(0xffffffff, chunk, 0);

        if ((int)chunk >= num_chunks) {
            return;
        }

        const int chunk_begin = (int)chunk * kChunkNnz;
        const int chunk_end = min(chunk_begin + kChunkNnz, nnz);

        int row = find_row_for_nnz(row_ptr, num_rows, chunk_begin);
        int pos = chunk_begin;

        while (pos < chunk_end && row < num_rows) {
            const int row_begin = row_ptr[row];
            const int row_end_full = row_ptr[row + 1];

            if (row_end_full <= pos) {
                ++row;
                continue;
            }

            const int run_end = min(row_end_full, chunk_end);

            float s0 = 0.0f;
            float s1 = 0.0f;
            float s2 = 0.0f;
            float s3 = 0.0f;
            float s4 = 0.0f;
            float s5 = 0.0f;
            float s6 = 0.0f;
            float s7 = 0.0f;

            int p = pos + lane;
            if (p < run_end) {
                int cur_col = __ldg(col_ind + p);
                float cur_a = __ldg(values + p);

                while (true) {
                    const int next_p = p + 32;
                    const bool has_next = next_p < run_end;

                    int next_col = 0;
                    float next_a = 0.0f;
                    if (has_next) {
                        next_col = __ldg(col_ind + next_p);
                        next_a = __ldg(values + next_p);
                    }

                    const float* bp = B + ((size_t)cur_col << 3);
                    float4 b0 = *reinterpret_cast<const float4*>(bp);

                    s0 = fmaf(cur_a, b0.x, s0);
                    s1 = fmaf(cur_a, b0.y, s1);
                    s2 = fmaf(cur_a, b0.z, s2);
                    s3 = fmaf(cur_a, b0.w, s3);

                    float4 b1 = *reinterpret_cast<const float4*>(bp + 4);

                    s4 = fmaf(cur_a, b1.x, s4);
                    s5 = fmaf(cur_a, b1.y, s5);
                    s6 = fmaf(cur_a, b1.z, s6);
                    s7 = fmaf(cur_a, b1.w, s7);

                    if (!has_next) {
                        break;
                    }

                    p = next_p;
                    cur_col = next_col;
                    cur_a = next_a;
                }
            }

            reduce8(s0, s1, s2, s3, s4, s5, s6, s7);

            if (lane == 0) {
                float* cp = C + ((size_t)row << 3);
                const bool partial =
                    (row_begin < chunk_begin) || (row_end_full > chunk_end);

                if (partial) {
                    atomicAdd(cp + 0, s0);
                    atomicAdd(cp + 1, s1);
                    atomicAdd(cp + 2, s2);
                    atomicAdd(cp + 3, s3);
                    atomicAdd(cp + 4, s4);
                    atomicAdd(cp + 5, s5);
                    atomicAdd(cp + 6, s6);
                    atomicAdd(cp + 7, s7);
                } else {
                    *reinterpret_cast<float4*>(cp) = make_float4(s0, s1, s2, s3);
                    *reinterpret_cast<float4*>(cp + 4) = make_float4(s4, s5, s6, s7);
                }
            }

            pos = run_end;
            ++row;
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
    int K) {
    (void)d_row_ptr;
    (void)d_col_ind;
    (void)d_values;
    (void)num_rows;
    (void)num_cols;
    (void)nnz;
    (void)K;

    SpmmContext* ctx = new SpmmContext;
    ctx->d_work_counter = nullptr;
    cudaMalloc(&ctx->d_work_counter, sizeof(unsigned int));
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
    int K) {
    (void)num_cols;
    (void)K;

    if (num_rows <= 0) {
        return;
    }

    SpmmContext* ctx = static_cast<SpmmContext*>(context);

    cudaMemsetAsync(d_C, 0, (size_t)num_rows * 8 * sizeof(float), 0);

    if (nnz <= 0 || ctx == nullptr || ctx->d_work_counter == nullptr) {
        return;
    }

    cudaMemsetAsync(ctx->d_work_counter, 0, sizeof(unsigned int), 0);

    const int num_chunks = (nnz + kChunkNnz - 1) / kChunkNnz;
    const int needed_blocks = (num_chunks + kWarpsPerBlock - 1) / kWarpsPerBlock;
    const int blocks = min(kPersistentBlocks, max(1, needed_blocks));

    csr_persistent_chunks_k8_kernel<<<blocks, 256>>>(
        d_row_ptr,
        d_col_ind,
        d_values,
        d_B,
        d_C,
        num_rows,
        nnz,
        ctx->d_work_counter,
        num_chunks);
}

void spmm_cleanup(void* context) {
    SpmmContext* ctx = static_cast<SpmmContext*>(context);
    if (ctx != nullptr) {
        if (ctx->d_work_counter != nullptr) {
            cudaFree(ctx->d_work_counter);
        }
        delete ctx;
    }
}