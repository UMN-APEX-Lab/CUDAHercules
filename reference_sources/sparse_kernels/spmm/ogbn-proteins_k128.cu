#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <cstddef>

struct WorkDesc {
    int row0;
    int row1;
    int row2;
    int row3;
    int begin;
    int end;
    int split;
};

struct SpmmContext {
    WorkDesc* d_desc;
    int* d_counter;
    int num_desc;
    int has_split;
};

__device__ __forceinline__ void load_metadata(
    const int* __restrict__ col_ind,
    const float* __restrict__ values,
    int p,
    int last,
    int num_cols,
    int& col,
    float& av,
    int& valid)
{
    col = 0;
    av = 0.0f;
    valid = 0;

    if (p < last) {
        const int candidate = __ldcs(col_ind + p);
        if ((unsigned)candidate < (unsigned)num_cols) {
            col = candidate;
            av = __ldcs(values + p);
            valid = 1;
        }
    }
}

__device__ __forceinline__ void consume_metadata(
    const float* __restrict__ B,
    int col,
    float av,
    int valid,
    int lane,
    float& x0,
    float& x1,
    float& x2,
    float& x3)
{
    #pragma unroll
    for (int src = 0; src < 32; ++src) {
        const int c = __shfl_sync(0xffffffff, col, src);
        const float a = __shfl_sync(0xffffffff, av, src);
        const int v = __shfl_sync(0xffffffff, valid, src);

        if (v) {
            const float4 bv = *reinterpret_cast<const float4*>(
                B + (size_t)c * 128 + (lane << 2));
            x0 = fmaf(a, bv.x, x0);
            x1 = fmaf(a, bv.y, x1);
            x2 = fmaf(a, bv.z, x2);
            x3 = fmaf(a, bv.w, x3);
        }
    }
}

__global__ __launch_bounds__(128, 12)
void spmm_csr_descriptor_k128(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_ind,
    const float* __restrict__ values,
    const float* __restrict__ B,
    float* __restrict__ C,
    const WorkDesc* __restrict__ descs,
    int num_desc,
    int num_cols,
    int* __restrict__ counter)
{
    __shared__ int s_row[4];
    __shared__ int s_begin;
    __shared__ int s_end;
    __shared__ int s_split;
    __shared__ float partial[4][32][4];

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    while (true) {
        if (tid == 0) {
            const int id = atomicAdd(counter, 1);
            if (id < num_desc) {
                const WorkDesc d = descs[id];
                s_row[0] = d.row0;
                s_row[1] = d.row1;
                s_row[2] = d.row2;
                s_row[3] = d.row3;
                s_begin = d.begin;
                s_end = d.end;
                s_split = d.split;
            } else {
                s_row[0] = -1;
            }
        }

        __syncthreads();

        if (s_row[0] < 0) return;

        const int split = s_split;
        const int row = split ? s_row[0] : s_row[warp];

        float x0 = 0.0f;
        float x1 = 0.0f;
        float x2 = 0.0f;
        float x3 = 0.0f;

        if (row >= 0) {
            int first;
            int last;
            int stride;

            if (split) {
                first = s_begin + (warp << 5);
                last = s_end;
                stride = 128;
            } else {
                first = row_ptr[row];
                last = row_ptr[row + 1];
                stride = 32;
            }

            int pbase = first;
            int col_cur;
            float av_cur;
            int valid_cur;
            load_metadata(col_ind, values, pbase + lane, last, num_cols,
                          col_cur, av_cur, valid_cur);

            while (pbase < last) {
                const int next_base = pbase + stride;

                int col_next;
                float av_next;
                int valid_next;

                load_metadata(col_ind, values, next_base + lane, last, num_cols,
                              col_next, av_next, valid_next);

                consume_metadata(B, col_cur, av_cur, valid_cur, lane,
                                 x0, x1, x2, x3);

                pbase = next_base;
                col_cur = col_next;
                av_cur = av_next;
                valid_cur = valid_next;
            }
        }

        if (!split) {
            if (row >= 0) {
                *reinterpret_cast<float4*>(
                    C + (size_t)row * 128 + (lane << 2)) =
                    make_float4(x0, x1, x2, x3);
            }
        } else {
            partial[warp][lane][0] = x0;
            partial[warp][lane][1] = x1;
            partial[warp][lane][2] = x2;
            partial[warp][lane][3] = x3;

            __syncthreads();

            if (warp == 0) {
                float r0 = partial[0][lane][0];
                float r1 = partial[0][lane][1];
                float r2 = partial[0][lane][2];
                float r3 = partial[0][lane][3];

                #pragma unroll
                for (int w = 1; w < 4; ++w) {
                    r0 += partial[w][lane][0];
                    r1 += partial[w][lane][1];
                    r2 += partial[w][lane][2];
                    r3 += partial[w][lane][3];
                }

                float* out = C + (size_t)row * 128 + (lane << 2);
                atomicAdd(out + 0, r0);
                atomicAdd(out + 1, r1);
                atomicAdd(out + 2, r2);
                atomicAdd(out + 3, r3);
            }
        }

        __syncthreads();
    }
}

__global__ void spmm_csr_generic(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_ind,
    const float* __restrict__ values,
    const float* __restrict__ B,
    float* __restrict__ C,
    int num_rows,
    int num_cols,
    int K)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    if (row >= num_rows) return;

    const int first = row_ptr[row];
    const int last = row_ptr[row + 1];

    for (int k = tid; k < K; k += blockDim.x) {
        float sum = 0.0f;
        for (int p = first; p < last; ++p) {
            const int col = col_ind[p];
            if ((unsigned)col < (unsigned)num_cols) {
                sum = fmaf(values[p], B[(size_t)col * K + k], sum);
            }
        }
        C[(size_t)row * K + k] = sum;
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
    (void)d_col_ind;
    (void)d_values;
    (void)num_cols;
    (void)nnz;
    (void)K;

    SpmmContext* ctx = new SpmmContext;
    ctx->d_desc = nullptr;
    ctx->d_counter = nullptr;
    ctx->num_desc = 0;
    ctx->has_split = 0;

    cudaMalloc(reinterpret_cast<void**>(&ctx->d_counter), sizeof(int));

    if (num_rows <= 0) return ctx;

    std::vector<int> h_ptr((size_t)num_rows + 1);
    cudaMemcpy(h_ptr.data(), d_row_ptr,
               ((size_t)num_rows + 1) * sizeof(int),
               cudaMemcpyDeviceToHost);

    constexpr int dense_threshold = 2461;
    constexpr int dense_chunk_nnz = 2048;

    std::vector<int> regular;
    regular.reserve(num_rows);

    std::vector<WorkDesc> work;
    work.reserve((size_t)num_rows / 4 + 8192);

    for (int r = 0; r < num_rows; ++r) {
        const int first = h_ptr[r];
        const int last = h_ptr[r + 1];
        const int len = last - first;

        if (len > dense_threshold) {
            for (int begin = first; begin < last; begin += dense_chunk_nnz) {
                WorkDesc d;
                d.row0 = r;
                d.row1 = -1;
                d.row2 = -1;
                d.row3 = -1;
                d.begin = begin;
                d.end = std::min(begin + dense_chunk_nnz, last);
                d.split = 1;
                work.push_back(d);
            }
            ctx->has_split = 1;
        } else {
            regular.push_back(r);
        }
    }

    for (size_t i = 0; i < regular.size(); i += 4) {
        WorkDesc d;
        d.row0 = regular[i];
        d.row1 = (i + 1 < regular.size()) ? regular[i + 1] : -1;
        d.row2 = (i + 2 < regular.size()) ? regular[i + 2] : -1;
        d.row3 = (i + 3 < regular.size()) ? regular[i + 3] : -1;
        d.begin = 0;
        d.end = 0;
        d.split = 0;
        work.push_back(d);
    }

    ctx->num_desc = static_cast<int>(work.size());

    if (!work.empty()) {
        cudaMalloc(reinterpret_cast<void**>(&ctx->d_desc),
                   work.size() * sizeof(WorkDesc));
        cudaMemcpy(ctx->d_desc, work.data(),
                   work.size() * sizeof(WorkDesc),
                   cudaMemcpyHostToDevice);
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
    (void)nnz;

    if (num_rows <= 0 || K <= 0) return;

    if (K != 128) {
        spmm_csr_generic<<<num_rows, 256>>>(
            d_row_ptr, d_col_ind, d_values, d_B, d_C,
            num_rows, num_cols, K);
        return;
    }

    SpmmContext* ctx = static_cast<SpmmContext*>(context);

    if (ctx == nullptr || ctx->d_counter == nullptr || ctx->num_desc <= 0) {
        cudaMemsetAsync(d_C, 0, (size_t)num_rows * 128 * sizeof(float), 0);
        return;
    }

    if (ctx->has_split) {
        cudaMemsetAsync(d_C, 0, (size_t)num_rows * 128 * sizeof(float), 0);
    }

    cudaMemsetAsync(ctx->d_counter, 0, sizeof(int), 0);

    constexpr int persistent_blocks = 2256;
    spmm_csr_descriptor_k128<<<persistent_blocks, 128>>>(
        d_row_ptr, d_col_ind, d_values, d_B, d_C,
        ctx->d_desc, ctx->num_desc, num_cols, ctx->d_counter);
}

void spmm_cleanup(void* context)
{
    SpmmContext* ctx = static_cast<SpmmContext*>(context);
    if (ctx != nullptr) {
        if (ctx->d_desc != nullptr) cudaFree(ctx->d_desc);
        if (ctx->d_counter != nullptr) cudaFree(ctx->d_counter);
        delete ctx;
    }
}