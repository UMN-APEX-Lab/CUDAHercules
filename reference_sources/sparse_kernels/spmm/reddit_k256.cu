#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <cstddef>

struct PacketDesc {
    int row0;
    int row1;
    int row2;
    int row3;
    int dense;
};

struct SpmmContext {
    PacketDesc* d_packets;
    unsigned int* d_work_counter;
    int num_packets;
    int num_rows;
};

__device__ __forceinline__ float load_ro(const float* p) {
    return __ldg(p);
}

__global__ __launch_bounds__(128, 12)
void persistent_csr_spmm_256(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_ind,
    const float* __restrict__ values,
    const float* __restrict__ B,
    float* __restrict__ C,
    const PacketDesc* __restrict__ packets,
    unsigned int* __restrict__ work_counter,
    int num_packets)
{
    __shared__ PacketDesc packet;
    __shared__ unsigned int packet_id;

    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    while (true) {
        if (tid == 0) {
            packet_id = atomicAdd(work_counter, 1u);
            if (packet_id < (unsigned int)num_packets) {
                packet = packets[packet_id];
            }
        }
        __syncthreads();

        if (packet_id >= (unsigned int)num_packets) {
            return;
        }

        if (packet.dense == 0) {
            int row = -1;
            if (warp == 0) row = packet.row0;
            else if (warp == 1) row = packet.row1;
            else if (warp == 2) row = packet.row2;
            else row = packet.row3;

            if (row >= 0) {
                int begin = 0;
                int end = 0;
                if (lane == 0) {
                    begin = __ldg(row_ptr + row);
                    end = __ldg(row_ptr + row + 1);
                }
                begin = __shfl_sync(0xffffffffu, begin, 0);
                end = __shfl_sync(0xffffffffu, end, 0);

                float c0 = 0.0f;
                float c1 = 0.0f;
                float c2 = 0.0f;
                float c3 = 0.0f;
                float c4 = 0.0f;
                float c5 = 0.0f;
                float c6 = 0.0f;
                float c7 = 0.0f;

                for (int p = begin; p < end; ++p) {
                    int col = 0;
                    float a = 0.0f;
                    if (lane == 0) {
                        col = __ldg(col_ind + p);
                        a = load_ro(values + p);
                    }
                    col = __shfl_sync(0xffffffffu, col, 0);
                    a = __shfl_sync(0xffffffffu, a, 0);

                    const float* b = B + ((size_t)col << 8) + lane;
                    c0 = fmaf(a, load_ro(b), c0);
                    c1 = fmaf(a, load_ro(b + 32), c1);
                    c2 = fmaf(a, load_ro(b + 64), c2);
                    c3 = fmaf(a, load_ro(b + 96), c3);
                    c4 = fmaf(a, load_ro(b + 128), c4);
                    c5 = fmaf(a, load_ro(b + 160), c5);
                    c6 = fmaf(a, load_ro(b + 192), c6);
                    c7 = fmaf(a, load_ro(b + 224), c7);
                }

                float* out = C + ((size_t)row << 8) + lane;
                out[0] = c0;
                out[32] = c1;
                out[64] = c2;
                out[96] = c3;
                out[128] = c4;
                out[160] = c5;
                out[192] = c6;
                out[224] = c7;
            }
        } else {
            const int row = packet.row0;
            int begin = 0;
            int end = 0;
            if (lane == 0) {
                begin = __ldg(row_ptr + row);
                end = __ldg(row_ptr + row + 1);
            }
            begin = __shfl_sync(0xffffffffu, begin, 0);
            end = __shfl_sync(0xffffffffu, end, 0);

            const int k0 = lane + (warp << 5);
            const int k1 = k0 + 128;
            float acc0 = 0.0f;
            float acc1 = 0.0f;

            for (int p = begin; p < end; ++p) {
                int col = 0;
                float a = 0.0f;
                if (lane == 0) {
                    col = __ldg(col_ind + p);
                    a = load_ro(values + p);
                }
                col = __shfl_sync(0xffffffffu, col, 0);
                a = __shfl_sync(0xffffffffu, a, 0);

                const float* b = B + ((size_t)col << 8);
                acc0 = fmaf(a, load_ro(b + k0), acc0);
                acc1 = fmaf(a, load_ro(b + k1), acc1);
            }

            float* out = C + ((size_t)row << 8);
            out[k0] = acc0;
            out[k1] = acc1;
        }

        __syncthreads();
    }
}

void* spmm_preprocess(const int* d_row_ptr, const int* d_col_ind,
                      const float* d_values,
                      int num_rows, int num_cols, int nnz, int K)
{
    (void)d_col_ind;
    (void)d_values;
    (void)num_cols;
    (void)nnz;
    (void)K;

    SpmmContext* ctx = new SpmmContext;
    ctx->d_packets = nullptr;
    ctx->d_work_counter = nullptr;
    ctx->num_packets = 0;
    ctx->num_rows = num_rows;

    if (num_rows <= 0) {
        return ctx;
    }

    std::vector<int> h_row_ptr((size_t)num_rows + 1);
    cudaMemcpy(h_row_ptr.data(), d_row_ptr,
               ((size_t)num_rows + 1) * sizeof(int),
               cudaMemcpyDeviceToHost);

    constexpr int DENSE_THRESHOLD = 2891;

    std::vector<int> normal_rows;
    normal_rows.reserve((size_t)num_rows);
    std::vector<PacketDesc> packets;
    packets.reserve((size_t)num_rows / 4 + 5000);

    for (int r = 0; r < num_rows; ++r) {
        const int len = h_row_ptr[r + 1] - h_row_ptr[r];
        if (len > DENSE_THRESHOLD) {
            PacketDesc p;
            p.row0 = r;
            p.row1 = -1;
            p.row2 = -1;
            p.row3 = -1;
            p.dense = 1;
            packets.push_back(p);
        } else {
            normal_rows.push_back(r);
        }
    }

    std::sort(normal_rows.begin(), normal_rows.end(),
              [&h_row_ptr](int a, int b) {
                  return (h_row_ptr[a + 1] - h_row_ptr[a]) >
                         (h_row_ptr[b + 1] - h_row_ptr[b]);
              });

    for (size_t i = 0; i < normal_rows.size(); i += 4) {
        PacketDesc p;
        p.row0 = normal_rows[i];
        p.row1 = (i + 1 < normal_rows.size()) ? normal_rows[i + 1] : -1;
        p.row2 = (i + 2 < normal_rows.size()) ? normal_rows[i + 2] : -1;
        p.row3 = (i + 3 < normal_rows.size()) ? normal_rows[i + 3] : -1;
        p.dense = 0;
        packets.push_back(p);
    }

    ctx->num_packets = (int)packets.size();

    if (ctx->num_packets > 0) {
        cudaMalloc((void**)&ctx->d_packets,
                   (size_t)ctx->num_packets * sizeof(PacketDesc));
        cudaMemcpy(ctx->d_packets, packets.data(),
                   (size_t)ctx->num_packets * sizeof(PacketDesc),
                   cudaMemcpyHostToDevice);
    }

    cudaMalloc((void**)&ctx->d_work_counter, sizeof(unsigned int));
    return ctx;
}

void spmm_compute(void* context,
                  const int* d_row_ptr, const int* d_col_ind,
                  const float* d_values,
                  const float* d_B, float* d_C,
                  int num_rows, int num_cols, int nnz, int K)
{
    (void)num_cols;
    (void)nnz;
    (void)K;

    SpmmContext* ctx = static_cast<SpmmContext*>(context);
    if (ctx == nullptr || ctx->num_packets <= 0 || num_rows <= 0) {
        return;
    }

    cudaMemsetAsync(ctx->d_work_counter, 0, sizeof(unsigned int), 0);

    constexpr int BLOCKS = 2256;
    persistent_csr_spmm_256<<<BLOCKS, 128>>>(
        d_row_ptr,
        d_col_ind,
        d_values,
        d_B,
        d_C,
        ctx->d_packets,
        ctx->d_work_counter,
        ctx->num_packets);
}

void spmm_cleanup(void* context)
{
    SpmmContext* ctx = static_cast<SpmmContext*>(context);
    if (ctx == nullptr) return;

    if (ctx->d_packets) cudaFree(ctx->d_packets);
    if (ctx->d_work_counter) cudaFree(ctx->d_work_counter);
    delete ctx;
}