/*
 * Flash Attention 3 Backward — hdim=96, bf16, non-causal
 *
 * Source: flash-attention/hopper/ (FA3 Hopper kernels)
 */

#include "flash_bwd_launch_template.h"
#include "flash_bwd_preprocess_kernel.h"

// Stub: referenced by dead varlen code path but never called in non-varlen mode
void prepare_varlen_num_blocks(Flash_fwd_params &params, cudaStream_t stream,
                               bool packgqa, int blockM, int blockN, bool enable_pdl) {}

// Template instantiation: <Arch=90, T, kHeadDim, Has_softcap=false>
template<>
void run_mha_bwd_<90, cutlass::bfloat16_t, 96, false>(Flash_bwd_params &params, cudaStream_t stream) {
    run_mha_bwd_hdim96<90, cutlass::bfloat16_t, false>(params, stream);
}

static inline int round_up(int x, int m) {
    return (x + m - 1) / m * m;
}

extern "C" void launch_fa3_bwd(
    const void* Q,          // [B, H, Sq, D] bf16
    const void* K,          // [B, H, Sk, D] bf16
    const void* V,          // [B, H, Sk, D] bf16
    const void* O,          // [B, H, Sq, D] bf16
    const void* dO,         // [B, H, Sq, D] bf16
    void* dQ,               // [B, H, Sq, D] bf16
    void* dK,               // [B, H, Sk, D] bf16
    void* dV,               // [B, H, Sk, D] bf16
    const float* lse,       // [B, H, Sq]
    float* dsoftmax_sum,    // [B, H, Sq]
    float* dq_accum,        // [B, H, Sq, D] fp32 accumulator
    int B,
    int H,
    int Sq,
    int Sk,
    int D,
    float scale,
    cudaStream_t stream
) {
    Flash_bwd_params params;
    memset(&params, 0, sizeof(params));

    params.q_ptr = const_cast<void*>(Q);
    params.k_ptr = const_cast<void*>(K);
    params.v_ptr = const_cast<void*>(V);
    params.o_ptr = const_cast<void*>(O);
    params.do_ptr = const_cast<void*>(dO);
    params.dq_ptr = dQ;
    params.dk_ptr = dK;
    params.dv_ptr = dV;

    params.q_batch_stride = static_cast<int64_t>(H) * Sq * D;
    params.k_batch_stride = static_cast<int64_t>(H) * Sk * D;
    params.v_batch_stride = static_cast<int64_t>(H) * Sk * D;
    params.o_batch_stride = static_cast<int64_t>(H) * Sq * D;

    params.q_row_stride = D;
    params.k_row_stride = D;
    params.v_row_stride = D;
    params.o_row_stride = D;

    params.q_head_stride = static_cast<int64_t>(Sq) * D;
    params.k_head_stride = static_cast<int64_t>(Sk) * D;
    params.v_head_stride = static_cast<int64_t>(Sk) * D;
    params.o_head_stride = static_cast<int64_t>(Sq) * D;

    params.v_dim_stride = 1;

    params.do_batch_stride = static_cast<int64_t>(H) * Sq * D;
    params.do_row_stride = D;
    params.do_head_stride = static_cast<int64_t>(Sq) * D;

    params.dq_batch_stride = static_cast<int64_t>(H) * Sq * D;
    params.dk_batch_stride = static_cast<int64_t>(H) * Sk * D;
    params.dv_batch_stride = static_cast<int64_t>(H) * Sk * D;
    params.dq_row_stride = D;
    params.dk_row_stride = D;
    params.dv_row_stride = D;
    params.dq_head_stride = static_cast<int64_t>(Sq) * D;
    params.dk_head_stride = static_cast<int64_t>(Sk) * D;
    params.dv_head_stride = static_cast<int64_t>(Sk) * D;

    params.h = H;
    params.h_k = H;

    params.b = B;
    params.seqlen_q = Sq;
    params.seqlen_k = Sk;
    params.d = D;
    params.dv = D;
    params.d_rounded = round_up(D, 32);
    params.dv_rounded = round_up(D, 32);
    params.seqlen_q_rounded = round_up(Sq, 128);
    params.seqlen_k_rounded = round_up(Sk, 128);
    params.total_q = B * Sq;
    params.total_k = B * Sk;

    params.scale_softmax = scale;

    params.softmax_lse_ptr = const_cast<float*>(lse);
    params.dsoftmax_sum = dsoftmax_sum;
    params.dq_accum_ptr = dq_accum;

    // Convert LSE to log2 scale for backward kernel
    static float* d_lse_log2 = nullptr;
    static size_t lse_log2_size = 0;
    size_t needed = (size_t)B * H * Sq * sizeof(float);
    if (needed > lse_log2_size) {
        if (d_lse_log2) cudaFree(d_lse_log2);
        cudaMalloc(&d_lse_log2, needed);
        lse_log2_size = needed;
    }
    params.softmax_lse_log2_ptr = d_lse_log2;

    params.p_dropout = 1.0f;
    params.p_dropout_in_uint8_t = 255;
    params.rp_dropout = 1.0f;

    params.is_causal = false;
    params.is_local = false;
    params.window_size_left = -1;
    params.window_size_right = -1;
    params.softcap = 0.0f;
    params.num_splits = 1;
    params.deterministic = false;

    params.is_bf16 = true;
    params.is_e4m3 = false;

    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    params.num_sm = prop.multiProcessorCount;
    params.arch = 90;

    // Semaphores for dq/dk/dv accumulation
    static int* d_dq_sem = nullptr;
    static int* d_dk_sem = nullptr;
    static int* d_dv_sem = nullptr;
    static int* d_tile_sem = nullptr;
    static size_t sem_size = 0;
    int num_blocks_m = (Sq + 127) / 128;
    int num_blocks_n = (Sk + 127) / 128;
    size_t needed_sem = (size_t)(num_blocks_m * B * H + 1) * sizeof(int);
    if (needed_sem > sem_size) {
        if (d_dq_sem) cudaFree(d_dq_sem);
        if (d_dk_sem) cudaFree(d_dk_sem);
        if (d_dv_sem) cudaFree(d_dv_sem);
        if (d_tile_sem) cudaFree(d_tile_sem);
        cudaMalloc(&d_dq_sem, needed_sem);
        cudaMalloc(&d_dk_sem, needed_sem);
        cudaMalloc(&d_dv_sem, needed_sem);
        cudaMalloc(&d_tile_sem, sizeof(int));
        sem_size = needed_sem;
    }
    cudaMemsetAsync(d_dq_sem, 0, needed_sem, stream);
    cudaMemsetAsync(d_dk_sem, 0, needed_sem, stream);
    cudaMemsetAsync(d_dv_sem, 0, needed_sem, stream);
    cudaMemsetAsync(d_tile_sem, 0, sizeof(int), stream);
    params.dq_semaphore = d_dq_sem;
    params.dk_semaphore = d_dk_sem;
    params.dv_semaphore = d_dv_sem;
    params.tile_count_semaphore = d_tile_sem;

    run_mha_bwd_<90, cutlass::bfloat16_t, 96, false>(params, stream);
}
