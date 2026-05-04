/*
 * SageAttention v3 reference: FP4 BlockScaled attention, head_dim=128, causal
 * Takes FP16 Q/K/V, internally quantizes to FP4, then runs BlockScaled attention.
 */

// ─── Part 1: FP4 Quantization Kernels ──────────────────────────────────────
#include "fp4_quant_kernels.cuh"

// ─── Part 2: Attention Kernel ──────────────────────────────────────────────
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cuda_bf16.h>

#ifndef KH_CUDA_CHECK
#define KH_CUDA_CHECK(err) do { \
    cudaError_t e_ = (err); \
    if (e_ != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)
#endif

#include "launch.h"

// ─── Part 3: Entry Point ──────────────────────────────────────────────────
extern "C"
void sageattn3_fp4_attn_hdim128_causal(
    const void* Q,
    const void* K,
    const void* V,
    void* O,
    int batch_size,
    int num_heads,
    int qo_len,
    int kv_len,
    float sm_scale,
    int is_causal_flag,
    cudaStream_t stream
) {
    constexpr int HEAD_DIM = 128;
    constexpr int BLOCK_SIZE = 128;

    auto round_up = [](int x, int m) { return (x + m - 1) / m * m; };
    int padded_kv = round_up(kv_len, BLOCK_SIZE);
    int seqlen_q_rounded = round_up(qo_len, 128);
    int seqlen_k_rounded = round_up(kv_len, 128);
    int sf_per_token = HEAD_DIM / 16;

    // ── Allocate temporary FP4 buffers ──────────────────────────────────
    uint8_t *q_fp4 = nullptr, *k_fp4 = nullptr, *v_fp4 = nullptr;
    uint8_t *sfq = nullptr, *sfk = nullptr, *sfv = nullptr;
    float *delta_s = nullptr, *softmax_lse = nullptr;

    size_t q_fp4_bytes = (size_t)batch_size * num_heads * qo_len * (HEAD_DIM / 2);
    size_t k_fp4_bytes = (size_t)batch_size * num_heads * kv_len * (HEAD_DIM / 2);
    size_t v_fp4_bytes = (size_t)batch_size * num_heads * HEAD_DIM * (padded_kv / 2);
    size_t sfq_bytes   = (size_t)batch_size * num_heads * qo_len * sf_per_token;
    size_t sfk_bytes   = (size_t)batch_size * num_heads * kv_len * sf_per_token;
    size_t sfv_bytes   = (size_t)batch_size * num_heads * HEAD_DIM * (padded_kv / 16);
    int n_groups       = (qo_len + 127) / 128;
    size_t ds_bytes    = (size_t)batch_size * num_heads * n_groups * kv_len * sizeof(float);
    size_t lse_bytes   = (size_t)batch_size * num_heads * qo_len * sizeof(float);

    KH_CUDA_CHECK(cudaMalloc(&q_fp4, q_fp4_bytes));
    KH_CUDA_CHECK(cudaMalloc(&k_fp4, k_fp4_bytes));
    KH_CUDA_CHECK(cudaMalloc(&v_fp4, v_fp4_bytes));
    KH_CUDA_CHECK(cudaMalloc(&sfq, sfq_bytes));
    KH_CUDA_CHECK(cudaMalloc(&sfk, sfk_bytes));
    KH_CUDA_CHECK(cudaMalloc(&sfv, sfv_bytes));
    KH_CUDA_CHECK(cudaMalloc(&delta_s, ds_bytes));
    KH_CUDA_CHECK(cudaMalloc(&softmax_lse, lse_bytes));

    // Zero all temp buffers (especially V and SFV for padding, delta_s for no-centering)
    KH_CUDA_CHECK(cudaMemsetAsync(v_fp4, 0, v_fp4_bytes, stream));
    KH_CUDA_CHECK(cudaMemsetAsync(sfv, 0, sfv_bytes, stream));
    KH_CUDA_CHECK(cudaMemsetAsync(delta_s, 0, ds_bytes, stream));

    // ── Quantize Q (permute=false) ──────────────────────────────────────
    {
        int stride_bz = num_heads * qo_len * HEAD_DIM;
        int stride_h  = qo_len * HEAD_DIM;
        int stride_seq = HEAD_DIM;
        int stride_bz_o  = num_heads * qo_len * (HEAD_DIM / 2);
        int stride_h_o   = qo_len * (HEAD_DIM / 2);
        int stride_seq_o = HEAD_DIM / 2;
        int stride_bz_sf  = num_heads * qo_len * sf_per_token;
        int stride_h_sf   = qo_len * sf_per_token;
        int stride_seq_sf = sf_per_token;

        dim3 block(BLOCK_SIZE * HEAD_DIM / CVT_FP4_ELTS_PER_THREAD);
        dim3 grid((qo_len + BLOCK_SIZE - 1) / BLOCK_SIZE, batch_size, num_heads);
        ref_fp4_quant_kernel<HEAD_DIM, BLOCK_SIZE, false, half>
            <<<grid, block, 0, stream>>>(
                (const half*)Q, q_fp4, sfq,
                batch_size, num_heads, qo_len,
                stride_bz, stride_h, stride_seq,
                stride_bz_o, stride_h_o, stride_seq_o,
                stride_bz_sf, stride_h_sf, stride_seq_sf);
    }

    // ── Quantize K (permute=true for MMA-friendly layout) ───────────────
    {
        int stride_bz = num_heads * kv_len * HEAD_DIM;
        int stride_h  = kv_len * HEAD_DIM;
        int stride_seq = HEAD_DIM;
        int stride_bz_o  = num_heads * kv_len * (HEAD_DIM / 2);
        int stride_h_o   = kv_len * (HEAD_DIM / 2);
        int stride_seq_o = HEAD_DIM / 2;
        int stride_bz_sf  = num_heads * kv_len * sf_per_token;
        int stride_h_sf   = kv_len * sf_per_token;
        int stride_seq_sf = sf_per_token;

        dim3 block(BLOCK_SIZE * HEAD_DIM / CVT_FP4_ELTS_PER_THREAD);
        dim3 grid((kv_len + BLOCK_SIZE - 1) / BLOCK_SIZE, batch_size, num_heads);
        ref_fp4_quant_kernel<HEAD_DIM, BLOCK_SIZE, true, half>
            <<<grid, block, 0, stream>>>(
                (const half*)K, k_fp4, sfk,
                batch_size, num_heads, kv_len,
                stride_bz, stride_h, stride_seq,
                stride_bz_o, stride_h_o, stride_seq_o,
                stride_bz_sf, stride_h_sf, stride_seq_sf);
    }

    // ── Quantize V (transpose) ──────────────────────────────────────────
    {
        int stride_bz = num_heads * kv_len * HEAD_DIM;
        int stride_h  = kv_len * HEAD_DIM;
        int stride_seq = HEAD_DIM;
        int stride_bz_o  = num_heads * HEAD_DIM * (padded_kv / 2);
        int stride_h_o   = HEAD_DIM * (padded_kv / 2);
        int stride_d_o   = padded_kv / 2;
        int stride_bz_sf  = num_heads * HEAD_DIM * (padded_kv / 16);
        int stride_h_sf   = HEAD_DIM * (padded_kv / 16);
        int stride_d_sf   = padded_kv / 16;

        dim3 block(BLOCK_SIZE * HEAD_DIM / CVT_FP4_ELTS_PER_THREAD);
        dim3 grid((kv_len + BLOCK_SIZE - 1) / BLOCK_SIZE, batch_size, num_heads);
        ref_fp4_quant_trans_kernel<HEAD_DIM, BLOCK_SIZE, half>
            <<<grid, block, 0, stream>>>(
                (const half*)V, v_fp4, sfv,
                batch_size, num_heads, kv_len,
                stride_bz, stride_h, stride_seq,
                stride_bz_o, stride_h_o, stride_d_o,
                stride_bz_sf, stride_h_sf, stride_d_sf);
    }

    // ── Set up Flash_fwd_params ─────────────────────────────────────────
    Flash_fwd_params params;
    memset(&params, 0, sizeof(params));

    params.q_ptr = q_fp4;
    params.k_ptr = k_fp4;
    params.v_ptr = v_fp4;
    params.o_ptr = O;
    params.sfq_ptr = sfq;
    params.sfk_ptr = sfk;
    params.sfv_ptr = sfv;
    params.delta_s_ptr = delta_s;
    params.softmax_lse_ptr = softmax_lse;

    // Q/K strides: [B, H, S, D] in FP4 element units
    params.q_row_stride = HEAD_DIM;
    params.q_head_stride = qo_len * HEAD_DIM;
    params.q_batch_stride = num_heads * qo_len * HEAD_DIM;
    params.k_row_stride = HEAD_DIM;
    params.k_head_stride = kv_len * HEAD_DIM;
    params.k_batch_stride = num_heads * kv_len * HEAD_DIM;

    // V transposed: [B, H, D, padded_kv] in FP4 element units
    params.v_row_stride = seqlen_k_rounded;
    params.v_head_stride = HEAD_DIM * seqlen_k_rounded;
    params.v_batch_stride = num_heads * HEAD_DIM * seqlen_k_rounded;

    // SF Q/K strides: [B, H, S, D/16] in bytes
    params.sfq_row_stride = sf_per_token;
    params.sfq_head_stride = qo_len * sf_per_token;
    params.sfq_batch_stride = num_heads * qo_len * sf_per_token;
    params.sfk_row_stride = sf_per_token;
    params.sfk_head_stride = kv_len * sf_per_token;
    params.sfk_batch_stride = num_heads * kv_len * sf_per_token;

    // SF V strides: [B, H, D, padded_kv/16] in bytes
    int sfv_per_dim = seqlen_k_rounded / 16;
    params.sfv_row_stride = sfv_per_dim;
    params.sfv_head_stride = HEAD_DIM * sfv_per_dim;
    params.sfv_batch_stride = num_heads * HEAD_DIM * sfv_per_dim;

    // Delta_s: [B, H, n_groups, kv_len]
    params.ds_row_stride = kv_len;
    params.ds_head_stride = n_groups * kv_len;
    params.ds_batch_stride = num_heads * n_groups * kv_len;

    // O: [B, H, S, D] bfloat16
    params.o_row_stride = HEAD_DIM;
    params.o_head_stride = qo_len * HEAD_DIM;
    params.o_batch_stride = num_heads * qo_len * HEAD_DIM;

    // Dimensions
    params.b = batch_size;
    params.h = num_heads;
    params.h_k = num_heads;
    params.h_h_k_ratio = 1;
    params.seqlen_q = qo_len;
    params.seqlen_k = kv_len;
    params.unpadded_seqlen_k = kv_len;
    params.seqlen_q_rounded = seqlen_q_rounded;
    params.seqlen_k_rounded = seqlen_k_rounded;
    params.d = HEAD_DIM;
    params.d_rounded = HEAD_DIM;

    params.head_divmod = cutlass::FastDivmod(num_heads);
    params.scale_softmax = sm_scale;
    params.scale_softmax_log2 = sm_scale * float(M_LOG2E);
    __half scale_softmax_log2_half = __float2half(params.scale_softmax_log2);
    __half2 scale_softmax_log2_half2 = __half2(scale_softmax_log2_half, scale_softmax_log2_half);
    params.scale_softmax_log2_half2 = reinterpret_cast<uint32_t&>(scale_softmax_log2_half2);

    params.is_causal = (is_causal_flag != 0);
    params.per_block_mean = false;
    params.seqlen_s = n_groups;
    params.is_bf16 = true;

    params.p_dropout = 1.0f;
    params.rp_dropout = 1.0f;
    params.scale_softmax_rp_dropout = params.scale_softmax;
    params.p_dropout_in_uint8_t = 255;
    params.window_size_left = -1;
    params.window_size_right = (is_causal_flag != 0) ? 0 : -1;
    params.is_seqlens_k_cumulative = true;

    // Tile count semaphore
    int* tile_count_semaphore = nullptr;
    KH_CUDA_CHECK(cudaMalloc(&tile_count_semaphore, sizeof(int)));
    if (is_causal_flag) {
        int val = 170;
        KH_CUDA_CHECK(cudaMemcpy(tile_count_semaphore, &val, sizeof(int), cudaMemcpyHostToDevice));
    }
    params.tile_count_semaphore = tile_count_semaphore;

    // ── Launch attention kernel ──────────────────────────────────────────
    using OType = cutlass::bfloat16_t;
    run_mha_fwd_<cutlass::nv_float4_t<cutlass::float_e2m1_t>, HEAD_DIM, OType>(params, stream);

    KH_CUDA_CHECK(cudaGetLastError());
    KH_CUDA_CHECK(cudaDeviceSynchronize());

    // ── Cleanup ─────────────────────────────────────────────────────────
    cudaFree(tile_count_semaphore);
    cudaFree(q_fp4);
    cudaFree(k_fp4);
    cudaFree(v_fp4);
    cudaFree(sfq);
    cudaFree(sfk);
    cudaFree(sfv);
    cudaFree(delta_s);
    cudaFree(softmax_lse);
}
