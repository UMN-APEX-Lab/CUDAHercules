/*
 * Flash Attention Backward — hdim=192, bf16, non-causal
 *
 * Source: flash-attention (Tri Dao)
 * Kernel source code in reference_sources/.
 */

#include "namespace_config.h"
#include "flash_bwd_launch_template.h"

namespace FLASH_NAMESPACE {

template<>
void run_mha_bwd_<cutlass::bfloat16_t, 192, false>(Flash_bwd_params &params, cudaStream_t stream) {
    run_mha_bwd_hdim192<cutlass::bfloat16_t, false>(params, stream);
}

} // namespace FLASH_NAMESPACE

static inline int round_up(int x, int m) {
    return (x + m - 1) / m * m;
}

extern "C" void launch_flash_attn_bwd(
    const void* dO,    // [B, H, Sq, D] bf16
    const void* Q,     // [B, H, Sq, D] bf16
    const void* K,     // [B, H, Sk, D] bf16
    const void* V,     // [B, H, Sk, D] bf16
    const void* O,     // [B, H, Sq, D] bf16
    const float* lse,  // [B, H, Sq]
    void* dQ,          // [B, H, Sq, D] bf16
    void* dK,          // [B, H, Sk, D] bf16
    void* dV,          // [B, H, Sk, D] bf16
    int B,
    int H,
    int Sq,
    int Sk,
    int D,
    float scale,
    cudaStream_t stream
) {
    FLASH_NAMESPACE::Flash_bwd_params params;
    memset(&params, 0, sizeof(params));

    int64_t q_batch_stride = static_cast<int64_t>(H) * Sq * D;
    int64_t k_batch_stride = static_cast<int64_t>(H) * Sk * D;

    params.q_ptr = const_cast<void*>(Q);
    params.k_ptr = const_cast<void*>(K);
    params.v_ptr = const_cast<void*>(V);
    params.o_ptr = const_cast<void*>(O);

    params.q_batch_stride = q_batch_stride;
    params.k_batch_stride = k_batch_stride;
    params.v_batch_stride = k_batch_stride;
    params.o_batch_stride = q_batch_stride;

    params.q_row_stride = D;
    params.k_row_stride = D;
    params.v_row_stride = D;
    params.o_row_stride = D;

    params.q_head_stride = static_cast<int64_t>(Sq) * D;
    params.k_head_stride = static_cast<int64_t>(Sk) * D;
    params.v_head_stride = static_cast<int64_t>(Sk) * D;
    params.o_head_stride = static_cast<int64_t>(Sq) * D;

    params.h = H;
    params.h_k = H;
    params.h_h_k_ratio = 1;
    params.b = B;
    params.seqlen_q = Sq;
    params.seqlen_k = Sk;
    params.seqlen_knew = 0;
    params.d = D;
    params.seqlen_q_rounded = round_up(Sq, 128);
    params.seqlen_k_rounded = round_up(Sk, 128);
    params.d_rounded = round_up(D, 32);
    params.rotary_dim = 0;
    params.total_q = B * Sq;

    params.scale_softmax = scale;
    params.scale_softmax_log2 = scale * M_LOG2E;

    params.softmax_lse_ptr = const_cast<float*>(lse);
    params.softmax_lseaccum_ptr = nullptr;
    params.oaccum_ptr = nullptr;
    params.p_ptr = nullptr;

    params.cu_seqlens_q = nullptr;
    params.cu_seqlens_k = nullptr;
    params.leftpad_k = nullptr;
    params.seqused_k = nullptr;
    params.blockmask = nullptr;
    params.knew_ptr = nullptr;
    params.vnew_ptr = nullptr;
    params.rotary_cos_ptr = nullptr;
    params.rotary_sin_ptr = nullptr;
    params.cache_batch_idx = nullptr;
    params.block_table = nullptr;
    params.alibi_slopes_ptr = nullptr;
    params.rng_state = nullptr;

    params.p_dropout = 1.0f;
    params.p_dropout_in_uint8_t = 255;
    params.rp_dropout = 1.0f;
    params.scale_softmax_rp_dropout = scale;

    params.is_causal = false;
    params.window_size_left = -1;
    params.window_size_right = -1;
    params.softcap = 0.0f;

    params.is_bf16 = true;
    params.is_seqlens_k_cumulative = true;
    params.is_rotary_interleaved = false;
    params.num_splits = 0;
    params.unpadded_lse = false;
    params.seqlenq_ngroups_swapped = false;

    // Backward-specific params
    params.do_ptr = const_cast<void*>(dO);
    params.dq_ptr = dQ;
    params.dk_ptr = dK;
    params.dv_ptr = dV;

    params.do_batch_stride = q_batch_stride;
    params.dq_batch_stride = q_batch_stride;
    params.dk_batch_stride = k_batch_stride;
    params.dv_batch_stride = k_batch_stride;

    params.do_row_stride = D;
    params.dq_row_stride = D;
    params.dk_row_stride = D;
    params.dv_row_stride = D;

    params.do_head_stride = static_cast<int64_t>(Sq) * D;
    params.dq_head_stride = static_cast<int64_t>(Sq) * D;
    params.dk_head_stride = static_cast<int64_t>(Sk) * D;
    params.dv_head_stride = static_cast<int64_t>(Sk) * D;

    params.deterministic = false;
    params.dq_accum_split_stride = 0;

    // Allocate dq_accum (fp32) and dsoftmax_sum
    int seqlen_q_rounded = round_up(Sq, 128);
    static float* dq_accum = nullptr; static size_t dq_accum_cached_size = 0;
    static float* dsoftmax_sum = nullptr; static size_t dsoftmax_sum_cached_size = 0;
    { size_t _need = (size_t)B * H * seqlen_q_rounded * D * sizeof(float);
        if (dq_accum_cached_size < _need) { if (dq_accum) cudaFree(dq_accum); cudaMalloc(&dq_accum, _need); dq_accum_cached_size = _need; } }
    { size_t _need = (size_t)B * H * seqlen_q_rounded * sizeof(float);
        if (dsoftmax_sum_cached_size < _need) { if (dsoftmax_sum) cudaFree(dsoftmax_sum); cudaMalloc(&dsoftmax_sum, _need); dsoftmax_sum_cached_size = _need; } }
    cudaMemsetAsync(dq_accum, 0, (size_t)B * H * seqlen_q_rounded * D * sizeof(float), stream);

    params.dq_accum_ptr = dq_accum;
    params.dk_accum_ptr = nullptr;
    params.dv_accum_ptr = nullptr;
    params.dsoftmax_sum = dsoftmax_sum;

    FLASH_NAMESPACE::run_mha_bwd_<cutlass::bfloat16_t, 192, false>(params, stream);

    cudaStreamSynchronize(stream);
    /* cudaFree(dq_accum); removed — buffer cached across invocations */
    /* cudaFree(dsoftmax_sum); removed — buffer cached across invocations */
}
