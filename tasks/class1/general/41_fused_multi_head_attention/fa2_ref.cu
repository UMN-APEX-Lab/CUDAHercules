// Reference FMHA forward for the 41_fused_multi_head_attention harness: FlashAttention-2 forward
// kernels (Dao-AILab/flash-attention, csrc/flash_attn/src, BSD-3-Clause) from
// reference_sources/flash_attn. Built as a separate translation unit (see Makefile); fa2_compat/
// provides the three small PyTorch headers the FlashAttention-2 sources include.
//
// Only the plain forward pass is instantiated (no dropout / ALiBi / softcap / sliding window).
// Tile configurations were chosen per problem shape by an offline sweep on an RTX PRO 6000
// Blackwell (SM120) over the FlashAttention-2 kernel-trait configurations.

#include <cmath>
#include <cstring>

#include "namespace_config.h"
#include "flash_fwd_launch_template.h"

#include "fa2_ref.h"

namespace kh_fa2_ref {

using Element = cutlass::half_t;
template <int D, int BM, int BN>
using Traits = ::Flash_fwd_kernel_traits<D, BM, BN, 4, false, false, Element>;

int round_up(int x, int m) { return (x + m - 1) / m * m; }

template <typename KernelTraits, bool Is_causal>
cudaError_t launch(FLASH_NAMESPACE::Flash_fwd_params const &params, cudaStream_t stream) {
  constexpr size_t smem_size = KernelTraits::kSmemSize;
  int const num_m_block = (params.seqlen_q + KernelTraits::kBlockM - 1) / KernelTraits::kBlockM;
  dim3 grid(num_m_block, params.b, params.h);
  bool const is_even_MN =
      params.seqlen_k % KernelTraits::kBlockN == 0 && params.seqlen_q % KernelTraits::kBlockM == 0;
  auto launch_one = [&](auto kernel) -> cudaError_t {
    if (smem_size >= 48 * 1024) {
      cudaError_t e = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(smem_size));
      if (e != cudaSuccess) return e;
    }
    kernel<<<grid, KernelTraits::kNThreads, smem_size, stream>>>(params);
    return cudaGetLastError();
  };
  // flash_fwd_kernel<Traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K,
  //                  Is_softcap, Return_softmax>
  if (is_even_MN) {
    return launch_one(&FLASH_NAMESPACE::flash_fwd_kernel<KernelTraits, false, Is_causal, false, false,
                                                         true, true, false, false>);
  }
  return launch_one(&FLASH_NAMESPACE::flash_fwd_kernel<KernelTraits, false, Is_causal, false, false,
                                                       false, true, false, false>);
}

template <typename KernelTraits>
cudaError_t launch(FLASH_NAMESPACE::Flash_fwd_params const &params, bool causal, cudaStream_t stream) {
  return causal ? launch<KernelTraits, true>(params, stream) : launch<KernelTraits, false>(params, stream);
}

}  // namespace kh_fa2_ref

using namespace kh_fa2_ref;

bool kh_ref_fmha_supported(int head_dim) { return head_dim == 64 || head_dim == 128; }

cudaError_t kh_ref_fmha_fwd(
    int batch, int seq_q, int seq_kv, int heads, int head_dim,
    void const *Q, void const *K, void const *V, void *O, float *lse,
    int64_t q_strideB, int64_t q_strideM, int64_t q_strideH,
    int64_t k_strideB, int64_t k_strideM, int64_t k_strideH,
    int64_t v_strideB, int64_t v_strideM, int64_t v_strideH,
    int64_t o_strideB, int64_t o_strideM, int64_t o_strideH,
    float scale, bool causal, cudaStream_t stream) {

  if (!kh_ref_fmha_supported(head_dim)) {
    return cudaErrorInvalidValue;
  }

  FLASH_NAMESPACE::Flash_fwd_params p;
  std::memset(&p, 0, sizeof(p));
  p.q_ptr = const_cast<void *>(Q);
  p.k_ptr = const_cast<void *>(K);
  p.v_ptr = const_cast<void *>(V);
  p.o_ptr = O;
  p.q_batch_stride = q_strideB; p.q_row_stride = q_strideM; p.q_head_stride = q_strideH;
  p.k_batch_stride = k_strideB; p.k_row_stride = k_strideM; p.k_head_stride = k_strideH;
  p.v_batch_stride = v_strideB; p.v_row_stride = v_strideM; p.v_head_stride = v_strideH;
  p.o_batch_stride = o_strideB; p.o_row_stride = o_strideM; p.o_head_stride = o_strideH;
  p.b = batch;
  p.h = heads;
  p.h_k = heads;
  p.h_h_k_ratio = 1;
  p.seqlen_q = seq_q;
  p.seqlen_k = seq_kv;
  p.seqlen_q_rounded = round_up(seq_q, 128);
  p.seqlen_k_rounded = round_up(seq_kv, 128);
  p.d = head_dim;
  p.d_rounded = round_up(head_dim, 32);
  p.total_q = batch * seq_q;
  p.scale_softmax = scale;
  p.scale_softmax_log2 = scale * float(M_LOG2E);
  p.softmax_lse_ptr = lse;
  p.p_dropout = 1.0f;
  p.p_dropout_in_uint8_t = 255;
  p.rp_dropout = 1.0f;
  p.scale_softmax_rp_dropout = scale;
  p.is_causal = causal;
  p.window_size_left = -1;
  p.window_size_right = causal ? 0 : -1;
  p.softcap = 0.0f;
  p.is_bf16 = false;
  p.is_seqlens_k_cumulative = true;
  p.num_splits = 1;

  if (head_dim == 64) {
    // Short sequences (e.g. ViT, 257 tokens) are fastest with 64 x 64 tiles, longer ones with
    // FlashAttention-2's default 128 x 128 tiles.
    if (seq_q <= 384) return launch<Traits<64, 64, 64>>(p, causal, stream);
    return launch<Traits<64, 128, 128>>(p, causal, stream);
  }
  // head_dim == 128: 128 x 64 tiles (FlashAttention-2's default outside SM8x), except 128 x 32
  // tiles (two CTAs per SM) for non-causal attention and for causal attention at >= 32K tokens.
  if (!causal || seq_q >= 32768) return launch<Traits<128, 128, 32>>(p, causal, stream);
  return launch<Traits<128, 128, 64>>(p, causal, stream);
}
