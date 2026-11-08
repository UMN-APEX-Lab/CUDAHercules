/*
 * Task deepgemm_mqa_logits: FP8 multi-query-attention scoring ("lightning indexer" logits of
 * DeepSeek-V3.2) on Blackwell (sm_100a / B200).
 *
 * Reference: DeepGEMM (DeepSeek, MIT license), the SM100 `fp8_fp4_mqa_logits` kernel
 * (deep_gemm/include/deep_gemm/impls/sm100_mqa_logits.cuh, upstream commit 5081f85) launched exactly
 * as DeepGEMM's runtime does: block_q = 128 / num_heads, 384-token KV splits, 3 Q / KV / TMEM
 * stages, one persistent CTA per SM, optional scheduling-metadata kernel.
 *
 *   logits[i, j] = sum_h  w[i, h] * relu( dot_mx(Q[i, h, :], KV[ks[i] + j, :]) ),   0 <= j < ke[i] - ks[i]
 *
 * with MXFP8 operands: E4M3 values, one UE8M0 scale per 32 head-dim elements (4 packed in a uint32),
 *   dot_mx(q, k) = sum_b 2^(sfq[b]-127) * 2^(sfk[b]-127) * sum_{d in block b} q[d] * k[d].
 *
 * Solution under test (KH_TEST_SOLUTION), implemented in solution.h:
 *
 *   cudaError_t solution_mqa_logits(
 *       const __nv_fp8_e4m3* q,         // [seq_len, num_heads, head_dim]
 *       const uint32_t* q_sf,           // [seq_len, num_heads]   4 packed UE8M0 scales
 *       const __nv_fp8_e4m3* kv,        // [seq_len_kv, head_dim]
 *       const uint32_t* kv_sf,          // [seq_len_kv]           4 packed UE8M0 scales
 *       const __nv_bfloat16* weights,   // [seq_len, num_heads], row stride weights_stride elements
 *       int weights_stride,
 *       const int* cu_seq_len_k_start,  // [seq_len]
 *       const int* cu_seq_len_k_end,    // [seq_len]
 *       __nv_bfloat16* logits,          // [seq_len, logits_stride], compressed, see below
 *       int logits_stride,
 *       int seq_len, int seq_len_kv, int num_heads, int head_dim, cudaStream_t stream);
 *
 * Only logits[i, 0 .. ke[i]-ks[i]) is defined and checked (column j <-> KV position ks[i] + j); the
 * rest of each row (and the padding rows) is "don't care": the buffers are pre-filled with NaN and
 * those elements are never looked at.
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#ifdef KH_TEST_SOLUTION
#include "solution.h"   // must stay before kh_verify.h and the DeepGEMM headers
#endif

#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <deep_gemm/impls/sm100_mqa_logits.cuh>

#include "kh_verify.h"

#define KH_DRV_CHECK(call)                                                                     \
  do {                                                                                         \
    CUresult kh_r_ = (call);                                                                   \
    if (kh_r_ != CUDA_SUCCESS) {                                                               \
      const char *msg_ = nullptr;                                                              \
      cuGetErrorString(kh_r_, &msg_);                                                          \
      std::cerr << "CUDA driver error " << (msg_ ? msg_ : "?") << " at " << __FILE__ << ":"    \
                << __LINE__ << std::endl;                                                      \
      std::exit(2);                                                                            \
    }                                                                                          \
  } while (0)

using namespace deep_gemm;

// DeepSeek-V3.2 indexer: 64 heads of dimension 128.
static constexpr int kHeads = 64;
static constexpr int kHeadDim = 128;
static constexpr int kBlockQ = 128 / kHeads;          // 2 queries per Q tile
static constexpr int kSplitKV = 384;                  // DeepGEMM kMQALogitsSplitKV
static constexpr int kUmmaN = kBlockQ * kHeads;       // 128
static constexpr int kQStages = 3, kKVStages = 3, kTmemStages = 3;
static constexpr int kSpecThreads = 128, kMathThreads = 384;
static constexpr int kMetaThreads = 256;
static constexpr int kSegmentCost = 3, kSplitCost = 2;

enum Pattern { kTailCausal = 0, kContextParallel = 1, kPackedDocs = 2 };

struct MqaCase {
  const char *label;
  int S, SKV;
  Pattern pattern;
  int scheduled;   // reference: 0 = grid-stride Q blocks, 1 = scheduling-metadata kernel + scheduled kernel
};

static int align_up(int a, int b) { return (a + b - 1) / b * b; }

// --- data generation (full-entropy random FP8 codes / scales / weights) ---------------------------

__global__ void kh_fill_fp8_kernel(__nv_fp8_e4m3 *p, size_t n, unsigned seed) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n;
       i += size_t(gridDim.x) * blockDim.x)
    p[i] = __nv_fp8_e4m3(kh_uniform(i, seed));
}

// 4 UE8M0 bytes per word, exponents in [124, 131] (scale factors 2^-3 .. 2^4, never all equal).
__global__ void kh_fill_sf_kernel(uint32_t *p, size_t n, unsigned seed) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n;
       i += size_t(gridDim.x) * blockDim.x) {
    unsigned long long h = kh_hash64(i, seed);
    uint32_t w = 0;
    for (int b = 0; b < 4; ++b) w |= uint32_t(124 + ((h >> (8 * b)) & 7)) << (8 * b);
    p[i] = w;
  }
}

// --- reference plan ---------------------------------------------------------------------------

static CUtensorMap make_tma_2d(void *ptr, CUtensorMapDataType dtype, int elem_bytes,
                               uint64_t inner, uint64_t outer, uint32_t smem_inner,
                               uint32_t smem_outer, uint64_t outer_stride_elems, int swizzle_bytes) {
  CUtensorMap m;
  cuuint64_t dims[2] = {inner, outer};
  cuuint32_t box[2] = {smem_inner, smem_outer};
  cuuint64_t strides[1] = {outer_stride_elems * elem_bytes};
  cuuint32_t es[2] = {1, 1};
  if (swizzle_bytes) box[0] = swizzle_bytes / elem_bytes;
  CUtensorMapSwizzle sw = swizzle_bytes == 0    ? CU_TENSOR_MAP_SWIZZLE_NONE
                          : swizzle_bytes == 32 ? CU_TENSOR_MAP_SWIZZLE_32B
                          : swizzle_bytes == 64 ? CU_TENSOR_MAP_SWIZZLE_64B
                                                : CU_TENSOR_MAP_SWIZZLE_128B;
  KH_DRV_CHECK(cuTensorMapEncodeTiled(&m, dtype, 2, ptr, dims, strides, box, es,
                                      CU_TENSOR_MAP_INTERLEAVE_NONE, sw,
                                      CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                                      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  return m;
}

/// One prepared DeepGEMM launch. All host work (tensor maps, attributes) happens in the
/// constructor; run() is the kernel launch(es): optionally the scheduling-metadata kernel, then the
/// logits kernel -- the same sequence deep_gemm.fp8_fp4_mqa_logits(..., schedule_meta) executes.
template <bool kScheduled>
struct RefMqa {
  static constexpr int kNumSMsConst = 148;
  using QK = cutlass::float_e4m3_t;
  CUtensorMap tq, tsfq, tkv, tsfkv, tw;
  int S, SKV, stride_logits, num_sms, meta_smem, smem;
  int const *ks, *ke;
  uint32_t *meta = nullptr;
  nv_bfloat16 *logits;

  using Shared = layout::MQALogitsSharedStorage<kHeads, kHeadDim, kBlockQ, kSplitKV, kUmmaN, kQStages,
                                                kKVStages, kTmemStages, QK>;
  static auto kernel() {
    return &sm100_mqa_logits<kHeads, kHeadDim, kScheduled, kBlockQ, kSplitKV, kUmmaN, kQStages, kKVStages,
                             kTmemStages, kNumSMsConst, kSpecThreads, kMathThreads, QK>;
  }
  static auto meta_kernel() {
    return &sched::sm100_mqa_logits_metadata<kBlockQ, kSplitKV, kNumSMsConst, kSegmentCost, kSplitCost,
                                             kMetaThreads>;
  }

  RefMqa(void *q, uint32_t *q_sf, void *kv, uint32_t *kv_sf, nv_bfloat16 *w, int w_stride,
         int const *ks_, int const *ke_, nv_bfloat16 *logits_, int stride_logits_, int S_, int SKV_)
      : S(S_), SKV(SKV_), stride_logits(stride_logits_), ks(ks_), ke(ke_), logits(logits_) {
    int dev = 0;
    KH_CHECK(cudaGetDevice(&dev));
    KH_CHECK(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev));
    if (num_sms != kNumSMsConst) {
      std::cerr << "This task is built for a 148-SM B200 (found " << num_sms << ")" << std::endl;
      std::exit(2);
    }
    // Tensor maps exactly as DeepGEMM's sm100_mqa_logits() builds them (FP8, MX scale factors).
    tq = make_tma_2d(q, CU_TENSOR_MAP_DATA_TYPE_UINT8, 1, kHeadDim, uint64_t(S) * kHeads, kHeadDim,
                     kBlockQ * kHeads, kHeadDim, kHeadDim);
    tkv = make_tma_2d(kv, CU_TENSOR_MAP_DATA_TYPE_UINT8, 1, kHeadDim, SKV, kHeadDim, 128, kHeadDim,
                      kHeadDim);
    tsfkv = make_tma_2d(kv_sf, CU_TENSOR_MAP_DATA_TYPE_UINT32, 4, align_up(SKV, 4), 1, 128, 1, 0, 0);
    tsfq = make_tma_2d(q_sf, CU_TENSOR_MAP_DATA_TYPE_UINT32, 4, kHeads, S, kHeads, kBlockQ, kHeads, 0);
    tw = make_tma_2d(w, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, kHeads, S, w_stride, kBlockQ, w_stride, 0);
    smem = int(sizeof(Shared));
    KH_CHECK(cudaFuncSetAttribute(kernel(), cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    if (kScheduled) {
      int nqb = (S + kBlockQ - 1) / kBlockQ;
      meta_smem = (2 * nqb + kMetaThreads / 32) * int(sizeof(int));
      KH_CHECK(cudaFuncSetAttribute(meta_kernel(), cudaFuncAttributeMaxDynamicSharedMemorySize, meta_smem));
      int words = align_up(3 * num_sms, 2) + 2 * nqb;
      KH_CHECK(cudaMalloc(&meta, words * sizeof(int)));
    }
  }
  ~RefMqa() { if (meta) cudaFree(meta); }

  void run() {
    if (kScheduled)
      meta_kernel()<<<1, kMetaThreads, meta_smem>>>(uint32_t(S), uint32_t(SKV),
                                                    reinterpret_cast<uint32_t const *>(ks),
                                                    reinterpret_cast<uint32_t const *>(ke), meta);
    kernel()<<<num_sms, kSpecThreads + kMathThreads, smem>>>(
        uint32_t(S), uint32_t(SKV), uint32_t(stride_logits), reinterpret_cast<uint32_t const *>(ks),
        reinterpret_cast<uint32_t const *>(ke), meta, logits, tq, tsfq, tkv, tsfkv, tw);
    KH_CHECK(cudaGetLastError());
  }
};

struct RefHolder {
  std::function<void()> run;
  std::shared_ptr<void> keep;
};

static RefHolder make_ref(bool scheduled, void *q, uint32_t *q_sf, void *kv, uint32_t *kv_sf,
                          nv_bfloat16 *w, int w_stride, int const *ks, int const *ke,
                          nv_bfloat16 *logits, int stride_logits, int S, int SKV) {
  RefHolder h;
  if (scheduled) {
    auto p = std::make_shared<RefMqa<true>>(q, q_sf, kv, kv_sf, w, w_stride, ks, ke, logits, stride_logits, S, SKV);
    h.run = [p] { p->run(); };
    h.keep = p;
  } else {
    auto p = std::make_shared<RefMqa<false>>(q, q_sf, kv, kv_sf, w, w_stride, ks, ke, logits, stride_logits, S, SKV);
    h.run = [p] { p->run(); };
    h.keep = p;
  }
  return h;
}

// Tolerance on the defined region: |x - ref| <= max_rms * RMS(ref) + rel * |ref| per element and a
// relative Frobenius bound.
//
// The DeepGEMM kernel reduces the 64 heads in BF16 arithmetic (relu -> BF16, packed BF16 FMA with the
// BF16 weights, two partial sums of 32 heads) and stores BF16, so the reference itself is only
// accurate to a relative Frobenius error of 4.7e-3 against an exact FP64 evaluation of the same
// quantized inputs (max element error 0.096 * RMS; pure BF16 rounding of the exact result would be
// 1.6e-3). Measured on B200 (O = defined region of the logits, "bad" = deliberately wrong
// solutions run through this harness):
//                                     rel. Frobenius error   max|err| / RMS(ref)
//   exact FP32 kernel, BF16 store        5.0e-3                0.12
//   BF16-rounded dot product             4.7e-3                0.12
//   BF16 head accumulation (63 adds)     7.7e-3                0.24     (still allowed)
//   FP16 partial dot accumulation        5.0e-3                0.12     (indistinguishable from the reference's own error)
//   tolerance used                       1.5e-2                0.4      (3x above a correct kernel)
//   bad: Q or KV rounded to FP4 (E2M1)   0.25                  2.8 .. 3.7
//   bad: one head skipped (of 64)        0.125                 4.4 .. 5.0
//   bad: scale factors ignored           0.91                  10 .. 13
//   bad: UE8M0 bias 126 instead of 127   0.75                  8 .. 10
//   bad: only head_dim 96 of 128         0.49                  8.5 .. 11.6
//   bad: no ReLU / ReLU after weighting  1.0                   11 .. 15
//   bad: weights ignored / wrong row     6.0 / 1.4             18 .. 44
//   bad: ks ignored (absolute column)    unwritten region -> NaN (packed-docs case, ks != 0)
//   bad: ks + 1                          1.29
//   bad: one 128-column tile unwritten / half of the columns unwritten -> NaN
#ifndef KH_TOL_SCALE
#define KH_TOL_SCALE 1.0   // development aid: scale all tolerances (compile-time only)
#endif
static KhTol kTol(0.4 * KH_TOL_SCALE, 1.5e-2 * KH_TOL_SCALE, 0.03 * KH_TOL_SCALE);

#ifdef KH_DUMP
static void dump(char const *name, void const *p, size_t bytes) {
  std::vector<char> h(bytes);
  KH_CHECK(cudaMemcpy(h.data(), p, bytes, cudaMemcpyDeviceToHost));
  FILE *f = std::fopen(name, "wb");
  std::fwrite(h.data(), 1, bytes, f);
  std::fclose(f);
}
#endif

static void make_ranges(MqaCase const &c, std::vector<int> &ks, std::vector<int> &ke) {
  int S = c.S, SKV = c.SKV;
  ks.assign(S, 0);
  ke.assign(S, 0);
  if (c.pattern == kTailCausal) {
    // The queries are the last S tokens of an SKV-token sequence: row i sees keys [0, SKV - S + i].
    for (int i = 0; i < S; ++i) ke[i] = SKV - S + i + 1;
  } else if (c.pattern == kContextParallel) {
    // Zigzag context-parallel layout (DeepGEMM's test): the S queries are two chunks of S/2 tokens of
    // a cp_size * S token sequence; an arbitrary rank (cp_id = cp_size / 3) holds chunks cp_id and
    // 2 * cp_size - 1 - cp_id.
    int chunk = S / 2, cp_size = SKV / S, cp_id = cp_size / 3;
    for (int i = 0; i < chunk; ++i) {
      ke[i] = cp_id * chunk + i + 1;
      ke[i + chunk] = (2 * cp_size - 1 - cp_id) * chunk + i + 1;
    }
  } else {
    // Documents of random length (600 .. 3000 tokens) packed into one sequence: row i attends to the
    // earlier tokens of its own document only, [doc_start, i], so ks is not zero.
    int pos = 0, d = 0;
    while (pos < S) {
      int len = 600 + int(kh_hash64(d++, 31u) % 2401);
      len = std::min(len, S - pos);
      for (int i = pos; i < pos + len; ++i) { ks[i] = pos; ke[i] = i + 1; }
      pos += len;
    }
  }
}

/// Returns false on failure. `timed` = also print Ref / Kernel timings.
static bool run_case(MqaCase const &c, bool timed, int trials) {
  using bf16 = nv_bfloat16;
  int S = c.S, SKV = c.SKV;
  int S_pad = align_up(S, kBlockQ);
  int w_stride = align_up(kHeads * 2, 16) / 2;

  std::vector<int> h_ks, h_ke, h_len(S);
  make_ranges(c, h_ks, h_ke);
  int max_len = 0;
  size_t total = 0;
  double pairs = 0;
  for (int i = 0; i < S; ++i) {
    h_len[i] = h_ke[i] - h_ks[i];
    max_len = std::max(max_len, h_len[i]);
    total += h_len[i];
    pairs += h_len[i];
  }
  int stride_logits = align_up(align_up(max_len, kSplitKV), 1024 / int(sizeof(bf16)));

  __nv_fp8_e4m3 *q, *kv;
  uint32_t *q_sf, *kv_sf;
  bf16 *w, *logits_ref;
  int *ks, *ke, *len;
  KH_CHECK(cudaMalloc(&q, size_t(S_pad) * kHeads * kHeadDim));
  KH_CHECK(cudaMalloc(&kv, size_t(SKV) * kHeadDim));
  KH_CHECK(cudaMalloc(&q_sf, size_t(S_pad) * kHeads * 4));
  KH_CHECK(cudaMalloc(&kv_sf, size_t(align_up(SKV, 4)) * 4));
  KH_CHECK(cudaMalloc(&w, size_t(S_pad) * w_stride * sizeof(bf16)));
  KH_CHECK(cudaMalloc(&ks, S * sizeof(int)));
  KH_CHECK(cudaMalloc(&ke, S * sizeof(int)));
  KH_CHECK(cudaMalloc(&len, S * sizeof(int)));
  size_t n_logits = size_t(S_pad) * stride_logits;
  KH_CHECK(cudaMalloc(&logits_ref, n_logits * sizeof(bf16)));
  // KV is the key cache and stays fixed. The queries (E4M3 data + their MX scales) and the per-token
  // head weights are per-call inputs: they are regenerated together with a new seed (`gen`) before
  // every timed call of the reference and of the solution (outside the timed region) and before the
  // first and the final correctness checks. An output (or any query-dependent intermediate) cached
  // across calls is wrong on the next call.
  unsigned gen = 0;
  auto refresh = [&] {
    kh_fill_fp8_kernel<<<4096, 256>>>(q, size_t(S_pad) * kHeads * kHeadDim, 11u + 3u * gen);
    kh_fill_sf_kernel<<<4096, 256>>>(q_sf, size_t(S_pad) * kHeads, 33u + 3u * gen);
    kh_fill_uniform(w, size_t(S_pad) * w_stride, 55u + 3u * gen);
    KH_CHECK(cudaGetLastError());
    ++gen;
  };
  refresh();
  kh_fill_fp8_kernel<<<4096, 256>>>(kv, size_t(SKV) * kHeadDim, 22u);
  kh_fill_sf_kernel<<<4096, 256>>>(kv_sf, size_t(align_up(SKV, 4)), 44u);
  KH_CHECK(cudaGetLastError());
  KH_CHECK(cudaMemcpy(ks, h_ks.data(), S * sizeof(int), cudaMemcpyHostToDevice));
  KH_CHECK(cudaMemcpy(ke, h_ke.data(), S * sizeof(int), cudaMemcpyHostToDevice));
  KH_CHECK(cudaMemcpy(len, h_len.data(), S * sizeof(int), cudaMemcpyHostToDevice));
  kh_poison(logits_ref, n_logits * sizeof(bf16));
  KH_CHECK(cudaDeviceSynchronize());

  auto ref = make_ref(c.scheduled, q, q_sf, kv, kv_sf, w, w_stride, ks, ke, logits_ref, stride_logits, S, SKV);
  ref.run();
  KH_CHECK(cudaDeviceSynchronize());

  bool ok = true;
  {   // The reference must have written the whole defined region (it was pre-filled with NaN).
    KhResult r = kh_compare_ragged(logits_ref, logits_ref, S, stride_logits, len, total, kTol);
    if (!r.pass) {
      std::cerr << "Reference output incomplete or non-finite (non-finite = " << r.nonfinite << ")" << std::endl;
      ok = false;
    }
  }

#ifdef KH_DUMP
  dump("dump_q.bin", q, size_t(S) * kHeads * kHeadDim); dump("dump_kv.bin", kv, size_t(SKV) * kHeadDim);
  dump("dump_qsf.bin", q_sf, size_t(S) * kHeads * 4); dump("dump_kvsf.bin", kv_sf, size_t(SKV) * 4);
  dump("dump_w.bin", w, size_t(S) * w_stride * 2); dump("dump_ks.bin", ks, S * 4); dump("dump_ke.bin", ke, S * 4);
  dump("dump_logits.bin", logits_ref, n_logits * 2);
  std::printf("stride_logits=%d w_stride=%d\n", stride_logits, w_stride);
#endif

#ifdef KH_TEST_SOLUTION
  bf16 *logits_sol = nullptr;
  KH_CHECK(cudaMalloc(&logits_sol, n_logits * sizeof(bf16)));
  auto reset_sol = [&] { kh_poison(logits_sol, n_logits * sizeof(bf16)); };
  auto call_sol = [&] {
    cudaError_t e = solution_mqa_logits(q, q_sf, kv, kv_sf, w, w_stride, ks, ke, logits_sol, stride_logits,
                                        S, SKV, kHeads, kHeadDim, 0);
    if (e != cudaSuccess) {
      std::cerr << "Solution kernel failed: " << cudaGetErrorString(e) << std::endl;
      std::exit(1);
    }
  };
  auto check_sol = [&](char const *tag) {
    KhResult r = kh_compare_ragged(logits_sol, logits_ref, S, stride_logits, len, total, kTol);
    kh_print_result(tag, r, kTol);
    return r.pass;
  };
  if (ok) {
    reset_sol();
    KH_CHECK(cudaDeviceSynchronize());
    call_sol();
    KH_CHECK(cudaDeviceSynchronize());
    ok = check_sol("solution");
  }
#endif

  if (ok && timed && trials > 0) {
    KhTime tr = kh_time([&] { ref.run(); }, trials, [&] { refresh(); kh_poison(logits_ref, n_logits * sizeof(bf16)); });
    kh_print_time("Ref", tr);
#ifdef KH_TEST_SOLUTION
    KhTime ts = kh_time(call_sol, trials, [&] { refresh(); reset_sol(); });
    kh_print_time("Kernel", ts);
    kh_print_speedup(tr, ts);
#else
    std::printf("  reference: %.0f TFLOPS (FP8, %.2f G (q,kv) pairs)\n",
                2.0 * pairs * kHeads * kHeadDim / (tr.min_ms * 1e-3) * 1e-12, pairs * 1e-9);
#endif
  }

#ifdef KH_TEST_SOLUTION
  // Re-check on freshly generated inputs; always runs (also when timing is disabled).
  if (ok) {
    // Recompute the reference output, then the
    // solution must produce the right answer for these new inputs from a poisoned output.
    refresh();
    kh_poison(logits_ref, n_logits * sizeof(bf16));
    ref.run();
    KH_CHECK(cudaDeviceSynchronize());
    reset_sol();
    KH_CHECK(cudaDeviceSynchronize());
    call_sol();
    KH_CHECK(cudaDeviceSynchronize());
    ok = check_sol("solution (re-check, fresh inputs)");
  }
#endif

#ifdef KH_TEST_SOLUTION
  cudaFree(logits_sol);
#endif
  cudaFree(q); cudaFree(kv); cudaFree(q_sf); cudaFree(kv_sf); cudaFree(w); cudaFree(ks);
  cudaFree(ke); cudaFree(len); cudaFree(logits_ref);
  return ok;
}

int main(int argc, char const **argv) {
  cudaDeviceProp props;
  KH_CHECK(cudaGetDeviceProperties(&props, 0));
  if (props.major != 10 || props.minor != 0) {
    std::cerr << "This task requires a Blackwell sm_100 GPU (B200)." << std::endl;
    return 0;
  }
  KH_DRV_CHECK(cuInit(0));
  int trials = kh_trials();
  bool const keep_going = std::getenv("KH_KEEP_GOING") != nullptr;   // development aid
  bool failed = false;

  // DeepSeek-V3.2 lightning indexer on one B200: 64 indexer heads of dimension 128; S query tokens
  // (a prefill chunk) score SKV cached key tokens.
  std::vector<MqaCase> cases = {
      {"chunk4k_kv16k_causal",  4096, 16384, kTailCausal,      0},
      {"chunk4k_kv32k_causal",  4096, 32768, kTailCausal,      0},
      {"cp8k_kv32k_zigzag",     8192, 32768, kContextParallel, 0},
      {"chunk8k_kv64k_causal",  8192, 65536, kTailCausal,      0},
      {"packed_docs_8k",        8192,  8192, kPackedDocs,      0},
  };

#ifdef KH_SWEEP
  for (auto s : cases) {
    for (int sched = 0; sched < 2; ++sched) {
      s.scheduled = sched;
      std::printf("\n=== %s scheduled=%d ===\n", s.label, sched);
      run_case(s, true, 20);
    }
  }
  return 0;
#endif

  if (argc >= 4) {   // ./test S SKV pattern(0|1|2) : one custom case
    MqaCase s{"custom", atoi(argv[1]), atoi(argv[2]), Pattern(atoi(argv[3])), 1};
    std::printf("\n=== custom: S=%d, SKV=%d, pattern=%d ===\n", s.S, s.SKV, int(s.pattern));
    if (!run_case(s, true, trials)) { std::cout << "Incorrect" << std::endl; return 1; }
    std::cout << "Passed" << std::endl;
    return 0;
  }

  for (auto const &s : cases) {
    std::printf("\n=== %s: S=%d, SKV=%d, heads=%d, D=%d ===\n", s.label, s.S, s.SKV, kHeads, kHeadDim);
    if (!run_case(s, true, trials)) { std::cout << "Incorrect" << std::endl; if (!keep_going) return 1; failed = true; }
    else std::cout << "Passed" << std::endl;
  }
  return failed ? 1 : 0;
}
