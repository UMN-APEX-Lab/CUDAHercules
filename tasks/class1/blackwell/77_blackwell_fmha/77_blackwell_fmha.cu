/*
  Task 77_blackwell_fmha: fused multi-head attention forward (GQA, optional causal mask, LSE
  output) on Blackwell (sm_100a / B200).

  Reference: cuDNN scaled-dot-product attention (cudnn-frontend graph API, ref_cudnn.cpp), the
  attention kernel that production Blackwell stacks (PyTorch SDPA, TRT-LLM, vLLM) call.

  Solution under test (KH_TEST_SOLUTION), implemented in solution.h, one overload per dtype:

    cudaError_t BlackwellFmha(int batch, int num_q_heads, int num_kv_heads, int seq_q, int seq_k,
                              int head_dim, bool causal,
                              __half const *Q, __half const *K, __half const *V,
                              __half *O, float *LSE);
    cudaError_t BlackwellFmha(... same with __nv_bfloat16 ...);

    Q, O: [batch, seq_q, num_q_heads, head_dim]   (row-major, contiguous)
    K, V: [batch, seq_k, num_kv_heads, head_dim]
    LSE : [batch, num_q_heads, seq_q]  FP32, natural log of sum_j exp(scale * q.k_j)
    O = softmax(scale * Q K^T [+ causal mask]) V,  scale = 1 / sqrt(head_dim),
    query head h uses KV head h / (num_q_heads / num_kv_heads).
    causal: key j is visible to query i iff j <= i (seq_q == seq_k in every causal test).

  Verification: inputs are full-mantissa random; the reference and the solution write to different
  buffers that are pre-filled with NaN; every element of O and of LSE must be finite and within
  tolerance of the reference. The solution is re-checked at the end on freshly regenerated inputs.
*/

#include <cstdio>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#ifdef KH_TEST_SOLUTION
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "solution.h"   // must stay before kh_verify.h: the solution must not see the helpers
#endif

#include "kh_verify.h"
#include "ref_cudnn.h"


enum DType { kFp16 = 0, kBf16 = 1 };

struct FmhaCase {
  const char *label;
  int B, Hq, Hkv, S, D;
  bool causal;
  DType dtype;
};

/// One prepared cuDNN SDPA graph; build / plan selection happens in the constructor, so run() is
/// just the execute call.
struct RefFmha {
  KhCudnnSdpa *p;
  RefFmha(FmhaCase const &c, void const *Q, void const *K, void const *V, void *O, float *LSE)
      : p(khcudnn_create(c.B, c.Hq, c.Hkv, c.S, c.D, c.causal, c.dtype == kBf16 ? 1 : 0, Q, K, V, O, LSE)) {}
  ~RefFmha() { khcudnn_destroy(p); }
  void run() { khcudnn_run(p); }
};

// Q is scaled up so that the attention distribution is peaked (logit std = kQScale / 3, about 2.7
// at hd128 after the 1/sqrt(d) scaling): a skipped key block, a wrong mask or a wrong KV head then
// changes the output visibly, which flat softmax data would hide behind the averaging.
static constexpr float kQScale = 8.0f;

// Tolerances: every element must satisfy |x - ref| <= max_rms * RMS(ref) + rel * |ref|, and the
// relative Frobenius error ||x - ref|| / ||ref|| must be <= fro. Same form for O and for LSE.
//
// Measured on B200 (S = 8K / 16K, hd128, GQA 64/8; exact FP32 FMHA with an FP32 online softmax vs
// the cuDNN reference; "bad" = deliberately wrong solutions run through this harness):
//                               O rel. Frobenius error      LSE rel. Frobenius error
//   exact FP32 kernel, BF16 out   2.4e-3 .. 2.5e-3             5e-7 .. 7e-7
//   exact FP32 kernel, FP16 out   3.0e-4                       7e-7
//   P rounded to BF16 / FP16      same (still allowed; flash kernels round P to 16 bits)
//   tolerance used (bf16 / fp16)  7.5e-3 / 1e-3                2e-5
//   bad: denominator +1%          1.0e-2          bad: scale 1.005/sqrt(d)  1.1e-2 .. 1.5e-2
//   bad: P in FP8 e4m3            1.4e-2 .. 1.7e-2   bad: V in FP8     2.4e-2
//   bad: BF16 PV accumulator      1.9e-2 .. 2.6e-2   bad: Q,K in FP8  6.6e-2 .. 7.3e-2
//   bad: skipped KV block         0.13 .. 0.21       bad: causal off by one row 7e-2 .. 9e-2
//   bad: ignored causal mask      0.85               bad: wrong KV head mapping 1.3
//   bad: LSE + 2e-3 / log2 LSE / LSE = row max   LSE Frobenius 1.6e-4 / 0.44 / 0.2 (O unaffected)
// Q,K rounded to BF16 on an FP16 problem: O 4.2e-3 (fails the fp16 tolerance; BF16 problems are
// unaffected because their inputs are already BF16). The per-element bounds are 3x above the worst
// element of a correct kernel (a correct kernel still passes with all tolerances scaled by 0.33).
#ifndef KH_TOL_SCALE
#define KH_TOL_SCALE 1.0   // development aid: scale all tolerances (compile-time only)
#endif
static KhTol kTolO_fp16(1.2e-2 * KH_TOL_SCALE, 1.0e-3 * KH_TOL_SCALE, 4.5e-3 * KH_TOL_SCALE);
static KhTol kTolO_bf16(6.0e-2 * KH_TOL_SCALE, 7.5e-3 * KH_TOL_SCALE, 2.4e-2 * KH_TOL_SCALE);
static KhTol kTolLSE(5.0e-5 * KH_TOL_SCALE, 2.0e-5 * KH_TOL_SCALE, 5.0e-5 * KH_TOL_SCALE);

// Q, K and V are all activations in prefill, so all three are regenerated with a new seed (`gen`)
// before every timed call of the reference and of the solution (outside the timed region), and
// before the first and the final correctness checks: an output (or any input-dependent
// intermediate) cached across calls is wrong on the next call.
template <class T>
static void fill_inputs(void *Q, void *K, void *V, size_t nq, size_t nkv, unsigned gen) {
  kh_fill_uniform(reinterpret_cast<T *>(Q), nq, 101u + 3u * gen, kQScale);
  kh_fill_uniform(reinterpret_cast<T *>(K), nkv, 202u + 3u * gen, 1.0f);
  kh_fill_uniform(reinterpret_cast<T *>(V), nkv, 303u + 3u * gen, 1.0f);
}

#ifdef KH_DUMP
static void dump(char const *name, void const *p, size_t bytes) {
  std::vector<char> h(bytes);
  KH_CHECK(cudaMemcpy(h.data(), p, bytes, cudaMemcpyDeviceToHost));
  FILE *f = std::fopen(name, "wb");
  std::fwrite(h.data(), 1, bytes, f);
  std::fclose(f);
}
#endif

/// Returns false on failure. `timed` = also print Ref / Kernel timings.
static bool run_case(FmhaCase const &c, bool timed, int trials) {
  size_t esz = c.dtype == kFp16 ? sizeof(__half) : sizeof(__nv_bfloat16);
  size_t nq = size_t(c.B) * c.S * c.Hq * c.D, nkv = size_t(c.B) * c.S * c.Hkv * c.D;
  size_t nl = size_t(c.B) * c.Hq * c.S;
  void *Q, *K, *V, *O_ref;
  float *LSE_ref;
  KH_CHECK(cudaMalloc(&Q, nq * esz));
  KH_CHECK(cudaMalloc(&K, nkv * esz));
  KH_CHECK(cudaMalloc(&V, nkv * esz));
  KH_CHECK(cudaMalloc(&O_ref, nq * esz));
  KH_CHECK(cudaMalloc(&LSE_ref, nl * sizeof(float)));
  unsigned gen = 0;   // input generation: every refresh() uses a new seed
  auto refresh = [&] {
    if (c.dtype == kFp16) fill_inputs<__half>(Q, K, V, nq, nkv, gen);
    else fill_inputs<__nv_bfloat16>(Q, K, V, nq, nkv, gen);
    ++gen;
  };
  refresh();
  kh_poison(O_ref, nq * esz);
  kh_poison(LSE_ref, nl * sizeof(float));
  KH_CHECK(cudaDeviceSynchronize());

  RefFmha ref(c, Q, K, V, O_ref, LSE_ref);
  ref.run();
  KH_CHECK(cudaDeviceSynchronize());

  KhTol tolO = c.dtype == kFp16 ? kTolO_fp16 : kTolO_bf16;
  bool ok = true;
  // The reference must have written every element (it was pre-filled with NaN).
  {
    bool f = c.dtype == kFp16;
    KhResult r = f ? kh_compare(reinterpret_cast<__half *>(O_ref), reinterpret_cast<__half *>(O_ref), nq, tolO)
                   : kh_compare(reinterpret_cast<__nv_bfloat16 *>(O_ref), reinterpret_cast<__nv_bfloat16 *>(O_ref), nq, tolO);
    KhResult l = kh_compare(LSE_ref, LSE_ref, nl, kTolLSE);
    if (!r.pass || !l.pass) {
      std::cerr << "Reference output incomplete or non-finite (O: " << r.nonfinite << ", LSE: "
                << l.nonfinite << ")" << std::endl;
      ok = false;
    }
  }

#ifdef KH_DUMP
  dump("dump_Q.bin", Q, nq * esz); dump("dump_K.bin", K, nkv * esz); dump("dump_V.bin", V, nkv * esz);
  dump("dump_O.bin", O_ref, nq * esz); dump("dump_LSE.bin", LSE_ref, nl * sizeof(float));
#endif

#ifdef KH_TEST_SOLUTION
  void *O_sol = nullptr;
  float *LSE_sol = nullptr;
  KH_CHECK(cudaMalloc(&O_sol, nq * esz));
  KH_CHECK(cudaMalloc(&LSE_sol, nl * sizeof(float)));
  auto reset_sol = [&] {
    kh_poison(O_sol, nq * esz);
    kh_poison(LSE_sol, nl * sizeof(float));
  };
  auto call_sol = [&] {
    cudaError_t e;
    if (c.dtype == kFp16)
      e = BlackwellFmha(c.B, c.Hq, c.Hkv, c.S, c.S, c.D, c.causal,
                        reinterpret_cast<__half const *>(Q), reinterpret_cast<__half const *>(K),
                        reinterpret_cast<__half const *>(V), reinterpret_cast<__half *>(O_sol), LSE_sol);
    else
      e = BlackwellFmha(c.B, c.Hq, c.Hkv, c.S, c.S, c.D, c.causal,
                        reinterpret_cast<__nv_bfloat16 const *>(Q), reinterpret_cast<__nv_bfloat16 const *>(K),
                        reinterpret_cast<__nv_bfloat16 const *>(V), reinterpret_cast<__nv_bfloat16 *>(O_sol), LSE_sol);
    if (e != cudaSuccess) {
      std::cerr << "Solution kernel failed: " << cudaGetErrorString(e) << std::endl;
      std::exit(1);
    }
  };
  auto check_sol = [&](char const *tag) {
    KhResult r = c.dtype == kFp16
        ? kh_compare(reinterpret_cast<__half *>(O_sol), reinterpret_cast<__half *>(O_ref), nq, tolO)
        : kh_compare(reinterpret_cast<__nv_bfloat16 *>(O_sol), reinterpret_cast<__nv_bfloat16 *>(O_ref), nq, tolO);
    KhResult l = kh_compare(LSE_sol, LSE_ref, nl, kTolLSE);
    std::string t = std::string(tag) + " O";
    kh_print_result(t.c_str(), r, tolO);
    t = std::string(tag) + " LSE";
    kh_print_result(t.c_str(), l, kTolLSE);
    return r.pass && l.pass;
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
    // Inputs are regenerated and outputs poisoned before every call (outside the timed region) on both sides.
    KhTime tr = kh_time([&] { ref.run(); }, trials, [&] {
      refresh();
      kh_poison(O_ref, nq * esz);
      kh_poison(LSE_ref, nl * sizeof(float));
    });
    kh_print_time("Ref", tr);
#ifdef KH_TEST_SOLUTION
    KhTime ts = kh_time(call_sol, trials, [&] { refresh(); reset_sol(); });
    kh_print_time("Kernel", ts);
    kh_print_speedup(tr, ts);
#else
    double flops = 4.0 * c.B * c.Hq * double(c.S) * c.S * c.D * (c.causal ? 0.5 : 1.0);
    std::printf("  reference: %.1f TFLOPS\n", flops / (tr.min_ms * 1e-3) * 1e-12);
#endif
  }

#ifdef KH_TEST_SOLUTION
  // Re-check on freshly generated inputs; always runs (also when timing is disabled).
  if (ok) {
    // Recompute the reference output, then the
    // solution must produce the right answer for these new inputs from a poisoned output.
    refresh();
    kh_poison(O_ref, nq * esz);
    kh_poison(LSE_ref, nl * sizeof(float));
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
  cudaFree(O_sol); cudaFree(LSE_sol);
#endif
  cudaFree(Q); cudaFree(K); cudaFree(V); cudaFree(O_ref); cudaFree(LSE_ref);
  return ok;
}

int main(int argc, char const **argv) {
  cudaDeviceProp props;
  KH_CHECK(cudaGetDeviceProperties(&props, 0));
  if (props.major != 10 || props.minor != 0) {
    std::cerr << "This task requires a Blackwell sm_100 GPU (B200)." << std::endl;
    return 0;
  }
  int trials = kh_trials();
  bool const keep_going = std::getenv("KH_KEEP_GOING") != nullptr;   // development aid
  bool failed = false;

  // Llama-3-70B and Qwen3-32B share the attention geometry: 64 query heads, 8 KV heads, head_dim
  // 128. Llama-3-8B: 32 / 8 heads. Prefill chunks of 8K / 16K tokens, one sequence.
  std::vector<FmhaCase> cases = {
      {"prefill_8k_causal_bf16",     1, 64, 8,  8192, 128, true,  kBf16},
      {"prefill_16k_causal_bf16",    1, 64, 8, 16384, 128, true,  kBf16},
      {"prefill_8k_full_bf16",       1, 64, 8,  8192, 128, false, kBf16},
      {"prefill_8k_causal_fp16",     1, 64, 8,  8192, 128, true,  kFp16},
      {"llama8b_8k_causal_bf16",     1, 32, 8,  8192, 128, true,  kBf16},
  };


  if (argc >= 7) {   // ./test B Hq Hkv S causal(0|1) dtype(0=fp16|1=bf16): one custom case
    FmhaCase s{"custom", atoi(argv[1]), atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), 128,
               atoi(argv[5]) != 0, atoi(argv[6]) ? kBf16 : kFp16};
    std::printf("\n=== custom: B=%d, Hq=%d, Hkv=%d, S=%d, causal=%d ===\n", s.B, s.Hq, s.Hkv, s.S, s.causal);
    if (!run_case(s, true, trials)) { std::cout << "Incorrect" << std::endl; return 1; }
    std::cout << "Passed" << std::endl;
    return 0;
  }

  for (auto const &s : cases) {
    std::printf("\n=== %s: B=%d, Hq=%d, Hkv=%d, S=%d, D=%d, causal=%d ===\n", s.label, s.B, s.Hq,
                s.Hkv, s.S, s.D, int(s.causal));
    if (!run_case(s, true, trials)) { std::cout << "Incorrect" << std::endl; if (!keep_going) return 1; failed = true; }
    else std::cout << "Passed" << std::endl;
  }
  return failed ? 1 : 0;
}
