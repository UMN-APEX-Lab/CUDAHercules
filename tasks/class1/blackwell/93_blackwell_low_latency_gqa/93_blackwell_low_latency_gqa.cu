/***************************************************************************************************
 * Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met: (BSD-3-Clause, see the CUTLASS LICENSE.txt)
 **************************************************************************************************/

/*
  Task 93_blackwell_low_latency_gqa: decode-phase grouped-query attention with attention sinks,
  sliding window and per-sequence KV lengths on Blackwell (sm_100a / B200).

  Reference: the CUTLASS example 93 "TGV" low-latency GQA kernel (split-KV flash decoding inside a
  thread-block cluster, tcgen05 MMA, BF16 in / BF16 out, FP32 accumulation), one configuration per
  problem chosen from a sweep on B200.

  Solution under test (KH_TEST_SOLUTION), implemented in solution.h:

    cudaError_t BlackwellGQA(int batch, int num_kv_heads, int num_q_heads, int max_seq_len,
                             int head_dim, int sliding_window,
                             __nv_bfloat16 const *Q, __nv_bfloat16 const *K,
                             __nv_bfloat16 const *V, int const *seq_lens,
                             float const *sinks, __nv_bfloat16 *O);

    Q, O  : [batch, num_q_heads, head_dim]                     (one query token per sequence)
    K, V  : [batch, max_seq_len, num_kv_heads, head_dim]       (KV cache, row-major, contiguous)
    seq_lens: [batch] int, valid cache length of each sequence (1 <= seq_len <= max_seq_len); the
            cache entries at positions >= seq_len are finite garbage and must not contribute
    sinks : [num_q_heads] FP32 attention-sink logits (natural-log domain), or nullptr (no sink)
    sliding_window: 0 = attend to all seq_len positions, W > 0 = only to the last W positions
            [max(0, seq_len - W), seq_len)
    O[b,h] = sum_j p_j V[b,j,h/g] / (sum_j p_j + exp(sink_h)),   p_j = exp(s_j - m) etc., i.e.
    softmax over {scores s_j = q.k_j / sqrt(head_dim)} U {sink_h} with the sink contributing no
    value; g = num_q_heads / num_kv_heads, query head h uses KV head h / g.

  Verification: Q/K/V/sinks are full-mantissa random, the reference and the solution read the same
  inputs and write different buffers that are pre-filled with NaN; every element of O must be
  finite and within tolerance of the reference output. The solution is re-checked at the end on freshly regenerated inputs.
*/

#include <cstdio>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#ifdef KH_TEST_SOLUTION
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "solution.h"   // must stay before kh_verify.h: the solution must not see the helpers
#endif

#include <cutlass/numeric_types.h>
#include <cute/tensor.hpp>

#include "tgv_gqa.cuh"
#include "kh_verify.h"

using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

struct GqaCase {
  const char *label;
  int B, kvH, qH, L, dH;
  int window;      // sliding window (0 = full attention)
  bool var_len;    // per-sequence lengths in [L/2, L] instead of all == L
  bool sinks;
  int cfg;         // reference configuration index (see make_ref)
};

// Q is scaled so the attention logits have std ~ kQScale/3 (= 2.7): the distribution is peaked,
// so skipping a block of keys, ignoring the window, the lengths or the sinks changes the result.
static constexpr float kQScale = 8.0f;
// Cache entries beyond seq_len are filled with finite garbage of this magnitude (relative to the
// valid data): a solution that reads them without masking is far off.
static constexpr float kGarbageScale = 64.0f;

template <typename T>
__global__ void kh_garble_kernel(T *p, size_t n, int L, int rowelems, int const *seq_lens,
                                 float scale) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n;
       i += size_t(gridDim.x) * blockDim.x) {
    size_t row = i / rowelems;           // (b, position)
    int pos = int(row % L), b = int(row / L);
    if (pos >= seq_lens[b]) p[i] = kh_from_float<T>(scale * kh_uniform(i, 777u));
  }
}

// Decode appends one new K/V entry per sequence at every step: refresh the last valid cache position
// (seq_len - 1) of every sequence with new random data.
template <typename T>
__global__ void kh_append_kernel(T *K, T *V, int L, int rowelems, int const *seq_lens, unsigned seed) {
  int b = blockIdx.x;
  size_t base = (size_t(b) * L + (seq_lens[b] - 1)) * rowelems;
  for (int e = threadIdx.x; e < rowelems; e += blockDim.x) {
    K[base + e] = kh_from_float<T>(kh_uniform(base + e, seed));
    V[base + e] = kh_from_float<T>(kh_uniform(base + e, seed + 1u));
  }
}

/// One prepared reference launch. The TGV host function builds the TMA descriptors and launches the
/// kernel on every call; that host work is captured once in a CUDA graph here, so run() is just the
/// (graph) launch, like the prepared CUTLASS plans of the GEMM tasks.
struct RefGqa {
  virtual ~RefGqa() {}
  virtual void run() = 0;
};

template <int CTA_dH, int MaxSplits, int NumRed, int Stage>
struct RefGqaImpl : RefGqa {
  using TypeQKV = cutlass::bfloat16_t;
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t exec = nullptr;

  RefGqaImpl(GqaCase const &c, void *K, void *Q, void *V, void *O, int *seq_lens, float *sinks) {
    int qHLocal = c.qH / c.kvH, qL = 1, kvL = c.L, dH = c.dH, kvH = c.kvH, BS = c.B;
    // K (kvL, dH, kvH, BS), V (dH, kvL, kvH, BS): [BS][kvL][kvH][dH];
    // Q ((qHLocal, qL), dH, kvH, BS), O (dH, (qHLocal, qL), kvH, BS): [BS][qL][kvH][qHLocal][dH]
    int sK_kvH = dH, sK_kvL = kvH * dH, sK_dH = 1, sK_BS = kvL * dH * kvH;
    int sQ_kvH = qHLocal * dH, sQ_qHLocal = dH, sQ_qL = kvH * qHLocal * dH, sQ_dH = 1,
        sQ_BS = kvH * qHLocal * dH * qL;
    float scale = 1.0f / std::sqrt(float(dH));
    cudaStream_t s;
    KH_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    KH_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
    TGV::gqa::gqa_host<TypeQKV, TypeQKV, float, 8, 1, 128, CTA_dH, Stage, Stage, MaxSplits, NumRed>(
        reinterpret_cast<TypeQKV *>(K), reinterpret_cast<TypeQKV *>(Q),
        reinterpret_cast<TypeQKV *>(V), reinterpret_cast<TypeQKV *>(O), seq_lens, sinks, kvH,
        qHLocal, qL, kvL, dH, BS, sK_kvH, sK_kvL, sK_dH, sK_BS, sQ_kvH, sQ_qHLocal, sQ_qL, sQ_dH,
        sQ_BS, sK_kvH, sK_kvL, sK_dH, sK_BS, sQ_kvH, sQ_qHLocal, sQ_qL, sQ_dH, sQ_BS, scale,
        c.window, /*pdl=*/false, -1, s);
    KH_CHECK(cudaStreamEndCapture(s, &graph));
    KH_CHECK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    KH_CHECK(cudaStreamDestroy(s));
  }
  ~RefGqaImpl() override {
    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
  }
  void run() override { KH_CHECK(cudaGraphLaunch(exec, 0)); }
};

static std::unique_ptr<RefGqa> make_ref(GqaCase const &c, void *K, void *Q, void *V, void *O,
                                        int *seq_lens, float *sinks) {
  // cfg: (MaxSplits, NumReductionCTA, pipeline stages) chosen from a sweep per case.
  // The swept configurations; the harness cases below name the best one per case.
  if (c.dH == 64) {
    switch (c.cfg) {
      case 0: return std::make_unique<RefGqaImpl<64, 8, 8, 3>>(c, K, Q, V, O, seq_lens, sinks);
      case 1: return std::make_unique<RefGqaImpl<64, 16, 8, 3>>(c, K, Q, V, O, seq_lens, sinks);
      case 2: return std::make_unique<RefGqaImpl<64, 4, 4, 3>>(c, K, Q, V, O, seq_lens, sinks);
      case 3: return std::make_unique<RefGqaImpl<64, 8, 4, 3>>(c, K, Q, V, O, seq_lens, sinks);
      case 4: return std::make_unique<RefGqaImpl<64, 16, 4, 3>>(c, K, Q, V, O, seq_lens, sinks);
      case 5: return std::make_unique<RefGqaImpl<64, 8, 8, 4>>(c, K, Q, V, O, seq_lens, sinks);
      case 6: return std::make_unique<RefGqaImpl<64, 2, 2, 3>>(c, K, Q, V, O, seq_lens, sinks);
      case 7: return std::make_unique<RefGqaImpl<64, 1, 1, 3>>(c, K, Q, V, O, seq_lens, sinks);
      case 8: return std::make_unique<RefGqaImpl<64, 4, 2, 3>>(c, K, Q, V, O, seq_lens, sinks);
      case 9: return std::make_unique<RefGqaImpl<64, 4, 1, 3>>(c, K, Q, V, O, seq_lens, sinks);
      default: return std::make_unique<RefGqaImpl<64, 16, 2, 3>>(c, K, Q, V, O, seq_lens, sinks);
    }
  }
  switch (c.cfg) {
    case 0: return std::make_unique<RefGqaImpl<128, 8, 8, 3>>(c, K, Q, V, O, seq_lens, sinks);
    case 1: return std::make_unique<RefGqaImpl<128, 16, 8, 3>>(c, K, Q, V, O, seq_lens, sinks);
    case 2: return std::make_unique<RefGqaImpl<128, 4, 4, 3>>(c, K, Q, V, O, seq_lens, sinks);
    case 3: return std::make_unique<RefGqaImpl<128, 8, 4, 3>>(c, K, Q, V, O, seq_lens, sinks);
    case 4: return std::make_unique<RefGqaImpl<128, 16, 4, 3>>(c, K, Q, V, O, seq_lens, sinks);
    case 5: return std::make_unique<RefGqaImpl<128, 2, 2, 3>>(c, K, Q, V, O, seq_lens, sinks);
    default: return std::make_unique<RefGqaImpl<128, 4, 2, 3>>(c, K, Q, V, O, seq_lens, sinks);
  }
}

// Tolerance: |x - ref| <= max_rms * RMS(ref) + rel * |ref| per element, relative Frobenius bound.
//
// Measured on B200 (O rel. Frobenius error vs the TGV reference; "bad" = deliberately wrong
// solutions run through this harness, worst-to-best over the cases that they affect):
//   exact FP32 kernel, BF16 out          0.9e-3 .. 1.7e-3   (BF16 rounding of the output)
//   P rounded to BF16 (still allowed)    0.9e-3 .. 1.5e-3
//   tolerance used                       6e-3 (3.5x above a correct kernel)
//   bad: denominator +1%                 1.0e-2           bad: scale 1.005/sqrt(d)   1.4e-2 .. 4.2e-2
//   bad: P rounded to FP8                2.6e-3 .. 1.6e-2 bad: V in FP8              2.4e-2
//   bad: K, Q in FP8                     7e-2 .. 8e-2     bad: BF16 PV accumulator   2.8e-3 (window) .. 2.7e-2
//   bad: skipped KV block                9e-3 .. 0.18     bad: last key dropped      1.9e-3 (B1, 128K keys) .. 0.19
//   bad: seq_lens ignored                350 .. 3700      bad: sliding window ignored 4.1 (window case)
//   bad: window off by +-1 key           4e-2 .. 8e-2 (window case)
//   bad: sinks ignored                   0.8 .. 26        bad: sink x0.5 / in log2 domain  0.2 .. 0.75 / 0.8 .. 9.7
//   bad: wrong GQA group mapping         1.3 .. 1.5       bad: half of the heads unwritten -> NaN
// The per-element bounds (max_rms, rel) are 3x above the worst element of a correct kernel; a bad
// variant that is too small to see on one case (e.g. one missing key out of 128K) is caught on
// the others. (Reference vs an independent FP64 computation: 1.7e-3, the same as pure BF16
// rounding of the exact result, 1.6e-3.)
#ifndef KH_TOL_SCALE
#define KH_TOL_SCALE 1.0   // development aid: scale all tolerances (compile-time only)
#endif
static KhTol kTolO(6.0e-2 * KH_TOL_SCALE, 6.0e-3 * KH_TOL_SCALE, 2.4e-2 * KH_TOL_SCALE);

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
static bool run_case(GqaCase const &c, bool timed, int trials) {
  using bf16 = __nv_bfloat16;
  size_t nQ = size_t(c.B) * c.qH * c.dH;
  size_t nKV = size_t(c.B) * c.L * c.kvH * c.dH;
  bf16 *Q, *K, *V, *O_ref;
  float *sinks = nullptr;
  int *seq_lens;
  KH_CHECK(cudaMalloc(&Q, nQ * sizeof(bf16)));
  KH_CHECK(cudaMalloc(&K, nKV * sizeof(bf16)));
  KH_CHECK(cudaMalloc(&V, nKV * sizeof(bf16)));
  KH_CHECK(cudaMalloc(&O_ref, nQ * sizeof(bf16)));
  KH_CHECK(cudaMalloc(&seq_lens, c.B * sizeof(int)));
  kh_fill_uniform(Q, nQ, 11u, kQScale);
  kh_fill_uniform(K, nKV, 22u, 1.0f);
  kh_fill_uniform(V, nKV, 33u, 1.0f);
  {
    std::vector<int> h(c.B);
    for (int b = 0; b < c.B; ++b)
      h[b] = c.var_len ? c.L / 2 + int(kh_hash64(b, 5u) % unsigned(c.L / 2 + 1)) : c.L;
    if (c.var_len) h[0] = c.L;   // the longest sequence is always present
    KH_CHECK(cudaMemcpy(seq_lens, h.data(), c.B * sizeof(int), cudaMemcpyHostToDevice));
    if (c.sinks) {
      // Sink logits near the typical log-sum-exp of the scores, so that the sink carries between
      // about 7% and 92% of the softmax mass and ignoring it (or mis-scaling it) is visible.
      float sigma = kQScale / 3.0f, avg_len = 0.75f * c.L;
      float base = std::log(c.var_len ? avg_len : float(c.L)) + 0.5f * sigma * sigma;
      std::vector<float> hs(c.qH);
      for (int i = 0; i < c.qH; ++i) hs[i] = base + 2.5f * kh_uniform(i, 99u);
      KH_CHECK(cudaMalloc(&sinks, c.qH * sizeof(float)));
      KH_CHECK(cudaMemcpy(sinks, hs.data(), c.qH * sizeof(float), cudaMemcpyHostToDevice));
    }
  }
  kh_garble_kernel<bf16><<<4096, 256>>>(K, nKV, c.L, c.kvH * c.dH, seq_lens, kGarbageScale);
  kh_garble_kernel<bf16><<<4096, 256>>>(V, nKV, c.L, c.kvH * c.dH, seq_lens, kGarbageScale);
  KH_CHECK(cudaGetLastError());
  // Per-call inputs: the query (new token) and the newest KV-cache entry of every sequence are
  // regenerated with a new seed (`gen`) before every timed call of the reference and of the
  // solution (outside the timed region) and before the first and the final correctness checks. The
  // rest of the KV cache is a cache and stays fixed; an output (or any query-dependent
  // intermediate) cached across calls is wrong on the next call.
  unsigned gen = 0;
  auto refresh = [&] {
    kh_fill_uniform(Q, nQ, 11u + 3u * gen, kQScale);
    kh_append_kernel<bf16><<<c.B, 256>>>(K, V, c.L, c.kvH * c.dH, seq_lens, 2000u + 3u * gen);
    KH_CHECK(cudaGetLastError());
    ++gen;
  };
  refresh();
  kh_poison(O_ref, nQ * sizeof(bf16));
  KH_CHECK(cudaDeviceSynchronize());

  auto ref = make_ref(c, K, Q, V, O_ref, seq_lens, sinks);
  ref->run();
  KH_CHECK(cudaDeviceSynchronize());

  bool ok = true;
  {   // The reference must have written every element (it was pre-filled with NaN).
    KhResult r = kh_compare(O_ref, O_ref, nQ, kTolO);
    if (!r.pass) {
      std::cerr << "Reference output incomplete or non-finite (non-finite = " << r.nonfinite
                << ")" << std::endl;
      ok = false;
    }
  }

#ifdef KH_DUMP
  dump("dump_Q.bin", Q, nQ * 2); dump("dump_K.bin", K, nKV * 2); dump("dump_V.bin", V, nKV * 2);
  dump("dump_O.bin", O_ref, nQ * 2);
  dump("dump_seq.bin", seq_lens, c.B * 4);
  if (sinks) dump("dump_sinks.bin", sinks, c.qH * 4);
#endif

#ifdef KH_TEST_SOLUTION
  bf16 *O_sol = nullptr;
  KH_CHECK(cudaMalloc(&O_sol, nQ * sizeof(bf16)));
  auto reset_sol = [&] { kh_poison(O_sol, nQ * sizeof(bf16)); };
  auto call_sol = [&] {
    cudaError_t e = BlackwellGQA(c.B, c.kvH, c.qH, c.L, c.dH, c.window, Q, K, V, seq_lens, sinks, O_sol);
    if (e != cudaSuccess) {
      std::cerr << "Solution kernel failed: " << cudaGetErrorString(e) << std::endl;
      std::exit(1);
    }
  };
  auto check_sol = [&](char const *tag) {
    KhResult r = kh_compare(O_sol, O_ref, nQ, kTolO);
    kh_print_result(tag, r, kTolO);
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
    KhTime tr = kh_time([&] { ref->run(); }, trials, [&] { refresh(); kh_poison(O_ref, nQ * sizeof(bf16)); });
    kh_print_time("Ref", tr);
#ifdef KH_TEST_SOLUTION
    KhTime ts = kh_time(call_sol, trials, [&] { refresh(); reset_sol(); });
    kh_print_time("Kernel", ts);
    kh_print_speedup(tr, ts);
#else
    double bytes = 0;   // KV bytes actually read
    {
      std::vector<int> h(c.B);
      KH_CHECK(cudaMemcpy(h.data(), seq_lens, c.B * sizeof(int), cudaMemcpyDeviceToHost));
      for (int b = 0; b < c.B; ++b) {
        int n = c.window ? std::min(h[b], c.window) : h[b];
        bytes += 2.0 * n * c.kvH * c.dH * 2;
      }
    }
    std::printf("  reference: %.0f GB/s KV read (%.0f MB)\n", bytes / (tr.min_ms * 1e-3) * 1e-9, bytes * 1e-6);
#endif
  }

#ifdef KH_TEST_SOLUTION
  // Re-check on freshly generated inputs; always runs (also when timing is disabled).
  if (ok) {
    // Recompute the reference output, then the
    // solution must produce the right answer for these new inputs from a poisoned output.
    refresh();
    kh_poison(O_ref, nQ * sizeof(bf16));
    ref->run();
    KH_CHECK(cudaDeviceSynchronize());
    reset_sol();
    KH_CHECK(cudaDeviceSynchronize());
    call_sol();
    KH_CHECK(cudaDeviceSynchronize());
    ok = check_sol("solution (re-check, fresh inputs)");
  }
#endif

#ifdef KH_TEST_SOLUTION
  cudaFree(O_sol);
#endif
  cudaFree(Q); cudaFree(K); cudaFree(V); cudaFree(O_ref); cudaFree(seq_lens);
  if (sinks) cudaFree(sinks);
  return ok;
}

#endif  // CUTLASS_ARCH_MMA_SM100_SUPPORTED

int main(int argc, char const **argv) {
  cudaDeviceProp props;
  KH_CHECK(cudaGetDeviceProperties(&props, 0));
  if (props.major != 10 || props.minor != 0) {
    std::cerr << "This task requires a Blackwell sm_100 GPU (B200)." << std::endl;
    return 0;
  }
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
  int trials = kh_trials();
  bool const keep_going = std::getenv("KH_KEEP_GOING") != nullptr;   // development aid
  bool failed = false;

  // gpt-oss-120b decode: 64 query heads, 8 KV heads, head_dim 64, attention sinks; its layers
  // alternate full attention (up to 128K tokens of context) and a 128-token sliding window.
  // The last case is a Llama-3-70B style layer (head_dim 128, no sinks).
  std::vector<GqaCase> cases = {
      // label                    B  kvH qH     L    dH  win  varlen sinks cfg
      {"b1_ctx128k",               1, 8, 64, 131072, 64,   0, false, true, 1},
      {"b8_ctx32k",                8, 8, 64,  32768, 64,   0, true,  true, 0},
      {"b32_ctx8k",               32, 8, 64,   8192, 64,   0, true,  true, 3},
      {"b64_ctx16k",              64, 8, 64,  16384, 64,   0, true,  true, 0},
      {"b64_ctx32k",              64, 8, 64,  32768, 64,   0, true,  true, 0},
      {"b64_ctx32k_window128",    64, 8, 64,  32768, 64, 128, true,  true, 7},
      {"llama70b_b16_ctx8k_hd128", 16, 8, 64,  8192, 128,  0, true,  false, 5},
  };

#ifdef KH_SWEEP
  for (auto s : cases) {
    int ncfg = s.dH == 64 ? 11 : 7;
    for (int cfg = 0; cfg < ncfg; ++cfg) {
      s.cfg = cfg;
      std::printf("\n=== %s cfg=%d ===\n", s.label, cfg);
      run_case(s, true, 30);
    }
  }
  return 0;
#endif

  if (argc >= 9) {   // ./test B kvH qH L dH window var_len sinks : one custom case
    GqaCase s{"custom", atoi(argv[1]), atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), atoi(argv[5]),
              atoi(argv[6]), atoi(argv[7]) != 0, atoi(argv[8]) != 0, atoi(argv[5]) == 64 ? 0 : 0};
    std::printf("\n=== custom: B=%d, kvH=%d, qH=%d, kvL=%d, D=%d, window=%d ===\n", s.B, s.kvH, s.qH, s.L, s.dH, s.window);
    if (!run_case(s, true, trials)) { std::cout << "Incorrect" << std::endl; return 1; }
    std::cout << "Passed" << std::endl;
    return 0;
  }

  for (auto const &s : cases) {
    std::printf("\n=== %s: B=%d, kvH=%d, qH=%d, kvL=%d, D=%d, window=%d, sinks=%d ===\n", s.label,
                s.B, s.kvH, s.qH, s.L, s.dH, s.window, int(s.sinks));
    if (!run_case(s, true, trials)) { std::cout << "Incorrect" << std::endl; if (!keep_going) return 1; failed = true; }
    else std::cout << "Passed" << std::endl;
  }
  return failed ? 1 : 0;
#endif
  return 0;
}
