/***************************************************************************************************
 * Copyright (c) 2025 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met: (BSD-3-Clause, see the CUTLASS LICENSE.txt)
 **************************************************************************************************/

/*
  Task 70_blackwell_fp16_gemm: FP16 GEMM with FP32 output on Blackwell (sm_100a / B200).

  Reference: the CUTLASS 3.x Blackwell GEMM (example 70) with one tile / cluster configuration
  chosen per problem shape from a sweep on B200 (prefill shapes: 256x256 tile, 2x1 cluster;
  decode shapes: a 1-SM tile). It reaches 87-109% of cuBLAS (FP16 in, FP32 out) on every shape.

  Solution under test (KH_TEST_SOLUTION), implemented in solution.h:

    cudaError_t BlackwellHgemm(int M, int N, int K, float alpha,
                               __half const *A, int lda, __half const *B, int ldb,
                               float beta, float *C, int ldc);

    A: row-major [M, K] (lda = K), B: column-major [K, N] (ldb = K),
    C: column-major [M, N] (ldc = M);  C = alpha * A * B + beta * C.

  Verification: inputs are full-mantissa random FP16; the reference and the solution write to
  different buffers that are pre-filled with NaN; every output element must be finite and within
  tolerance of the reference. The solution is re-checked after the timing loop.
*/

#include <cstdio>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "cutlass/cutlass.h"

#ifdef KH_TEST_SOLUTION
#include <cuda_fp16.h>
#include "solution.h"   // must stay before kh_verify.h: the solution must not see the helpers
#endif

#include "cute/tensor.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

#include "kh_verify.h"

using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

#define CUTLASS_CHECK_KH(status)                                                         \
  {                                                                                      \
    cutlass::Status error = status;                                                      \
    if (error != cutlass::Status::kSuccess) {                                            \
      std::cerr << "Got cutlass error: " << cutlassGetStatusString(error) << " at: "     \
                << __LINE__ << std::endl;                                                \
      std::exit(2);                                                                      \
    }                                                                                    \
  }

using ElementA = cutlass::half_t;
using LayoutA = cutlass::layout::RowMajor;
constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
using ElementB = cutlass::half_t;
using LayoutB = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
using ElementC = float;
using LayoutC = cutlass::layout::ColumnMajor;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
using ElementAccumulator = float;
using ArchTag = cutlass::arch::Sm100;
using OperatorClass = cutlass::arch::OpClassTensorOp;

template <class MmaTile, class ClusterShape>
struct GemmCfg {
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag, OperatorClass, MmaTile, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator, ElementAccumulator,
      ElementC, LayoutC, AlignmentC, ElementC, LayoutC, AlignmentC,
      cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OperatorClass, ElementA, LayoutA, AlignmentA, ElementB, LayoutB, AlignmentB,
      ElementAccumulator, MmaTile, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
          static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>,
                                                          CollectiveMainloop, CollectiveEpilogue,
                                                          void>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

// Reference configurations (selected per shape below).
using CfgLarge  = GemmCfg<Shape<_256, _128, _64>, Shape<_2, _2, _1>>;   // prefill-sized GEMMs
using CfgLarge2 = GemmCfg<Shape<_256, _256, _64>, Shape<_2, _1, _1>>;
using CfgSmall  = GemmCfg<Shape<_128, _128, _64>, Shape<_1, _1, _1>>;   // small-M (decode) GEMMs
using CfgSmall2 = GemmCfg<Shape<_128, _64, _64>, Shape<_1, _1, _1>>;
using CfgSmall3 = GemmCfg<Shape<_128, _256, _64>, Shape<_1, _1, _1>>;

/// One prepared CUTLASS GEMM; initialization (argument packing, workspace) is done in the
/// constructor, so run() is just the kernel launch.
struct RefGemm {
  virtual ~RefGemm() {}
  virtual void run() = 0;
};

template <class Cfg>
struct RefGemmImpl : RefGemm {
  using Gemm = typename Cfg::Gemm;
  Gemm gemm;
  void *workspace = nullptr;
  RefGemmImpl(int M, int N, int K, float alpha, float beta, ElementA const *A, ElementB const *B,
              float const *C, float *D) {
    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideC = typename Gemm::GemmKernel::StrideC;
    using StrideD = typename Gemm::GemmKernel::StrideD;
    StrideA sa = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
    StrideB sb = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
    StrideC sc = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});
    StrideD sd = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});
    typename Gemm::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm,
                                  {M, N, K, 1},
                                  {A, sa, B, sb},
                                  {{alpha, beta}, C, sc, D, sd}};
    size_t ws = Gemm::get_workspace_size(args);
    if (ws) KH_CHECK(cudaMalloc(&workspace, ws));
    CUTLASS_CHECK_KH(gemm.can_implement(args));
    CUTLASS_CHECK_KH(gemm.initialize(args, workspace));
  }
  ~RefGemmImpl() override { cudaFree(workspace); }
  void run() override { CUTLASS_CHECK_KH(gemm.run()); }
};

static std::unique_ptr<RefGemm> make_ref(int cfg, int M, int N, int K, float alpha, float beta,
                                         ElementA const *A, ElementB const *B, float const *C,
                                         float *D) {
  switch (cfg) {
    case 0: return std::make_unique<RefGemmImpl<CfgLarge>>(M, N, K, alpha, beta, A, B, C, D);
    case 1: return std::make_unique<RefGemmImpl<CfgLarge2>>(M, N, K, alpha, beta, A, B, C, D);
    case 2: return std::make_unique<RefGemmImpl<CfgSmall>>(M, N, K, alpha, beta, A, B, C, D);
    case 3: return std::make_unique<RefGemmImpl<CfgSmall2>>(M, N, K, alpha, beta, A, B, C, D);
    default: return std::make_unique<RefGemmImpl<CfgSmall3>>(M, N, K, alpha, beta, A, B, C, D);
  }
}

// Tolerances, relative to the RMS of the reference output. Reference and solution both accumulate
// exact FP16 products in FP32 with different summation orders (a plain sequential FP32 kernel
// measures max|err|/rms <= 2.0e-4 and rel. Frobenius error <= 3.4e-5 at the largest K, so these
// bounds leave >= 5x margin). Rounding the inputs to BF16 gives 1.1e-2 / 2.1e-3, FP16
// accumulation 7e-2 / 6.6e-3, FP8 inputs 1.5e-1 / 3.3e-2: all fail.
static KhTol kTol(1.0e-3, 2.0e-4);

struct Shape3 { const char *label; int M, N, K, cfg; };

/// Returns false on failure. `timed` = also print Ref / Kernel timings.
static bool run_case(Shape3 const &s, float alpha, float beta, bool timed, int trials) {
  int M = s.M, N = s.N, K = s.K;
  size_t nA = size_t(M) * K, nB = size_t(K) * N, nC = size_t(M) * N;
  __half *A, *B;
  float *C_in = nullptr, *D_ref;
  KH_CHECK(cudaMalloc(&A, nA * sizeof(__half)));
  KH_CHECK(cudaMalloc(&B, nB * sizeof(__half)));
  KH_CHECK(cudaMalloc(&D_ref, nC * sizeof(float)));
  kh_fill_uniform(A, nA, 11u);
  kh_fill_uniform(B, nB, 22u);
  if (beta != 0.f) {
    KH_CHECK(cudaMalloc(&C_in, nC * sizeof(float)));
    kh_fill_uniform(C_in, nC, 33u);
  }
  kh_poison(D_ref, nC * sizeof(float));
  KH_CHECK(cudaDeviceSynchronize());

  auto ref = make_ref(s.cfg, M, N, K, alpha, beta, reinterpret_cast<ElementA const *>(A),
                      reinterpret_cast<ElementB const *>(B), C_in, D_ref);
  ref->run();
  KH_CHECK(cudaDeviceSynchronize());

  bool ok = true;
  // The reference must have written every element (it was pre-filled with NaN).
  {
    KhResult r = kh_compare(D_ref, D_ref, nC, kTol);
    if (!r.pass) {
      std::cerr << "Reference output incomplete or non-finite (non-finite = " << r.nonfinite
                << ")" << std::endl;
      ok = false;
    }
  }

#ifdef KH_TEST_SOLUTION
  float *D_sol = nullptr;
  KH_CHECK(cudaMalloc(&D_sol, nC * sizeof(float)));
  auto reset_sol = [&] {
    if (beta != 0.f) KH_CHECK(cudaMemcpy(D_sol, C_in, nC * sizeof(float), cudaMemcpyDeviceToDevice));
    else kh_poison(D_sol, nC * sizeof(float));
  };
  auto call_sol = [&] {
    cudaError_t e = BlackwellHgemm(M, N, K, alpha, A, K, B, K, beta, D_sol, M);
    if (e != cudaSuccess) {
      std::cerr << "Solution kernel failed: " << cudaGetErrorString(e) << std::endl;
      std::exit(1);
    }
  };
  if (ok) {
    reset_sol();
    KH_CHECK(cudaDeviceSynchronize());
    call_sol();
    KH_CHECK(cudaDeviceSynchronize());
    KhResult r = kh_compare(D_sol, D_ref, nC, kTol);
    kh_print_result("solution", r, kTol);
    ok = r.pass;
  }
#endif

  if (ok && timed) {   // the fresh-input re-check also runs when timing is disabled (KH_BENCHMARK=0)
    // The activations A are regenerated (new seed, outside the timed region, before the L2 flush) before
    // every call of both the reference and the solution, so a result or an A-dependent intermediate cannot
    // be memoized across calls; the weights B stay fixed. Reference outputs are poisoned before every call
    // so both sides do identical work.
    unsigned seed = 1000u;
    auto refresh_A = [&] { if (beta == 0.f) kh_fill_uniform(A, nA, seed++); };
    if (trials > 0) {
    KhTime tr = kh_time([&] { ref->run(); }, trials,
                        [&] { if (beta == 0.f) kh_poison(D_ref, nC * sizeof(float)); refresh_A(); });
    kh_print_time("Ref", tr);
#ifdef KH_TEST_SOLUTION
    KhTime ts = kh_time(call_sol, trials, [&] { reset_sol(); refresh_A(); });
    kh_print_time("Kernel", ts);
    kh_print_speedup(tr, ts);
#endif
    }
#ifdef KH_TEST_SOLUTION
    // Re-check after timing with yet another fresh A: recompute the reference for it, re-poison the
    // solution output, and require the solution to produce the right answer from scratch.
    refresh_A();
    kh_poison(D_ref, nC * sizeof(float));
    ref->run();
    KH_CHECK(cudaDeviceSynchronize());
    reset_sol();
    KH_CHECK(cudaDeviceSynchronize());
    call_sol();
    KH_CHECK(cudaDeviceSynchronize());
    KhResult r = kh_compare(D_sol, D_ref, nC, kTol);
    kh_print_result("solution (re-check after timing, fresh A)", r, kTol);
    ok = r.pass;
#endif
  }

#ifdef KH_TEST_SOLUTION
  cudaFree(D_sol);
#endif
  cudaFree(A); cudaFree(B); cudaFree(D_ref);
  if (C_in) cudaFree(C_in);
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

  // Llama-3-70B linear layers on one B200 (hidden 8192, 64 q / 8 kv heads x 128 -> fused QKV
  // 10240, FFN 28672 -> fused gate+up 57344, vocab 128256); M = tokens: an 8K-token prefill
  // chunk or one decode step of 128 sequences.
  std::vector<Shape3> shapes = {
      {"decode_qkv",        128,  10240,  8192, 2},
      {"decode_mlp_down",   128,   8192, 28672, 3},
      {"decode_lm_head",    128, 128256,  8192, 2},
      {"prefill_qkv",      8192,  10240,  8192, 1},
      {"prefill_o_proj",   8192,   8192,  8192, 1},
      {"prefill_mlp_up",   8192,  57344,  8192, 1},
      {"prefill_mlp_down", 8192,   8192, 28672, 1},
  };

#ifdef KH_SWEEP
  // Development aid: time every reference configuration on every shape.
  for (auto s : shapes) {
    for (int cfg = 0; cfg < 5; ++cfg) {
      s.cfg = cfg;
      std::printf("\n=== %s cfg=%d: M=%d, N=%d, K=%d ===\n", s.label, cfg, s.M, s.N, s.K);
      run_case(s, 1.f, 0.f, true, 10);
    }
  }
  return 0;
#endif

  if (argc >= 4) {   // ./test M N K : single custom shape
    Shape3 s{"custom", atoi(argv[1]), atoi(argv[2]), atoi(argv[3]), atoi(argv[1]) <= 512 ? 2 : 1};
    std::printf("\n=== custom: M=%d, N=%d, K=%d ===\n", s.M, s.N, s.K);
    if (!run_case(s, 1.f, 0.f, true, trials)) { std::cout << "Incorrect" << std::endl; return 1; }
    std::cout << "Passed" << std::endl;
    return 0;
  }

  // alpha / beta handling (correctness only): C = 0.75 * A*B + 0.5 * C
  {
    Shape3 s{"alpha_beta", 384, 1280, 2048, 2};
    // "---" header, not "===": the runner pairs each "=== name ===" block with timing lines.
    std::printf("\n--- %s: M=%d, N=%d, K=%d (alpha=0.75, beta=0.5, correctness only) ---\n",
                s.label, s.M, s.N, s.K);
    if (!run_case(s, 0.75f, 0.5f, false, 0)) { std::cout << "Incorrect" << std::endl; if (!keep_going) return 1; failed = true; }
    else std::cout << "Passed" << std::endl;
  }

  for (auto const &s : shapes) {
    std::printf("\n=== %s: M=%d, N=%d, K=%d ===\n", s.label, s.M, s.N, s.K);
    if (!run_case(s, 1.f, 0.f, true, trials)) { std::cout << "Incorrect" << std::endl; if (!keep_going) return 1; failed = true; }
    else std::cout << "Passed" << std::endl;
  }
  return failed ? 1 : 0;
#endif
  return 0;
}
