/***************************************************************************************************
 * Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met: (BSD-3-Clause, see the CUTLASS LICENSE.txt)
 **************************************************************************************************/

/*
  Task 81_blackwell_gemm_blockwise: FP8 (E4M3) GEMM with block-wise scaling, BF16 output, on Blackwell
  (sm_100a / B200) -- the DeepSeek-V3 / "FP8 blockwise" serving recipe:

      D[m,n] = sum_kb  SFA[kb,m] * SFB[kb, n/128] * ( sum_{k in block kb} A[m,k] * B[n,k] )

  activations A are scaled per (row, 128-wide K block) ("1x128"), weights B per (128 N) x (128 K)
  block ("128x128"); the per-block partial sums are promoted to FP32 and accumulated in FP32.

  Reference: the CUTLASS Blackwell blockwise-scaled GEMM (example 81, KernelScheduleSm100Blockwise)
  with one tile / cluster configuration (and, for decode shapes, swapped operands) chosen per problem
  shape from a sweep on B200. The reference was validated offline (and can be re-validated with
  `KH_VALIDATE_REF=1 ./ref`) against an independent device kernel that applies the documented
  scale layouts by hand.

  Solution under test (KH_TEST_SOLUTION), implemented in solution.h:

    cudaError_t BlackwellBlockwiseGemm(int M, int N, int K,
                                       __nv_fp8_e4m3 const *A, float const *SFA,
                                       __nv_fp8_e4m3 const *B, float const *SFB,
                                       __nv_bfloat16 *D);

  The data / scale layouts are documented in description.txt.

  Verification: random full-mantissa FP8 data and random positive FP32 scales; the reference and the
  solution write to different buffers pre-filled with NaN; every output element must be finite and
  within tolerance of the reference (device-side comparison). The solution is re-checked after timing.
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
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include "solution.h"   // must stay before kh_verify.h: the solution must not see the helpers
#endif

#include "cute/tensor.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/detail/blockwise_scale_layout.hpp"
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

constexpr int kBlk = 128;   // scale block size along K, and along N for B

using ElementAB = cutlass::float_e4m3_t;
using LayoutATag = cutlass::layout::RowMajor;
using LayoutBTag = cutlass::layout::ColumnMajor;
constexpr int AlignmentAB = 128 / cutlass::sizeof_bits<ElementAB>::value;
using ElementD = cutlass::bfloat16_t;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
using ElementAccumulator = float;

// Swap = false: D[M,N] = A[M,K] * B[K,N]  (row-major D), A scaled 1x128, B scaled 128x128.
// Swap = true : operands exchange roles, D^T[N,M] = B[N,K] * A^T (a column-major D^T is exactly the
//               row-major D[M,N] of the task): then the CUTLASS "A" operand carries the 128x128 scales.
template <class MmaTile, class ClusterShape, bool Swap>
struct GemmCfg {
  using ScaleConfig = cute::conditional_t<Swap, cutlass::detail::Sm100BlockwiseScaleConfig<128, 1, 128>,
                                          cutlass::detail::Sm100BlockwiseScaleConfig<1, 128, 128>>;
  using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
  using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());
  using LayoutDTag = cute::conditional_t<Swap, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, MmaTile, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator, float,
      void, LayoutDTag, AlignmentD, ElementD, LayoutDTag, AlignmentD,
      cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
      ElementAB, cute::tuple<LayoutATag, LayoutSFA>, AlignmentAB,
      ElementAB, cute::tuple<LayoutBTag, LayoutSFB>, AlignmentAB,
      ElementAccumulator, MmaTile, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
          static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::KernelScheduleSm100Blockwise>::CollectiveOp;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>,
                                                          CollectiveMainloop, CollectiveEpilogue,
                                                          void>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  static constexpr bool kSwap = Swap;
};

// Reference configurations, selected per shape from a sweep on B200 (see KH_SWEEP; swizzle is the
// CUTLASS tile-scheduler raster swizzle):
//   0: prefill (M >= 8K): 256x128 MMA tile, 2-SM, cluster 2x1, swizzle 8
//   1: decode (M = 128 tokens): 128x128 tile, 1-SM, cluster 1x2 (two CTAs along N share the A tile)
using Cfg0 = GemmCfg<Shape<_256, _128, _128>, Shape<_2, _1, _1>, false>;
using Cfg1 = GemmCfg<Shape<_128, _128, _128>, Shape<_1, _2, _1>, false>;
#ifdef KH_SWEEP
using Cfg2 = GemmCfg<Shape<_128, _128, _128>, Shape<_1, _1, _1>, false>;
using Cfg3 = GemmCfg<Shape<_256, _128, _128>, Shape<_2, _2, _1>, false>;
using Cfg4 = GemmCfg<Shape<_256, _128, _128>, Shape<_4, _1, _1>, false>;
using Cfg5 = GemmCfg<Shape<_128, _128, _128>, Shape<_1, _1, _1>, true>;    // swapped operands
using Cfg6 = GemmCfg<Shape<_256, _128, _128>, Shape<_2, _1, _1>, true>;
using Cfg7 = GemmCfg<Shape<_128, _128, _128>, Shape<_1, _4, _1>, false>;
using Cfg8 = GemmCfg<Shape<_256, _128, _128>, Shape<_4, _1, _1>, true>;
using Cfg9 = GemmCfg<Shape<_128, _256, _128>, Shape<_1, _1, _1>, false>;
using Cfg10 = GemmCfg<Shape<_128, _32, _128>, Shape<_1, _1, _1>, true>;
using Cfg11 = GemmCfg<Shape<_256, _64, _128>, Shape<_2, _1, _1>, true>;
using Cfg12 = GemmCfg<Shape<_256, _256, _128>, Shape<_2, _1, _1>, false>;
constexpr int kNumCfg = 13;
#endif

// ---------------------------------------------------------------------------------------------
// Operand layouts (see description.txt)
//   A   : fp8 e4m3 [M, K]   row-major (K contiguous)
//   B   : fp8 e4m3 [N, K]   row-major (== column-major [K, N])
//   SFA : fp32 [K/128][M]            : SFA[kb * M + m]
//   SFB : fp32 [K/128][ceil(N/128)]  : SFB[kb * ceil(N/128) + n / 128]
// ---------------------------------------------------------------------------------------------
static inline int kh_cdiv(int a, int b) { return (a + b - 1) / b; }

// Scales: log-uniform in [2^-2, 2^2): random positive full-mantissa floats.
__global__ void gen_scales_kernel(float *p, size_t n, unsigned seed) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x)
    p[i] = exp2f(kh_uniform_pos(i, seed, 0.f, 1.f) * 4.f - 2.f);
}

// ---------------------------------------------------------------------------------------------
// Independent reference (device): applies the scale layouts by hand; validates the CUTLASS kernel
// ---------------------------------------------------------------------------------------------
__global__ void indep_ref_kernel(int M, int N, int K, __nv_fp8_e4m3 const *A, float const *SFA,
                                 __nv_fp8_e4m3 const *B, float const *SFB, __nv_bfloat16 *D) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y;
  if (j >= N) return;
  int nbk = (N + kBlk - 1) / kBlk;
  float acc = 0.f;
  for (int kb = 0; kb < K / kBlk; ++kb) {
    float part = 0.f;
    for (int k = kb * kBlk; k < (kb + 1) * kBlk; ++k)
      part = fmaf(float(A[size_t(i) * K + k]), float(B[size_t(j) * K + k]), part);
    acc = fmaf(part, SFA[size_t(kb) * M + i] * SFB[size_t(kb) * nbk + j / kBlk], acc);
  }
  D[size_t(i) * N + j] = __float2bfloat16_rn(acc);
}

/// One prepared CUTLASS GEMM; argument packing / workspace are done in the constructor.
struct RefGemm {
  virtual ~RefGemm() {}
  virtual void run() = 0;
};

template <class Cfg>
struct RefGemmImpl : RefGemm {
  using Gemm = typename Cfg::Gemm;
  Gemm gemm;
  void *workspace = nullptr;
  RefGemmImpl(int M, int N, int K, __nv_fp8_e4m3 const *A, float const *SFA, __nv_fp8_e4m3 const *B,
              float const *SFB, __nv_bfloat16 *D, int swizzle) {
    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideD = typename Gemm::GemmKernel::StrideD;
    constexpr bool S = Cfg::kSwap;
    int Mg = S ? N : M, Ng = S ? M : N;   // GEMM problem handed to CUTLASS
    auto const *Ag = S ? B : A;
    auto const *Bg = S ? A : B;
    float const *SFAg = S ? SFB : SFA, *SFBg = S ? SFA : SFB;
    StrideA sa = cutlass::make_cute_packed_stride(StrideA{}, {Mg, K, 1});
    StrideB sb = cutlass::make_cute_packed_stride(StrideB{}, {Ng, K, 1});
    StrideD sd = cutlass::make_cute_packed_stride(StrideD{}, {Mg, Ng, 1});
    auto lsfa = Cfg::ScaleConfig::tile_atom_to_shape_SFA(cute::make_shape(Mg, Ng, K, 1));
    auto lsfb = Cfg::ScaleConfig::tile_atom_to_shape_SFB(cute::make_shape(Mg, Ng, K, 1));
    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {Mg, Ng, K, 1},
        {reinterpret_cast<ElementAB const *>(Ag), sa, reinterpret_cast<ElementAB const *>(Bg), sb,
         SFAg, lsfa, SFBg, lsfb},
        {{1.f, 0.f}, nullptr, sd, reinterpret_cast<ElementD *>(D), sd}};
    args.scheduler.max_swizzle_size = swizzle;
    size_t ws = Gemm::get_workspace_size(args);
    if (ws) KH_CHECK(cudaMalloc(&workspace, ws));
    CUTLASS_CHECK_KH(gemm.can_implement(args));
    CUTLASS_CHECK_KH(gemm.initialize(args, workspace));
  }
  ~RefGemmImpl() override { cudaFree(workspace); }
  void run() override { CUTLASS_CHECK_KH(gemm.run()); }
};

static std::unique_ptr<RefGemm> make_ref(int cfg, int M, int N, int K, __nv_fp8_e4m3 const *A,
                                         float const *SFA, __nv_fp8_e4m3 const *B, float const *SFB,
                                         __nv_bfloat16 *D, int swz) {
#define KH_CASE(i) case i: return std::make_unique<RefGemmImpl<Cfg##i>>(M, N, K, A, SFA, B, SFB, D, swz);
  switch (cfg) {
    KH_CASE(1)
#ifdef KH_SWEEP
    KH_CASE(2) KH_CASE(3) KH_CASE(4) KH_CASE(5) KH_CASE(6) KH_CASE(7) KH_CASE(8) KH_CASE(9) KH_CASE(10) KH_CASE(11) KH_CASE(12)
#endif
    default: return std::make_unique<RefGemmImpl<Cfg0>>(M, N, K, A, SFA, B, SFB, D, swz);
  }
#undef KH_CASE
}

// Tolerances: |x - ref| <= 1e-3 * RMS(ref) + 0.03125 * |ref| per element, and relative Frobenius error
// <= 3e-4. Reference and solution round an FP32 result to BF16; summation-order differences flip single BF16
// roundings (<= 1 ulp = 0.0078 |ref|, hence the 4x relative term). Measured vs the CUTLASS reference: a
// sequential FP32 solution with the same block promotion is bit-identical (FP8 products and the per-block
// sums are exact in FP32); a correct variant with a different order (reverse K, promotion every 32 K,
// (part*sb)*sa) measures rel. Frobenius 8e-7 .. 5.3e-5 and max|err|/rms <= 2.5e-2, so 3e-4 is >= 5.7x
// above. Deliberately wrong solutions (Frobenius error): FP16 accumulation 1.6e-3 .. 3.2e-3, BF16 accumulation
// 4.6e-3 .. 1.8e-2, last 128 K skipped 6e-2 .. 0.26, inputs rounded to E5M2 8.2e-2, FP4 inputs 0.40, scales
// of B per N-column instead of 128x128 blocks 0.60, A scales ignored 0.64, B scales ignored 0.64, A scales
// read K-major 0.85, half the outputs / everything unwritten (NaN): all fail on every size.
static KhTol kTol(1.0e-3, 3.0e-4, 0.03125);

struct Shape3 { const char *label; int M, N, K, cfg; int swz = 0; };

/// Returns false on failure. `timed` = also print Ref / Kernel timings.
static bool run_case(Shape3 const &s, bool timed, int trials) {
  int M = s.M, N = s.N, K = s.K;
  size_t nD = size_t(M) * N;
  __nv_fp8_e4m3 *A, *B;
  float *SFA, *SFB;
  __nv_bfloat16 *D_ref;
  size_t nSFA = size_t(K / kBlk) * M, nSFB = size_t(K / kBlk) * kh_cdiv(N, kBlk);
  KH_CHECK(cudaMalloc(&A, size_t(M) * K));
  KH_CHECK(cudaMalloc(&B, size_t(N) * K));
  KH_CHECK(cudaMalloc(&SFA, nSFA * sizeof(float)));
  KH_CHECK(cudaMalloc(&SFB, nSFB * sizeof(float)));
  KH_CHECK(cudaMalloc(&D_ref, nD * sizeof(__nv_bfloat16)));
  kh_fill_uniform(A, size_t(M) * K, 11u);
  kh_fill_uniform(B, size_t(N) * K, 22u);
  gen_scales_kernel<<<1024, 256>>>(SFA, nSFA, 33u);
  gen_scales_kernel<<<1024, 256>>>(SFB, nSFB, 44u);
  KH_CHECK(cudaGetLastError());
  kh_poison(D_ref, nD * sizeof(__nv_bfloat16));
  KH_CHECK(cudaDeviceSynchronize());

  auto ref = make_ref(s.cfg, M, N, K, A, SFA, B, SFB, D_ref, s.swz);
  ref->run();
  KH_CHECK(cudaDeviceSynchronize());

  bool ok = true;
  // The reference must have written every element (it was pre-filled with NaN).
  {
    KhResult r = kh_compare(D_ref, D_ref, nD, kTol);
    if (!r.pass) {
      std::cerr << "Reference output incomplete or non-finite (non-finite = " << r.nonfinite << ")" << std::endl;
      ok = false;
    }
  }
  if (std::getenv("KH_VALIDATE_REF")) {   // development aid: CUTLASS reference vs independent kernel
    __nv_bfloat16 *D_chk;
    KH_CHECK(cudaMalloc(&D_chk, nD * sizeof(__nv_bfloat16)));
    kh_poison(D_chk, nD * sizeof(__nv_bfloat16));
    indep_ref_kernel<<<dim3((N + 127) / 128, M), 128>>>(M, N, K, A, SFA, B, SFB, D_chk);
    KH_CHECK(cudaGetLastError());
    KH_CHECK(cudaDeviceSynchronize());
    KhResult r = kh_compare(D_chk, D_ref, nD, kTol);
    kh_print_result("independent kernel vs CUTLASS reference", r, kTol);
    ok = ok && r.pass;
    cudaFree(D_chk);
  }

#ifdef KH_TEST_SOLUTION
  __nv_bfloat16 *D_sol = nullptr;
  KH_CHECK(cudaMalloc(&D_sol, nD * sizeof(__nv_bfloat16)));
  auto reset_sol = [&] { kh_poison(D_sol, nD * sizeof(__nv_bfloat16)); };
  auto call_sol = [&] {
    cudaError_t e = BlackwellBlockwiseGemm(M, N, K, A, SFA, B, SFB, D_sol);
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
    KhResult r = kh_compare(D_sol, D_ref, nD, kTol);
    kh_print_result("solution", r, kTol);
    ok = r.pass;
  }
#endif

  // Anti-memoisation: the operand that varies per call in deployment (the activations and their scale
  // factors; the weights stay fixed) is regenerated with a fresh seed before EVERY timed call of both the
  // reference and the solution, outside the timed region (before the L2 flush), so nothing derived from the
  // input can be cached across calls. The first check above and the final re-check use their own seeds.
  auto regen_input = [&](unsigned seed) {
    kh_fill_uniform(A, size_t(M) * K, seed);
    gen_scales_kernel<<<1024, 256>>>(SFA, nSFA, seed + 1u);
    KH_CHECK(cudaGetLastError());
  };
  unsigned seed_ctr = 1000u;
  auto fresh = [&] { regen_input(++seed_ctr); };

  if (ok && timed && trials > 0) {
    KhTime tr = kh_time([&] { ref->run(); }, trials, [&] { fresh(); kh_poison(D_ref, nD * sizeof(__nv_bfloat16)); });
    kh_print_time("Ref", tr);
#ifdef KH_TEST_SOLUTION
    KhTime ts = kh_time(call_sol, trials, [&] { fresh(); reset_sol(); });
    kh_print_time("Kernel", ts);
    kh_print_speedup(tr, ts);
#endif
  }

#ifdef KH_TEST_SOLUTION
  if (ok) {
    // Re-check on a brand-new input (also for the correctness-only cases): regenerate the activations,
    // recompute the reference for them, re-poison the solution output, call the solution, compare.
    regen_input(777777u);
    kh_poison(D_ref, nD * sizeof(__nv_bfloat16));
    KH_CHECK(cudaDeviceSynchronize());
    ref->run();
    KH_CHECK(cudaDeviceSynchronize());
    reset_sol();
    KH_CHECK(cudaDeviceSynchronize());
    call_sol();
    KH_CHECK(cudaDeviceSynchronize());
    KhResult r = kh_compare(D_sol, D_ref, nD, kTol);
    kh_print_result("solution (re-check, fresh input)", r, kTol);
    ok = r.pass;
  }
  cudaFree(D_sol);
#endif
  cudaFree(A); cudaFree(B); cudaFree(SFA); cudaFree(SFB); cudaFree(D_ref);
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

  // Llama-3-70B linear layers in FP8 on one B200 (hidden 8192, fused QKV 10240, FFN 28672 -> fused
  // gate+up 57344, vocab 128256). M = tokens: one decode step of 128 sequences, or an 8K-token prefill
  // chunk. (The decode fused-QKV layer, 84 MB of weights, is ~10 us and too short to time reliably.)
  std::vector<Shape3> shapes = {
      {"decode_mlp_up",    128,  57344,  8192, 1, 0},
      {"decode_mlp_down",  128,   8192, 28672, 1, 0},
      {"decode_lm_head",   128, 128256,  8192, 1, 0},
      {"prefill_qkv",     8192,  10240,  8192, 0, 8},
      {"prefill_o_proj",  8192,   8192,  8192, 0, 8},
      {"prefill_mlp_up",  8192,  57344,  8192, 0, 8},
      {"prefill_mlp_down",8192,   8192, 28672, 0, 8},
  };

#ifdef KH_SWEEP
  // Development aid: time every reference configuration on every shape.
  for (auto s : shapes) {
    for (int cfg = 0; cfg < kNumCfg; ++cfg) {
      bool swapped_cfg = cfg == 5 || cfg == 6 || cfg == 8 || cfg == 10 || cfg == 11;
      if (swapped_cfg && s.M > 512) continue;
      for (int swz : {0, 4, 8}) {
        s.cfg = cfg; s.swz = swz;
        std::printf("\n=== %s cfg=%d swz=%d: M=%d, N=%d, K=%d ===\n", s.label, cfg, swz, s.M, s.N, s.K);
        run_case(s, true, 10);
      }
    }
  }
  return 0;
#endif

  if (argc >= 4) {   // ./test M N K : single custom shape
    Shape3 s{"custom", atoi(argv[1]), atoi(argv[2]), atoi(argv[3]), atoi(argv[1]) <= 512 ? 1 : 0, 8};
    std::printf("\n=== custom: M=%d, N=%d, K=%d ===\n", s.M, s.N, s.K);
    if (!run_case(s, true, trials)) { std::cout << "Incorrect" << std::endl; return 1; }
    std::cout << "Passed" << std::endl;
    return 0;
  }

  // Correctness-only cases ("---" header, not "===": the runner pairs each "=== name ===" block with
  // timing lines). M = 200 is not a multiple of 128 and N = 1152 / K = 1408 are not multiples of
  // the tile; M = 16 is a tiny decode batch.
  {
    Shape3 extra[] = {{"odd_m", 200, 1152, 1408, 0, 0}, {"tiny_m", 16, 2048, 4096, 1, 0}};
    for (auto const &s : extra) {
      std::printf("\n--- %s: M=%d, N=%d, K=%d (correctness only) ---\n", s.label, s.M, s.N, s.K);
      if (!run_case(s, false, 0)) { std::cout << "Incorrect" << std::endl; if (!keep_going) return 1; failed = true; }
      else std::cout << "Passed" << std::endl;
    }
  }

  for (auto const &s : shapes) {
    std::printf("\n=== %s: M=%d, N=%d, K=%d ===\n", s.label, s.M, s.N, s.K);
    if (!run_case(s, true, trials)) { std::cout << "Incorrect" << std::endl; if (!keep_going) return 1; failed = true; }
    else std::cout << "Passed" << std::endl;
  }
  return failed ? 1 : 0;
#endif
  return 0;
}
