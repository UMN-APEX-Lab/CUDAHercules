/***************************************************************************************************
 * Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met: (BSD-3-Clause, see the CUTLASS LICENSE.txt)
 **************************************************************************************************/

/*
  Task 76_blackwell_conv: 3D convolution forward (fprop), FP16 in / FP16 out, FP32 accumulation,
  on Blackwell (sm_100a / B200).

  Workload: the 3x3x3 / 3x1x1 convolutions of a causal video-VAE (Wan2.1 / HunyuanVideo style:
  channel widths 96..384, several frames, hundreds of pixels per side, temporal extent T > 1 with
  real temporal padding and stride) plus the 3x3 convolutions of ResNet-50 at batch 256 (2D
  convolutions expressed as D = T = 1).

  Reference: the CUTLASS 3.x Blackwell implicit-GEMM convolution (example 76; TMA im2col loads, UMMA)
  with one tile / cluster configuration chosen per layer from a sweep on B200, compared with cuDNN
  (torch conv3d, channels-last-3d, FP16) in description.txt.

  Solution under test (KH_TEST_SOLUTION), implemented in solution.h:

    cudaError_t BlackwellConv3dFprop(int N, int D, int H, int W, int C,
                                     int K, int T, int R, int S,
                                     int pad_d, int pad_h, int pad_w,
                                     int stride_d, int stride_h, int stride_w,
                                     __half const *input, __half const *filter, __half *output);

    input  NDHWC [N, D, H, W, C], filter KTRSC [K, T, R, S, C], output NDHWC [N, Z, P, Q, K],
    Z = (D + 2 pad_d - T) / stride_d + 1, P = (H + 2 pad_h - R) / stride_h + 1,
    Q = (W + 2 pad_w - S) / stride_w + 1, zero padding, dilation 1, cross-correlation:
      out[n,z,p,q,k] = sum_{t,r,s,c} in[n, z*sd+t-pd, p*sh+r-ph, q*sw+s-pw, c] * flt[k,t,r,s,c]

  Verification: inputs are full-mantissa random FP16; the reference and the solution write to
  different buffers that are pre-filled with NaN; every output element must be finite and within
  tolerance of the reference. The solution is re-checked after the timing loop.
*/

#include <cstdio>
#include <cstring>
#include <algorithm>
#include <cmath>
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
#include "cutlass/kernel_hardware_info.hpp"
#include "cutlass/conv/convolution.h"
#include "cutlass/conv/convnd_problem_shape.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/conv/dispatch_policy.hpp"
#include "cutlass/conv/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/conv/device/conv_universal_adapter.hpp"
#include "cutlass/conv/kernel/conv_universal.hpp"

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

using ElementAct = cutlass::half_t;
using ElementFlt = cutlass::half_t;
using ElementOut = cutlass::half_t;
constexpr int AlignmentAct = 128 / cutlass::sizeof_bits<ElementAct>::value;
constexpr int AlignmentFlt = 128 / cutlass::sizeof_bits<ElementFlt>::value;
constexpr int AlignmentOut = 128 / cutlass::sizeof_bits<ElementOut>::value;
using LayoutA = cutlass::layout::TensorNDHWC;
using LayoutB = cutlass::layout::TensorNDHWC;
using LayoutC = cutlass::layout::TensorNDHWC;
using ElementAccumulator = float;
using ElementCompute = float;
using ArchTag = cutlass::arch::Sm100;
using OperatorClass = cutlass::arch::OpClassTensorOp;
constexpr cutlass::conv::Operator ConvOp = cutlass::conv::Operator::kFprop;

template <class TileShape, class ClusterShape>
struct ConvCfg {
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag, OperatorClass, TileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator, ElementCompute,
      void, LayoutC, AlignmentOut,   // no C operand: D = A * B
      ElementOut, LayoutC, AlignmentOut,
      cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;
  using CollectiveMainloop = typename cutlass::conv::collective::CollectiveBuilder<
      ArchTag, OperatorClass, ConvOp, ElementAct, LayoutA, AlignmentAct, ElementFlt, LayoutB,
      AlignmentFlt, ElementAccumulator, TileShape, ClusterShape,
      cutlass::conv::collective::StageCountAutoCarveout<
          static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::conv::collective::KernelScheduleAuto>::CollectiveOp;
  using ProblemShape =
      cutlass::conv::ConvProblemShape<ConvOp, CollectiveMainloop::DispatchPolicy::NumSpatialDimensions>;
  using ConvKernel = cutlass::conv::kernel::ConvUniversal<ProblemShape, CollectiveMainloop,
                                                          CollectiveEpilogue>;
  using Conv = cutlass::conv::device::ConvUniversalAdapter<ConvKernel>;
};

// Reference configurations (MMA tile M x N x K, cluster), selected per layer below.
using Cfg0 = ConvCfg<Shape<_128, _128, Shape<_64>>, Shape<_1, _1, _1>>;
using Cfg1 = ConvCfg<Shape<_128, _256, Shape<_64>>, Shape<_1, _1, _1>>;
using Cfg2 = ConvCfg<Shape<_256, _128, Shape<_64>>, Shape<_2, _1, _1>>;
using Cfg3 = ConvCfg<Shape<_256, _256, Shape<_64>>, Shape<_2, _1, _1>>;
using Cfg4 = ConvCfg<Shape<_256, _128, Shape<_64>>, Shape<_4, _1, _1>>;
using Cfg5 = ConvCfg<Shape<_128, _128, Shape<_64>>, Shape<_2, _1, _1>>;
using Cfg6 = ConvCfg<Shape<_256, _64, Shape<_64>>, Shape<_2, _1, _1>>;
using Cfg7 = ConvCfg<Shape<_128, _64, Shape<_64>>, Shape<_1, _1, _1>>;
using Cfg8 = ConvCfg<Shape<_128, _64, Shape<_64>>, Shape<_2, _1, _1>>;
using Cfg9 = ConvCfg<Shape<_256, _64, Shape<_64>>, Shape<_4, _1, _1>>;
using Cfg10 = ConvCfg<Shape<_128, _128, Shape<_64>>, Shape<_4, _1, _1>>;
constexpr int kNumCfg = 11;

/// Convolution problem.
struct ConvDesc {
  int N, D, H, W, C, K, T, R, S, pd, ph, pw, sd, sh, sw;
  int Z() const { return (D + 2 * pd - T) / sd + 1; }
  int P() const { return (H + 2 * ph - R) / sh + 1; }
  int Q() const { return (W + 2 * pw - S) / sw + 1; }
  double flops() const { return 2.0 * N * Z() * P() * Q() * K * T * R * S * C; }
};

struct RefConv {
  virtual ~RefConv() {}
  virtual void run() = 0;
};

template <class Cfg>
struct RefConvImpl : RefConv {
  using Conv = typename Cfg::Conv;
  using ProblemShape = typename Cfg::ProblemShape;
  using StrideC = typename Conv::ConvKernel::StrideC;
  using StrideD = typename Conv::ConvKernel::StrideD;
  Conv conv;
  void *workspace = nullptr;
  RefConvImpl(ConvDesc const &d, void const *A, void const *B, void *Dout) {
    ProblemShape problem_shape(
        cutlass::conv::Mode::kCrossCorrelation, {d.N, d.D, d.H, d.W, d.C},
        {d.K, d.T, d.R, d.S, d.C}, {d.pd, d.ph, d.pw}, {d.pd, d.ph, d.pw}, {d.sd, d.sh, d.sw},
        {1, 1, 1}, 1);
    StrideC stride_C;
    StrideD stride_D;
    cute::for_each(cute::make_seq<cute::rank<0>(StrideC{})>{}, [&](auto i) {
      cute::get<0, i>(stride_C) = problem_shape.stride_C[ProblemShape::RankT - 2 - i];
    });
    cute::for_each(cute::make_seq<cute::rank<0>(StrideD{})>{}, [&](auto i) {
      cute::get<0, i>(stride_D) = problem_shape.stride_C[ProblemShape::RankT - 2 - i];
    });
    typename Conv::Arguments args{
        problem_shape,
        {static_cast<ElementAct const *>(A), static_cast<ElementFlt const *>(B)},
        {{1.f, 0.f}, nullptr, stride_C, static_cast<ElementOut *>(Dout), stride_D}};
    size_t ws = Conv::get_workspace_size(args);
    if (ws) KH_CHECK(cudaMalloc(&workspace, ws));
    CUTLASS_CHECK_KH(conv.can_implement(args));
    CUTLASS_CHECK_KH(conv.initialize(args, workspace));
  }
  ~RefConvImpl() override { cudaFree(workspace); }
  void run() override { CUTLASS_CHECK_KH(conv.run()); }
};

static std::unique_ptr<RefConv> make_ref(int cfg, ConvDesc const &d, void const *A, void const *B,
                                         void *Dout) {
  switch (cfg) {
    case 0: return std::make_unique<RefConvImpl<Cfg0>>(d, A, B, Dout);
    case 1: return std::make_unique<RefConvImpl<Cfg1>>(d, A, B, Dout);
    case 2: return std::make_unique<RefConvImpl<Cfg2>>(d, A, B, Dout);
    case 3: return std::make_unique<RefConvImpl<Cfg3>>(d, A, B, Dout);
    case 4: return std::make_unique<RefConvImpl<Cfg4>>(d, A, B, Dout);
    case 5: return std::make_unique<RefConvImpl<Cfg5>>(d, A, B, Dout);
    case 6: return std::make_unique<RefConvImpl<Cfg6>>(d, A, B, Dout);
    case 7: return std::make_unique<RefConvImpl<Cfg7>>(d, A, B, Dout);
    case 8: return std::make_unique<RefConvImpl<Cfg8>>(d, A, B, Dout);
    case 9: return std::make_unique<RefConvImpl<Cfg9>>(d, A, B, Dout);
    default: return std::make_unique<RefConvImpl<Cfg10>>(d, A, B, Dout);
  }
}

struct Layer {
  char const *label;
  ConvDesc d;
  int cfg;   // reference configuration (from the sweep)
};

/// Tolerances for the FP16 output: every element must satisfy |x - ref| <= 1e-2 * RMS(ref) + |ref|/1024
/// (the second term is one FP16 ulp) and the relative Frobenius error must be <= 4e-4.
/// Reference and solution accumulate exact FP16 products in FP32 in different orders and round the
/// result to FP16. Measured on B200 over all layers (max|err|/RMS, rel. Frobenius error):
///   plain sequential FP32 kernel (shipped naive solution):   <= 5e-3 (1 ulp of the largest
///        elements, covered by the |ref|/1024 term),                  <= 8.6e-5   -> bound 4.7x above
///   (the CUTLASS reference itself differs from an FP32 cuDNN-class result by 2.1e-4: FP16 rounding)
///   inputs rounded to BF16:                    1.2e-2 .. 1.6e-2,    2.28e-3  (fails, 5.7x above bound)
///   FP16 accumulation:                         0.05 .. 0.25,        3.5e-3 .. 1.4e-2  (fails)
///   inputs rounded to FP8 E4M3:                0.20 .. 0.24,        3.6e-2  (fails)
///   skipped last 64 of the reduction (T*R*S*C): 0.47 .. 2.0,        0.075 .. 0.33  (fails)
/// Wrong temporal extent / padding / stride, wrong spatial padding / stride, replicate instead of
/// zero padding, correlation vs convolution (flipped filter) and unwritten outputs fail by O(1)
/// errors or the NaN / non-finite check on the layers they affect (see description.txt).
static KhTol kTol(1.0e-2, 4.0e-4, 1.0 / 1024);

/// Returns false on failure. `timed` = also print Ref / Kernel timings.
static bool run_case(Layer const &L, bool timed, int trials) {
  ConvDesc const &d = L.d;
  int Z = d.Z(), P = d.P(), Q = d.Q();
  size_t nA = size_t(d.N) * d.D * d.H * d.W * d.C;
  size_t nB = size_t(d.K) * d.T * d.R * d.S * d.C;
  size_t nD = size_t(d.N) * Z * P * Q * d.K;
  __half *A, *B, *Dref, *Dsol = nullptr;
  KH_CHECK(cudaMalloc(&A, nA * 2));
  KH_CHECK(cudaMalloc(&B, nB * 2));
  KH_CHECK(cudaMalloc(&Dref, nD * 2));
  // The input activation tensor changes on every call in deployment: it is regenerated with a new
  // seed outside the timed region before every timed call of the reference and of the solution, so
  // outputs and input-dependent intermediates cannot be cached. The filters are fixed.
  auto refresh_input = [&](unsigned seed) { kh_fill_uniform(A, nA, seed); };
  unsigned seed_counter = 1000;
  auto next_input = [&] { refresh_input(++seed_counter); };
  refresh_input(11u);
  kh_fill_uniform(B, nB, 22u, 0.1f);   // weight scale: keeps the outputs O(1..10), well inside FP16
  kh_poison(Dref, nD * 2);
  KH_CHECK(cudaDeviceSynchronize());
  std::printf("  out = %d x %d x %d x %d x %d, %.1f GFLOP\n", d.N, Z, P, Q, d.K, d.flops() * 1e-9);

  auto ref = make_ref(L.cfg, d, A, B, Dref);
  ref->run();
  KH_CHECK(cudaDeviceSynchronize());

  bool ok = true;
  {
    KhResult r = kh_compare(Dref, Dref, nD, kTol);
    if (!r.pass) {
      std::cerr << "Reference output incomplete or non-finite (non-finite = " << r.nonfinite << ")"
                << std::endl;
      ok = false;
    }
  }

#ifdef KH_TEST_SOLUTION
  KH_CHECK(cudaMalloc(&Dsol, nD * 2));
  auto reset_sol = [&] { kh_poison(Dsol, nD * 2); };
  auto call_sol = [&] {
    cudaError_t e = BlackwellConv3dFprop(d.N, d.D, d.H, d.W, d.C, d.K, d.T, d.R, d.S, d.pd, d.ph,
                                         d.pw, d.sd, d.sh, d.sw, A, B, Dsol);
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
    KhResult r = kh_compare(Dsol, Dref, nD, kTol);
    kh_print_result("solution", r, kTol);
    ok = r.pass;
  }
#endif

  if (ok && timed && trials > 0) {
    KhTime tr = kh_time([&] { ref->run(); }, trials,
                        [&] { next_input(); kh_poison(Dref, nD * 2); });
    kh_print_time("Ref", tr);
    std::printf("  reference: %.1f TFLOP/s\n", d.flops() / (tr.min_ms * 1e-3) * 1e-12);
#ifdef KH_TEST_SOLUTION
    KhTime ts = kh_time(call_sol, trials, [&] { next_input(); reset_sol(); });
    kh_print_time("Kernel", ts);
    kh_print_speedup(tr, ts);
#endif
  }

#ifdef KH_TEST_SOLUTION
  // Re-check with yet another fresh input (also when timing is disabled): recompute the reference
  // output for it and make the solution produce the right answer again from a freshly poisoned buffer.
  if (ok) {
    refresh_input(777777u);
    kh_poison(Dref, nD * 2);
    ref->run();
    KH_CHECK(cudaDeviceSynchronize());
    reset_sol();
    KH_CHECK(cudaDeviceSynchronize());
    call_sol();
    KH_CHECK(cudaDeviceSynchronize());
    KhResult r = kh_compare(Dsol, Dref, nD, kTol);
    kh_print_result("solution (re-check, fresh input)", r, kTol);
    ok = r.pass;
  }
#endif

  if (Dsol) cudaFree(Dsol);
  cudaFree(A); cudaFree(B); cudaFree(Dref);
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

  //            N,  D,   H,   W,   C,   K,  T, R, S, pd,ph,pw, sd,sh,sw
  // Video VAE (batch 1, frames chunk D, 3x3x3 with padding 1, or the temporal (3,1,1) layers):
  // Wan2.1-style decoder widths 384 / 384 / 192 / 96 at latent, 2x, 4x, 8x resolution of a
  // 480x832 clip, and encoder-side down-sampling layers (temporal stride 2, spatial stride 2).
  // ResNet-50 at batch 256: the 3x3 convolutions of stages 1-3 (D = T = 1).
  std::vector<Layer> layers = {
      {"vae_3x3x3_t5_60x104_c384",    {1, 5, 60, 104, 384, 384, 3, 3, 3, 1, 1, 1, 1, 1, 1}, 2},
      {"vae_3x3x3_t9_120x208_c384",   {1, 9, 120, 208, 384, 384, 3, 3, 3, 1, 1, 1, 1, 1, 1}, 2},
      {"vae_3x3x3_t9_240x416_c192",   {1, 9, 240, 416, 192, 192, 3, 3, 3, 1, 1, 1, 1, 1, 1}, 3},
      {"vae_3x3x3_t5_480x832_c96",    {1, 5, 480, 832, 96, 96, 3, 3, 3, 1, 1, 1, 1, 1, 1}, 4},
      {"vae_3x1x1_tstride2_c384",     {1, 17, 120, 208, 384, 384, 3, 1, 1, 0, 0, 0, 2, 1, 1}, 2},
      {"vae_3x3x3_s122_c192",         {1, 9, 240, 416, 192, 192, 3, 3, 3, 1, 1, 1, 1, 2, 2}, 3},
      {"resnet50_b256_3x3_28x28_c128", {256, 1, 28, 28, 128, 128, 1, 3, 3, 0, 1, 1, 1, 1, 1}, 2},
      {"resnet50_b256_3x3_s2_c128",   {256, 1, 56, 56, 128, 128, 1, 3, 3, 0, 1, 1, 1, 2, 2}, 2},
      {"resnet50_b256_3x3_14x14_c256", {256, 1, 14, 14, 256, 256, 1, 3, 3, 0, 1, 1, 1, 1, 1}, 3},
      {"resnet50_b256_3x3_7x7_c512",  {256, 1, 7, 7, 512, 512, 1, 3, 3, 0, 1, 1, 1, 1, 1}, 2},
  };

  std::string only = (argc >= 2) ? argv[1] : "";   // dev aid: ./test <label substring>

#ifdef KH_SWEEP
  // Development aid: time every reference configuration on every layer.
  for (auto l : layers) {
    if (!only.empty() && std::string(l.label).find(only) == std::string::npos) continue;
    for (int cfg = std::getenv("KH_SWEEP_MIN") ? std::atoi(std::getenv("KH_SWEEP_MIN")) : 0; cfg < kNumCfg; ++cfg) {
      l.cfg = cfg;
      std::printf("\n=== %s cfg=%d ===\n", l.label, cfg);
      run_case(l, true, 10);
    }
  }
  return 0;
#endif

  for (auto const &l : layers) {
    if (!only.empty() && std::string(l.label).find(only) == std::string::npos) continue;
    ConvDesc const &d = l.d;
    std::printf("\n=== %s: N=%d, D=%d, H=%d, W=%d, C=%d, K=%d, T=%d, R=%d, S=%d ===\n", l.label,
                d.N, d.D, d.H, d.W, d.C, d.K, d.T, d.R, d.S);
    if (!run_case(l, true, trials)) { std::cout << "Incorrect" << std::endl; if (!keep_going) return 1; failed = true; }
    else std::cout << "Passed" << std::endl;
  }
  return failed ? 1 : 0;
#endif
  return 0;
}
