/***************************************************************************************************
 * Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met: (BSD-3-Clause, see the CUTLASS LICENSE.txt)
 **************************************************************************************************/

/*
  Task 86_blackwell_mixed_dtype_gemm: weight-only INT4 GEMM (W4A16) on Blackwell (sm_100a / B200).

  Reference: the CUTLASS 3.x Blackwell mixed-input GEMM (example 86, ScaleOnly mode: A is swapped
  with B and transposed so that the narrow weights are the MMA "A" operand), with one tile /
  cluster configuration chosen per problem shape from a sweep on B200.

  Solution under test (KH_TEST_SOLUTION), implemented in solution.h:

    cudaError_t BlackwellMixedDtypeGemm(int M, int N, int K, float alpha,
                                        __nv_bfloat16 const *A, int lda,
                                        uint8_t const *B, int ldb,
                                        __nv_bfloat16 const *B_scales, int scale_ldb,
                                        float beta, __nv_bfloat16 *D, int ldd);

    A        row-major [M, K] BF16 (lda = K)
    B        packed INT4 weights, column-major [K, N]: column j occupies K/2 contiguous bytes
             at B + j * ldb (ldb = K / 2 bytes); element (k, j) is in byte k / 2, low nibble for
             even k, high nibble for odd k; two's complement, values in [-8, 7]
    B_scales BF16 group scales, row-major [N, K / 128]: scale of (k, j) at
             B_scales[j * scale_ldb + k / 128] (scale_ldb = K / 128)
    D        row-major [M, N] BF16 (ldd = N);    D = alpha * A * (B * scale) + beta * D

  Verification: A and the scales are full-mantissa random BF16, B random nibbles; the reference
  and the solution write to different buffers pre-filled with NaN (beta != 0: with the same random
  C); every output element must be finite and within tolerance of the reference. The solution is
  re-checked after the timing loop.
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
#include "solution.h"   // must stay before kh_verify.h: the solution must not see the helpers
#endif

#include "cute/tensor.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/detail/collective/mixed_input_utils.hpp"

#include "kh_verify.h"
#include "marlin_ref.h"

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

using MmaType = cutlass::bfloat16_t;
using QuantType = cutlass::int4b_t;
using ElementA = MmaType;
using LayoutA = cutlass::layout::RowMajor;
constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
using ElementB = QuantType;
using LayoutB = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
using LayoutA_Transpose = typename cutlass::layout::LayoutTranspose<LayoutA>::type;
using LayoutB_Transpose = typename cutlass::layout::LayoutTranspose<LayoutB>::type;
using ElementScale = MmaType;
using ElementC = cutlass::bfloat16_t;
using LayoutC = cutlass::layout::RowMajor;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
using ElementD = ElementC;
using LayoutD = LayoutC;
constexpr int AlignmentD = AlignmentC;
using ElementAccumulator = float;
using ArchTag = cutlass::arch::Sm100;
using OperatorClass = cutlass::arch::OpClassTensorOp;
using EpilogueTileType = cutlass::epilogue::collective::EpilogueTileAuto;

constexpr int kGroup = 128;   // scale granularity along K
using ScaleConfig = cutlass::detail::Sm100MixedInputBlockwiseScaleConfig<1, kGroup>;
using LayoutScale = decltype(ScaleConfig::deduce_layout_scale());

/// MmaTile is given in the swapped problem: (weights N tile, tokens M tile, K tile).
template <class MmaTile, class ClusterShape, bool TwoSm>
struct GemmCfg {
  using MainloopSchedule = cute::conditional_t<
      TwoSm, cutlass::gemm::KernelTmaWarpSpecialized2SmMixedInputSm100,
      cutlass::gemm::KernelTmaWarpSpecialized1SmMixedInputSm100>;
  using EpilogueSchedule = cute::conditional_t<TwoSm, cutlass::epilogue::TmaWarpSpecialized2Sm,
                                               cutlass::epilogue::TmaWarpSpecialized1Sm>;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag, OperatorClass, MmaTile, ClusterShape, EpilogueTileType, ElementAccumulator,
      ElementAccumulator, ElementC, typename cutlass::layout::LayoutTranspose<LayoutC>::type,
      AlignmentC, ElementD, typename cutlass::layout::LayoutTranspose<LayoutD>::type, AlignmentD,
      EpilogueSchedule>::CollectiveOp;
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OperatorClass, cute::tuple<ElementB, ElementScale>,
      cute::tuple<LayoutB_Transpose, LayoutScale>, AlignmentB, ElementA, LayoutA_Transpose,
      AlignmentA, ElementAccumulator, MmaTile, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
          static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      MainloopSchedule>::CollectiveOp;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>,
                                                          CollectiveMainloop, CollectiveEpilogue>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

// Reference configurations (selected per shape below; see the KH_SWEEP development target).
// (weights tile, token tile, K tile), cluster, 2-SM?
using Cfg0 = GemmCfg<Shape<_256, _128, _128>, Shape<_2, _1, _1>, true>;
using Cfg1 = GemmCfg<Shape<_256, _64, _128>, Shape<_2, _1, _1>, true>;
using Cfg2 = GemmCfg<Shape<_128, _128, _128>, Shape<_1, _1, _1>, false>;
using Cfg3 = GemmCfg<Shape<_128, _64, _128>, Shape<_1, _1, _1>, false>;
using Cfg4 = GemmCfg<Shape<_128, _32, _128>, Shape<_1, _1, _1>, false>;
constexpr int kNumCfg = 5;

/// Transposes the harness scale layout [N, G] (row-major) into the CUTLASS ScaleOnly layout, in
/// which the scale of (n, group g) lives at g * N + n.
__global__ void kh_scale_to_cutlass(__nv_bfloat16 const *src, __nv_bfloat16 *dst, int N, int G) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < size_t(N) * G;
       i += size_t(gridDim.x) * blockDim.x) {
    int g = int(i / N), n = int(i % N);
    dst[i] = src[size_t(n) * G + g];
  }
}

/// One prepared CUTLASS GEMM (arguments, TMA descriptors and workspace are set up in the
/// constructor, so run() is just the kernel launch).
struct RefGemm {
  virtual ~RefGemm() {}
  virtual void run() = 0;
};

template <class Cfg>
struct RefGemmImpl : RefGemm {
  using Gemm = typename Cfg::Gemm;
  Gemm gemm;
  void *workspace = nullptr;
  RefGemmImpl(int M, int N, int K, float alpha, float beta, void const *Bq, void const *A,
              void const *scale_cutlass, void const *C, void *D) {
    using StrideA = cutlass::detail::TagToStrideA_t<LayoutA>;
    using StrideB = cutlass::detail::TagToStrideB_t<LayoutB>;
    using StrideC = typename Gemm::GemmKernel::StrideC;
    using StrideD = typename Gemm::GemmKernel::StrideD;
    StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
    StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1));
    // reversed because of the swap + transpose
    StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(N, M, 1));
    StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(N, M, 1));
    auto layout_S = ScaleConfig::tile_atom_to_shape_scale(make_shape(N, K, 1));
    // Pin down the assumption behind kh_scale_to_cutlass.
    if (layout_S(make_coord(5 % N, (K / kGroup - 1) * kGroup, 0)) != size_t(K / kGroup - 1) * N + 5 % N) {
      std::cerr << "unexpected CUTLASS scale layout" << std::endl;
      std::exit(2);
    }
    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {N, M, K, 1},
        {reinterpret_cast<ElementB const *>(Bq), stride_B,
         reinterpret_cast<ElementA const *>(A), stride_A,
         reinterpret_cast<ElementScale const *>(scale_cutlass), layout_S},
        {{alpha, beta}, reinterpret_cast<ElementC const *>(C), stride_C,
         reinterpret_cast<ElementD *>(D), stride_D}};
    size_t ws = Gemm::get_workspace_size(args);
    if (ws) KH_CHECK(cudaMalloc(&workspace, ws));
    CUTLASS_CHECK_KH(gemm.can_implement(args));
    CUTLASS_CHECK_KH(gemm.initialize(args, workspace));
  }
  ~RefGemmImpl() override { cudaFree(workspace); }
  void run() override { CUTLASS_CHECK_KH(gemm.run()); }
};

/// vLLM Marlin as the reference for small-batch decode (see marlin_ref.h); kMarlinCfg selects it.
constexpr int kMarlinCfg = 100;
struct MarlinRefWrap : RefGemm {
  MarlinRef m;
  MarlinRefWrap(int M, int N, int K, void const *Bq, void const *A, void const *S, void *D)
      : m(reinterpret_cast<uint8_t const *>(Bq), reinterpret_cast<__nv_bfloat16 const *>(S),
          reinterpret_cast<__nv_bfloat16 const *>(A), reinterpret_cast<__nv_bfloat16 *>(D), M, N, K) {}
  void run() override { m.run(); }
};

static std::unique_ptr<RefGemm> make_ref(int cfg, int M, int N, int K, float alpha, float beta,
                                         void const *Bq, void const *A, void const *S_harness,
                                         void const *S_cutlass, void const *C, void *D) {
  if (cfg == kMarlinCfg) return std::make_unique<MarlinRefWrap>(M, N, K, Bq, A, S_harness, D);
#define KH_CASE(i) case i: return std::make_unique<RefGemmImpl<Cfg##i>>(M, N, K, alpha, beta, Bq, A, S_cutlass, C, D)
  switch (cfg) {
    KH_CASE(0); KH_CASE(1); KH_CASE(2); KH_CASE(3); KH_CASE(4);
    default: std::cerr << "bad config" << std::endl; std::exit(2);
  }
#undef KH_CASE
}

// Tolerances (measured; D = BF16, rel. Frobenius error vs the reference, all sizes below). Both the
// CUTLASS and the Marlin kernel dequantize int4 * scale to BF16 before the MMA, so an exact FP32
// dequantization (the shipped baseline) differs from them by 2.4e-3 (max|err|/rms <= 3.8e-2), while a
// solution that also rounds the dequantized weight to BF16 matches to 6e-5 .. 3.7e-4 (Marlin shapes
// 6e-5 .. 1.9e-4). Both are correct. Bounds:
// Frobenius 7.5e-3 (3.1x the exact-dequant error), elements 6e-2*RMS + 1.6e-2*|ref| (2x /
// two BF16 ulps). Bad variants (all FAIL on every shape): FP16 accumulation 1.3e-2 .. 2.4e-2 (6.9e-3
// at K = 2048), BF16 accumulation 5e-2 .. 1.8e-1, FP8 E4M3 activations 2.4e-2, FP8 scales 2.5e-2,
// FP4 activations 2.5e-1, skipped K group / ignored scales / wrong scale layout / swapped or
// unsigned nibbles (O(1)), half the outputs unwritten (NaN), ignoring alpha / beta.
static KhTol kTol(6.0e-2, 7.5e-3, 1.6e-2);

struct Shape3 { const char *label; int M, N, K, cfg; };

/// Fills n bytes with random bytes (every byte = two uniformly random int4 values).
__global__ void kh_fill_bytes(uint8_t *p, size_t n, unsigned int seed) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n;
       i += size_t(gridDim.x) * blockDim.x)
    p[i] = uint8_t(kh_hash64(i, seed) >> 40);
}

__global__ void kh_fill_scales(__nv_bfloat16 *p, size_t n, unsigned int seed) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n;
       i += size_t(gridDim.x) * blockDim.x)
    p[i] = __float2bfloat16_rn(kh_uniform_pos(i, seed, 0.004f, 0.06f));
}

/// Returns false on failure. `timed` = also print Ref / Kernel timings.
static bool run_case(Shape3 const &s, float alpha, float beta, bool timed, int trials) {
  int M = s.M, N = s.N, K = s.K, G = K / kGroup;
  size_t nA = size_t(M) * K, nB = size_t(K) * N / 2, nS = size_t(N) * G, nC = size_t(M) * N;
  __nv_bfloat16 *A, *S, *S_cut, *C_in = nullptr, *D_ref;
  uint8_t *B;
  KH_CHECK(cudaMalloc(&A, nA * 2));
  KH_CHECK(cudaMalloc(&B, nB));
  KH_CHECK(cudaMalloc(&S, nS * 2));
  KH_CHECK(cudaMalloc(&S_cut, nS * 2));
  KH_CHECK(cudaMalloc(&D_ref, nC * 2));
  kh_fill_uniform(A, nA, 11u);
  kh_fill_bytes<<<4096, 256>>>(B, nB, 22u);
  kh_fill_scales<<<1024, 256>>>(S, nS, 44u);
  kh_scale_to_cutlass<<<1024, 256>>>(S, S_cut, N, G);
  KH_CHECK(cudaGetLastError());
  if (beta != 0.f) {
    KH_CHECK(cudaMalloc(&C_in, nC * 2));
    kh_fill_uniform(C_in, nC, 33u);
  }
  kh_poison(D_ref, nC * 2);
  KH_CHECK(cudaDeviceSynchronize());

  auto ref = make_ref(s.cfg, M, N, K, alpha, beta, B, A, S, S_cut, C_in, D_ref);
  ref->run();
  KH_CHECK(cudaDeviceSynchronize());

  bool ok = true;
  {
    KhResult r = kh_compare(D_ref, D_ref, nC, kTol);
    if (!r.pass) {
      std::cerr << "Reference output incomplete or non-finite (non-finite = " << r.nonfinite
                << ")" << std::endl;
      ok = false;
    }
  }

#ifdef KH_TEST_SOLUTION
  __nv_bfloat16 *D_sol = nullptr;
  KH_CHECK(cudaMalloc(&D_sol, nC * 2));
  auto reset_sol = [&] {
    if (beta != 0.f) KH_CHECK(cudaMemcpy(D_sol, C_in, nC * 2, cudaMemcpyDeviceToDevice));
    else kh_poison(D_sol, nC * 2);
  };
  auto call_sol = [&] {
    cudaError_t e = BlackwellMixedDtypeGemm(M, N, K, alpha, A, K, B, K / 2, S, G, beta, D_sol, N);
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

  // The activations are refreshed (new random values, outside the timed region) before every call
  // of both the reference and the solution, so a result cannot be memoized across calls; the weights
  // and scales stay fixed (a solution may repack them once, see the description).
  unsigned seed = 1000u;
  auto refresh_A = [&] { kh_fill_uniform(A, nA, seed++); };

  if (ok && timed && trials > 0) {
    KhTime tr = kh_time([&] { ref->run(); }, trials,
                        [&] { kh_poison(D_ref, nC * 2); refresh_A(); });
    kh_print_time("Ref", tr);
#ifdef KH_TEST_SOLUTION
    KhTime ts = kh_time(call_sol, trials, [&] { reset_sol(); refresh_A(); });
    kh_print_time("Kernel", ts);
    kh_print_speedup(tr, ts);
#endif
  }

#ifdef KH_TEST_SOLUTION
  // Re-check with yet another fresh A (runs for every case, also when timing is off): the solution
  // must still produce the right answer from scratch, so a memoized result fails here.
  if (ok) {
    refresh_A();
    kh_poison(D_ref, nC * 2);
    ref->run();
    KH_CHECK(cudaDeviceSynchronize());
    reset_sol();
    KH_CHECK(cudaDeviceSynchronize());
    call_sol();
    KH_CHECK(cudaDeviceSynchronize());
    KhResult r = kh_compare(D_sol, D_ref, nC, kTol);
    kh_print_result("solution (re-check, fresh A)", r, kTol);
    ok = r.pass;
  }
#endif

#ifdef KH_TEST_SOLUTION
  cudaFree(D_sol);
#endif
  cudaFree(A); cudaFree(B); cudaFree(S); cudaFree(S_cut); cudaFree(D_ref);
  if (C_in) cudaFree(C_in);
  return ok;
}

#endif  // CUTLASS_ARCH_MMA_SM100_SUPPORTED

int main(int argc, char const **argv) {
  // Never exit 0 without having run the tests: a skipped run must not look like a pass.
#if !defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
  std::cerr << "CUTLASS SM100 support not compiled in (build with -arch sm_100a)." << std::endl;
  return 1;
#else
  cudaDeviceProp props;
  KH_CHECK(cudaGetDeviceProperties(&props, 0));
  if (props.major != 10 || props.minor != 0) {
    std::cerr << "This task requires a Blackwell sm_100 GPU (B200)." << std::endl;
    return 1;
  }
  int trials = kh_trials();
  bool const keep_going = std::getenv("KH_KEEP_GOING") != nullptr;   // development aid
  bool failed = false;

  // Llama-3-70B weight-only INT4 (W4A16) linear layers on one B200 (hidden 8192, fused QKV
  // 10240, FFN 28672 -> fused gate+up 57344); M = tokens: decode batches of 1..128 sequences
  // (weight-bandwidth bound) and one 8K-token prefill chunk.
  std::vector<Shape3> shapes = {
      {"decode_m1_mlp_up",      1, 57344,  8192, kMarlinCfg},
      {"decode_m16_qkv",       16, 10240,  8192, kMarlinCfg},
      {"decode_m64_mlp_down",  64,  8192, 28672, kMarlinCfg},
      {"decode_m128_o_proj",  128,  8192,  8192, 2},
      {"decode_m128_mlp_up",  128, 57344,  8192, 2},
      {"prefill_qkv",        8192, 10240,  8192, 0},
  };

#ifdef KH_SWEEP
  // Development aid: time every reference configuration on every shape.
  for (auto s : shapes) {
    for (int cfg = 0; cfg < kNumCfg; ++cfg) {
      s.cfg = cfg;
      std::printf("\n=== %s cfg=%d: M=%d, N=%d, K=%d ===\n", s.label, cfg, s.M, s.N, s.K);
      run_case(s, 1.f, 0.f, true, 10);
    }
    if (s.M <= 128 && marlin_ref_supported(s.M, s.N, s.K)) {
      s.cfg = kMarlinCfg;
      std::printf("\n=== %s cfg=marlin: M=%d, N=%d, K=%d ===\n", s.label, s.M, s.N, s.K);
      run_case(s, 1.f, 0.f, true, 10);
    }
  }
  return 0;
#endif

  if (argc >= 4) {   // ./test M N K : single custom shape
    Shape3 s{"custom", atoi(argv[1]), atoi(argv[2]), atoi(argv[3]), (atoi(argv[1]) <= 64 && marlin_ref_supported(atoi(argv[1]), atoi(argv[2]), atoi(argv[3]))) ? kMarlinCfg : (atoi(argv[1]) <= 64 ? 3 : (atoi(argv[1]) <= 128 ? 2 : 0))};
    std::printf("\n=== custom: M=%d, N=%d, K=%d ===\n", s.M, s.N, s.K);
    if (!run_case(s, 1.f, 0.f, true, trials)) { std::cout << "Incorrect" << std::endl; return 1; }
    std::cout << "Passed" << std::endl;
    return 0;
  }

  // alpha / beta handling (correctness only): D = 0.75 * A*dequant(B) + 0.5 * C
  {
    Shape3 s{"alpha_beta", 48, 1280, 8192, 3};
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
}
