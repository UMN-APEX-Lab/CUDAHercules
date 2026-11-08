/***************************************************************************************************
 * Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met: (BSD-3-Clause, see the CUTLASS LICENSE.txt)
 **************************************************************************************************/

/*
  Task 92_blackwell_moe_gemm: uneven-token Mixture-of-Experts grouped GEMM with DeepSeek-V3 FP8
  block scaling, FP8 E4M3 in / BF16 out, on Blackwell (sm_100a / B200).

  Workload: the routed-expert GEMMs of a DeepSeek-V3 MoE layer (256 routed experts, top-8, hidden
  7168, expert FFN 2048) as they look after a token dispatcher: the routed token rows are sorted by
  expert and the dispatcher hands over per-expert counts and offsets on the device. The counts are
  uneven (Zipf-skewed router, some experts empty); decode batches, prefill chunks and an
  expert-parallel shard (32 local experts, 16K dispatched tokens) are tested.

  Quantisation (DeepSeek-V3): activations are FP8 E4M3 with one FP32 scale per token and per 128
  channels of K (1x128 blocks), weights are FP8 E4M3 with one FP32 scale per 128x128 block (N x K) of
  each expert.

  Reference: the CUTLASS 3.x Blackwell grouped (pointer-array) blockwise-scaled GEMM (example 81,
  KernelPtrArrayTmaWarpSpecializedBlockwise{1,2}SmSm100: the scales are applied per 128-wide K block
  inside the mainloop) -- the same kernel family real MoE stacks use on sm100. The group tables
  (problem sizes, operand / scale pointers, strides, scale layouts) are built on the device from the
  dispatcher's counts and offsets by a small kernel that is part of every timed reference call.
  Tile / cluster configuration per scenario from a sweep on B200.

  Solution under test (KH_TEST_SOLUTION), implemented in solution.h:

    cudaError_t BlackwellMoeGemm(int num_experts, int total_rows, int N, int K,
                                 __nv_fp8_e4m3 const *X,       // [total_rows, K]  row-major
                                 float const *X_scale,         // [total_rows, K/128] row-major
                                 __nv_fp8_e4m3 const *W,       // [num_experts, N, K] row-major
                                 float const *W_scale,         // [num_experts, N/128, K/128] row-major
                                 int const *expert_offsets,    // [num_experts + 1] (device)
                                 int const *tokens_per_expert, // [num_experts] (device)
                                 __nv_bfloat16 *Y);            // [total_rows, N]  row-major

    for every expert e and row r in [expert_offsets[e], expert_offsets[e+1]):
        Y[r, n] = sum_{kb=0}^{K/128-1}  X_scale[r, kb] * W_scale[e, n/128, kb]
                      * ( sum_{k=128 kb}^{128 kb + 127}  X[r, k] * W[e, n, k] )     (FP32 accumulation)

  Verification: inputs are full-range random FP8, scales are random positive; the reference and the
  solution write to different buffers that are pre-filled with NaN; every output element must be
  finite and within tolerance of the reference. The solution is re-checked after the timing loop.
*/

#include <cstdio>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "cutlass/cutlass.h"

#ifdef KH_TEST_SOLUTION
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include "solution.h"   // must stay before kh_verify.h: the solution must not see the helpers
#endif

#include "cute/tensor.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
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

using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;   // <M,N,K> per group
using ElementIn = cutlass::float_e4m3_t;
using ElementD = cutlass::bfloat16_t;
constexpr int AlignmentIn = 128 / cutlass::sizeof_bits<ElementIn>::value;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
using ElementAccumulator = float;
using ArchTag = cutlass::arch::Sm100;
using OperatorClass = cutlass::arch::OpClassTensorOp;
using ClusterShape = Shape<int32_t, int32_t, _1>;   // runtime (preferred / fallback) cluster
constexpr int kBlk = 128;                           // FP8 scaling block size (K, and N of the weights)

struct Sm1 {
  using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedBlockwise1SmSm100;
  using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm;
};
struct Sm2 {
  using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedBlockwise2SmSm100;
  using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized2Sm;
};

// Swap == false: MMA-M = tokens (A = activations, scale granularity 1 x 128; B = weights, 128 x 128),
//                D row-major [tokens, N].
// Swap == true : "swap-AB": MMA-M = output features (A = weights, granularity 128 x 128; B =
//                activations, granularity 1 x 128), D column-major [N, tokens] == the row-major
//                [tokens, N] output; the small / ragged token dimension maps to the flexible MMA-N.
// All scale tensors are K-major (K/128 scales of a row contiguous), exactly the layouts of the
// task: X_scale [rows, K/128] and W_scale [E, N/128, K/128].
template <class MmaTile, class SmCfg, bool Swap_>
struct MoeCfg {
  static constexpr bool Swap = Swap_;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutD = std::conditional_t<Swap_, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>;
  using ScaleConfig = std::conditional_t<
      Swap_,
      cutlass::detail::Sm100BlockwiseScaleConfig<128, 1, 128, cute::UMMA::Major::K, cute::UMMA::Major::K>,
      cutlass::detail::Sm100BlockwiseScaleConfig<1, 128, 128, cute::UMMA::Major::K, cute::UMMA::Major::K>>;
  using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
  using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag, OperatorClass, MmaTile, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator, ElementAccumulator,
      void, LayoutD *, AlignmentD,   // no C operand
      ElementD, LayoutD *, AlignmentD, typename SmCfg::EpilogueSchedule>::CollectiveOp;
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OperatorClass, ElementIn, cute::tuple<LayoutA *, LayoutSFA *>, AlignmentIn, ElementIn,
      cute::tuple<LayoutB *, LayoutSFB *>, AlignmentIn, ElementAccumulator, MmaTile, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
          static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      typename SmCfg::KernelSchedule>::CollectiveOp;
  using GemmKernel =
      cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloop, CollectiveEpilogue>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

// Reference configurations (MMA tile M x N x K), selected per scenario below.
using Cfg0 = MoeCfg<Shape<_256, _128, _128>, Sm2, false>;
using Cfg1 = MoeCfg<Shape<_128, _128, _128>, Sm1, false>;
using Cfg2 = MoeCfg<Shape<_256, _256, _128>, Sm2, false>;
using Cfg3 = MoeCfg<Shape<_128, _256, _128>, Sm1, false>;
using Cfg4 = MoeCfg<Shape<_256, _128, _128>, Sm2, true>;
using Cfg5 = MoeCfg<Shape<_256, _64, _128>, Sm2, true>;
using Cfg6 = MoeCfg<Shape<_128, _128, _128>, Sm1, true>;
using Cfg7 = MoeCfg<Shape<_128, _64, _128>, Sm1, true>;
using Cfg8 = MoeCfg<Shape<_128, _32, _128>, Sm1, true>;
using Cfg9 = MoeCfg<Shape<_256, _256, _128>, Sm2, true>;
constexpr int kNumCfg = 10;
constexpr int kNumCluster = 5;
static bool cfg_is_2sm(int cfg) { return cfg == 0 || cfg == 2 || cfg == 4 || cfg == 5 || cfg == 9; }
// preferred cluster x,y / fallback x,y
static const int kClusters[kNumCluster][4] = {{2, 1, 2, 1}, {4, 1, 2, 1}, {4, 2, 2, 1},
                                              {1, 1, 1, 1}, {2, 2, 2, 1}};

/// Device-side builder of the per-group tables of the CUTLASS grouped kernel from the dispatcher's
/// offsets / counts (part of every timed reference call, like the tables a real MoE stack builds).
template <class Cfg>
struct RefTables {
  using Gemm = typename Cfg::Gemm;
  using StrideA = typename Gemm::GemmKernel::InternalStrideA;
  using StrideB = typename Gemm::GemmKernel::InternalStrideB;
  using StrideD = typename Gemm::GemmKernel::InternalStrideD;
  using LayoutSFA = typename Cfg::LayoutSFA;
  using LayoutSFB = typename Cfg::LayoutSFB;
  Shape<int, int, int> *sizes;
  ElementIn const **pA, **pB;
  float const **pSFA, **pSFB;
  ElementD **pD;
  StrideA *sA;
  StrideB *sB;
  StrideD *sD;
  LayoutSFA *lSFA;
  LayoutSFB *lSFB;
};

template <class Cfg>
__global__ void ref_build_tables(RefTables<Cfg> t, int E, int N, int K, int const *offsets,
                                 int const *counts, ElementIn const *X, float const *Xs,
                                 ElementIn const *W, float const *Ws, ElementD *Y) {
  int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= E) return;
  using StrideA = typename RefTables<Cfg>::StrideA;
  using StrideB = typename RefTables<Cfg>::StrideB;
  using StrideD = typename RefTables<Cfg>::StrideD;
  int M = counts[e], KB = K / kBlk, NB = N / kBlk;
  size_t off = offsets[e];
  ElementIn const *xa = X + off * K;
  ElementIn const *wa = W + size_t(e) * N * K;
  float const *xs = Xs + off * KB;
  float const *ws = Ws + size_t(e) * NB * KB;
  if constexpr (!Cfg::Swap) {
    t.sizes[e] = {M, N, K};
    t.pA[e] = xa; t.pB[e] = wa; t.pSFA[e] = xs; t.pSFB[e] = ws;
    t.sA[e] = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
    t.sB[e] = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
    t.sD[e] = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});
    t.lSFA[e] = Cfg::ScaleConfig::tile_atom_to_shape_SFA(make_shape(M, N, K, 1));
    t.lSFB[e] = Cfg::ScaleConfig::tile_atom_to_shape_SFB(make_shape(M, N, K, 1));
  } else {
    t.sizes[e] = {N, M, K};
    t.pA[e] = wa; t.pB[e] = xa; t.pSFA[e] = ws; t.pSFB[e] = xs;
    t.sA[e] = cutlass::make_cute_packed_stride(StrideA{}, {N, K, 1});
    t.sB[e] = cutlass::make_cute_packed_stride(StrideB{}, {M, K, 1});
    t.sD[e] = cutlass::make_cute_packed_stride(StrideD{}, {N, M, 1});
    t.lSFA[e] = Cfg::ScaleConfig::tile_atom_to_shape_SFA(make_shape(N, M, K, 1));
    t.lSFB[e] = Cfg::ScaleConfig::tile_atom_to_shape_SFB(make_shape(N, M, K, 1));
  }
  t.pD[e] = Y + off * N;
}

struct RefMoe {
  virtual ~RefMoe() {}
  virtual void run() = 0;
};

template <class Cfg>
struct RefMoeImpl : RefMoe {
  using Gemm = typename Cfg::Gemm;
  Gemm gemm;
  void *workspace = nullptr;
  RefTables<Cfg> t;
  int E, N, K;
  int const *offsets, *counts;
  ElementIn const *X, *W;
  float const *Xs, *Ws;
  ElementD *Y;

  template <class T> static T *alloc(size_t n) {
    T *p;
    KH_CHECK(cudaMalloc(&p, sizeof(T) * std::max<size_t>(n, 1)));
    return p;
  }

  RefMoeImpl(int E_, int N_, int K_, void const *X_, void const *Xs_, void const *W_, void const *Ws_,
             int const *counts_, int const *offsets_, void *Y_, int cluster)
      : E(E_), N(N_), K(K_), offsets(offsets_), counts(counts_),
        X(static_cast<ElementIn const *>(X_)), W(static_cast<ElementIn const *>(W_)),
        Xs(static_cast<float const *>(Xs_)), Ws(static_cast<float const *>(Ws_)),
        Y(static_cast<ElementD *>(Y_)) {
    t.sizes = alloc<Shape<int, int, int>>(E);
    t.pA = alloc<ElementIn const *>(E); t.pB = alloc<ElementIn const *>(E);
    t.pSFA = alloc<float const *>(E); t.pSFB = alloc<float const *>(E);
    t.pD = alloc<ElementD *>(E);
    t.sA = alloc<typename RefTables<Cfg>::StrideA>(E);
    t.sB = alloc<typename RefTables<Cfg>::StrideB>(E);
    t.sD = alloc<typename RefTables<Cfg>::StrideD>(E);
    t.lSFA = alloc<typename RefTables<Cfg>::LayoutSFA>(E);
    t.lSFB = alloc<typename RefTables<Cfg>::LayoutSFB>(E);
    build_tables();
    KH_CHECK(cudaDeviceSynchronize());

    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
    hw_info.cluster_shape = dim3(kClusters[cluster][0], kClusters[cluster][1], 1);
    hw_info.cluster_shape_fallback = dim3(kClusters[cluster][2], kClusters[cluster][3], 1);

    typename Gemm::Arguments args;
    decltype(args.epilogue.thread) fusion_args;
    fusion_args.alpha_ptr = nullptr;
    fusion_args.beta_ptr = nullptr;
    fusion_args.alpha = 1.f;
    fusion_args.beta = 0.f;
    fusion_args.alpha_ptr_array = nullptr;
    fusion_args.beta_ptr_array = nullptr;
    fusion_args.dAlpha = {_0{}, _0{}, 0};
    fusion_args.dBeta = {_0{}, _0{}, 0};
    typename Gemm::GemmKernel::TileSchedulerArguments scheduler;
    args = typename Gemm::Arguments{cutlass::gemm::GemmUniversalMode::kGrouped,
                                    {E, t.sizes, nullptr},
                                    {t.pA, t.sA, t.pB, t.sB, t.pSFA, t.lSFA, t.pSFB, t.lSFB},
                                    {fusion_args, nullptr, nullptr, t.pD, t.sD},
                                    hw_info, scheduler};
    size_t ws = Gemm::get_workspace_size(args);
    if (ws) KH_CHECK(cudaMalloc(&workspace, ws));
    CUTLASS_CHECK_KH(gemm.can_implement(args));
    CUTLASS_CHECK_KH(gemm.initialize(args, workspace));
  }
  void build_tables() {
    ref_build_tables<Cfg><<<(E + 127) / 128, 128>>>(t, E, N, K, offsets, counts, X, Xs, W, Ws, Y);
  }
  ~RefMoeImpl() override {
    cudaFree(workspace); cudaFree(t.sizes); cudaFree(t.pA); cudaFree(t.pB); cudaFree(t.pSFA);
    cudaFree(t.pSFB); cudaFree(t.pD); cudaFree(t.sA); cudaFree(t.sB); cudaFree(t.sD);
    cudaFree(t.lSFA); cudaFree(t.lSFB);
  }
  void run() override {
    build_tables();
    CUTLASS_CHECK_KH(gemm.run());
  }
};

static std::unique_ptr<RefMoe> make_ref(int cfg, int cluster, int E, int N, int K, void const *X,
                                        void const *Xs, void const *W, void const *Ws,
                                        int const *cnt, int const *off, void *Y) {
#define KH_MK(i) \
  case i: return std::make_unique<RefMoeImpl<Cfg##i>>(E, N, K, X, Xs, W, Ws, cnt, off, Y, cluster);
  switch (cfg) {
    KH_MK(0) KH_MK(1) KH_MK(2) KH_MK(3) KH_MK(4) KH_MK(5) KH_MK(6) KH_MK(7) KH_MK(8)
    default: return std::make_unique<RefMoeImpl<Cfg9>>(E, N, K, X, Xs, W, Ws, cnt, off, Y, cluster);
  }
#undef KH_MK
}

// ---------------------------------------------------------------------------------------------
// Scenarios: DeepSeek-V3 routed-expert GEMMs with skewed top-k routing
// ---------------------------------------------------------------------------------------------

/// `tokens` tokens are routed to `topk` of `E` experts by a router with Zipf-skewed popularity
/// (exponent `skew`, random expert order) plus `cold` experts that receive nothing; per token the
/// top-k of (log-popularity + Gumbel noise) are chosen, i.e. sampling without replacement as a real
/// top-k router does. Expert e then owns the contiguous rows [offsets[e], offsets[e+1]).
struct Scenario {
  char const *label;
  int E, tokens, topk, N, K;
  double skew;
  int cold;
  unsigned seed;
  int cfg, cluster;   // reference configuration (from the sweep)
};

static std::vector<int> route_counts(Scenario const &s) {
  int E = s.E;
  auto uni = [&](unsigned long long i, unsigned salt) {   // uniform in (0,1)
    return (double(kh_hash64(i, s.seed * 131u + salt) >> 11) + 0.5) * (1.0 / 9007199254740992.0);
  };
  std::vector<int> perm(E);
  std::iota(perm.begin(), perm.end(), 0);
  for (int i = E - 1; i > 0; --i) std::swap(perm[i], perm[int(uni(i, 1) * (i + 1))]);
  std::vector<double> logp(E);
  for (int r = 0; r < E; ++r) logp[perm[r]] = -s.skew * std::log(double(r + 1));
  std::vector<int> perm2(E);
  std::iota(perm2.begin(), perm2.end(), 0);
  for (int i = E - 1; i > 0; --i) std::swap(perm2[i], perm2[int(uni(i, 2) * (i + 1))]);
  for (int c = 0; c < s.cold; ++c) logp[perm2[c]] = -1e30;
  std::vector<int> cnt(E, 0);
  std::vector<std::pair<double, int>> sc(E);
  for (int t = 0; t < s.tokens; ++t) {
    for (int e = 0; e < E; ++e)
      sc[e] = {logp[e] - std::log(-std::log(uni(size_t(t) * E + e, 3))), e};
    std::partial_sort(sc.begin(), sc.begin() + s.topk, sc.end(),
                      [](auto const &a, auto const &b) { return a.first > b.first; });
    for (int i = 0; i < s.topk; ++i) cnt[sc[i].second]++;
  }
  return cnt;
}

__global__ void kh_fill_scale(float *p, size_t n, unsigned seed, float lo, float hi) {
  size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
  if (i < n) p[i] = kh_uniform_pos(i, seed, lo, hi);   // random positive, full-mantissa FP32
}

/// Tolerances for the BF16 output. Every element must satisfy |x - ref| <= 1e-2 * RMS(ref) + |ref|/128
/// (the second term is one BF16 ulp) and the relative Frobenius error must be <= 1e-4.
/// Reference and solution accumulate each 128-wide K block in FP32 (exact FP8 products), apply the
/// block scales in FP32 and round the result to BF16. Measured on B200 over all scenarios
/// (max|err|/RMS, rel. Frobenius error):
///   plain sequential FP32 kernel (shipped naive solution):   <= 1.2e-2 (1 ulp of the largest
///        elements, covered by the |ref|/128 term),                   <= 5.7e-6   -> bound 17x above
///   FP16 accumulation of the products:         2.4e-2 .. 4.8e-2,    2.6e-3  (fails, 26x above bound)
///   inputs rounded to FP4 E2M1 / FP8 E5M2:     0.86 .. 1.2 / 0.50 .. 0.65,  0.156 / 0.089 (fail)
///   skipped last 128-wide K block:             1.5 .. 3.2,          0.13 .. 0.25  (fails)
///   activation scales ignored / weight scales ignored:       4e2 / 9e3,      81 / 1.6e3  (fail)
///   per-expert (first block) weight scale, per-token (first block) activation scale, scale tables
///   transposed / MN-major / 64-wide N blocks / not offset by expert / wrong expert:  3.3 .. 6.2,
///                                                                    0.56 .. 0.63  (fail)
/// Structural errors (unwritten rows, wrong expert, dropped expert, ragged tail, equal-size
/// assumption, padded-offset assumption) fail through the NaN / non-finite check or by O(1) errors.
static KhTol kTol(1.0e-2, 1.0e-4, 1.0 / 128);

/// Returns false on failure. `timed` = also print Ref / Kernel timings.
static bool run_case(Scenario const &s, bool timed, int trials) {
  int E = s.E, N = s.N, K = s.K;
  std::vector<int> cnt = route_counts(s);
  std::vector<int> off(E + 1, 0);
  for (int e = 0; e < E; ++e) off[e + 1] = off[e] + cnt[e];
  int R = off[E];
  int empty = 0, maxrows = 0, minrows = 1 << 30;
  for (int e = 0; e < E; ++e) {
    empty += cnt[e] == 0; maxrows = std::max(maxrows, cnt[e]); minrows = std::min(minrows, cnt[e]);
  }
  double flops = 2.0 * R * N * K;
  // minimum DRAM traffic: weights of the non-empty experts + all tokens in + all outputs out
  double bytes = double(E - empty) * N * K + double(R) * K + double(R) * N * 2 +
                 4.0 * (double(R) * (K / kBlk) + double(E - empty) * (N / kBlk) * (K / kBlk));
  std::printf("  experts = %d (empty = %d), rows per expert: min %d, mean %.1f, max %d, total %d; "
              "%.1f GFLOP\n", E, empty, minrows, double(R) / E, maxrows, R, flops * 1e-9);

  size_t nX = size_t(R) * K, nW = size_t(E) * N * K, nY = size_t(R) * N;
  size_t nXs = size_t(R) * (K / kBlk), nWs = size_t(E) * (N / kBlk) * (K / kBlk);
  __nv_fp8_e4m3 *X, *W;
  __nv_bfloat16 *Yref, *Ysol = nullptr;
  float *Xs, *Ws;
  int *d_cnt, *d_off;
  KH_CHECK(cudaMalloc(&X, nX));
  KH_CHECK(cudaMalloc(&W, nW));
  KH_CHECK(cudaMalloc(&Yref, nY * 2));
  KH_CHECK(cudaMalloc(&Xs, nXs * sizeof(float)));
  KH_CHECK(cudaMalloc(&Ws, nWs * sizeof(float)));
  KH_CHECK(cudaMalloc(&d_cnt, E * sizeof(int)));
  KH_CHECK(cudaMalloc(&d_off, (E + 1) * sizeof(int)));
  KH_CHECK(cudaMemcpy(d_cnt, cnt.data(), E * sizeof(int), cudaMemcpyHostToDevice));
  KH_CHECK(cudaMemcpy(d_off, off.data(), (E + 1) * sizeof(int), cudaMemcpyHostToDevice));
  // Quantised data use the whole FP8 range (a block's amax maps to 448); scales are amax / 448 of
  // realistic activations (about 2e-3 .. 2e-2) and weights (about 1e-4 .. 1e-3).
  // The activations (FP8 data and their per-token scales) change on every call in deployment: they
  // are regenerated with a new seed outside the timed region before every timed call of the
  // reference and of the solution, so outputs and input-dependent intermediates cannot be cached.
  // Weights and weight scales are fixed.
  auto refresh_input = [&](unsigned seed) {
    kh_fill_uniform(X, nX, 11u + s.seed + seed, 448.f);
    kh_fill_scale<<<(unsigned)((nXs + 255) / 256), 256>>>(Xs, nXs, 33u + s.seed + seed, 2.0e-3f, 2.0e-2f);
    KH_CHECK(cudaGetLastError());
  };
  unsigned seed_counter = 1000;
  auto next_input = [&] { refresh_input(++seed_counter); };
  refresh_input(0);
  kh_fill_uniform(W, nW, 22u + s.seed, 448.f);
  kh_fill_scale<<<(unsigned)((nWs + 255) / 256), 256>>>(Ws, nWs, 44u + s.seed, 1.0e-4f, 1.0e-3f);
  KH_CHECK(cudaGetLastError());
  kh_poison(Yref, nY * 2);
  KH_CHECK(cudaDeviceSynchronize());

  auto ref = make_ref(s.cfg, s.cluster, E, N, K, X, Xs, W, Ws, d_cnt, d_off, Yref);
  ref->run();
  KH_CHECK(cudaDeviceSynchronize());

  bool ok = true;
  // The reference must have written every element (it was pre-filled with NaN).
  {
    KhResult r = kh_compare(Yref, Yref, nY, kTol);
    if (!r.pass) {
      std::cerr << "Reference output incomplete or non-finite (non-finite = " << r.nonfinite << ")"
                << std::endl;
      ok = false;
    }
  }

#ifdef KH_TEST_SOLUTION
  KH_CHECK(cudaMalloc(&Ysol, nY * 2));
  auto reset_sol = [&] { kh_poison(Ysol, nY * 2); };
  auto call_sol = [&] {
    cudaError_t e = BlackwellMoeGemm(E, R, N, K, X, Xs, W, Ws, d_off, d_cnt, Ysol);
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
    KhResult r = kh_compare(Ysol, Yref, nY, kTol);
    kh_print_result("solution", r, kTol);
    ok = r.pass;
  }
#endif

  if (ok && timed && trials > 0) {
    KhTime tr = kh_time([&] { ref->run(); }, trials,
                        [&] { next_input(); kh_poison(Yref, nY * 2); });
    kh_print_time("Ref", tr);
    std::printf("  reference: %.1f TFLOP/s, %.2f TB/s of minimum DRAM traffic\n",
                flops / (tr.min_ms * 1e-3) * 1e-12, bytes / (tr.min_ms * 1e-3) * 1e-12);
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
    kh_poison(Yref, nY * 2);
    ref->run();
    KH_CHECK(cudaDeviceSynchronize());
    reset_sol();
    KH_CHECK(cudaDeviceSynchronize());
    call_sol();
    KH_CHECK(cudaDeviceSynchronize());
    KhResult r = kh_compare(Ysol, Yref, nY, kTol);
    kh_print_result("solution (re-check, fresh input)", r, kTol);
    ok = r.pass;
  }
#endif

  if (Ysol) cudaFree(Ysol);
  cudaFree(X); cudaFree(W); cudaFree(Yref); cudaFree(Xs); cudaFree(Ws); cudaFree(d_cnt); cudaFree(d_off);
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

  // DeepSeek-V3: 256 routed experts, top-8, hidden 7168, expert FFN 2048. Fused gate+up
  // projection: [rows, 7168] x [7168 -> 4096]; down projection: [rows, 2048] x [2048 -> 7168].
  const int E = 256, H = 7168, I2 = 4096, I = 2048;
  // {label, E, tokens, topk, N, K, skew, cold experts, seed, ref cfg, ref cluster}
  // Reference configuration per scenario (cfg, cluster) from the KH_SWEEP build on B200.
  std::vector<Scenario> scenarios = {
      {"decode128_gate_up",   E, 128, 8, I2, H, 1.0, 0, 21, 8, 1},   // 1024 routed rows
      {"decode128_down",      E, 128, 8, H, I, 1.0, 0, 21, 8, 1},
      {"decode512_gate_up",   E, 512, 8, I2, H, 1.0, 0, 22, 8, 1},   // 4096 routed rows
      {"prefill4k_gate_up",   E, 4096, 8, I2, H, 0.8, 16, 23, 1, 3}, // 32768 routed rows
      {"prefill4k_down",      E, 4096, 8, H, I, 0.8, 16, 23, 1, 3},
      {"ep8_prefill16k_gate_up", 32, 16384, 1, I2, H, 0.7, 2, 24, 1, 3},  // 32 local experts
      {"ep8_prefill16k_down", 32, 16384, 1, H, I, 0.7, 2, 24, 1, 3},
  };

  std::string only = (argc >= 2) ? argv[1] : "";   // dev aid: ./test <label substring>

#ifdef KH_SWEEP
  // Development aid: time every reference configuration on every scenario.
  for (auto s : scenarios) {
    if (!only.empty() && std::string(s.label).find(only) == std::string::npos) continue;
    for (int cfg = 0; cfg < kNumCfg; ++cfg)
      for (int cl = 0; cl < kNumCluster; ++cl) {
        if (cfg_is_2sm(cfg) && kClusters[cl][0] < 2) continue;
        s.cfg = cfg; s.cluster = cl;
        std::printf("\n=== %s cfg=%d cluster=%d ===\n", s.label, cfg, cl);
        run_case(s, true, 10);
      }
  }
  return 0;
#endif

  for (auto const &s : scenarios) {
    if (!only.empty() && std::string(s.label).find(only) == std::string::npos) continue;
    std::printf("\n=== %s: M=%d, N=%d, K=%d ===\n", s.label, s.tokens * s.topk, s.N, s.K);
    if (!run_case(s, true, trials)) { std::cout << "Incorrect" << std::endl; if (!keep_going) return 1; failed = true; }
    else std::cout << "Passed" << std::endl;
  }
  return failed ? 1 : 0;
#endif
  return 0;
}
