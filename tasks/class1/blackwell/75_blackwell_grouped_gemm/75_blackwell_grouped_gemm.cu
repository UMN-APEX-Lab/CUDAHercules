/***************************************************************************************************
 * Copyright (c) 2024 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met: (BSD-3-Clause, see the CUTLASS LICENSE.txt)
 **************************************************************************************************/

/*
  Task 75_blackwell_grouped_gemm: pointer-array grouped GEMM, BF16 in / BF16 out, on Blackwell
  (sm_100a / B200).

  Workload: the expert GEMMs of a Qwen3-30B-A3B MoE layer (128 experts, top-8 routing, hidden 2048,
  expert intermediate 768) with the token-to-expert assignment produced by a skewed router. Group
  sizes are therefore uneven (a few hot experts, many small ones, some empty).

  Reference: the CUTLASS 3.x Blackwell pointer-array grouped GEMM (example 75, all group scheduling
  on the device), with one tile / cluster configuration chosen per scenario from a sweep on B200.

  Solution under test (KH_TEST_SOLUTION), implemented in solution.h:

    cudaError_t BlackwellGroupedGemm(int num_groups,
        int const *Ms, int const *Ns, int const *Ks,                  // device arrays [num_groups]
        __nv_bfloat16 const *const *A_ptrs, int const *ldas,          // device arrays
        __nv_bfloat16 const *const *B_ptrs, int const *ldbs,
        __nv_bfloat16 *const *D_ptrs, int const *ldds);

    group g:  D_g[M,N] = A_g[M,K] * B_g[K,N]      (FP32 accumulation, BF16 output)
      A_g row-major  (element (i,k) at A_g[i*lda + k],  lda = K)
      B_g column-major (element (k,j) at B_g[k + j*ldb], ldb = K)   (an [N,K] weight matrix)
      D_g row-major  (element (i,j) at D_g[i*ldd + j],  ldd = N)
    M may be 0 (empty group), is otherwise arbitrary; N and K are multiples of 64 and may differ
    from group to group.

  Verification: inputs are full-mantissa random BF16; the reference and the solution write to
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
#include <numeric>
#include <string>
#include <vector>

#include "cutlass/cutlass.h"

#ifdef KH_TEST_SOLUTION
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

using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;  // <M,N,K> per group
using ElementA = cutlass::bfloat16_t;
using LayoutA = cutlass::layout::RowMajor;
constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
using ElementB = cutlass::bfloat16_t;
using LayoutB = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
using ElementD = cutlass::bfloat16_t;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
using ElementAccumulator = float;
using ArchTag = cutlass::arch::Sm100;
using OperatorClass = cutlass::arch::OpClassTensorOp;
using ClusterShape = Shape<int32_t, int32_t, _1>;   // runtime (preferred / fallback) cluster

struct Sm1 {
  using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecialized1SmSm100;
  using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm;
};
struct Sm2 {
  using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecialized2SmSm100;
  using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized2Sm;
};

// Swap == false: M-dim of the MMA = tokens (A = activations, B = weights, D row-major).
// Swap == true : "swap-AB": M-dim of the MMA = output features (A = weights, B = activations,
//                D column-major [N, M]), which is the same memory as the row-major [M, N] output;
//                the small / ragged token dimension then maps to the flexible MMA N tile.
template <class MmaTile, class SmCfg, bool Swap_ = false>
struct GroupedCfg {
  static constexpr bool Swap = Swap_;
  using LayoutD = std::conditional_t<Swap_, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag, OperatorClass, MmaTile, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator, ElementAccumulator,
      void, LayoutD *, AlignmentD,   // no C operand: D = A * B
      ElementD, LayoutD *, AlignmentD, typename SmCfg::EpilogueSchedule>::CollectiveOp;
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OperatorClass, ElementA, LayoutA *, AlignmentA, ElementB, LayoutB *, AlignmentB,
      ElementAccumulator, MmaTile, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
          static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      typename SmCfg::KernelSchedule>::CollectiveOp;
  using GemmKernel =
      cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloop, CollectiveEpilogue>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

// Reference configurations (selected per scenario below, see the KH_SWEEP build).
using Cfg0 = GroupedCfg<Shape<_256, _256, _64>, Sm2>;
using Cfg1 = GroupedCfg<Shape<_256, _128, _64>, Sm2>;
using Cfg2 = GroupedCfg<Shape<_128, _256, _64>, Sm1>;
using Cfg3 = GroupedCfg<Shape<_128, _128, _64>, Sm1>;
using Cfg4 = GroupedCfg<Shape<_128, _64, _64>, Sm1>;
// swap-AB variants (MMA M = output features, MMA N = tokens)
using Cfg5 = GroupedCfg<Shape<_256, _128, _64>, Sm2, true>;
using Cfg6 = GroupedCfg<Shape<_256, _64, _64>, Sm2, true>;
using Cfg7 = GroupedCfg<Shape<_128, _128, _64>, Sm1, true>;
using Cfg8 = GroupedCfg<Shape<_128, _64, _64>, Sm1, true>;
using Cfg9 = GroupedCfg<Shape<_128, _32, _64>, Sm1, true>;
constexpr int kNumCfg = 10;
constexpr int kNumCluster = 5;
// preferred cluster x,y / fallback x,y
static const int kClusters[kNumCluster][4] = {{2, 1, 2, 1}, {4, 1, 2, 1}, {4, 2, 2, 1},
                                              {1, 1, 1, 1}, {2, 2, 2, 1}};

/// One problem of the group.
struct Prob {
  int M, N, K;
  __nv_bfloat16 const *A;   // [M,K] row-major
  __nv_bfloat16 const *B;   // [N,K] row-major == [K,N] column-major
  __nv_bfloat16 *D;         // [M,N] row-major
};

/// One prepared CUTLASS grouped GEMM: all device-side argument arrays and the initialization are
/// done in the constructor, so run() is just the kernel launch.
struct RefGrouped {
  virtual ~RefGrouped() {}
  virtual void run() = 0;
};

template <class Cfg>
struct RefGroupedImpl : RefGrouped {
  using Gemm = typename Cfg::Gemm;
  static constexpr bool Swap = Cfg::Swap;
  using StrideA = typename Gemm::GemmKernel::InternalStrideA;
  using StrideB = typename Gemm::GemmKernel::InternalStrideB;
  using StrideD = typename Gemm::GemmKernel::InternalStrideD;
  Gemm gemm;
  void *workspace = nullptr;
  Shape<int, int, int> *d_sizes = nullptr;
  ElementA const **d_A = nullptr;
  ElementB const **d_B = nullptr;
  ElementD **d_D = nullptr;
  StrideA *d_sa = nullptr;
  StrideB *d_sb = nullptr;
  StrideD *d_sd = nullptr;

  template <class T> static T *upload(std::vector<T> const &v) {
    T *p;
    KH_CHECK(cudaMalloc(&p, sizeof(T) * std::max<size_t>(v.size(), 1)));
    KH_CHECK(cudaMemcpy(p, v.data(), sizeof(T) * v.size(), cudaMemcpyHostToDevice));
    return p;
  }

  RefGroupedImpl(std::vector<Prob> const &probs, int cluster) {
    int G = int(probs.size());
    std::vector<Shape<int, int, int>> sizes;
    std::vector<ElementA const *> pa;
    std::vector<ElementB const *> pb;
    std::vector<ElementD *> pd;
    std::vector<StrideA> sa;
    std::vector<StrideB> sb;
    std::vector<StrideD> sd;
    for (auto const &p : probs) {
      int m = Swap ? p.N : p.M, n = Swap ? p.M : p.N;   // MMA-level M, N
      sizes.push_back({m, n, p.K});
      pa.push_back(reinterpret_cast<ElementA const *>(Swap ? p.B : p.A));
      pb.push_back(reinterpret_cast<ElementB const *>(Swap ? p.A : p.B));
      pd.push_back(reinterpret_cast<ElementD *>(p.D));
      sa.push_back(cutlass::make_cute_packed_stride(StrideA{}, {m, p.K, 1}));
      sb.push_back(cutlass::make_cute_packed_stride(StrideB{}, {n, p.K, 1}));
      sd.push_back(cutlass::make_cute_packed_stride(StrideD{}, {m, n, 1}));
    }
    d_sizes = upload(sizes); d_A = upload(pa); d_B = upload(pb); d_D = upload(pd);
    d_sa = upload(sa); d_sb = upload(sb); d_sd = upload(sd);

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
                                    {G, d_sizes, nullptr},
                                    {d_A, d_sa, d_B, d_sb},
                                    {fusion_args, nullptr, nullptr, d_D, d_sd},
                                    hw_info, scheduler};
    size_t ws = Gemm::get_workspace_size(args);
    if (ws) KH_CHECK(cudaMalloc(&workspace, ws));
    CUTLASS_CHECK_KH(gemm.can_implement(args));
    CUTLASS_CHECK_KH(gemm.initialize(args, workspace));
  }
  ~RefGroupedImpl() override {
    cudaFree(workspace); cudaFree(d_sizes); cudaFree(d_A); cudaFree(d_B); cudaFree(d_D);
    cudaFree(d_sa); cudaFree(d_sb); cudaFree(d_sd);
  }
  void run() override { CUTLASS_CHECK_KH(gemm.run()); }
};

static std::unique_ptr<RefGrouped> make_ref(int cfg, int cluster, std::vector<Prob> const &p) {
  switch (cfg) {
    case 0: return std::make_unique<RefGroupedImpl<Cfg0>>(p, cluster);
    case 1: return std::make_unique<RefGroupedImpl<Cfg1>>(p, cluster);
    case 2: return std::make_unique<RefGroupedImpl<Cfg2>>(p, cluster);
    case 3: return std::make_unique<RefGroupedImpl<Cfg3>>(p, cluster);
    case 4: return std::make_unique<RefGroupedImpl<Cfg4>>(p, cluster);
    case 5: return std::make_unique<RefGroupedImpl<Cfg5>>(p, cluster);
    case 6: return std::make_unique<RefGroupedImpl<Cfg6>>(p, cluster);
    case 7: return std::make_unique<RefGroupedImpl<Cfg7>>(p, cluster);
    case 8: return std::make_unique<RefGroupedImpl<Cfg8>>(p, cluster);
    default: return std::make_unique<RefGroupedImpl<Cfg9>>(p, cluster);
  }
}

// ---------------------------------------------------------------------------------------------
// Scenarios: Qwen3-30B-A3B MoE expert GEMMs with skewed top-k routing
// ---------------------------------------------------------------------------------------------

/// One set of expert GEMMs sharing (N, K): `tokens` tokens are routed to `topk` of `E` experts by a
/// router with Zipf-skewed popularity (exponent `skew`, random expert order) plus `cold` experts
/// that receive nothing; per token the top-k of (log-popularity + Gumbel noise) are chosen, i.e.
/// sampling without replacement as a real top-k router does. group g has M_g = tokens routed to
/// expert g.
struct Segment {
  int E, tokens, topk, N, K;
  double skew;
  int cold;
  unsigned seed;
};

struct Scenario {
  char const *label;
  std::vector<Segment> segs;
  int cfg, cluster;   // reference configuration (from the sweep)
};

static std::vector<int> route_counts(Segment const &s) {
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

struct SegBuf {
  int E, N, K, R;
  std::vector<int> cnt, off;
  __nv_bfloat16 *A, *W, *Dref, *Dsol;
  size_t nA, nW, nD;
};

/// Tolerances for the BF16 output. Every element must satisfy |x - ref| <= 1e-2 * RMS(ref) + |ref|/128
/// (the second term is one BF16 ulp) and the relative Frobenius error must be <= 5e-4, per segment.
/// Reference and solution accumulate exact BF16 products in FP32 in different orders and round to
/// BF16. Measured on B200 over all scenarios (max|err|/RMS, rel. Frobenius error):
///   plain sequential FP32 kernel (shipped naive solution):   <= 2.7e-2 (1 ulp of the largest
///        elements, covered by the |ref|/128 term),                   <= 9.7e-5   -> bound 5x above
///   FP16 accumulation:                         5.4e-2 .. 9.9e-2,    4.7e-3 .. 7.0e-3  (fails)
///   inputs rounded to FP8 E4M3 / E5M2:         0.18 .. 0.20 / 0.32 .. 0.40,  3.4e-2 / 6.7e-2 (fail)
///   skipped last 64 of K:                      0.9 .. 1.7,          0.18 .. 0.29  (fails)
/// Structural errors (unwritten rows, wrong expert, dropped group, ragged tail, wrong group size)
/// fail through the NaN / non-finite check or by O(1) errors.
static KhTol kTol(1.0e-2, 5.0e-4, 1.0 / 128);

/// Returns false on failure. `timed` = also print Ref / Kernel timings.
static bool run_case(Scenario const &sc, bool timed, int trials) {
  std::vector<SegBuf> sb;
  std::vector<Prob> probs;
  long long total_rows = 0;
  int total_groups = 0, empty = 0, maxrows = 0, minrows = 1 << 30;
  double flops = 0, bytes = 0;   // bytes: minimum DRAM traffic (weights of non-empty groups + A + D)
  for (size_t si = 0; si < sc.segs.size(); ++si) {
    Segment const &s = sc.segs[si];
    SegBuf b;
    b.E = s.E; b.N = s.N; b.K = s.K;
    b.cnt = route_counts(s);
    b.off.assign(s.E + 1, 0);
    for (int e = 0; e < s.E; ++e) b.off[e + 1] = b.off[e] + b.cnt[e];
    b.R = b.off[s.E];
    b.nA = size_t(b.R) * s.K; b.nW = size_t(s.E) * s.N * s.K; b.nD = size_t(b.R) * s.N;
    KH_CHECK(cudaMalloc(&b.A, b.nA * 2));
    KH_CHECK(cudaMalloc(&b.W, b.nW * 2));
    KH_CHECK(cudaMalloc(&b.Dref, b.nD * 2));
    KH_CHECK(cudaMalloc(&b.Dsol, b.nD * 2));
    kh_fill_uniform(b.A, b.nA, 11u + 100u * unsigned(si));
    kh_fill_uniform(b.W, b.nW, 22u + 100u * unsigned(si));
    kh_poison(b.Dref, b.nD * 2);
    kh_poison(b.Dsol, b.nD * 2);
    for (int e = 0; e < s.E; ++e) {
      probs.push_back({b.cnt[e], s.N, s.K, b.A + size_t(b.off[e]) * s.K,
                       b.W + size_t(e) * s.N * s.K, b.Dref + size_t(b.off[e]) * s.N});
      empty += b.cnt[e] == 0;
      maxrows = std::max(maxrows, b.cnt[e]);
      minrows = std::min(minrows, b.cnt[e]);
      flops += 2.0 * b.cnt[e] * s.N * s.K;
      if (b.cnt[e] > 0) bytes += 2.0 * s.N * s.K + 2.0 * b.cnt[e] * (double(s.K) + s.N);
    }
    total_rows += b.R; total_groups += s.E;
    sb.push_back(std::move(b));
  }
  KH_CHECK(cudaDeviceSynchronize());
  std::printf("  groups = %d (empty = %d), rows per group: min %d, mean %.1f, max %d, total %lld; "
              "%.1f GFLOP\n", total_groups, empty, minrows, double(total_rows) / total_groups,
              maxrows, total_rows, flops * 1e-9);

  // The token matrix (A of every group) changes on every call in deployment: it is regenerated with a
  // new seed outside the timed region before every timed call of the reference and of the solution,
  // so outputs and input-dependent intermediates cannot be cached across calls.
  unsigned seed_counter = 1000;
  auto refresh_input = [&](unsigned seed) {
    for (size_t si = 0; si < sb.size(); ++si)
      kh_fill_uniform(sb[si].A, sb[si].nA, seed + 100u * unsigned(si));
  };
  auto next_input = [&] { refresh_input(++seed_counter); };

  auto ref = make_ref(sc.cfg, sc.cluster, probs);
  ref->run();
  KH_CHECK(cudaDeviceSynchronize());

  auto check = [&](char const *who) {
    bool ok = true;
    for (size_t si = 0; si < sb.size(); ++si) {
      KhResult r = kh_compare(sb[si].Dsol, sb[si].Dref, sb[si].nD, kTol);
      char name[96];
      std::snprintf(name, sizeof(name), "%s [segment %zu]", who, si);
      kh_print_result(name, r, kTol);
      ok = ok && r.pass;
    }
    return ok;
  };

  bool ok = true;
  // The reference must have written every element (it was pre-filled with NaN).
  for (auto &b : sb) {
    KhResult r = kh_compare(b.Dref, b.Dref, b.nD, kTol);
    if (!r.pass) {
      std::cerr << "Reference output incomplete or non-finite (non-finite = " << r.nonfinite << ")"
                << std::endl;
      ok = false;
    }
  }

#ifdef KH_TEST_SOLUTION
  // Device-side argument arrays for the solution (they point into the solution's own D buffers).
  std::vector<int> hM, hN, hK, hlda, hldb, hldd;
  std::vector<__nv_bfloat16 const *> hA, hB;
  std::vector<__nv_bfloat16 *> hD;
  for (auto &b : sb) {
    for (int e = 0; e < b.E; ++e) {
      hM.push_back(b.cnt[e]); hN.push_back(b.N); hK.push_back(b.K);
      hlda.push_back(b.K); hldb.push_back(b.K); hldd.push_back(b.N);
      hA.push_back(b.A + size_t(b.off[e]) * b.K);
      hB.push_back(b.W + size_t(e) * b.N * b.K);
      hD.push_back(b.Dsol + size_t(b.off[e]) * b.N);
    }
  }
  int G = int(hM.size());
  auto up = [](auto const &v) {
    using T = typename std::decay_t<decltype(v)>::value_type;
    T *p;
    KH_CHECK(cudaMalloc(&p, sizeof(T) * v.size()));
    KH_CHECK(cudaMemcpy(p, v.data(), sizeof(T) * v.size(), cudaMemcpyHostToDevice));
    return p;
  };
  int *dM = up(hM), *dN = up(hN), *dK = up(hK), *dlda = up(hlda), *dldb = up(hldb),
      *dldd = up(hldd);
  __nv_bfloat16 const **dA = up(hA), **dB = up(hB);
  __nv_bfloat16 **dD = up(hD);
  auto reset_sol = [&] { for (auto &b : sb) kh_poison(b.Dsol, b.nD * 2); };
  auto call_sol = [&] {
    cudaError_t e = BlackwellGroupedGemm(G, dM, dN, dK, dA, dlda, dB, dldb, dD, dldd);
    if (e != cudaSuccess) {
      std::cerr << "Solution kernel failed: " << cudaGetErrorString(e) << std::endl;
      std::exit(1);
    }
  };
  if (ok) {
    call_sol();
    KH_CHECK(cudaDeviceSynchronize());
    ok = check("solution");
  }
#endif

  if (ok && timed && trials > 0) {
    KhTime tr = kh_time([&] { ref->run(); }, trials,
                        [&] { next_input(); for (auto &b : sb) kh_poison(b.Dref, b.nD * 2); });
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
  // Re-check with yet another fresh input (also when timing is disabled): recompute D_ref for it and
  // make the solution produce the right answer again from a freshly poisoned buffer.
  if (ok) {
    refresh_input(777777u);
    for (auto &b : sb) kh_poison(b.Dref, b.nD * 2);
    ref->run();
    KH_CHECK(cudaDeviceSynchronize());
    reset_sol();
    KH_CHECK(cudaDeviceSynchronize());
    call_sol();
    KH_CHECK(cudaDeviceSynchronize());
    ok = check("solution (re-check, fresh input)");
  }
#endif

#ifdef KH_TEST_SOLUTION
  cudaFree(dM); cudaFree(dN); cudaFree(dK); cudaFree(dlda); cudaFree(dldb); cudaFree(dldd);
  cudaFree(dA); cudaFree(dB); cudaFree(dD);
#endif
  for (auto &b : sb) { cudaFree(b.A); cudaFree(b.W); cudaFree(b.Dref); cudaFree(b.Dsol); }
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

  // Qwen3-30B-A3B: 128 experts, top-8, hidden 2048, expert FFN 768. Fused gate+up projection:
  // [rows, 2048] x [2048 -> 1536]; down projection: [rows, 768] x [768 -> 2048].
  const int E = 128, H = 2048, I2 = 1536, I = 768;
  // {E, tokens, topk, N, K, skew, cold experts, seed}
  Segment pf8k_gu{E, 8192, 8, I2, H, 0.9, 8, 7};    // 8K-token prefill chunk: 65536 routed rows
  Segment pf8k_dn{E, 8192, 8, H, I, 0.9, 8, 7};     // same routing
  Segment pf2k_gu{E, 2048, 8, I2, H, 1.1, 12, 8};
  Segment pf2k_dn{E, 2048, 8, H, I, 1.1, 12, 8};
  Segment dec_gu{E, 128, 8, I2, H, 1.0, 0, 9};      // decode, batch 128: 1024 routed rows
  Segment dec_dn{E, 128, 8, H, I, 1.0, 0, 9};
  // Two-micro-batch overlap: the gate+up GEMMs of micro-batch B and the down GEMMs of micro-batch A
  // run as one grouped launch: 256 groups with two different (N, K) and different routings.
  Segment ovl_gu{E, 2048, 8, I2, H, 1.1, 12, 10};
  Segment ovl_dn{E, 2048, 8, H, I, 1.1, 12, 11};

  // Reference configuration per scenario (cfg, cluster) from the KH_SWEEP build on B200.
  std::vector<Scenario> scenarios = {
      {"prefill8k_gate_up", {pf8k_gu}, 0, 4},
      {"prefill8k_down",    {pf8k_dn}, 0, 4},
      {"prefill2k_gate_up", {pf2k_gu}, 2, 3},
      {"prefill2k_down",    {pf2k_dn}, 2, 3},
      {"overlap_mixed_nk",  {ovl_gu, ovl_dn}, 5, 0},
      {"decode128_gate_up", {dec_gu}, 9, 1},
      {"decode128_down",    {dec_dn}, 9, 1},
  };

  std::string only = (argc >= 2) ? argv[1] : "";   // dev aid: ./test <label substring>

#ifdef KH_SWEEP
  // Development aid: time every reference configuration on every scenario.
  for (auto s : scenarios) {
    if (!only.empty() && std::string(s.label).find(only) == std::string::npos) continue;
    for (int cfg = std::getenv("KH_SWEEP_MIN") ? std::atoi(std::getenv("KH_SWEEP_MIN")) : 0; cfg < kNumCfg; ++cfg)
      for (int cl = 0; cl < kNumCluster; ++cl) {
        bool two_sm = cfg < 2 || cfg == 5 || cfg == 6;
        if (two_sm && kClusters[cl][0] < 2) continue;
        s.cfg = cfg; s.cluster = cl;
        std::printf("\n=== %s cfg=%d cluster=%d ===\n", s.label, cfg, cl);
        run_case(s, true, 10);
      }
  }
  return 0;
#endif

  for (auto const &s : scenarios) {
    if (!only.empty() && std::string(s.label).find(only) == std::string::npos) continue;
    long long rows = 0;
    for (auto const &g : s.segs) rows += (long long)g.tokens * g.topk;
    std::printf("\n=== %s: M=%lld, N=%d, K=%d ===\n", s.label, rows, s.segs[0].N, s.segs[0].K);
    if (!run_case(s, true, trials)) { std::cout << "Incorrect" << std::endl; if (!keep_going) return 1; failed = true; }
    else std::cout << "Passed" << std::endl;
  }
  return failed ? 1 : 0;
#endif
  return 0;
}
