/***************************************************************************************************
 * Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met: (BSD-3-Clause, see the CUTLASS LICENSE.txt)
 **************************************************************************************************/

/*
  Task 84_blackwell_narrow_precision_sparse_gemm: structured-sparse NVFP4 x dense NVFP4 -> BF16
  block-scaled GEMM on Blackwell (sm_100a / B200).

  A (the weights) is stored 2:4-compressed exactly as the Blackwell sparse tensor-core instruction
  (tcgen05.mma.sp ... kind::mxf4nvf4) wants it: values + metadata. For 4-bit data the hardware
  sparsity granule is a *pair* of adjacent K elements: in every group of 8 consecutive K elements
  (4 pairs), exactly 2 pairs are kept (2:4 over pairs, i.e. 4:8 over elements). B (activations) is
  dense. Both carry NVFP4 block scales (one UE4M3 per 16 logical K elements).

  Reference: the CUTLASS Blackwell sparse block-scaled GEMM (example 84, sparse NVFP4 kernel
  schedule) consuming exactly the buffers handed to the solution. The reference was validated
  offline (and can be re-validated with `KH_VALIDATE_REF=1 ./ref`) against an independent device
  kernel that decodes the documented value / metadata / scale layouts by hand.

  Solution under test (KH_TEST_SOLUTION), implemented in solution.h:

    cudaError_t BlackwellNvfp4SparseGemm(int M, int N, int K, float alpha,
                                         uint8_t const *A, uint8_t const *E, uint8_t const *SFA,
                                         uint8_t const *B, uint8_t const *SFB, __nv_bfloat16 *D);

  The exact formats are documented in description.txt.
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
#include <cstdint>
#include "solution.h"   // must stay before kh_verify.h: the solution must not see the helpers
#endif

#include "cute/tensor.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
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

constexpr int kSfVec = 32;   // sparse NVFP4 MMA: one UE4M3 scale per 32 logical K elements (= 16 stored A values)

using ElementA = cutlass::float_e2m1_t;
using ElementAPair = cutlass::nv_float4_t<ElementA>;
using LayoutTagA = cutlass::layout::RowMajor;
using ElementB = cutlass::float_e2m1_t;
using ElementBPair = cutlass::nv_float4_t<ElementB>;
using LayoutTagB = cutlass::layout::ColumnMajor;
constexpr int AlignmentA = 64;   // elements; 2x for the compression along K
constexpr int AlignmentB = 32;
using ElementD = cutlass::bfloat16_t;
using LayoutTagD = cutlass::layout::RowMajor;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
using ElementAccumulator = float;
using ArchTag = cutlass::arch::Sm100;
using OperatorClass = cutlass::arch::OpClassBlockScaledSparseTensorOp;
using ProblemShape = Shape<int, int, int, int>;

template <class MmaTile, class ClusterShape, bool TwoSm>
struct GemmCfg {
  using EpiSched = cute::conditional_t<TwoSm, cutlass::epilogue::TmaWarpSpecialized2SmNvf4,
                                       cutlass::epilogue::TmaWarpSpecialized1SmNvf4>;
  using MainSched = cute::conditional_t<TwoSm, cutlass::gemm::KernelSparseTmaWarpSpecialized2SmNvf4Sm100,
                                        cutlass::gemm::KernelSparseTmaWarpSpecialized1SmNvf4Sm100>;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag, OperatorClass, MmaTile, ClusterShape, cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementAccumulator, void, LayoutTagD, AlignmentD, ElementD, LayoutTagD,
      AlignmentD, EpiSched>::CollectiveOp;
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OperatorClass, ElementAPair, LayoutTagA, AlignmentA, ElementBPair, LayoutTagB,
      AlignmentB, ElementAccumulator, MmaTile, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveoutEpi<CollectiveEpilogue>,
      MainSched>::CollectiveOp;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloop,
                                                          CollectiveEpilogue, void>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

// Reference configurations, selected per shape from a sweep on B200 (see KH_SWEEP; swizzle is the
// CUTLASS tile-scheduler raster swizzle):
//   0: decode (N = 128 tokens): 256x128 MMA tile, 2-SM, cluster 2x1
//   1: prefill: 256x256 MMA tile, 2-SM, cluster 2x1
using Cfg0 = GemmCfg<Shape<_256, _128, _256>, Shape<_2, _1, _1>, true>;
using Cfg1 = GemmCfg<Shape<_256, _256, _256>, Shape<_2, _1, _1>, true>;
#ifdef KH_SWEEP
using Cfg2 = GemmCfg<Shape<_128, _128, _256>, Shape<_1, _1, _1>, false>;
using Cfg3 = GemmCfg<Shape<_256, _192, _256>, Shape<_2, _1, _1>, true>;
using Cfg4 = GemmCfg<Shape<_256, _128, _256>, Shape<_4, _1, _1>, true>;
using Cfg5 = GemmCfg<Shape<_128, _192, _256>, Shape<_1, _1, _1>, false>;
using Cfg6 = GemmCfg<Shape<_128, _256, _256>, Shape<_1, _1, _1>, false>;
using Cfg7 = GemmCfg<Shape<_256, _128, _256>, Shape<_2, _2, _1>, true>;
using Cfg8 = GemmCfg<Shape<_256, _256, _256>, Shape<_4, _1, _1>, true>;
constexpr int kNumCfg = 9;
#endif

// ---------------------------------------------------------------------------------------------
// Operand formats (see description.txt)
//   A  : uint8 [M, K/4]   compressed values: byte j of row m = one KEPT PAIR (low nibble = even k)
//   E  : metadata bytes,  Mp = roundup(M,128):  offset(m, g) = (g/16)*Mp*16 + (m/128)*2048 + (m%128)*16 + g%16
//        for the group g = k/16 (16 logical K elements = 2 chunks of 8); low nibble = first chunk
//        (idx0 | idx1 << 2: the two kept pairs of the chunk, idx0 < idx1)
//   B  : uint8 [N, K/2]   dense FP4 (low nibble = even k)
//   SF*: UE4M3, one scale per 32 logical K elements, swizzled 128x4 blocks as in task 72:
//          ((r/128) * (Kb_pad/4) + c/4) * 512 + (r%32)*16 + ((r/32)%4)*4 + c%4
// ---------------------------------------------------------------------------------------------
__host__ __device__ inline int kh_pad(int x, int m) { return (x + m - 1) / m * m; }
__host__ __device__ inline size_t sf_offset(int r, int c, int kb_pad) {
  return (size_t(r / 128) * (kb_pad / 4) + c / 4) * 512 + (r % 32) * 16 + ((r / 32) % 4) * 4 + (c % 4);
}
__host__ __device__ inline size_t e_offset(int m, int g, int Mp) {
  return size_t(g / 16) * Mp * 16 + size_t(m / 128) * 2048 + (m % 128) * 16 + (g % 16);
}
static size_t sf_bytes(int rows, int K) { return size_t(kh_pad(rows, 128)) * kh_pad(K / kSfVec, 4); }
static size_t e_bytes(int M, int K) { return size_t(kh_pad(M, 128)) * (K / 16); }

__global__ void gen_bytes_kernel(uint8_t *p, size_t n, unsigned seed) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x)
    p[i] = uint8_t(kh_hash64(i, seed) >> 40);
}

// Valid metadata nibbles (idx0 | idx1 << 2 with idx0 < idx1): the 6 ways to keep 2 of 4 pairs.
__global__ void gen_meta_kernel(uint8_t *e, int M, int Mp, int K, unsigned seed) {
  const unsigned char valid[6] = {0x4, 0x8, 0x9, 0xC, 0xD, 0xE};
  int G = K / 16;
  size_t n = size_t(Mp) * G;
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
    int m = int(i / G), g = int(i % G);
    unsigned long long h = kh_hash64(i, seed);
    unsigned lo = valid[(h >> 20) % 6], hi = valid[(h >> 40) % 6];
    e[e_offset(m, g, Mp)] = uint8_t(lo | (hi << 4));
  }
}

// Scales: log-uniform in [2^-3, 2^2) rounded to E4M3 (full 3-bit mantissa), 0 in the padding.
__global__ void gen_sf_kernel(uint8_t *sf, int rows, int kb, int rows_pad, int kb_pad, unsigned seed) {
  size_t n = size_t(rows_pad) * kb_pad;
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
    int r = int(i / kb_pad), c = int(i % kb_pad);
    uint8_t b = 0;
    if (r < rows && c < kb) {
      float u = kh_uniform_pos(i, seed, 0.f, 1.f);
      __nv_fp8_e4m3 q(exp2f(u * 5.f - 3.f));
      b = q.__x;
    }
    sf[sf_offset(r, c, kb_pad)] = b;
  }
}

static void gen_sf(uint8_t *sf, int rows, int K, unsigned seed) {
  int kb = K / kSfVec, kb_pad = kh_pad(kb, 4), rp = kh_pad(rows, 128);
  gen_sf_kernel<<<4096, 256>>>(sf, rows, kb, rp, kb_pad, seed);
  KH_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------------------------
// Independent reference (device): decodes values / metadata / scales by hand
// ---------------------------------------------------------------------------------------------
__device__ inline float e2m1_decode(unsigned code) {
  const float mag[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
  float v = mag[code & 7];
  return (code & 8) ? -v : v;
}
__device__ inline float ue4m3_decode(unsigned b) {
  unsigned e = (b >> 3) & 15, m = b & 7;
  return e == 0 ? float(m) * (1.f / 512.f) : (1.f + float(m) * 0.125f) * exp2f(float(int(e) - 7));
}
__global__ void indep_ref_kernel(int M, int N, int K, float alpha, uint8_t const *A, uint8_t const *E,
                                 uint8_t const *SFA, uint8_t const *B, uint8_t const *SFB,
                                 __nv_bfloat16 *D) {
  int j = blockIdx.y * blockDim.x + threadIdx.x;
  int i = blockIdx.x;
  if (j >= N) return;
  int kb_pad = kh_pad(K / kSfVec, 4), Mp = kh_pad(M, 128);
  float acc = 0.f;
  for (int c = 0; c < K / 8; ++c) {                       // chunk of 8 logical K elements
    unsigned eb = E[e_offset(i, c / 2, Mp)];
    unsigned nib = (c & 1) ? (eb >> 4) & 15 : eb & 15;
    for (int s = 0; s < 2; ++s) {                         // the two kept pairs
      unsigned idx = (nib >> (2 * s)) & 3;
      unsigned ab = A[size_t(i) * (K / 4) + 2 * c + s];
      int k = 8 * c + 2 * idx;                            // logical k of the pair's first element
      for (int h = 0; h < 2; ++h) {
        int kk = k + h;
        float a = e2m1_decode(h ? ab >> 4 : ab);
        unsigned bb = B[size_t(j) * (K / 2) + kk / 2];
        float b = e2m1_decode((kk & 1) ? bb >> 4 : bb);
        float sa = ue4m3_decode(SFA[sf_offset(i, kk / kSfVec, kb_pad)]);
        float sb = ue4m3_decode(SFB[sf_offset(j, kk / kSfVec, kb_pad)]);
        acc = fmaf(a * sa, b * sb, acc);
      }
    }
  }
  D[size_t(i) * N + j] = __float2bfloat16_rn(alpha * acc);
}

/// One prepared CUTLASS GEMM; argument packing / workspace are done in the constructor.
struct RefGemm {
  virtual ~RefGemm() {}
  virtual void run() = 0;
};

template <class Cfg>
struct RefGemmImpl : RefGemm {
  using Gemm = typename Cfg::Gemm;
  using CM = typename Gemm::GemmKernel::CollectiveMainloop;
  Gemm gemm;
  void *workspace = nullptr;
  RefGemmImpl(int M, int N, int K, float alpha, uint8_t const *A, uint8_t const *E, uint8_t const *SFA,
              uint8_t const *B, uint8_t const *SFB, __nv_bfloat16 *D, int swizzle) {
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideD = typename Gemm::GemmKernel::StrideD;
    using SparseConfig = typename CM::SparseConfig;
    using Sm1xxBlkScaledConfig = typename CM::Sm1xxBlkScaledConfig;
    using ArrayElementA = typename CM::ArrayElementA;
    using ArrayElementB = typename CM::ArrayElementB;
    ProblemShape ps = cute::make_tuple(M, N, K, 1);
    auto layout_A = SparseConfig::fill_layoutA(ps);
    auto layout_E = SparseConfig::fill_layoutE(ps);
    auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(ps);
    auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(ps);
    StrideB sb = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
    StrideD sd = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});
    using ES = typename ElementAPair::ScaleFactorType;
    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        {reinterpret_cast<ArrayElementA *>(const_cast<uint8_t *>(A)), layout_A,
         reinterpret_cast<ArrayElementB *>(const_cast<uint8_t *>(B)), sb,
         const_cast<uint8_t *>(E), layout_E,
         reinterpret_cast<ES *>(const_cast<uint8_t *>(SFA)), layout_SFA,
         reinterpret_cast<ES *>(const_cast<uint8_t *>(SFB)), layout_SFB},
        {{alpha, 0.f}, nullptr, sd, reinterpret_cast<ElementD *>(D), sd}};
    args.scheduler.max_swizzle_size = swizzle;
    size_t ws = Gemm::get_workspace_size(args);
    if (ws) KH_CHECK(cudaMalloc(&workspace, ws));
    CUTLASS_CHECK_KH(gemm.can_implement(args));
    CUTLASS_CHECK_KH(gemm.initialize(args, workspace));
  }
  ~RefGemmImpl() override { cudaFree(workspace); }
  void run() override { CUTLASS_CHECK_KH(gemm.run()); }
};

static std::unique_ptr<RefGemm> make_ref(int cfg, int M, int N, int K, float alpha, uint8_t const *A,
                                         uint8_t const *E, uint8_t const *SFA, uint8_t const *B,
                                         uint8_t const *SFB, __nv_bfloat16 *D, int swz) {
#define KH_CASE(i) case i: return std::make_unique<RefGemmImpl<Cfg##i>>(M, N, K, alpha, A, E, SFA, B, SFB, D, swz);
  switch (cfg) {
    KH_CASE(1)
#ifdef KH_SWEEP
    KH_CASE(2) KH_CASE(3) KH_CASE(4) KH_CASE(5) KH_CASE(6) KH_CASE(7) KH_CASE(8)
#endif
    default: return std::make_unique<RefGemmImpl<Cfg0>>(M, N, K, alpha, A, E, SFA, B, SFB, D, swz);
  }
#undef KH_CASE
}

// Tolerances: |x - ref| <= 1e-3 * RMS(ref) + 0.03125 * |ref| per element, and relative Frobenius error
// <= 3e-4 (BF16 output: single BF16 roundings flip between summation orders, <= 1 ulp = 0.0078 |ref|, hence
// the 4x relative term). Measured vs the CUTLASS reference with a plain FP32-FMA solution: rel. Frobenius
// error 2.2e-5 .. 6.1e-5 (0 on the tiny cases), max|err|/rms <= 2.9e-2, so 3e-4 is >= 4.9x above. Deliberately
// wrong solutions (Frobenius error): FP16 accumulation 2.0e-3 .. 5.0e-3, BF16 accumulation 7.7e-3 .. 3.2e-2,
// operands rounded to FP8 2.3e-2, last 32 K skipped 3.3e-2 .. 0.17, FP4 mantissa dropped 0.45, scales ignored
// 0.85, 16-element scale blocks >= 1.1, sparsity metadata ignored (first two pairs assumed kept) 1.15, kept-pair
// slots swapped 1.41, alpha ignored 136, half the outputs / everything unwritten (NaN): all fail on every size.
static KhTol kTol(1.0e-3, 3.0e-4, 0.03125);

struct Shape3 { const char *label; int M, N, K, cfg; int swz = 0; };

static float const kAlpha = 0.00731f;   // product of the two per-tensor (global) FP32 scales

/// Returns false on failure. `timed` = also print Ref / Kernel timings.
static bool run_case(Shape3 const &s, bool timed, int trials) {
  int M = s.M, N = s.N, K = s.K;
  size_t nD = size_t(M) * N;
  uint8_t *A, *E, *B, *SFA, *SFB;
  __nv_bfloat16 *D_ref;
  int Mp = kh_pad(M, 128);
  KH_CHECK(cudaMalloc(&A, size_t(M) * K / 4));
  KH_CHECK(cudaMalloc(&E, e_bytes(M, K)));
  KH_CHECK(cudaMalloc(&B, size_t(N) * K / 2));
  KH_CHECK(cudaMalloc(&SFA, sf_bytes(M, K)));
  KH_CHECK(cudaMalloc(&SFB, sf_bytes(N, K)));
  KH_CHECK(cudaMalloc(&D_ref, nD * sizeof(__nv_bfloat16)));
  gen_bytes_kernel<<<4096, 256>>>(A, size_t(M) * K / 4, 11u);
  gen_meta_kernel<<<4096, 256>>>(E, M, Mp, K, 55u);
  gen_bytes_kernel<<<4096, 256>>>(B, size_t(N) * K / 2, 22u);
  KH_CHECK(cudaGetLastError());
  gen_sf(SFA, M, K, 33u);
  gen_sf(SFB, N, K, 44u);
  kh_poison(D_ref, nD * sizeof(__nv_bfloat16));
  KH_CHECK(cudaDeviceSynchronize());

  auto ref = make_ref(s.cfg, M, N, K, kAlpha, A, E, SFA, B, SFB, D_ref, s.swz);
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
    indep_ref_kernel<<<dim3(M, (N + 127) / 128), 128>>>(M, N, K, kAlpha, A, E, SFA, B, SFB, D_chk);
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
    cudaError_t e = BlackwellNvfp4SparseGemm(M, N, K, kAlpha, A, E, SFA, B, SFB, D_sol);
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
    gen_bytes_kernel<<<4096, 256>>>(B, size_t(N) * K / 2, seed);
    KH_CHECK(cudaGetLastError());
    gen_sf(SFB, N, K, seed + 1u);
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
  cudaFree(A); cudaFree(E); cudaFree(B); cudaFree(SFA); cudaFree(SFB); cudaFree(D_ref);
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

  // Llama-3.1-70B linear layers with 2:4-pruned NVFP4 weights served on one B200 (hidden 8192, fused
  // QKV 10240, FFN 28672 -> fused gate+up 57344, vocab 128256). Weights are the sparse operand A
  // [M = output features, K = input features]; B holds the N tokens: one decode step of 128
  // sequences, or an 8K-token prefill chunk. Output D[M, N] is feature-major.
  std::vector<Shape3> shapes = {
      {"decode_mlp_up",   57344,   128,  8192, 0, 0},
      {"decode_mlp_down",  8192,   128, 28672, 0, 0},
      {"decode_lm_head", 128256,   128,  8192, 0, 0},
      {"prefill_qkv",     10240,  8192,  8192, 1, 0},
      {"prefill_o_proj",   8192,  8192,  8192, 1, 0},
      {"prefill_mlp_up",  57344,  8192,  8192, 1, 0},
      {"prefill_mlp_down", 8192,  8192, 28672, 1, 8},
  };

#ifdef KH_SWEEP
  for (auto s : shapes) {
    for (int cfg = 0; cfg < kNumCfg; ++cfg) {
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
    Shape3 s{"custom", atoi(argv[1]), atoi(argv[2]), atoi(argv[3]), atoi(argv[2]) <= 512 ? 0 : 1, 0};
    std::printf("\n=== custom: M=%d, N=%d, K=%d ===\n", s.M, s.N, s.K);
    if (!run_case(s, true, trials)) { std::cout << "Incorrect" << std::endl; return 1; }
    std::cout << "Passed" << std::endl;
    return 0;
  }

  // Correctness-only cases ("---" header, not "===": the runner pairs each "=== name ===" block with
  // timing lines). M = 200 exercises padded metadata / scale rows (not a multiple of 128).
  {
    Shape3 extra[] = {{"odd_m", 200, 1152, 1280, 1, 0}, {"tiny_n", 1152, 16, 2048, 0, 0}};
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
