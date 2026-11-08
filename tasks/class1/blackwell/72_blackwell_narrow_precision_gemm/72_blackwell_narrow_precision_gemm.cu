/***************************************************************************************************
 * Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met: (BSD-3-Clause, see the CUTLASS LICENSE.txt)
 **************************************************************************************************/

/*
  Task 72_blackwell_narrow_precision_gemm: NVFP4 x NVFP4 -> BF16 block-scaled GEMM on Blackwell
  (sm_100a / B200).

  Reference: the CUTLASS 3.x/4.x Blackwell block-scaled GEMM (example 72), nv_float4_t operands
  (FP4 E2M1 data, one UE4M3 scale per 16 elements along K), FP32 accumulation, BF16 output, with one
  tile / cluster configuration (and, for decode shapes, swapped operands) chosen per problem shape
  from a sweep on B200.  The reference was validated offline (and can be re-validated with
  `KH_VALIDATE_REF=1 ./ref`) against an independent dequantize-and-FMA device kernel that decodes
  the nibbles and E4M3 scales by hand from the documented byte layouts.

  Solution under test (KH_TEST_SOLUTION), implemented in solution.h:

    cudaError_t BlackwellNvfp4Gemm(int M, int N, int K, float alpha,
                                   uint8_t const *A, uint8_t const *SFA,
                                   uint8_t const *B, uint8_t const *SFB,
                                   __nv_bfloat16 *D);

  The exact data / scale-factor layouts are documented in description.txt.

  Verification: inputs are random FP4 codes and random UE4M3 scales over a realistic range; the
  reference and the solution write to different buffers pre-filled with NaN; every output element
  must be finite and within tolerance of the reference. The solution is re-checked after timing.
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

constexpr int kSfVec = 16;   // NVFP4: one UE4M3 scale per 16 consecutive K elements

using ElementAB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using LayoutATag = cutlass::layout::RowMajor;
using LayoutBTag = cutlass::layout::ColumnMajor;
constexpr int AlignmentAB = 32;
using ElementD = cutlass::bfloat16_t;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
using ElementAccumulator = float;
using ArchTag = cutlass::arch::Sm100;
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

// Swap = false: D[M,N] = A[M,K] * B[K,N]  (row-major D).
// Swap = true : the operands exchange roles, D^T[N,M] = B[N,K] * A^T (column-major D^T is exactly
//               the row-major D[M,N] of the task); used for small-M (decode) problems so that the
//               large dimension N fills the MMA's M side.
template <class MmaTile, class ClusterShape, bool Swap>
struct GemmCfg {
  using LayoutDTag = cute::conditional_t<Swap, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag, OperatorClass, MmaTile, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator, ElementAccumulator,
      void, LayoutDTag, AlignmentD, ElementD, LayoutDTag, AlignmentD,
      cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OperatorClass, ElementAB, LayoutATag, AlignmentAB, ElementAB, LayoutBTag,
      AlignmentAB, ElementAccumulator, MmaTile, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
          static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>,
                                                          CollectiveMainloop, CollectiveEpilogue,
                                                          void>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  static constexpr bool kSwap = Swap;
};

// Reference configurations, selected per shape from a sweep on B200 (see KH_SWEEP; swizzle is
// the CUTLASS tile-scheduler raster swizzle):
//   0: prefill (M >= 8K): 256x192 MMA tile, 2-SM, cluster 2x1
//   1: decode, swapped operands, 256x128 tile, cluster 2x1
//   2: decode, swapped operands, 256x128 tile, cluster 4x1
using Cfg0 = GemmCfg<Shape<_256, _192, _256>, Shape<_2, _1, _1>, false>;
using Cfg1 = GemmCfg<Shape<_256, _128, _256>, Shape<_2, _1, _1>, true>;
using Cfg2 = GemmCfg<Shape<_256, _128, _256>, Shape<_4, _1, _1>, true>;
#ifdef KH_SWEEP
using Cfg3 = GemmCfg<Shape<_256, _256, _256>, Shape<_2, _1, _1>, false>;
using Cfg4 = GemmCfg<Shape<_256, _256, _256>, Shape<_2, _4, _1>, false>;
using Cfg5 = GemmCfg<Shape<_256, _128, _256>, Shape<_2, _2, _1>, false>;
using Cfg6 = GemmCfg<Shape<_128, _128, _256>, Shape<_1, _1, _1>, false>;
using Cfg7 = GemmCfg<Shape<_256, _256, _256>, Shape<_4, _1, _1>, false>;
using Cfg8 = GemmCfg<Shape<_128, _128, _256>, Shape<_1, _1, _1>, true>;
using Cfg9 = GemmCfg<Shape<_128, _64, _256>, Shape<_1, _1, _1>, true>;
constexpr int kNumCfg = 10;
#endif

// ---------------------------------------------------------------------------------------------
// Operand layouts (see description.txt)
//   A:   uint8 [M, K/2], row-major; byte j of row i holds k = 2j (low nibble) and 2j+1 (high nibble)
//   B:   uint8 [N, K/2], same layout (K-contiguous == column-major [K, N])
//   SF*: UE4M3 bytes, "swizzled 128x4" blocks:  rows padded to a multiple of 128, K/16 scale
//        columns padded to a multiple of 4; byte offset of (row r, scale column c) =
//          ((r / 128) * (Kb_pad / 4) + c / 4) * 512 + (r % 32) * 16 + ((r / 32) % 4) * 4 + (c % 4)
// ---------------------------------------------------------------------------------------------
__host__ __device__ inline int kh_pad(int x, int m) { return (x + m - 1) / m * m; }
__host__ __device__ inline size_t sf_offset(int r, int c, int kb_pad) {
  return (size_t(r / 128) * (kb_pad / 4) + c / 4) * 512 + (r % 32) * 16 + ((r / 32) % 4) * 4 + (c % 4);
}
static size_t sf_bytes(int rows, int K) { return size_t(kh_pad(rows, 128)) * kh_pad(K / kSfVec, 4); }

__global__ void gen_bytes_kernel(uint8_t *p, size_t n, unsigned seed) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x)
    p[i] = uint8_t(kh_hash64(i, seed) >> 40);   // two independent random FP4 codes per byte
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

static void gen_operand(uint8_t *data, uint8_t *sf, int rows, int K, unsigned seed) {
  gen_bytes_kernel<<<4096, 256>>>(data, size_t(rows) * (K / 2), seed);
  KH_CHECK(cudaGetLastError());
  int kb = K / kSfVec, kb_pad = kh_pad(kb, 4), rp = kh_pad(rows, 128);
  gen_sf_kernel<<<4096, 256>>>(sf, rows, kb, rp, kb_pad, seed + 7);
  KH_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------------------------
// Independent reference (device): decodes everything by hand; used to validate the CUTLASS kernel
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
__global__ void indep_ref_kernel(int M, int N, int K, float alpha, uint8_t const *A, uint8_t const *SFA,
                                 uint8_t const *B, uint8_t const *SFB, __nv_bfloat16 *D) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y;
  if (j >= N) return;
  int kb_pad = kh_pad(K / kSfVec, 4);
  float acc = 0.f;
  for (int k = 0; k < K; ++k) {
    unsigned ba = A[size_t(i) * (K / 2) + k / 2], bb = B[size_t(j) * (K / 2) + k / 2];
    float a = e2m1_decode((k & 1) ? ba >> 4 : ba);
    float b = e2m1_decode((k & 1) ? bb >> 4 : bb);
    float sa = ue4m3_decode(SFA[sf_offset(i, k / kSfVec, kb_pad)]);
    float sb = ue4m3_decode(SFB[sf_offset(j, k / kSfVec, kb_pad)]);
    acc = fmaf(a * sa, b * sb, acc);
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
  using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
  Gemm gemm;
  void *workspace = nullptr;
  RefGemmImpl(int M, int N, int K, float alpha, uint8_t const *A, uint8_t const *SFA,
              uint8_t const *B, uint8_t const *SFB, __nv_bfloat16 *D, int swizzle) {
    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideD = typename Gemm::GemmKernel::StrideD;
    constexpr bool S = Cfg::kSwap;
    int Mg = S ? N : M, Ng = S ? M : N;   // GEMM problem handed to CUTLASS
    uint8_t const *Ag = S ? B : A, *SFAg = S ? SFB : SFA;
    uint8_t const *Bg = S ? A : B, *SFBg = S ? SFA : SFB;
    StrideA sa = cutlass::make_cute_packed_stride(StrideA{}, {Mg, K, 1});
    StrideB sb = cutlass::make_cute_packed_stride(StrideB{}, {Ng, K, 1});
    StrideD sd = cutlass::make_cute_packed_stride(StrideD{}, {Mg, Ng, 1});
    auto lsfa = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(Mg, Ng, K, 1));
    auto lsfb = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(Mg, Ng, K, 1));
    using EA = typename ElementAB::DataType;
    using ES = typename ElementAB::ScaleFactorType;
    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {Mg, Ng, K, 1},
        {reinterpret_cast<EA const *>(Ag), sa, reinterpret_cast<EA const *>(Bg), sb,
         reinterpret_cast<ES const *>(SFAg), lsfa, reinterpret_cast<ES const *>(SFBg), lsfb},
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
                                         uint8_t const *SFA, uint8_t const *B, uint8_t const *SFB,
                                         __nv_bfloat16 *D, int swz) {
#define KH_CASE(i) case i: return std::make_unique<RefGemmImpl<Cfg##i>>(M, N, K, alpha, A, SFA, B, SFB, D, swz);
  switch (cfg) {
    KH_CASE(0) KH_CASE(1)
#ifdef KH_SWEEP
    KH_CASE(3) KH_CASE(4) KH_CASE(5) KH_CASE(6) KH_CASE(7) KH_CASE(8) KH_CASE(9)
#endif
    default: return std::make_unique<RefGemmImpl<Cfg2>>(M, N, K, alpha, A, SFA, B, SFB, D, swz);
  }
#undef KH_CASE
}

// Tolerances: |x - ref| <= 1e-3 * RMS(ref) + 0.03125 * |ref| per element, and relative Frobenius error
// <= 3e-4. Reference and solution round an FP32 sum of exact products to BF16; different summation orders
// (tensor core vs. sequential FP32) flip single BF16 roundings (<= 1 ulp = 0.0078 |ref|, hence the 4x
// relative term). Measured vs the CUTLASS reference with a plain FP32-FMA solution: rel. Frobenius error
// 3.5e-5 .. 9.1e-5 (0 on the tiny cases), max|err|/rms <= 2.0e-2 (a BF16 ulp of the largest outputs); the
// 3e-4 bound is >= 3.3x above that. Deliberately wrong solutions (Frobenius error): FP16 accumulation
// 2.0e-3 .. 5.0e-3, BF16 accumulation 8e-3 .. 3.4e-2, dequantized operands rounded to FP8 3.2e-2, last 32 K
// skipped 3.4e-2 .. 0.15, power-of-two (UE8M0-like) scales 0.30, FP4 mantissa dropped 0.45, scales ignored
// 0.85, 32-element scale blocks 0.84, linear scale layout 1.0 .. 1.2, A nibbles swapped 1.4, alpha ignored
// 136, half the outputs / everything unwritten (NaN): all fail on every size.
static KhTol kTol(1.0e-3, 3.0e-4, 0.03125);

struct Shape3 { const char *label; int M, N, K, cfg; int swz = 0; };

static float const kAlpha = 0.00731f;   // product of the two per-tensor (global) FP32 scales

/// Returns false on failure. `timed` = also print Ref / Kernel timings.
static bool run_case(Shape3 const &s, bool timed, int trials) {
  int M = s.M, N = s.N, K = s.K;
  size_t nD = size_t(M) * N;
  uint8_t *A, *B, *SFA, *SFB;
  __nv_bfloat16 *D_ref;
  KH_CHECK(cudaMalloc(&A, size_t(M) * K / 2));
  KH_CHECK(cudaMalloc(&B, size_t(N) * K / 2));
  KH_CHECK(cudaMalloc(&SFA, sf_bytes(M, K)));
  KH_CHECK(cudaMalloc(&SFB, sf_bytes(N, K)));
  KH_CHECK(cudaMalloc(&D_ref, nD * sizeof(__nv_bfloat16)));
  gen_operand(A, SFA, M, K, 11u);
  gen_operand(B, SFB, N, K, 22u);
  kh_poison(D_ref, nD * sizeof(__nv_bfloat16));
  KH_CHECK(cudaDeviceSynchronize());

  auto ref = make_ref(s.cfg, M, N, K, kAlpha, A, SFA, B, SFB, D_ref, s.swz);
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
    indep_ref_kernel<<<dim3((N + 127) / 128, M), 128>>>(M, N, K, kAlpha, A, SFA, B, SFB, D_chk);
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
    cudaError_t e = BlackwellNvfp4Gemm(M, N, K, kAlpha, A, SFA, B, SFB, D_sol);
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
    gen_operand(A, SFA, M, K, seed);
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

  // Llama-3.1-70B linear layers quantized to NVFP4 and served on one B200 (hidden 8192, fused QKV
  // 10240, FFN 28672 -> fused gate+up 57344, vocab 128256). M = tokens: one decode step of 128
  // sequences, or an 8K-token prefill chunk. (The decode fused-QKV layer, 42 MB of weights, is
  // ~5 us and too short to time reliably, so it is not listed.)
  std::vector<Shape3> shapes = {
      {"decode_mlp_up",    128,  57344,  8192, 2, 0},
      {"decode_mlp_down",  128,   8192, 28672, 1, 0},
      {"decode_lm_head",   128, 128256,  8192, 1, 0},
      {"prefill_qkv",     8192,  10240,  8192, 0, 0},
      {"prefill_o_proj",  8192,   8192,  8192, 0, 2},
      {"prefill_mlp_up",  8192,  57344,  8192, 0, 0},
      {"prefill_mlp_down",8192,   8192, 28672, 0, 8},
  };

#ifdef KH_SWEEP
  // Development aid: time every reference configuration on every shape.
  for (auto s : shapes) {
    for (int cfg = 0; cfg < kNumCfg; ++cfg) {
      bool swapped_cfg = cfg == 1 || cfg == 2 || cfg == 8 || cfg == 9;
      if (swapped_cfg != (s.M <= 512)) continue;
      for (int swz : {0, 2, 4, 8}) {
        s.cfg = cfg; s.swz = swz;
        std::printf("\n=== %s cfg=%d swz=%d: M=%d, N=%d, K=%d ===\n", s.label, cfg, swz, s.M, s.N, s.K);
        run_case(s, true, 10);
      }
    }
  }
  return 0;
#endif

  if (argc >= 4) {   // ./test M N K : single custom shape
    Shape3 s{"custom", atoi(argv[1]), atoi(argv[2]), atoi(argv[3]), atoi(argv[1]) <= 512 ? 1 : 0};
    std::printf("\n=== custom: M=%d, N=%d, K=%d ===\n", s.M, s.N, s.K);
    if (!run_case(s, true, trials)) { std::cout << "Incorrect" << std::endl; return 1; }
    std::cout << "Passed" << std::endl;
    return 0;
  }

  // Correctness-only cases ("---" header, not "===": the runner pairs each "=== name ===" block
  // with timing lines). M = 200 exercises padded scale-factor rows (200 is not a multiple of 128)
  // and a K that is not a multiple of the 256-wide MMA tile; M = 16 is a tiny decode batch.
  {
    Shape3 extra[] = {{"odd_m", 200, 1152, 1344, 0}, {"tiny_m", 16, 2048, 4096, 1}};
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
