/*
Tencent is pleased to support the open source community by making hpc-ops available.

Copyright (C) 2026 Tencent. All rights reserved.

hpc-ops is licensed under MIT, except for the third-party component(s) listed below, which remain licensed under their respective original terms. hpc-ops does not impose any additional restrictions beyond those specified in the original licenses of these third-party component(s). Users are required to comply with all applicable terms and conditions of these original licenses and to ensure that the use of these third-party component(s) conforms to all relevant laws and regulations.


Terms of the MIT License:
--------------------------------------------------------------------
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the " Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice (including the next paragraph) shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.



*/
// Copyright (C) 2026 Tencent. MIT license reproduced above.
// Adapted from Tencent/hpc-ops at 2a2e26562433a8ba4b504858f1c938eb7612c901.
// ABI changes: mixed E4M3/E5M2, device pointer arrays, FP16 column-major D.
#pragma once
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cutlass/numeric_types.h>
#include <cstdlib>
// Copyright (C) 2026 Tencent.

#ifndef SRC_GROUP_GEMM_SM90_CP_ASYNC_CONFIG_H_
#define SRC_GROUP_GEMM_SM90_CP_ASYNC_CONFIG_H_

#include "cute/tensor.hpp"


namespace kh_hpc_ops_reference {
namespace group_gemm_cp_async {

namespace config {

using namespace cute;  // NOLINT

template <int kTileM>
CUTE_HOST_DEVICE constexpr auto mma_selector() {
  if constexpr (kTileM == 8) {
    return SM90_64x8x32_F32E5M2E4M3_SS_TN<>{};
  } else if constexpr (kTileM == 16) {
    return SM90_64x16x32_F32E5M2E4M3_SS_TN<>{};
  } else if constexpr (kTileM == 32) {
    return SM90_64x32x32_F32E5M2E4M3_SS_TN<>{};
  } else if constexpr (kTileM == 48) {
    return SM90_64x48x32_F32E5M2E4M3_SS_TN<>{};
  } else if constexpr (kTileM == 64) {
    return SM90_64x64x32_F32E5M2E4M3_SS_TN<>{};
  } else {
    static_assert(kTileM == 8 || kTileM == 16 || kTileM == 32 || kTileM == 48 || kTileM == 64,
                  "mma_selector: kTileM must be one of {8, 16, 32, 48, 64}");
    return SM90_64x64x32_F32E5M2E4M3_SS_TN<>{};
  }
}

template <typename Tin_, typename Tout_, int kTileM_, int kTileN_, int kTileK_, int kStage_ = 2>
struct FP8GemmConfig {
  using Tin = Tin_;
  using Tout = Tout_;

  static constexpr int kTileM = kTileM_;
  static constexpr int kTileN = kTileN_;
  static constexpr int kTileK = kTileK_;
  static constexpr int kStage = kStage_;

  using WarpGroupLayout = decltype(make_layout(make_shape(Int<1>{}, Int<1>{}, Int<1>{})));
  using TiledMMA = decltype(make_tiled_mma(mma_selector<kTileM_>(), WarpGroupLayout{}));

  using SmemAtomA = std::conditional_t<kTileK_ >= 128, GMMA::Layout_K_SW128_Atom<Tin>,
                                       GMMA::Layout_K_SW64_Atom<Tin>>;
  using SmemAtomB = std::conditional_t<kTileK_ >= 128, GMMA::Layout_K_SW128_Atom<Tin>,
                                       GMMA::Layout_K_SW64_Atom<Tin>>;

  using SmemLayoutA =
      decltype(tile_to_shape(SmemAtomA{}, make_shape(Int<kTileM>{}, Int<kTileK>{}, Int<kStage>{})));
  using SmemLayoutB =
      decltype(tile_to_shape(SmemAtomB{}, make_shape(Int<kTileN>{}, Int<kTileK>{}, Int<kStage>{})));

  using SmemLayoutCT = decltype(tile_to_shape(GMMA::Layout_MN_SW64_Atom<Tout>{},
                                              make_shape(Int<kTileN>{}, Int<kTileM>{})));

  using CpAsyncAtom = Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, Tin>;
  using G2SCopyA_FOR_TK_128 = decltype(make_tiled_copy(
      CpAsyncAtom{}, make_layout(make_shape(Int<16>{}, Int<8>{}), make_stride(Int<8>{}, Int<1>{})),
      make_layout(make_shape(Int<1>{}, Int<16>{}), make_stride(Int<0>{}, Int<1>{}))));
  using G2SCopyA_FOR_TK_64 = decltype(make_tiled_copy(
      CpAsyncAtom{}, make_layout(make_shape(Int<32>{}, Int<4>{}), make_stride(Int<4>{}, Int<1>{})),
      make_layout(make_shape(Int<1>{}, Int<16>{}), make_stride(Int<0>{}, Int<1>{}))));
  using G2SCopyA = std::conditional_t<kTileK_ >= 128, G2SCopyA_FOR_TK_128, G2SCopyA_FOR_TK_64>;
  using G2SCopyB = G2SCopyA;

  // swapAB R2S: SM90_U16x8_STSM_T for kTileM>=16, SM90_U16x4_STSM_T for kTileM<16
  using R2SCopyAtomC = std::conditional_t<kTileM_ >= 16, Copy_Atom<SM90_U16x8_STSM_T, Tout>,
                                          Copy_Atom<SM90_U16x4_STSM_T, Tout>>;

  using S2GCopyC = decltype(make_tiled_copy(
      Copy_Atom<UniversalCopy<cute::uint128_t>, Tout>{},
      make_layout(make_shape(Int<8>{}, Int<16>{}), make_stride(Int<1>{}, Int<8>{})),
      make_layout(make_shape(Int<8>{}, Int<1>{}), make_stride(Int<1>{}, Int<0>{}))));

  static constexpr int shm_xw = sizeof(Tin) * (cosize(SmemLayoutA{}) + cosize(SmemLayoutB{}));
  static constexpr int shm_y = sizeof(Tout) * cosize(SmemLayoutCT{});
  static constexpr int kShmSize = shm_xw > shm_y ? shm_xw : shm_y;
};

}  // namespace config
}  // namespace group_gemm_cp_async
}  // namespace hpc

#endif  // SRC_GROUP_GEMM_SM90_CP_ASYNC_CONFIG_H_

namespace kh_hpc_ops_reference {
namespace group_gemm_cp_async {
namespace kernels {

// kMinBlocks is a plain compile-time template parameter so __launch_bounds__
// sees a literal integer; the value is chosen by MultistageLaunchPolicy in
// the host dispatcher (see below).
template <typename Config, bool kUseTaskMap, bool kUsePDL, int kMinBlocks>
__global__ void __launch_bounds__(128, kMinBlocks)
    group_gemm_fp8_multistage_kernel(int num_group, const int *Ms, const int *Ns, const int *Ks,
      const __nv_fp8_e4m3 *const *As, const int *ldas,
      const __nv_fp8_e5m2 *const *Bs, const int *ldbs,
      __half *const *Ds, const int *ldds, float alpha) {
  using namespace cute;  // NOLINT

  using Tin = typename Config::Tin;
  using Tout = typename Config::Tout;
  using SmemLayoutA = typename Config::SmemLayoutA;
  using SmemLayoutB = typename Config::SmemLayoutB;
  using SmemLayoutCT = typename Config::SmemLayoutCT;
  using G2SCopyA = typename Config::G2SCopyA;
  using G2SCopyB = typename Config::G2SCopyB;
  using S2GCopyC = typename Config::S2GCopyC;
  using TiledMMA = typename Config::TiledMMA;

  constexpr int kTileM = Config::kTileM;
  constexpr int kTileN = Config::kTileN;
  constexpr int kTileK = Config::kTileK;
  constexpr int kStage = Config::kStage;

  extern __shared__ Tin shm_data[];
  Tin *Ashm = shm_data;
  Tin *Bshm = Ashm + cosize(SmemLayoutA{});
  int *shm_tiles = reinterpret_cast<int *>(Bshm + cosize(SmemLayoutB{}));
  Tout *Cshm = (Tout *)(shm_data);

  int idx = threadIdx.x;
  int iblock = blockIdx.x;

  if constexpr (kUsePDL) {
    cudaGridDependencySynchronize();
  }

  int total_tiles = 0;
  for (int g = 0; g < num_group; ++g)
    total_tiles += ((Ms[g] + kTileM - 1) / kTileM) * ((Ns[g] + kTileN - 1) / kTileN);
  while (iblock < total_tiles) {
    int igroup = 0, rem = iblock;
    for (; igroup < num_group; ++igroup) {
      int count = ((Ms[igroup] + kTileM - 1) / kTileM) * ((Ns[igroup] + kTileN - 1) / kTileN);
      if (rem < count) break;
      rem -= count;
    }
    int m = Ms[igroup], n = Ns[igroup], k = Ks[igroup];
    int num_tile_n = (n + kTileN - 1) / kTileN;
    int itile_m = rem / num_tile_n, itile_n = rem % num_tile_n;
    int ntile = k / kTileK;
    iblock += gridDim.x;
    // Global tensors
    Tensor A = make_tensor(make_gmem_ptr((Tin *)As[igroup]), make_shape(m, k),
                           make_stride(ldas[igroup], Int<1>{}));
    Tensor B = make_tensor(make_gmem_ptr((Tin *)Bs[igroup]), make_shape(n, k),
                           make_stride(ldbs[igroup], Int<1>{}));
    Tensor C = make_tensor(make_gmem_ptr((Tout *)Ds[igroup]),
                           make_shape(n, m), make_stride(ldds[igroup], Int<1>{}));

    Tensor gA = local_tile(A, make_tile(Int<kTileM>{}, Int<kTileK>{}), make_coord(itile_m, _));
    Tensor gB = local_tile(B, make_tile(Int<kTileN>{}, Int<kTileK>{}), make_coord(itile_n, _));
    Tensor gCT =
        local_tile(C, make_tile(Int<kTileN>{}, Int<kTileM>{}), make_coord(itile_n, itile_m));

    // Shared memory tensors
    auto sA = make_tensor(make_smem_ptr(Ashm), SmemLayoutA{});  // (kTileM, kTileK, kStage)
    auto sB = make_tensor(make_smem_ptr(Bshm), SmemLayoutB{});  // (kTileN, kTileK, kStage)
    auto sCT = make_tensor(make_smem_ptr((Tout *)Cshm), SmemLayoutCT{});  // (kTileN, kTileM)

    // G2S
    G2SCopyA g2s_tiled_copy_a;
    auto g2s_thr_copy_a = g2s_tiled_copy_a.get_slice(idx);
    auto tgA_copy = g2s_thr_copy_a.partition_S(gA);
    auto tsA_copy = g2s_thr_copy_a.partition_D(sA);

    G2SCopyB g2s_tiled_copy_b;
    auto g2s_thr_copy_b = g2s_tiled_copy_b.get_slice(idx);
    auto tgB_copy = g2s_thr_copy_b.partition_S(gB);
    auto tsB_copy = g2s_thr_copy_b.partition_D(sB);

    // S2G
    S2GCopyC s2g_tiled_copy_c;
    auto s2g_thr_copy_c = s2g_tiled_copy_c.get_slice(idx);
    auto tCs2g = s2g_thr_copy_c.partition_S(sCT);
    auto tCg2s = s2g_thr_copy_c.partition_D(gCT);

    // swapAB
    TiledMMA tiled_mma;
    auto thr_mma = tiled_mma.get_slice(idx);
    auto tBr = thr_mma.make_fragment_A(thr_mma.partition_A(sB));
    auto tAr = thr_mma.make_fragment_B(thr_mma.partition_B(sA));
    auto tCr = thr_mma.partition_fragment_C(gCT);

    // Predicates
    auto tIA =
        g2s_thr_copy_a.partition_S(make_identity_tensor(make_shape(Int<kTileM>{}, Int<kTileK>{})));
    auto tIC =
        s2g_thr_copy_c.partition_S(make_identity_tensor(make_shape(Int<kTileN>{}, Int<kTileM>{})));
    auto pred_a = make_tensor<bool>(shape(tIA));
    auto pred_c = make_tensor<bool>(shape(tIC));
#pragma unroll
    for (int i = 0; i < size(tIA); ++i) {
      pred_a(i) = get<0>(tIA(i)) < kTileM && itile_m * kTileM + get<0>(tIA(i)) < m;
    }
#pragma unroll
    for (int i = 0; i < size(tIC); ++i) {
      pred_c(i) = get<1>(tIC(i)) < kTileM && itile_m * kTileM + get<1>(tIC(i)) < m;
    }

    // Prologue
    int itile_to_read = 0;
    int ismem_write = 0;
#pragma unroll
    for (int i = 0; i < kStage - 1; i++) {
      if (i < ntile) {
        cute::copy_if(g2s_tiled_copy_a, pred_a, tgA_copy(_, _, _, i), tsA_copy(_, _, _, i));
        cute::copy(g2s_tiled_copy_b, tgB_copy(_, _, _, i), tsB_copy(_, _, _, i));
      }
      cp_async_fence();
      ++itile_to_read;
      ++ismem_write;
    }

    // Main loop
    auto tDr = make_tensor_like(tCr);
    clear(tDr);
    float scale = alpha;
    for (int itile = 0; itile < ntile; itile++) {
      if (itile_to_read < ntile) {
        cute::copy_if(g2s_tiled_copy_a, pred_a, tgA_copy(_, _, _, itile_to_read),
                      tsA_copy(_, _, _, ismem_write));
        cute::copy(g2s_tiled_copy_b, tgB_copy(_, _, _, itile_to_read),
                   tsB_copy(_, _, _, ismem_write));
        ++itile_to_read;
        ismem_write = (ismem_write + 1) % kStage;
      }
      cp_async_fence();

      cp_async_wait<kStage - 1>();
      __syncthreads();

      tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;
      warpgroup_fence_operand(tCr);
      warpgroup_arrive();
#pragma unroll
      for (int ik = 0; ik < size<2>(tBr); ++ik) {
        cute::gemm(tiled_mma, tBr(_, _, ik, itile % kStage), tAr(_, _, ik, itile % kStage), tCr);
        tiled_mma.accumulate_ = GMMA::ScaleOut::One;
      }
      warpgroup_commit_batch();
      warpgroup_wait<0>();
      warpgroup_fence_operand(tCr);
#pragma unroll
      for (int i = 0; i < size(tCr); ++i) {
        tDr(i) = tCr(i) * scale + tDr(i);
      }
    }

    // Task ABI requires FP16 column-major output, without BF16 rounding.
    auto tCg = thr_mma.partition_C(gCT);
    auto coords = thr_mma.partition_C(make_identity_tensor(make_shape(Int<kTileN>{}, Int<kTileM>{})));
#pragma unroll
    for (int i = 0; i < size(tDr); ++i) {
      if (itile_n * kTileN + get<0>(coords(i)) < n &&
          itile_m * kTileM + get<1>(coords(i)) < m)
        tCg(i) = Tout(tDr(i));
    }
    __syncthreads();
  }

  // PDL release
  if constexpr (kUsePDL) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
}


} } }

namespace kh_hpc_ops_reference_adapter {
template<int TM,int TK,int ST,int MIN>
cudaError_t launch(int grid_multiplier, int count, const int* Ms,const int* Ns,const int* Ks,
 const __nv_fp8_e4m3*const* As,const int* ldas,const __nv_fp8_e5m2*const* Bs,const int* ldbs,
 __half*const* Ds,const int* ldds,float alpha) {
 using C=kh_hpc_ops_reference::group_gemm_cp_async::config::FP8GemmConfig<cutlass::float_e4m3_t,cutlass::half_t,TM,64,TK,ST>;
 auto kernel=kh_hpc_ops_reference::group_gemm_cp_async::kernels::group_gemm_fp8_multistage_kernel<C,false,false,MIN>;
 static bool configured=false;
 static int sms=0;
 if(!configured) {
   auto e=cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,C::kShmSize);
   if(e!=cudaSuccess)return e;
   int device=0;cudaGetDevice(&device);cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,device);
   configured=true;
 }
 kernel<<<sms*grid_multiplier,128,C::kShmSize>>>(count,Ms,Ns,Ks,As,ldas,Bs,ldbs,Ds,ldds,alpha);
 return cudaGetLastError();
}
}
inline cudaError_t kh_run_hpc_grouped_gemm(int count,const int* Ms,const int* Ns,const int* Ks,
 float alpha,const __nv_fp8_e4m3** As,const int* ldas,const __nv_fp8_e5m2** Bs,const int* ldbs,
 __half** Ds,const int* ldds) {
 return kh_hpc_ops_reference_adapter::launch<64,128,3,3>(3,count,Ms,Ns,Ks,As,ldas,Bs,ldbs,Ds,ldds,alpha);
}
