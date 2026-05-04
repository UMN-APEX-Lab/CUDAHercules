/***************************************************************************************************
 * Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/

/*! \file
    \brief Hopper GEMM with Collective Builder -- half precision.
    A row-major, B col-major, D col-major. Half I/O, float accumulation.
*/

#include <iostream>
#include <sstream>
#include <vector>
#include <cmath>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"

#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/reference/device/tensor_fill.h"

#include "helper.h"

#ifdef KH_TEST_SOLUTION
#include "solution.h"
#endif

using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

using LayoutA49 = cutlass::layout::RowMajor;
using LayoutB49 = cutlass::layout::ColumnMajor;
using LayoutD49 = cutlass::layout::ColumnMajor;

using ElementA49 = cutlass::half_t;
using ElementB49 = cutlass::half_t;
using ElementD49 = cutlass::half_t;
using ElementAcc49 = float;
using ElementComp49 = float;

static constexpr int Align49 = 16 / sizeof(ElementA49);

using DefaultOp49 = cutlass::epilogue::fusion::LinearCombination<ElementD49, ElementComp49, ElementD49, float>;

using Epi49 = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    Shape<_128,_128,_64>, Shape<_1,_1,_1>,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAcc49, ElementComp49,
    ElementD49, LayoutD49, Align49,
    ElementD49, LayoutD49, Align49,
    cutlass::epilogue::collective::EpilogueScheduleAuto,
    DefaultOp49
  >::CollectiveOp;

using Main49 = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElementA49, LayoutA49, Align49,
    ElementB49, LayoutB49, Align49,
    ElementAcc49,
    Shape<_128,_128,_64>, Shape<_2,_1,_1>,
    cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(sizeof(typename Epi49::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto
  >::CollectiveOp;

using GK49 = cutlass::gemm::kernel::GemmUniversal<Shape<int,int,int,int>, Main49, Epi49, cutlass::gemm::PersistentScheduler>;
using G49 = cutlass::gemm::device::GemmUniversalAdapter<GK49>;

using SA49 = typename G49::GemmKernel::StrideA;
using SB49 = typename G49::GemmKernel::StrideB;
using SC49 = typename G49::GemmKernel::StrideC;
using SD49 = typename G49::GemmKernel::StrideD;

cudaError_t CutlassGemm49(int M, int N, int K, float alpha,
  cutlass::half_t const *A, cutlass::half_t const *B, float beta, cutlass::half_t *D, int ldd) {
  int lda = K, ldb = K;
  auto sA = cutlass::make_cute_packed_stride(SA49{}, {M, K, 1});
  auto sB = cutlass::make_cute_packed_stride(SB49{}, {N, K, 1});
  auto sC = cutlass::make_cute_packed_stride(SC49{}, {M, N, 1});
  auto sD = cutlass::make_cute_packed_stride(SD49{}, {M, N, 1});

  cutlass::KernelHardwareInfo hw; hw.device_id = 0;
  hw.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);

  typename G49::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm, {M,N,K,1},
    {A, sA, B, sB}, {{}, nullptr, sC, D, sD}, hw};
  args.epilogue.thread.alpha = alpha;
  args.epilogue.thread.beta = beta;

  G49 gemm;
  size_t ws = G49::get_workspace_size(args);
  cutlass::device_memory::allocation<uint8_t> workspace(ws);
  if (gemm.can_implement(args) != cutlass::Status::kSuccess) return cudaErrorUnknown;
  if (gemm.initialize(args, workspace.get()) != cutlass::Status::kSuccess) return cudaErrorUnknown;
  if (gemm.run() != cutlass::Status::kSuccess) return cudaErrorUnknown;
  return cudaSuccess;
}
#endif

///////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void InitHalfMatrix_kernel(__half *matrix, int size, int seed) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < size) {
    float value = float(((idx + seed) * 16807 % 16) - 8) * 0.01f;
    matrix[idx] = __float2half(value);
  }
}

cudaError_t AllocHalf(__half **m, int size, int seed) {
  cudaError_t r = cudaMalloc(reinterpret_cast<void**>(m), sizeof(__half)*size);
  if (r != cudaSuccess) return r;
  InitHalfMatrix_kernel<<<(size+255)/256, 256>>>(*m, size, seed);
  return cudaGetLastError();
}

__global__ void NaiveHGemm_kernel(int M, int N, int K, float alpha,
  __half const *A, int lda, __half const *B, int ldb, float beta, __half *D, int ldd) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  if (i < M && j < N) {
    float acc = 0.0f;
    for (int k = 0; k < K; ++k)
      acc += __half2float(A[i*lda+k]) * __half2float(B[k+j*ldb]);
    D[i+j*ldd] = __float2half(alpha*acc + beta*__half2float(D[i+j*ldd]));
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

cudaError_t TestCorrectness(int M, int N, int K, float alpha, float beta) {
  int lda=K, ldb=K, ldd=M;
  size_t szD = sizeof(__half)*M*N;
  __half *A, *B, *D_ref, *D_cut;
  AllocHalf(&A, M*K, 0); AllocHalf(&B, K*N, 17);
  cudaMalloc(&D_ref, szD); cudaMemset(D_ref, 0, szD);
  cudaMalloc(&D_cut, szD); cudaMemset(D_cut, 0, szD);

  {dim3 bl(16,16), gr((M+15)/16,(N+15)/16);
   NaiveHGemm_kernel<<<gr,bl>>>(M,N,K,alpha,A,lda,B,ldb,beta,D_ref,ldd);}
  cudaDeviceSynchronize();

  bool cutlass_ok = false;
#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
  if (CutlassGemm49(M,N,K,alpha,(cutlass::half_t const*)A,(cutlass::half_t const*)B,beta,(cutlass::half_t*)D_cut,ldd)==cudaSuccess) {
    cudaDeviceSynchronize(); cutlass_ok = true;
  } else { cudaGetLastError(); }
#endif

#ifdef KH_TEST_SOLUTION
  __half *D_sol; cudaMalloc(&D_sol, szD); cudaMemset(D_sol, 0, szD);
  cudaError_t sr = HopperCollectiveGemm(M,N,K,alpha,A,lda,B,ldb,beta,D_sol,ldd);
  if (sr != cudaSuccess) {
    std::cerr<<"Solution failed: "<<cudaGetErrorString(sr)<<std::endl;
    cudaFree(D_sol); cudaFree(D_ref); cudaFree(D_cut); cudaFree(B); cudaFree(A);
    return sr;
  }
  cudaDeviceSynchronize();

  __half *D_chk = cutlass_ok ? D_cut : D_ref;
  std::vector<__half> hs(M*N), hr(M*N);
  cudaMemcpy(hs.data(), D_sol, szD, cudaMemcpyDeviceToHost);
  cudaMemcpy(hr.data(), D_chk, szD, cudaMemcpyDeviceToHost);
  float md=0; for (int i=0;i<M*N;++i) md=fmaxf(md,fabsf(__half2float(hs[i])-__half2float(hr[i])));
  if (md > 1.0f) {
    std::cerr<<"Solution incorrect. Max diff: "<<md<<std::endl;
    cudaFree(D_sol); cudaFree(D_ref); cudaFree(D_cut); cudaFree(B); cudaFree(A);
    return cudaErrorUnknown;
  }
  cudaFree(D_sol);
#endif
  cudaFree(D_ref); cudaFree(D_cut); cudaFree(B); cudaFree(A);
  return cudaSuccess;
}

void Profile(int M, int N, int K, float alpha, float beta, int iters) {
  int ldd=M; __half *A, *B, *D;
  AllocHalf(&A, M*K, 0); AllocHalf(&B, K*N, 17);
  cudaMalloc(&D, sizeof(__half)*M*N); cudaMemset(D, 0, sizeof(__half)*M*N);
  cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
  float ref_min=1e30f; bool rok=false;
#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
  if (CutlassGemm49(M,N,K,alpha,(cutlass::half_t const*)A,(cutlass::half_t const*)B,beta,(cutlass::half_t*)D,ldd)==cudaSuccess) {
    cudaDeviceSynchronize();
    float rt=0;
    for (int i=0;i<iters;++i) {
      cudaEventRecord(t0);
      CutlassGemm49(M,N,K,alpha,(cutlass::half_t const*)A,(cutlass::half_t const*)B,beta,(cutlass::half_t*)D,ldd);
      cudaEventRecord(t1); cudaEventSynchronize(t1);
      float ms; cudaEventElapsedTime(&ms,t0,t1); rt+=ms; if(ms<ref_min)ref_min=ms;
    }
    fprintf(stdout,"Ref time: %.4f ms (avg over %d trials, min: %.4f ms)\n",rt/iters,iters,ref_min);
    rok=true;
  } else { cudaGetLastError(); }
#endif
#ifdef KH_TEST_SOLUTION
  // Save reference output for correctness check
  __half *D_ref_saved = nullptr;
  if (rok) {
    cudaMalloc(&D_ref_saved, sizeof(__half)*M*N);
    cudaMemcpy(D_ref_saved, D, sizeof(__half)*M*N, cudaMemcpyDeviceToDevice);
  }

  cudaMemset(D,0,sizeof(__half)*M*N);
  HopperCollectiveGemm(M,N,K,alpha,A,K,B,K,beta,D,ldd);
  cudaDeviceSynchronize();

  if (rok && D_ref_saved) {
    std::vector<__half> h_sol(M*N), h_ref(M*N);
    cudaMemcpy(h_sol.data(), D, sizeof(__half)*M*N, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ref.data(), D_ref_saved, sizeof(__half)*M*N, cudaMemcpyDeviceToHost);
    cudaFree(D_ref_saved);
    float md=0;
    for(int i=0;i<M*N;++i) md=fmaxf(md,fabsf(__half2float(h_sol[i])-__half2float(h_ref[i])));
    if (md > 1.0f) {
      fprintf(stderr,"Solution incorrect in Profile: max_diff=%.6f\n",md);
      std::cout<<"Incorrect"<<std::endl;
      cudaEventDestroy(t0); cudaEventDestroy(t1); cudaFree(D); cudaFree(B); cudaFree(A);
      exit(-1);
    }
  }

  for(int i=0;i<3;i++) HopperCollectiveGemm(M,N,K,alpha,A,K,B,K,beta,D,ldd);
  cudaDeviceSynchronize();
  float st=0,sm=1e30f;
  for(int i=0;i<iters;++i){
    cudaEventRecord(t0); HopperCollectiveGemm(M,N,K,alpha,A,K,B,K,beta,D,ldd);
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms; cudaEventElapsedTime(&ms,t0,t1); st+=ms; if(ms<sm)sm=ms;
  }
  fprintf(stdout,"Kernel time: %.4f ms (avg over %d trials, min: %.4f ms)\n",st/iters,iters,sm);
  if(rok) fprintf(stdout,"Speedup: %.4fx (ref_min / kernel_min)\n",ref_min/sm);
#endif
  cudaEventDestroy(t0); cudaEventDestroy(t1); cudaFree(D); cudaFree(B); cudaFree(A);
}

int main(int argc, const char *arg[]) {
  struct TC { const char* l; int m,n,k; };
  std::vector<TC> cfgs;
  float alpha=1.0f, beta=0.0f; int iters=20;
  if (argc>=4) {
    int M=128,N=128,K=128;
    std::stringstream(arg[1])>>M; std::stringstream(arg[2])>>N; std::stringstream(arg[3])>>K;
    if(argc>4)std::stringstream(arg[4])>>alpha; if(argc>5)std::stringstream(arg[5])>>beta;
    if(argc>6)std::stringstream(arg[6])>>iters;
    cfgs.push_back({"custom",M,N,K});
  } else {
    cfgs={{"small",1024,1024,1024},{"medium",4096,4096,4096},{"large",8192,8192,8192}};
  }
  for (auto& c : cfgs) {
    fprintf(stdout,"\n=== %s: M=%d, N=%d, K=%d ===\n",c.l,c.m,c.n,c.k);
    if (TestCorrectness(c.m,c.n,c.k,alpha,beta) != cudaSuccess) {
      std::cout<<"Incorrect"<<std::endl;
      return -1;
    }
    std::cout<<"Passed"<<std::endl;
    if(iters>0) Profile(c.m,c.n,c.k,alpha,beta,iters);
  }
  return 0;
}
