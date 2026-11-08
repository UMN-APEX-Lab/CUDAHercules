// FP8 E4M3 GEMM with fp32 accumulation/output. No auxiliary epilogue work.
#include "hopper_validation.cuh"
#include <cuda_fp8.h>
#include <cublas_v2.h>
#include <algorithm>
#include <vector>
#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"
#ifdef KH_TEST_SOLUTION
#include "solution.h"
#endif
#define CHECK(call) do { auto err=(call); if(int(err)!=0) { fprintf(stderr,"%s failed: %d\n",#call,int(err)); exit(1); } } while(0)
using namespace cute;
using Element54=cutlass::float_e4m3_t;
using TileShape54=Shape<_128,_128,_128>;
using Cluster=Shape<_1,_2,_1>;
using Epilogue=typename cutlass::epilogue::collective::CollectiveBuilder<
  cutlass::arch::Sm90,cutlass::arch::OpClassTensorOp,TileShape54,Cluster,
  cutlass::epilogue::collective::EpilogueTileAuto,float,float,
  float,cutlass::layout::ColumnMajor,4,float,cutlass::layout::ColumnMajor,4,
  cutlass::epilogue::TmaWarpSpecializedCooperative>::CollectiveOp;
using Mainloop=typename cutlass::gemm::collective::CollectiveBuilder<
  cutlass::arch::Sm90,cutlass::arch::OpClassTensorOp,
  Element54,cutlass::layout::RowMajor,16,Element54,cutlass::layout::ColumnMajor,16,
  float,TileShape54,Cluster,cutlass::gemm::collective::StageCountAutoCarveout<sizeof(typename Epilogue::SharedStorage)>,
  cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;
using Kernel=cutlass::gemm::kernel::GemmUniversal<Shape<int,int,int,int>,Mainloop,Epilogue>;
using Gemm=cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
__global__ void init_fp8(Element54* a,__half* decoded,size_t count,int seed) {
  size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i<count) { Element54 x=Element54(kh_random(i,seed)); a[i]=x; decoded[i]=__float2half(float(x)); }
}
template<class F> float bench(const char* label,F fn) {
  for(int i=0;i<3;++i) fn();
  CHECK(cudaDeviceSynchronize());
  cudaEvent_t a,b; CHECK(cudaEventCreate(&a)); CHECK(cudaEventCreate(&b));
  float sum=0,best=1e30f;
  // Identical cold-L2 policy for the reference and solution.
  void* cache; CHECK(cudaMalloc(&cache,128u<<20));
  for(int i=0;i<20;++i) {
    CHECK(cudaMemset(cache,i,128u<<20));
    CHECK(cudaEventRecord(a)); fn(); CHECK(cudaEventRecord(b)); CHECK(cudaEventSynchronize(b));
    float ms; CHECK(cudaEventElapsedTime(&ms,a,b)); sum+=ms; best=std::min(best,ms);
  }
  printf("%s time: %.6f ms (avg over 20 trials, min: %.6f ms)\n",label,sum/20,best);
  CHECK(cudaFree(cache)); CHECK(cudaEventDestroy(a)); CHECK(cudaEventDestroy(b));
  return best;
}
void run(int M,int N,int K) {
  printf("\n=== fp8_linear: M=%d, N=%d, K=%d ===\n",M,N,K);
  Element54 *A,*B; __half *Ah,*Bh; float *ref,*oracle,*sol;
  size_t na=size_t(M)*K,nb=size_t(N)*K,nc=size_t(M)*N;
  CHECK(cudaMalloc(&A,na)); CHECK(cudaMalloc(&B,nb));
  CHECK(cudaMalloc(&Ah,na*2)); CHECK(cudaMalloc(&Bh,nb*2));
  CHECK(cudaMalloc(&ref,nc*4)); CHECK(cudaMalloc(&oracle,nc*4)); CHECK(cudaMalloc(&sol,nc*4));
  init_fp8<<<(na+255)/256,256>>>(A,Ah,na,42);
  init_fp8<<<(nb+255)/256,256>>>(B,Bh,nb,43);
  auto da=cutlass::make_cute_packed_stride(Kernel::StrideA{},make_shape(M,K,1));
  auto db=cutlass::make_cute_packed_stride(Kernel::StrideB{},make_shape(N,K,1));
  auto dc=cutlass::make_cute_packed_stride(Kernel::StrideC{},make_shape(M,N,1));
  Gemm::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm,{M,N,K,1},
                      {A,da,B,db},{{1.0f,0.0f},ref,dc,ref,dc}};
  Gemm gemm;
  void* workspace; CHECK(cudaMalloc(&workspace,std::max(size_t(1),Gemm::get_workspace_size(args))));
  CHECK(gemm.can_implement(args)); CHECK(gemm.initialize(args,workspace));
  auto reference=[&](){CHECK(gemm.run());};
  reference(); CHECK(cudaDeviceSynchronize());
  cublasHandle_t handle; CHECK(cublasCreate(&handle));
  float alpha=1,beta=0;
  CHECK(cublasGemmEx(handle,CUBLAS_OP_T,CUBLAS_OP_N,M,N,K,&alpha,Ah,CUDA_R_16F,K,
      Bh,CUDA_R_16F,K,&beta,oracle,CUDA_R_32F,M,CUBLAS_COMPUTE_32F_PEDANTIC,CUBLAS_GEMM_DEFAULT));
  CHECK(cudaDeviceSynchronize());
  auto compare=[&](float* actual,float* expected) {
    std::vector<float> x(nc),y(nc);
    CHECK(cudaMemcpy(x.data(),actual,nc*4,cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(y.data(),expected,nc*4,cudaMemcpyDeviceToHost));
    for(size_t i=0;i<nc;++i) if(!std::isfinite(x[i])||!std::isfinite(y[i])||std::abs(x[i]-y[i])>0.06f+0.002f*std::abs(y[i])) {
      fprintf(stderr,"Incorrect at %zu: got %g expected %g\n",i,x[i],y[i]); exit(1);
    }
  };
  compare(ref,oracle);
#ifdef KH_TEST_SOLUTION
  kh_poison(sol,nc*4);
  auto solution=[&](){CHECK(HopperFp8Gemm(M,N,K,1.0f,reinterpret_cast<__nv_fp8_e4m3*>(A),K,
                reinterpret_cast<__nv_fp8_e4m3*>(B),K,0.0f,sol,M));};
  solution(); CHECK(cudaDeviceSynchronize()); compare(sol,oracle);
#endif
  float rt=bench("Ref",reference);
#ifdef KH_TEST_SOLUTION
  float st=bench("Kernel",solution); printf("Speedup: %.6fx\n",rt/st);
#endif
  puts("Passed");
  CHECK(cublasDestroy(handle));
  for(void* p:{(void*)A,(void*)B,(void*)Ah,(void*)Bh,(void*)ref,(void*)oracle,(void*)sol,workspace}) CHECK(cudaFree(p));
}
int main(int argc,char** argv) {
  if(argc==4) run(atoi(argv[1]),atoi(argv[2]),atoi(argv[3]));
  else { run(128,8192,8192); run(8192,8192,8192); run(8192,28672,8192); }
}
