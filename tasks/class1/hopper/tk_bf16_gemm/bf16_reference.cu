// Keep the original TK kernel translation unit unchanged for large GEMMs.
#include "hopper_cublaslt_reference.cuh"
#include <tuple>
void kh_small_bf16_reference(void* a,void* b,void* d,int M,int N,int K) {
  auto key=std::make_tuple(M,N,K);
  static decltype(key) previous{};
  static std::unique_ptr<kh_lt_plan> plan;
  if(!plan||key!=previous) {
    kh_lt_layout layout{M,N,K,K,N,N,CUDA_R_16BF,CUDA_R_16BF,CUDA_R_16BF};
    layout.b_row=true;layout.d_row=true;
    plan=std::make_unique<kh_lt_plan>(layout);previous=key;
  }
  if(plan->run(a,b,d)!=cudaSuccess)exit(1);
}
