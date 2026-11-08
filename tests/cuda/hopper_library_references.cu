// Standalone GPU controls for the new production-library reference machinery.
#include "../../tasks/common/hopper_cublaslt_reference.cuh"
#include "../../tasks/common/hopper_tf32_validation.cuh"
#include <cuda_bf16.h>
#include <cassert>
#include <vector>

__global__ void input(float* a,float* b,int M,int N,int K) {
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<M*K;i+=blockDim.x*gridDim.x)
        a[i]=ldexpf(kh_random(i,1),int(kh_hash(i%K)%41)-20);
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<N*K;i+=blockDim.x*gridDim.x)
        b[i]=ldexpf(kh_random(i,2),20-int(kh_hash(i%K)%41));
}
__global__ void bf16_round(float* a,int count) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count)a[i]=float(__float2bfloat16(a[i]));
}
int main() {
    constexpr int M=32,N=64,K=256;float *a,*b,*d,*initial,*aq,*bq;
    cudaMalloc(&a,M*K*4);cudaMalloc(&b,N*K*4);cudaMalloc(&d,M*N*4);cudaMalloc(&initial,M*N*4);
    cudaMalloc(&aq,M*K*4);cudaMalloc(&bq,N*K*4);
    input<<<32,256>>>(a,b,M,N,K);
    std::vector<float> init(M*N,0.375f);cudaMemcpy(initial,init.data(),M*N*4,cudaMemcpyHostToDevice);
    kh_tf32_contract contract(M,N,K,1,a,b,0,initial);
    cublasHandle_t handle;kh_lt_check(cublasCreate(&handle));float alpha=1,beta=0;
    auto gemm=[&](float* x,float* y){kh_lt_check(cublasGemmEx(handle,CUBLAS_OP_T,CUBLAS_OP_N,M,N,K,&alpha,x,CUDA_R_32F,K,y,CUDA_R_32F,K,&beta,d,CUDA_R_32F,M,CUBLAS_COMPUTE_32F_PEDANTIC,CUBLAS_GEMM_DEFAULT));};
    gemm(a,b);assert(contract.accepts(d));
    for(bool nearest:{false,true}) {
        kh_tf32_round<<<32,256>>>(a,aq,M*K,nearest);kh_tf32_round<<<64,256>>>(b,bq,N*K,nearest);
        gemm(aq,bq);assert(contract.accepts(d));
    }
    cudaMemset(d,0,M*N*4);assert(!contract.accepts(d));
    kh_poison(d,M*N*4);assert(!contract.accepts(d));
    cudaMemcpy(aq,a,M*K*4,cudaMemcpyDeviceToDevice);cudaMemcpy(bq,b,N*K*4,cudaMemcpyDeviceToDevice);
    bf16_round<<<32,256>>>(aq,M*K);bf16_round<<<64,256>>>(bq,N*K);gemm(aq,bq);assert(!contract.accepts(d));
    // Selection must preserve the beta input and the first invocation must
    // apply beta exactly once, even though many algorithms are benchmarked.
    cudaMemcpy(d,initial,M*N*4,cudaMemcpyDeviceToDevice);
    kh_lt_layout layout{M,N,K,K,K,M,CUDA_R_32F,CUDA_R_32F,CUDA_R_32F};
    layout.compute=CUBLAS_COMPUTE_32F_FAST_TF32;
    kh_lt_plan plan(layout);alpha=0;beta=2;plan.run(a,b,d,alpha,beta);
    std::vector<float> actual(M*N);cudaMemcpy(actual.data(),d,M*N*4,cudaMemcpyDeviceToHost);
    for(auto value:actual)assert(value==0.75f);
    cublasDestroy(handle);for(auto p:{a,b,d,initial,aq,bq})cudaFree(p);
    puts("FP32/RN/RZ accepted; zero/NaN/BF16 rejected; beta input preserved: Passed");
}
