// nvcc -O3 -std=c++17 -arch=sm_90a this.cu -lcublasLt -lcublas
#include "../tasks/common/hopper_cublaslt_reference.cuh"
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <iostream>
#include <string>
#include <type_traits>

template<class T> __global__ void fill(T* x,size_t n,int seed) {
    for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=size_t(blockDim.x)*gridDim.x)
        x[i]=T(kh_random(i,seed)*0.1f);
}
template<> __global__ void fill(int8_t* x,size_t n,int seed) {
    for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=size_t(blockDim.x)*gridDim.x)x[i]=int8_t(int(kh_random(i,seed)*7));
}
template<class T> __global__ void decode(const T* x,float* y,size_t n) {
    for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=size_t(blockDim.x)*gridDim.x)y[i]=float(x[i]);
}
__global__ void scale_out(float* x,int M,int N) {
    for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<size_t(M)*N;i+=size_t(blockDim.x)*gridDim.x)
        x[i]*=(0.5f+float((i/N)%17)/32)*(0.5f+float((i%N)%13)/32);
}
template<class T> __global__ void check(const T* x,const float* y,int M,int N,bool row,bool scaled,int* bad) {
    for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<size_t(M)*N;i+=size_t(blockDim.x)*gridDim.x) {
        size_t m=i%M,n=i/M,idx=row?m*N+n:i;
        float z=y[i];if(scaled)z*=(0.5f+float(m%17)/32)*(0.5f+float(n%13)/32);
        float expected=float(T(z)),error=fabsf(float(x[idx])-expected);
        float limit=std::is_integral<T>::value ? 0.f : 0.002f+0.008f*fabsf(expected);
        if(!isfinite(float(x[idx])) || error > limit)atomicExch(bad,1);
    }
}
template<class I,class O>
void run(const std::string& task,int M,int N,int K,cudaDataType_t input,cudaDataType_t output,bool brow=false,bool drow=false,bool scaled=false) {
    I *a,*b;O* d;float *af,*bf,*gold;int* bad;
    cudaMalloc(&a,size_t(M)*K*sizeof(I));cudaMalloc(&b,size_t(N)*K*sizeof(I));cudaMalloc(&d,size_t(M)*N*sizeof(O));
    cudaMalloc(&af,size_t(M)*K*4);cudaMalloc(&bf,size_t(N)*K*4);cudaMalloc(&gold,size_t(M)*N*4);cudaMalloc(&bad,4);
    fill<<<1024,256>>>(a,size_t(M)*K,42);fill<<<1024,256>>>(b,size_t(N)*K,59);
    decode<<<1024,256>>>(a,af,size_t(M)*K);decode<<<1024,256>>>(b,bf,size_t(N)*K);
    cublasHandle_t oracle;cublasCreate(&oracle);float alpha=1,beta=0;
    kh_lt_check(cublasGemmEx(oracle,CUBLAS_OP_T,brow?CUBLAS_OP_T:CUBLAS_OP_N,M,N,K,&alpha,
        af,CUDA_R_32F,K,bf,CUDA_R_32F,brow?N:K,&beta,gold,CUDA_R_32F,M,CUBLAS_COMPUTE_32F_PEDANTIC,CUBLAS_GEMM_DEFAULT));
    kh_lt_layout layout{M,N,K,K,brow?N:K,drow?N:M,input,input,output};layout.b_row=brow;layout.d_row=drow;
    if(input==CUDA_R_8I)layout.compute=CUBLAS_COMPUTE_32I;
    kh_lt_plan plan(layout);
    auto launch=[&](){if(plan.run(a,b,d)!=cudaSuccess)exit(1);if(scaled)scale_out<<<1024,256>>>((float*)d,M,N);};
    launch();cudaDeviceSynchronize();cudaMemset(bad,0,4);check<<<1024,256>>>(d,gold,M,N,drow,scaled,bad);
    int failed=1;cudaMemcpy(&failed,bad,4,cudaMemcpyDeviceToHost);
    cudaEvent_t start,stop;cudaEventCreate(&start);cudaEventCreate(&stop);std::vector<float> times;
    for(int i=0;i<23;++i){kh_cold_l2();cudaEventRecord(start);launch();cudaEventRecord(stop);cudaEventSynchronize(stop);float ms;cudaEventElapsedTime(&ms,start,stop);if(i>=3)times.push_back(ms);}
    std::sort(times.begin(),times.end());
    std::cout<<"{\"task\":\""<<task<<"\",\"M\":"<<M<<",\"N\":"<<N<<",\"K\":"<<K<<",\"passed\":"<<(failed?"false":"true")<<",\"min_ms\":"<<times[0]<<",\"median_ms\":"<<times[10]<<"}"<<std::endl;
    cublasDestroy(oracle);cudaEventDestroy(start);cudaEventDestroy(stop);cudaFree(a);cudaFree(b);cudaFree(d);cudaFree(af);cudaFree(bf);cudaFree(gold);cudaFree(bad);
    if(failed)exit(1);
}
int main(int argc,char** argv) {
    std::string filter=argc>1?argv[1]:"";
    for(auto shape:std::vector<std::array<int,3>>{{128,8192,8192},{8192,8192,8192},{8192,28672,8192}}) {
        auto [m,n,k]=shape;
        if(filter.empty()||filter=="49")run<__half,__half>("49",m,n,k,CUDA_R_16F,CUDA_R_16F);
        if(filter.empty()||filter=="50")run<int8_t,int32_t>("50",m,n,k,CUDA_R_8I,CUDA_R_32I);
        if(filter.empty()||filter=="54")run<__nv_fp8_e4m3,float>("54",m,n,k,CUDA_R_8F_E4M3,CUDA_R_32F);
    }
    if(filter.empty()||filter=="tk_bf16_gemm") {
        run<__nv_bfloat16,__nv_bfloat16>("tk_bf16_gemm",128,5120,5120,CUDA_R_16BF,CUDA_R_16BF,true,true);
        run<__nv_bfloat16,__nv_bfloat16>("tk_bf16_gemm",8192,27648,5120,CUDA_R_16BF,CUDA_R_16BF,true,true);
    }
    for(auto shape:std::vector<std::array<int,3>>{{128,8192,8192},{8192,28672,8192}}) {
        auto [m,n,k]=shape;
        if(filter.empty()||filter=="tk_fp8_gemm")run<__nv_fp8_e4m3,__nv_bfloat16>("tk_fp8_gemm",m,n,k,CUDA_R_8F_E4M3,CUDA_R_16BF,false,true);
        if(filter.empty()||filter=="tk_fp8_gemm_scaled")run<__nv_fp8_e4m3,float>("tk_fp8_gemm_scaled",m,n,k,CUDA_R_8F_E4M3,CUDA_R_32F,false,true,true);
    }
}
