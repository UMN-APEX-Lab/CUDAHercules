#pragma once
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>

// fmax silently discards NaNs: map every nonfinite error to +infinity first.
template<class T> __host__ __device__ inline double kh_abs(T x) {
    double value = static_cast<double>(x);
    return isfinite(value) ? fabs(value) : INFINITY;
}
__host__ __device__ inline uint32_t kh_hash(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15;
    x *= 0x846ca68bU; x ^= x >> 16; return x;
}
__host__ __device__ inline float kh_random(uint64_t index, uint32_t seed) {
    uint32_t bits = kh_hash(uint32_t(index) ^ kh_hash(seed + uint32_t(index >> 32)));
    return float(bits >> 8) * (2.0f / 16777216.0f) - 1.0f;
}
// Validation-only poison: called after saving the reference and before solution.
// 0xff is a NaN in fp32/fp16/bf16/e4m3 and a sentinel for integer outputs.
inline void kh_poison(void* ptr, size_t bytes) {
    auto err = cudaMemset(ptr, 0xff, bytes);
    if (err != cudaSuccess) { fprintf(stderr,"poison: %s\n",cudaGetErrorString(err)); exit(1); }
}

// Independent sampled fp64 dot products keep validation linear in K, even for
// large prefill shapes. The full solution output is still compared separately.
template<class A,class B,class C,class D,bool ColumnA=false,bool PermutedB=false>
__global__ void kh_sample_gemm_kernel(int M,int N,int K,float alpha,const A* a,
    const B* b,float beta,const C* c,const D* d,float atol,float rtol,int* failed) {
    int sample=threadIdx.x;
    size_t count=size_t(M)*N;
    size_t index=sample<4 ? (sample==0?0:sample==1?count-1:sample==2?M-1:count-M) : kh_hash(sample*7933u)%count;
    int row=index%M,col=index/M;
    double dot=0;
    for(int k=0;k<K;++k) {
      size_t bi=PermutedB ? (k%128)+size_t(128)*(col+size_t(N)*(k/128)) : k+size_t(col)*K;
      dot+=double(float(a[ColumnA ? row+size_t(k)*M : size_t(row)*K+k]))*double(float(b[bi]));
    }
    double expected=alpha*dot+(beta==0?0:beta*double(float(c[index])));
    // Account for the contract's output rounding.
    expected=double(float(D(expected)));
    double actual=double(float(d[index]));
    if(!isfinite(actual)||fabs(actual-expected)>atol+rtol*fabs(expected)) atomicExch(failed,1);
}
template<class A,class B,class C,class D,bool ColumnA=false,bool PermutedB=false>
inline void kh_sample_gemm(int M,int N,int K,float alpha,const A* a,const B* b,
    float beta,const C* c,const D* d,float atol=0.01f,float rtol=0.002f) {
    int* failed;cudaMalloc(&failed,sizeof(int));cudaMemset(failed,0,sizeof(int));
    kh_sample_gemm_kernel<A,B,C,D,ColumnA,PermutedB><<<1,128>>>(M,N,K,alpha,a,b,beta,c,d,atol,rtol,failed);
    int bad=1;auto err=cudaMemcpy(&bad,failed,sizeof(int),cudaMemcpyDeviceToHost);cudaFree(failed);
    if(err!=cudaSuccess||bad) {fputs("Independent sampled GEMM oracle failed\n",stderr);exit(1);}
}

inline void kh_cold_l2() {
    static void* cache=nullptr;
    if(!cache) {
      auto err=cudaMalloc(&cache,128u<<20);
      if(err!=cudaSuccess) {fprintf(stderr,"L2 allocation: %s\n",cudaGetErrorString(err));exit(1);}
    }
    auto err=cudaMemset(cache,0x5a,128u<<20);
    if(err!=cudaSuccess) {fprintf(stderr,"L2 eviction: %s\n",cudaGetErrorString(err));exit(1);}
}

template<class VectorA,class VectorB>
inline void kh_compare_outputs(const VectorA& actual,const VectorB& expected,
    double atol,double rtol,double rms_limit) {
    if(actual.size()!=expected.size() || actual.empty()) {fputs("Invalid output sizes\n",stderr);exit(1);}
    double error2=0,magnitude2=0;
    for(size_t i=0;i<actual.size();++i) {
      double x=double(float(actual[i])),y=double(float(expected[i]));
      double e=kh_abs(x-y);
      if(!std::isfinite(x)||!std::isfinite(y)||e>atol+rtol*std::abs(y)) {
        fprintf(stderr,"Incorrect element %zu: got %.9g expected %.9g\n",i,x,y);exit(1);
      }
      error2+=e*e;magnitude2+=y*y;
    }
    if(std::sqrt(error2/(magnitude2+1e-30))>rms_limit) {
      fprintf(stderr,"Incorrect normalized RMS: %.9g (limit %.9g)\n",std::sqrt(error2/(magnitude2+1e-30)),rms_limit);exit(1);
    }
}

#include <vector>
template<class T>
inline void kh_check_device(const T* actual,const T* expected,size_t count,
    double atol,double rtol,double rms_limit) {
    std::vector<T> a(count),b(count);
    auto err=cudaMemcpy(a.data(),actual,count*sizeof(T),cudaMemcpyDeviceToHost);
    if(err==cudaSuccess) err=cudaMemcpy(b.data(),expected,count*sizeof(T),cudaMemcpyDeviceToHost);
    if(err!=cudaSuccess) {fprintf(stderr,"output read: %s\n",cudaGetErrorString(err));exit(1);}
    kh_compare_outputs(a,b,atol,rtol,rms_limit);
}
