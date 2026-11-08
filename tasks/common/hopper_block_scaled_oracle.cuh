#pragma once
#include "hopper_validation.cuh"

// Independent fp64 samples of the public block-scale contract. No CUTLASS
// layouts, native reference kernels, or candidate helpers are used here.
template<bool PerRowA, class A, class B, class D>
__global__ void kh_block_scaled_samples(int M,int N,int K,float alpha,
    const A* a,const B* b,const float* sa,const float* sb,const D* output,
    double atol,double rtol,int* failed) {
    unsigned sample=threadIdx.x;
    size_t count=size_t(M)*N;
    size_t index=sample==0?0:sample==1?count-1:kh_hash(sample*7919u)%count;
    int m=index%M,n=index/M;
    double sum=0;
    for(int k=0;k<K;++k) {
        size_t ai=PerRowA ? m+size_t(k/128)*M : m/128+size_t(k/128)*((M+127)/128);
        size_t bi=n/128+size_t(k/128)*((N+127)/128);
        sum+=double(float(a[size_t(m)*K+k]))*double(float(b[size_t(n)*K+k]))*sa[ai]*sb[bi];
    }
    double expected=double(float(D(alpha*sum)));
    double actual=double(float(output[index]));
    if(!isfinite(actual)||fabs(actual-expected)>atol+rtol*fabs(expected)) {
        if(atomicExch(failed,1)==0)
            printf("Block-scale oracle (%d,%d): got %.9g expected %.9g\n",m,n,actual,expected);
    }
}

template<bool PerRowA, class A, class B, class D>
inline void kh_sample_block_scaled(int M,int N,int K,float alpha,
    const A* a,const B* b,const float* sa,const float* sb,const D* output,
    double atol,double rtol) {
    if(M==0||N==0)return;
    int* failed;cudaMalloc(&failed,sizeof(int));cudaMemset(failed,0,sizeof(int));
    kh_block_scaled_samples<PerRowA><<<1,128>>>(M,N,K,alpha,a,b,sa,sb,output,atol,rtol,failed);
    int bad=1;auto error=cudaMemcpy(&bad,failed,sizeof(int),cudaMemcpyDeviceToHost);cudaFree(failed);
    if(error!=cudaSuccess||bad) {fputs("Independent block-scale oracle failed\n",stderr);exit(1);}
}
