#pragma once
#include "hopper_validation.cuh"
#include <cublas_v2.h>

// Check the public FP32/TF32 contract independently of the performance kernel.
// FP32, TF32 round-to-nearest and TF32 round-toward-zero are all admissible.
// This avoids rejecting accurate FP32 due to TF32 cancellation error.
__global__ void kh_tf32_round(const float* in,float* out,size_t count,bool nearest) {
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=size_t(blockDim.x)*gridDim.x) {
        uint32_t bits=__float_as_uint(in[i]);
        if(nearest)bits+=0xfffu+((bits>>13)&1u);
        out[i]=__uint_as_float(bits&0xffffe000u);
    }
}
struct kh_tf32_error {double error2,magnitude2;int failed;};
__global__ void kh_tf32_measure_error(const float* actual,const float* expected,size_t n,
                                     double atol,kh_tf32_error* stats) {
    double e2=0,r2=0;int bad=0;
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=size_t(blockDim.x)*gridDim.x) {
        double a=actual[i],b=expected[i],error=kh_abs(a-b);
        bad|=!isfinite(a)||!isfinite(b)||error>atol+3e-5*fabs(b);
        e2+=error*error;r2+=b*b;
    }
    for(int step=16;step;step/=2) {
        e2+=__shfl_down_sync(0xffffffff,e2,step);r2+=__shfl_down_sync(0xffffffff,r2,step);
        bad|=__shfl_down_sync(0xffffffff,bad,step);
    }
    if(threadIdx.x%32==0){atomicAdd(&stats->error2,e2);atomicAdd(&stats->magnitude2,r2);if(bad)atomicExch(&stats->failed,1);}
}
class kh_tf32_contract {
    int m_,n_;float alpha_;float* expected_[3]{};kh_tf32_error* stats_{};
    kh_tf32_error error(const float* actual,int mode) {
        cudaMemset(stats_,0,sizeof(kh_tf32_error));
        kh_tf32_measure_error<<<256,256>>>(actual,expected_[mode],size_t(m_)*n_,0.003*fmaxf(1,fabsf(alpha_)),stats_);
        kh_tf32_error result{};
        auto status=cudaMemcpy(&result,stats_,sizeof(result),cudaMemcpyDeviceToHost);
        if(status!=cudaSuccess){fprintf(stderr,"TF32 validation: %s\n",cudaGetErrorString(status));exit(1);}
        return result;
    }
public:
    kh_tf32_contract(int m,int n,int k,float alpha,const float* a,const float* b,float beta,const float* initial_c)
        :m_(m),n_(n),alpha_(alpha) {
        float *aq,*bq;cudaMalloc(&aq,size_t(m)*k*4);cudaMalloc(&bq,size_t(n)*k*4);cudaMalloc(&stats_,sizeof(kh_tf32_error));
        cublasHandle_t handle;if(cublasCreate(&handle)!=CUBLAS_STATUS_SUCCESS)exit(1);
        for(int mode=0;mode<3;++mode) {
            cudaMalloc(&expected_[mode],size_t(m)*n*4);
            if(beta!=0)cudaMemcpy(expected_[mode],initial_c,size_t(m)*n*4,cudaMemcpyDeviceToDevice);
            if(mode){kh_tf32_round<<<1024,256>>>(a,aq,size_t(m)*k,mode==1);kh_tf32_round<<<1024,256>>>(b,bq,size_t(n)*k,mode==1);}
            const float* aa=mode?aq:a;const float* bb=mode?bq:b;
            auto status=cublasGemmEx(handle,CUBLAS_OP_T,CUBLAS_OP_N,m,n,k,&alpha,aa,CUDA_R_32F,k,
                bb,CUDA_R_32F,k,&beta,expected_[mode],CUDA_R_32F,m,CUBLAS_COMPUTE_32F_PEDANTIC,CUBLAS_GEMM_DEFAULT);
            if(status!=CUBLAS_STATUS_SUCCESS){fprintf(stderr,"TF32 oracle error: %d\n",int(status));exit(1);}
            kh_sample_gemm(m,n,k,alpha,aa,bb,beta,initial_c,expected_[mode],0.001f*fmaxf(1,fabsf(alpha)),0.00001f);
        }
        cublasDestroy(handle);cudaFree(aq);cudaFree(bq);
    }
    ~kh_tf32_contract(){for(auto p:expected_)cudaFree(p);cudaFree(stats_);}
    bool accepts(const float* actual) {
        auto raw=error(actual,0);
        double nrms=sqrt(raw.error2/(raw.magnitude2+1e-30));
        if(!std::isfinite(nrms)||nrms>0.001)return false;
        for(int mode=0;mode<3;++mode) {
            auto stats=mode?error(actual,mode):raw;
            if(!stats.failed&&sqrt(stats.error2/(stats.magnitude2+1e-30))<=0.0001)return true;
        }
        return false;
    }
    void check(const float* actual) {
        if(!accepts(actual)){fputs("FP32/TF32 contract validation failed\n",stderr);exit(1);}
    }
};
