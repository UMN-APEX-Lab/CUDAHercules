#pragma once
#include "hopper_validation.cuh"
#include <cublasLt.h>
#include <algorithm>
#include <array>
#include <memory>
#include <vector>

// A performance reference, separate from the independent correctness oracle.
// Descriptors, workspace and algorithm selection are prepared before timing.
inline void kh_lt_check(cublasStatus_t status) {
    if(status!=CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr,"cuBLASLt reference error: %d\n",int(status));exit(1);
    }
}
struct kh_lt_layout {
    int m,n,k,lda,ldb,ldd;
    cudaDataType_t a_type,b_type,d_type;
    cublasComputeType_t compute=CUBLAS_COMPUTE_32F;
    bool a_row=true,b_row=false,d_row=false;
};
class kh_lt_plan {
    kh_lt_layout shape_;
    cublasLtHandle_t handle_{};
    cublasLtMatmulDesc_t operation_{};
    cublasLtMatrixLayout_t a_{},b_{},c_{};
    cublasLtMatmulAlgo_t best_{};
    void* workspace_{};
    size_t workspace_bytes_=256u<<20;
    bool prepared_=false;
public:
    explicit kh_lt_plan(kh_lt_layout shape):shape_(shape) {
        kh_lt_check(cublasLtCreate(&handle_));
        auto scalar=shape.compute==CUBLAS_COMPUTE_32I?CUDA_R_32I:CUDA_R_32F;
        kh_lt_check(cublasLtMatmulDescCreate(&operation_,shape.compute,scalar));
        auto ta=shape.a_row?CUBLAS_OP_T:CUBLAS_OP_N;
        auto tb=shape.b_row?CUBLAS_OP_T:CUBLAS_OP_N;
        kh_lt_check(cublasLtMatmulDescSetAttribute(operation_,CUBLASLT_MATMUL_DESC_TRANSA,&ta,sizeof(ta)));
        kh_lt_check(cublasLtMatmulDescSetAttribute(operation_,CUBLASLT_MATMUL_DESC_TRANSB,&tb,sizeof(tb)));
        kh_lt_check(cublasLtMatrixLayoutCreate(&a_,shape.a_type,shape.a_row?shape.k:shape.m,shape.a_row?shape.m:shape.k,shape.lda));
        kh_lt_check(cublasLtMatrixLayoutCreate(&b_,shape.b_type,shape.b_row?shape.n:shape.k,shape.b_row?shape.k:shape.n,shape.ldb));
        kh_lt_check(cublasLtMatrixLayoutCreate(&c_,shape.d_type,shape.m,shape.n,shape.ldd));
        if(shape.d_row) {
            auto order=CUBLASLT_ORDER_ROW;
            kh_lt_check(cublasLtMatrixLayoutSetAttribute(c_,CUBLASLT_MATRIX_LAYOUT_ORDER,&order,sizeof(order)));
        }
        auto error=cudaMalloc(&workspace_,workspace_bytes_);
        if(error!=cudaSuccess){fprintf(stderr,"cuBLASLt workspace: %s\n",cudaGetErrorString(error));exit(1);}
    }
    kh_lt_plan(const kh_lt_plan&)=delete;
    kh_lt_plan& operator=(const kh_lt_plan&)=delete;
    ~kh_lt_plan() {
        cudaFree(workspace_);cublasLtMatrixLayoutDestroy(a_);cublasLtMatrixLayoutDestroy(b_);
        cublasLtMatrixLayoutDestroy(c_);cublasLtMatmulDescDestroy(operation_);cublasLtDestroy(handle_);
    }
    cublasStatus_t execute(const cublasLtMatmulAlgo_t& algo,const void* a,const void* b,
                          const void* c,void* d,const void* alpha,const void* beta,cudaStream_t stream=nullptr) {
        return cublasLtMatmul(handle_,operation_,alpha,a,a_,b,b_,beta,c,c_,d,c_,&algo,workspace_,workspace_bytes_,stream);
    }
    void prepare(const void* a,const void* b,const void* c,const void* alpha,const void* beta) {
        if(prepared_)return;
        cublasLtMatmulPreference_t preference{};
        kh_lt_check(cublasLtMatmulPreferenceCreate(&preference));
        kh_lt_check(cublasLtMatmulPreferenceSetAttribute(preference,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&workspace_bytes_,sizeof(workspace_bytes_)));
        std::array<cublasLtMatmulHeuristicResult_t,64> candidates{};int count=0;
        kh_lt_check(cublasLtMatmulAlgoGetHeuristic(handle_,operation_,a_,b_,c_,c_,preference,candidates.size(),candidates.data(),&count));
        cublasLtMatmulPreferenceDestroy(preference);
        // Selection writes a scratch destination: beta!=0 must not mutate C.
        void* scratch{};
        size_t elements=size_t(shape_.ldd)*(shape_.d_row?shape_.m:shape_.n);
        size_t bytes=elements*(shape_.d_type==CUDA_R_16F||shape_.d_type==CUDA_R_16BF?2:4);
        if(cudaMalloc(&scratch,bytes)!=cudaSuccess)exit(1);
        cudaEvent_t start,stop;cudaEventCreate(&start);cudaEventCreate(&stop);
        float best_time=INFINITY;
        for(int index=0;index<count;++index) {
            auto& candidate=candidates[index];
            if(candidate.state!=CUBLAS_STATUS_SUCCESS)continue;
            if(shape_.compute==CUBLAS_COMPUTE_32F_FAST_TF32) {
                uint64_t flags=0;size_t written=0;
                kh_lt_check(cublasLtMatmulAlgoCapGetAttribute(&candidate.algo,CUBLASLT_ALGO_CAP_NUMERICAL_IMPL_FLAGS,&flags,sizeof(flags),&written));
                if((flags&(CUBLASLT_NUMERICAL_IMPL_FLAGS_INPUT_16F|CUBLASLT_NUMERICAL_IMPL_FLAGS_INPUT_16BF))||
                   !(flags&CUBLASLT_NUMERICAL_IMPL_FLAGS_ACCUMULATOR_32F))continue;
            }
            if(execute(candidate.algo,a,b,c,scratch,alpha,beta)!=CUBLAS_STATUS_SUCCESS)continue;
            if(cudaDeviceSynchronize()!=cudaSuccess)exit(1);
            std::array<float,5> trials{};bool ok=true;
            for(float& elapsed:trials) {
                kh_cold_l2();cudaEventRecord(start);
                auto status=execute(candidate.algo,a,b,c,scratch,alpha,beta);
                cudaEventRecord(stop);
                if(status!=CUBLAS_STATUS_SUCCESS||cudaEventSynchronize(stop)!=cudaSuccess){ok=false;break;}
                cudaEventElapsedTime(&elapsed,start,stop);
            }
            std::sort(trials.begin(),trials.end());
            if(ok&&trials[2]<best_time){best_time=trials[2];best_=candidate.algo;prepared_=true;}
        }
        cudaEventDestroy(start);cudaEventDestroy(stop);cudaFree(scratch);
        if(!prepared_){fputs("No supported cuBLASLt reference algorithm\n",stderr);exit(1);}
        int id=-1;uint32_t tile=0;size_t written=0;
        kh_lt_check(cublasLtMatmulAlgoConfigGetAttribute(&best_,CUBLASLT_ALGO_CONFIG_ID,&id,sizeof(id),&written));
        kh_lt_check(cublasLtMatmulAlgoConfigGetAttribute(&best_,CUBLASLT_ALGO_CONFIG_TILE_ID,&tile,sizeof(tile),&written));
        printf("cuBLASLt reference config: M=%d N=%d K=%d library=%zu algorithm=%d tile=%u workspace_cap=%zu\n",
               shape_.m,shape_.n,shape_.k,cublasLtGetVersion(),id,tile,workspace_bytes_);
    }
    cudaError_t run(const void* a,const void* b,void* d,float alpha=1,float beta=0,cudaStream_t stream=nullptr) {
        if(shape_.compute==CUBLAS_COMPUTE_32I) {
            int ai=int(alpha),bi=int(beta);prepare(a,b,d,&ai,&bi);
            return execute(best_,a,b,d,d,&ai,&bi,stream)==CUBLAS_STATUS_SUCCESS?cudaSuccess:cudaErrorUnknown;
        }
        prepare(a,b,d,&alpha,&beta);
        return execute(best_,a,b,d,d,&alpha,&beta,stream)==CUBLAS_STATUS_SUCCESS?cudaSuccess:cudaErrorUnknown;
    }
    cudaError_t run_integer(const void* a,const void* b,void* d,int alpha,int beta,cudaStream_t stream=nullptr) {
        prepare(a,b,d,&alpha,&beta);
        return execute(best_,a,b,d,d,&alpha,&beta,stream)==CUBLAS_STATUS_SUCCESS?cudaSuccess:cudaErrorUnknown;
    }
};

// Both implementations perform the same complete operation. Alternate trials
// during preparation so a clock ramp does not systematically favor one side.
template<class A,class B> bool kh_prefer_second_reference(A first,B second) {
    first();second();cudaDeviceSynchronize();
    std::array<float,5> times[2];cudaEvent_t start,stop;cudaEventCreate(&start);cudaEventCreate(&stop);
    for(int trial=0;trial<5;++trial)for(int j=0;j<2;++j) {
        int which=(trial+j)%2;kh_cold_l2();cudaEventRecord(start);
        if(which)second();else first();cudaEventRecord(stop);
        if(cudaEventSynchronize(stop)!=cudaSuccess)exit(1);
        cudaEventElapsedTime(&times[which][trial],start,stop);
    }
    cudaEventDestroy(start);cudaEventDestroy(stop);
    for(auto& t:times)std::sort(t.begin(),t.end());
    // Retain the existing implementation when the difference is within the
    // clock/noise band. Tiny apparent wins do not justify changing backends.
    bool prefer_second=times[1][2]<0.95f*times[0][2];
    printf("Reference backend selection: first=%.6f ms second=%.6f ms selected=%s\n",
           times[0][2],times[1][2],prefer_second?"second":"first");
    return prefer_second;
}

__global__ void kh_channel_scale(float* d,const float* a_scale,const float* b_scale,int M,int N) {
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<size_t(M)*N;i+=size_t(blockDim.x)*gridDim.x)
        d[i]*=a_scale[i/N]*b_scale[i%N];
}
