// Build with nvcc -O3 -std=c++17 -arch=sm_90a -lcublasLt -lcublas.
// KH_TUNE_CUTLASS additionally benchmarks prepared CUTLASS configurations.
// Runs the public task-48 layout and exponent-range input contract unchanged.
#ifndef NDEBUG
#define NDEBUG
#endif
#include "../tasks/common/hopper_validation.cuh"
#include <cublasLt.h>
#include <cublas_v2.h>
#include <algorithm>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#ifdef KH_TUNE_CUTLASS
#define main kh_legacy_main
#include "../tasks/class1/hopper/48_hopper_warp_specialized_gemm/48_hopper_warp_specialized_gemm.cu"
#undef main
#endif

// Optional frozen submission for a fresh comparison; never used to construct
// or select the vendor/CUTLASS reference implementations.
#ifdef KH_TUNE_SOLUTION
#include KH_TUNE_SOLUTION
#endif

#define CUDA_OK(call) do {auto s=(call);if(s!=cudaSuccess){std::cerr<<#call<<": "<<cudaGetErrorString(s)<<'\n';exit(1);}}while(0)
#define BLAS_OK(call) do {auto s=(call);if(s!=CUBLAS_STATUS_SUCCESS){std::cerr<<#call<<": "<<int(s)<<'\n';exit(1);}}while(0)

int experiment_seed() {const char* value=std::getenv("KH_TUNE_SEED");return value?std::atoi(value):0;}
bool selected(const std::string& name) {
    const char* value=std::getenv("KH_TUNE_CANDIDATE");return !value||name==value;
}

__global__ void fill_tf32(float* a,size_t count,int K,int seed,bool inverse) {
    for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<count;i+=size_t(blockDim.x)*gridDim.x) {
        int exponent=int(kh_hash(uint32_t(i%K)+998)%41)-20;
        a[i]=ldexpf(kh_random(i,seed),inverse?-exponent:exponent);
    }
}
__global__ void round_tf32(const float* input,float* output,size_t count,bool nearest) {
    for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<count;i+=size_t(blockDim.x)*gridDim.x) {
        uint32_t bits=__float_as_uint(input[i]);
        if(nearest)bits+=0xfffu+((bits>>13)&1u);
        output[i]=__uint_as_float(bits&0xffffe000u);
    }
}
struct ErrorStats {double error2,reference2,max_abs,max_fraction,raw_nrms;int failed,rounding;};
__global__ void validate_tf32(const float* actual,const float* expected,size_t n,ErrorStats* stats,double atol,double rtol) {
    double e2=0,r2=0,ma=0,mf=0;int failed=0;
    for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=size_t(blockDim.x)*gridDim.x) {
        double x=actual[i],y=expected[i],e=kh_abs(x-y),f=e/(atol+rtol*fabs(y));
        failed|=!isfinite(x)||!isfinite(y)||f>1; e2+=e*e;r2+=y*y;ma=fmax(ma,e);mf=fmax(mf,f);
    }
    for(int offset=16;offset;offset/=2) {
        e2+=__shfl_down_sync(0xffffffff,e2,offset);r2+=__shfl_down_sync(0xffffffff,r2,offset);
        ma=fmax(ma,__shfl_down_sync(0xffffffff,ma,offset));mf=fmax(mf,__shfl_down_sync(0xffffffff,mf,offset));
        failed|=__shfl_down_sync(0xffffffff,failed,offset);
    }
    if(threadIdx.x%32==0) {
        atomicAdd(&stats->error2,e2);atomicAdd(&stats->reference2,r2);
        atomicMax((unsigned long long*)&stats->max_abs,__double_as_longlong(ma));
        atomicMax((unsigned long long*)&stats->max_fraction,__double_as_longlong(mf));
        if(failed)atomicExch(&stats->failed,1);
    }
}
struct Problem {
    int m,n,k;float *a,*b,*d,*gold,*gold_rn,*gold_rz,*aq,*bq;void* workspace;size_t workspace_bytes=256u<<20;
    ErrorStats* stats;cublasHandle_t oracle;
    Problem(int M,int N,int K):m(M),n(N),k(K) {
        CUDA_OK(cudaMalloc(&a,size_t(m)*k*4));CUDA_OK(cudaMalloc(&b,size_t(n)*k*4));
        CUDA_OK(cudaMalloc(&d,size_t(m)*n*4));CUDA_OK(cudaMalloc(&gold,size_t(m)*n*4));
        CUDA_OK(cudaMalloc(&gold_rn,size_t(m)*n*4));CUDA_OK(cudaMalloc(&gold_rz,size_t(m)*n*4));
        CUDA_OK(cudaMalloc(&aq,size_t(m)*k*4));CUDA_OK(cudaMalloc(&bq,size_t(n)*k*4));
        CUDA_OK(cudaMalloc(&workspace,workspace_bytes));CUDA_OK(cudaMalloc(&stats,sizeof(ErrorStats)));
        BLAS_OK(cublasCreate(&oracle));refill(experiment_seed());
    }
    void refill(int seed) {
        fill_tf32<<<1024,256>>>(a,size_t(m)*k,k,seed,false);
        fill_tf32<<<1024,256>>>(b,size_t(n)*k,k,seed+17,true);
        float alpha=1,beta=0;
        BLAS_OK(cublasGemmEx(oracle,CUBLAS_OP_T,CUBLAS_OP_N,m,n,k,&alpha,a,CUDA_R_32F,k,
                    b,CUDA_R_32F,k,&beta,gold,CUDA_R_32F,m,CUBLAS_COMPUTE_32F_PEDANTIC,CUBLAS_GEMM_DEFAULT));
        CUDA_OK(cudaDeviceSynchronize());
        kh_sample_gemm(m,n,k,1.f,a,b,0.f,gold,gold,0.001f,0.00001f);
        for(bool nearest:{true,false}) {
            round_tf32<<<1024,256>>>(a,aq,size_t(m)*k,nearest);
            round_tf32<<<1024,256>>>(b,bq,size_t(n)*k,nearest);
            float* target=nearest?gold_rn:gold_rz;
            BLAS_OK(cublasGemmEx(oracle,CUBLAS_OP_T,CUBLAS_OP_N,m,n,k,&alpha,aq,CUDA_R_32F,k,
                       bq,CUDA_R_32F,k,&beta,target,CUDA_R_32F,m,CUBLAS_COMPUTE_32F_PEDANTIC,CUBLAS_GEMM_DEFAULT));
            CUDA_OK(cudaDeviceSynchronize());
            kh_sample_gemm(m,n,k,1.f,aq,bq,0.f,target,target,0.001f,0.00001f);
        }
    }
    ErrorStats compare(const float* target,double atol,double rtol) {
        CUDA_OK(cudaMemset(stats,0,sizeof(ErrorStats)));
        validate_tf32<<<256,256>>>(d,target,size_t(m)*n,stats,atol,rtol);
        ErrorStats out{};CUDA_OK(cudaMemcpy(&out,stats,sizeof(out),cudaMemcpyDeviceToHost));
        return out;
    }
    ErrorStats check() {
        // A TF32 contract permits operand rounding. Check FP32 accumulation
        // against independently rounded operands, plus total RMS vs real FP32.
        auto best=compare(gold,0.003,0.00003);
        double raw_nrms=sqrt(best.error2/(best.reference2+1e-30));
        for(int mode=1;mode<=2;++mode) {
            auto candidate=compare(mode==1?gold_rn:gold_rz,0.003,0.00003);
            candidate.rounding=mode;
            if(candidate.error2<best.error2)best=candidate;
        }
        best.raw_nrms=raw_nrms;
        // Different FP32 reduction trees can differ after 8192 products;
        // retain the strict elementwise accumulation bound as well as RMS.
        if(raw_nrms>0.001||sqrt(best.error2/(best.reference2+1e-30))>0.0001)best.failed=1;
        return best;
    }
    ~Problem(){cublasDestroy(oracle);cudaFree(a);cudaFree(b);cudaFree(d);cudaFree(gold);cudaFree(gold_rn);cudaFree(gold_rz);cudaFree(aq);cudaFree(bq);cudaFree(workspace);cudaFree(stats);}
};

struct Timing {float minimum,median,mean;};
cudaStream_t test_stream=nullptr;
Timing measure(const std::function<bool()>& run,bool graph=false,int trials=10) {
    if(const char* value=std::getenv("KH_TUNE_TRIALS"))trials=std::max(1,std::atoi(value));
    if(const char* value=std::getenv("KH_TUNE_GRAPH"))graph=std::atoi(value)!=0;
    for(int i=0;i<3;++i)if(!run())return {0,0,0};
    CUDA_OK(cudaDeviceSynchronize());
    cudaGraph_t g{};cudaGraphExec_t executable{};cudaStream_t stream{};
    if(graph) {
        CUDA_OK(cudaStreamCreate(&stream));
        test_stream=stream;
        CUDA_OK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
        if(!run())exit(1);
        CUDA_OK(cudaStreamEndCapture(stream,&g));
        CUDA_OK(cudaGraphInstantiate(&executable,g,nullptr,nullptr,0));
    }
    cudaEvent_t start,stop;CUDA_OK(cudaEventCreate(&start));CUDA_OK(cudaEventCreate(&stop));
    std::vector<float> times;
    for(int i=0;i<trials;++i) {
        kh_cold_l2();CUDA_OK(cudaEventRecord(start,stream));
        if(graph)CUDA_OK(cudaGraphLaunch(executable,stream));else if(!run())return {0,0,0};
        CUDA_OK(cudaEventRecord(stop,stream));CUDA_OK(cudaEventSynchronize(stop));
        float ms;CUDA_OK(cudaEventElapsedTime(&ms,start,stop));times.push_back(ms);
    }
    CUDA_OK(cudaEventDestroy(start));CUDA_OK(cudaEventDestroy(stop));
    if(graph){CUDA_OK(cudaGraphExecDestroy(executable));CUDA_OK(cudaGraphDestroy(g));CUDA_OK(cudaStreamDestroy(stream));test_stream=nullptr;}
    std::sort(times.begin(),times.end());float sum=0;for(float t:times)sum+=t;
    return {times.front(),times[times.size()/2],sum/times.size()};
}

void report(const Problem& p,const std::string& name,const ErrorStats& error,Timing timing,
            const std::string& extra="") {
    std::cout<<"{\"M\":"<<p.m<<",\"N\":"<<p.n<<",\"K\":"<<p.k
        <<",\"seed\":"<<experiment_seed()<<",\"cuda_graph\":"<<(std::getenv("KH_TUNE_GRAPH")&&std::atoi(std::getenv("KH_TUNE_GRAPH"))?"true":"false")
        <<",\"candidate\":\""<<name<<"\",\"passed\":"<<(error.failed?"false":"true")
        <<",\"max_abs\":"<<error.max_abs<<",\"max_fraction\":"<<error.max_fraction
        <<",\"normalized_rms\":"<<sqrt(error.error2/(error.reference2+1e-30))
        <<",\"normalized_rms_to_fp32\":"<<error.raw_nrms<<",\"operand_rounding\":"<<error.rounding
        <<",\"min_ms\":"<<timing.minimum<<",\"median_ms\":"<<timing.median<<",\"mean_ms\":"<<timing.mean
        <<extra<<"}"<<std::endl;
}

struct LtRunner {
    Problem& p;cublasLtHandle_t handle;cublasLtMatmulDesc_t op;cublasLtMatrixLayout_t a,b,c;
    cublasLtMatmulPreference_t preference;
    std::vector<cublasLtMatmulHeuristicResult_t> candidates;
    LtRunner(Problem& problem):p(problem) {
        BLAS_OK(cublasLtCreate(&handle));
        BLAS_OK(cublasLtMatmulDescCreate(&op,CUBLAS_COMPUTE_32F_FAST_TF32,CUDA_R_32F));
        BLAS_OK(cublasLtMatrixLayoutCreate(&a,CUDA_R_32F,p.m,p.k,p.k));
        cublasLtOrder_t row=CUBLASLT_ORDER_ROW;
        BLAS_OK(cublasLtMatrixLayoutSetAttribute(a,CUBLASLT_MATRIX_LAYOUT_ORDER,&row,sizeof(row)));
        BLAS_OK(cublasLtMatrixLayoutCreate(&b,CUDA_R_32F,p.k,p.n,p.k));
        BLAS_OK(cublasLtMatrixLayoutCreate(&c,CUDA_R_32F,p.m,p.n,p.m));
        BLAS_OK(cublasLtMatmulPreferenceCreate(&preference));
        BLAS_OK(cublasLtMatmulPreferenceSetAttribute(preference,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&p.workspace_bytes,sizeof(p.workspace_bytes)));
        candidates.resize(200);int count=0;
        BLAS_OK(cublasLtMatmulAlgoGetHeuristic(handle,op,a,b,c,c,preference,candidates.size(),candidates.data(),&count));
        candidates.resize(count);
    }
    bool run(const cublasLtMatmulAlgo_t& algo) {
        float alpha=1,beta=0;
        return cublasLtMatmul(handle,op,&alpha,p.a,a,p.b,b,&beta,p.d,c,p.d,c,&algo,p.workspace,p.workspace_bytes,test_stream)==CUBLAS_STATUS_SUCCESS;
    }
    ~LtRunner(){cublasLtMatmulPreferenceDestroy(preference);cublasLtMatrixLayoutDestroy(a);cublasLtMatrixLayoutDestroy(b);cublasLtMatrixLayoutDestroy(c);cublasLtMatmulDescDestroy(op);cublasLtDestroy(handle);}
};

#ifdef KH_TUNE_CUTLASS
template<int BM,int BN,int BK,int CM,int CN>
void cutlass_candidate(Problem& p) {
    std::string name="cutlass_"+std::to_string(BM)+"x"+std::to_string(BN)+"x"+std::to_string(BK)+"_c"+std::to_string(CM)+"x"+std::to_string(CN);
    if(!selected(name))return;
    using Tile=cute::Shape<cute::Int<BM>,cute::Int<BN>,cute::Int<BK>>;
    using Cluster=cute::Shape<cute::Int<CM>,cute::Int<CN>,cute::_1>;
    using Epi=typename cutlass::epilogue::collective::CollectiveBuilder<ArchTag,OperatorClass,
        Tile,Cluster,cutlass::epilogue::collective::EpilogueTileAuto,float,float,
        float,LayoutC,4,float,LayoutC,4,cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;
    using Loop=typename cutlass::gemm::collective::CollectiveBuilder<ArchTag,OperatorClass,
        float,LayoutA,4,float,LayoutB,4,float,Tile,Cluster,
        cutlass::gemm::collective::StageCountAutoCarveout<sizeof(typename Epi::SharedStorage)>,
        cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
    using Kernel=cutlass::gemm::kernel::GemmUniversal<cute::Shape<int,int,int>,Loop,Epi>;
    using Device=cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
    auto sa=cutlass::make_cute_packed_stride(typename Kernel::StrideA{},{p.m,p.k,1});
    auto sb=cutlass::make_cute_packed_stride(typename Kernel::StrideB{},{p.n,p.k,1});
    auto sc=cutlass::make_cute_packed_stride(typename Kernel::StrideC{},{p.m,p.n,1});
    auto sd=cutlass::make_cute_packed_stride(typename Kernel::StrideD{},{p.m,p.n,1});
    auto hardware=cutlass::KernelHardwareInfo::make_kernel_hardware_info<Kernel>(0);
    typename Device::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm,{p.m,p.n,p.k},
        {p.a,sa,p.b,sb},{{1.f,0.f},p.d,sc,p.d,sd},hardware};
    Device gemm;
    if(gemm.can_implement(args)!=cutlass::Status::kSuccess||Device::get_workspace_size(args)>p.workspace_bytes)return;
    if(gemm.initialize(args,p.workspace)!=cutlass::Status::kSuccess)return;
    auto run=[&](){return gemm.run(test_stream)==cutlass::Status::kSuccess;};
    kh_poison(p.d,size_t(p.m)*p.n*4);if(!run())return;
    auto errors=p.check();auto timing=errors.failed?Timing{0,0,0}:measure(run);
    report(p,name,errors,timing);
}
#endif

int main(int argc,char** argv) {
    std::vector<std::vector<int>> shapes={{128,8192,8192},{8192,8192,8192},{8192,28672,8192}};
    if(argc>=4)shapes={{atoi(argv[1]),atoi(argv[2]),atoi(argv[3])}};
    bool cutlass_only=argc>=5&&std::string(argv[4])=="cutlass";
    for(auto shape:shapes) {
        Problem p(shape[0],shape[1],shape[2]);
        if(!cutlass_only) {
            LtRunner lt(p);int index=0;
            for(auto& candidate:lt.candidates) {
                int current=index++;
                if(!selected("cublaslt_"+std::to_string(current)))continue;
                if(candidate.state!=CUBLAS_STATUS_SUCCESS)continue;
                uint64_t flags=0;size_t written=0;
                BLAS_OK(cublasLtMatmulAlgoCapGetAttribute(&candidate.algo,CUBLASLT_ALGO_CAP_NUMERICAL_IMPL_FLAGS,&flags,sizeof(flags),&written));
                if((flags&(CUBLASLT_NUMERICAL_IMPL_FLAGS_INPUT_16F|CUBLASLT_NUMERICAL_IMPL_FLAGS_INPUT_16BF))||
                   !(flags&CUBLASLT_NUMERICAL_IMPL_FLAGS_ACCUMULATOR_32F))continue;
                kh_poison(p.d,size_t(p.m)*p.n*4);if(!lt.run(candidate.algo))continue;
                auto errors=p.check();Timing timing=errors.failed?Timing{0,0,0}:measure([&](){return lt.run(candidate.algo);});
                int id=-1;uint32_t tile=0,split=0,stages=0;
                cublasLtMatmulAlgoConfigGetAttribute(&candidate.algo,CUBLASLT_ALGO_CONFIG_ID,&id,sizeof(id),&written);
                cublasLtMatmulAlgoConfigGetAttribute(&candidate.algo,CUBLASLT_ALGO_CONFIG_TILE_ID,&tile,sizeof(tile),&written);
                cublasLtMatmulAlgoConfigGetAttribute(&candidate.algo,CUBLASLT_ALGO_CONFIG_SPLITK_NUM,&split,sizeof(split),&written);
                cublasLtMatmulAlgoConfigGetAttribute(&candidate.algo,CUBLASLT_ALGO_CONFIG_STAGES_ID,&stages,sizeof(stages),&written);
                report(p,"cublaslt_"+std::to_string(current),errors,timing,
                    ",\"algo_id\":"+std::to_string(id)+",\"tile_id\":"+std::to_string(tile)+",\"split_k\":"+std::to_string(split)+",\"stages_id\":"+std::to_string(stages)+",\"numerical_flags\":"+std::to_string(flags)+",\"workspace_bytes\":"+std::to_string(candidate.workspaceSize));
            }
        }
// Evaluate an archived candidate under the same inputs and timing protocol.
#ifdef KH_TUNE_SOLUTION
        if(selected("kda_saved") && !(std::getenv("KH_TUNE_GRAPH")&&std::atoi(std::getenv("KH_TUNE_GRAPH")))) {
            auto run=[&](){return HopperGemm(p.m,p.n,p.k,1,p.a,p.k,p.b,p.k,0,p.d,p.m)==cudaSuccess;};
            kh_poison(p.d,size_t(p.m)*p.n*4);if(!run())return 1;
            auto errors=p.check();report(p,"kda_saved",errors,errors.failed?Timing{0,0,0}:measure(run));
        }
#endif
#ifdef KH_TUNE_CUTLASS
        if(selected("current_reference") && !(std::getenv("KH_TUNE_GRAPH")&&std::atoi(std::getenv("KH_TUNE_GRAPH")))) {
            auto old=[&](){return kh_tf32_reference48(p.m,p.n,p.k,1,p.a,p.k,p.b,p.k,0,p.d,p.m)==cudaSuccess;};
            kh_poison(p.d,size_t(p.m)*p.n*4);if(!old())return 1;auto errors=p.check();report(p,"current_reference",errors,measure(old));
        }
#define TUNE(M,N,K,CM,CN) cutlass_candidate<M,N,K,CM,CN>(p)
        TUNE(128,128,32,4,2);TUNE(64,64,32,1,1);TUNE(64,128,32,1,1);TUNE(64,256,32,1,1);
        TUNE(128,64,32,1,1);TUNE(128,128,32,1,1);TUNE(128,256,32,1,1);
        TUNE(64,128,64,1,1);TUNE(128,128,64,1,1);TUNE(128,256,64,1,1);
        TUNE(128,128,32,1,2);TUNE(128,128,32,2,1);TUNE(128,256,32,1,2);TUNE(128,256,32,2,1);
#undef TUNE
#endif
    }
}
