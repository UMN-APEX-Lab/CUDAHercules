#include "kittens.cuh"
#include "prototype.cuh"
#include "common.cuh"

#ifdef KH_TEST_SOLUTION
#include "solution.h"
#endif

#ifdef TORCH_COMPILE
#define TK_COMPILE_FP8_GEMM
#endif

using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::lcf;
struct matmul_layout {
    using  a_tile         = st_fp8e4m3<64,  128>; // SA: note that if we could accum in fp16, then we could use <64, 256>
    using  b_tile         = st_fp8e4m3<256, 128>;
    using  c_tile         = st_fp8e4m3<64,  256>;
    using  a_layout       = gl<fp8e4m3, 1, 1, -1, -1, a_tile>;
    using  b_layout       = gl<fp8e4m3, 1, 1, -1, -1, b_tile>;
    using  c_layout       = gl<fp8e4m3, 1, 1, -1, -1, c_tile>;
    struct globals        { a_layout A; b_layout B; c_layout C; };
    struct input_block    { a_tile a[2]; b_tile b; };
    struct finish_block   { c_tile c[2]; };
    struct common_state   { int2 coord; };
    struct consumer_state {
        rt_fl<16, c_tile::cols> accum;  // Changed to single tall accumulator
        rt_fp8e4m3<16, c_tile::cols> accum_fp8;  // Changed to match tall format
    };
};
template<int _SUPER_M=12>
struct matmul_template {
    static constexpr int SUPER_M = _SUPER_M;
    using layout    = matmul_layout;
    static constexpr int NUM_CONSUMER_WARPS=8, INPUT_PIPE_STAGES=4, PRODUCER_BARRIER_ARRIVALS=1;
    // Helper functions
    template<bool PERISISTENT_GRID=true> __host__ static inline dim3 grid(int M, int N, int K) {
        return dim3(PERISISTENT_GRID ? 132 : M*N/(2*layout::c_tile::num_elements));
    }
    // ThunderKittens template functions
    __device__ static inline void common_setup(common_setup_args<layout> args) {
        int Rblocks = args.globals.C.rows() / (2*layout::c_tile::rows), Cblocks = args.globals.C.cols() / layout::c_tile::cols;
        int super_rows = (Rblocks/SUPER_M)*SUPER_M,
            final_rows = Rblocks - super_rows,
            super_repeat = SUPER_M*Cblocks;
        int task_id = args.task_iter*gridDim.x + blockIdx.x;
        if (task_id < super_rows * Cblocks)
            args.common.coord = { SUPER_M*(task_id/super_repeat) + task_id%SUPER_M, (task_id%super_repeat)/SUPER_M };
        else if (task_id < Rblocks*Cblocks) {
            int remainder_id = task_id - super_rows*Cblocks;
            args.common.coord = { super_rows + (remainder_id%final_rows), remainder_id/final_rows };
        }
        else { // Id is too high, no more work to do
            args.num_iters = -1;
            return;
        }
        args.num_iters = args.globals.A.cols()/layout::a_tile::cols;
        int id = warpgroup::groupid() == NUM_CONSUMER_WARPS/4 ? 0 : warpgroup::groupid(); // producer sets as 0
        args.common.coord = { args.common.coord.x*2 + id, args.common.coord.y };
    }

    struct producer {
        __device__ static void setup(producer_setup_args<layout> args) {
            warpgroup::decrease_registers<40>(); // decrease registers for producers
        }
        __device__ static void load(producer_load_args<layout> args) {
            if(warpgroup::laneid() == 0) {
                tma::expect(args.inputs_arrived, args.input);
                for(int i = 0; i < 2; i++) {
                    tma::load_async(args.input.a[i], args.globals.A,
                                    {args.common.coord.x+i, args.iter}, args.inputs_arrived);
                }
                tma::load_async(args.input.b, args.globals.B,
                                {args.common.coord.y, args.iter}, args.inputs_arrived);
            }
        }
    };

    struct consumer {
        __device__ static void setup(consumer_setup_args<layout> args) {
            warpgroup::increase_registers<232>(); // increase registers for consumers
            warp::zero(args.state.accum);
        }
        __device__ static void compute(consumer_compute_args<layout> args) {
            warpgroup::mma_ABt(
                args.state.accum,
                args.input.a[warpgroup::groupid()],
                args.input.b
            );
            warpgroup::mma_async_wait();
            if(warp::laneid() == 0) arrive(args.inputs_finished); // TODO REVIEW
        }
        __device__ static void finish(consumer_finish_args<layout> args) {
            kittens::warp::copy(args.state.accum_fp8, args.state.accum);
            warpgroup::store(args.finish.c[warpgroup::groupid()], args.state.accum_fp8);
            warpgroup::sync(warpgroup::groupid()+4);
            if(warpgroup::laneid() == 0) {
                tma::store_async(args.globals.C, args.finish.c[warpgroup::groupid()],
                                 {args.common.coord.x, args.common.coord.y});
                tma::store_async_read_wait();
            }
            warp::zero(args.state.accum);
            if(warp::laneid() == 0) arrive(args.finish_finished); // TODO REVIEW
        }
    };
};

#include <iostream>
#include <random>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

template<typename mmt>
void inner_run(fp8e4m3 *d_A, fp8e4m3 *d_B, fp8e4m3 *d_C, size_t M, size_t N, size_t K, dim3 grid, dim3 block) {
    using a_layout = typename mmt::layout::a_layout;
    using b_layout = typename mmt::layout::b_layout;
    using c_layout = typename mmt::layout::c_layout;
    using globals  = typename mmt::layout::globals;
    a_layout Ag{d_A, nullptr, nullptr, M, K};
    b_layout Bg{d_B, nullptr, nullptr, N, K};
    c_layout Cg{d_C, nullptr, nullptr, M, N};
    globals G{Ag, Bg, Cg};
    prototype::lcf::kernel<mmt><<<grid, block, MAX_SHARED_MEMORY-1024>>>(G);
}

#ifdef TK_COMPILE_FP8_GEMM
#include <ATen/cuda/CUDAContext.h>
#include <ATen/Functions.h>
#include "pyutils/torchutils.cuh"
#include <iostream>

at::Tensor fp8_gemm(at::Tensor A, at::Tensor B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);

    auto M = A.size(0);
    auto N = B.size(0);
    auto K = A.size(1);
    printf("M=%d N=%d K=%d\n", M, N, K);
    at::Tensor C = at::empty({M, N}, A.options());

    // convert to bf16
    c10::Float8_e4m3fn *A_fp8 = A.data_ptr<c10::Float8_e4m3fn>();
    c10::Float8_e4m3fn *B_fp8 = B.data_ptr<c10::Float8_e4m3fn>();

    fp8e4m3 *d_A = reinterpret_cast<fp8e4m3*>(A_fp8);
    fp8e4m3 *d_B = reinterpret_cast<fp8e4m3*>(B_fp8);
    fp8e4m3 *d_C = reinterpret_cast<fp8e4m3*>(C.data_ptr<c10::Float8_e4m3fn>());

    dim3 grid(matmul_template<8>::grid(M, N, K));
    dim3 block(kittens::prototype::detail::NUM_THREADS_v<matmul_template<8>>);

    inner_run<matmul_template<8>>(d_A, d_B, d_C, M, N, K, grid, block);

    CHECK_CUDA_ERROR(cudaGetLastError());
    return C;
}
#else

#define CUDACHECK(err) do { \
    cudaError_t err_ = (err); \
    if (err_ != cudaSuccess) { \
        fprintf(stderr, "CUDA error %d at %s:%d: %s\n", err_, __FILE__, __LINE__, cudaGetErrorString(err_)); \
        exit(1); \
    } \
} while(0)

template<typename mmt>
void run_benchmark(const char* label, size_t M, size_t N, size_t K) {
    std::cout << "\n=== " << label << ": M=" << M << ", N=" << N << ", K=" << K << " ===" << std::endl;

    // Allocate host memory
    float *h_A = new float[M * K];
    float *h_B = new float[K * N];

    // Initialize random number generator
    std::mt19937 gen(42);
    std::normal_distribution dis(0.0f, 1.0f);

    // Initialize matrices with random values
    for (size_t i = 0; i < M * K; ++i) h_A[i] = dis(gen) * 0.1f;
    for (size_t i = 0; i < K * N; ++i) h_B[i] = dis(gen) * 0.1f;

    // Allocate device memory
    fp8e4m3 *d_A, *d_B, *d_C, *d_C_ref;
    CUDACHECK(cudaMalloc(&d_A, M*K*sizeof(fp8e4m3)));
    CUDACHECK(cudaMalloc(&d_B, K*N*sizeof(fp8e4m3)));
    CUDACHECK(cudaMalloc(&d_C, M*N*sizeof(fp8e4m3)));
    CUDACHECK(cudaMalloc(&d_C_ref, M*N*sizeof(fp8e4m3)));

    // Convert to fp8 and copy to device
    __nv_fp8_e4m3 *h_A_fp8 = new __nv_fp8_e4m3[M * K];
    __nv_fp8_e4m3 *h_B_fp8 = new __nv_fp8_e4m3[K * N];
    for (size_t i = 0; i < M * K; ++i) h_A_fp8[i] = __nv_fp8_e4m3(h_A[i]);
    for (size_t i = 0; i < K * N; ++i) h_B_fp8[i] = __nv_fp8_e4m3(h_B[i]);
    // Round-trip to get exact fp8 values
    for (size_t i = 0; i < M * K; ++i) h_A[i] = float(h_A_fp8[i]);
    for (size_t i = 0; i < K * N; ++i) h_B[i] = float(h_B_fp8[i]);

    CUDACHECK(cudaMemcpy(d_A, h_A_fp8, M*K*sizeof(fp8e4m3), cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(d_B, h_B_fp8, K*N*sizeof(fp8e4m3), cudaMemcpyHostToDevice));

    // Compute reference GEMM on GPU (transpose_b=true for ABt layout)
    reference_gemm<fp8e4m3, fp8e4m3, true>(d_C_ref, d_A, d_B, M, N, K);
    CUDACHECK(cudaDeviceSynchronize());
    std::cout << "Computed reference GEMM" << std::endl;

    unsigned long mem_size = MAX_SHARED_MEMORY - 1024;
    CUDACHECK(cudaFuncSetAttribute(prototype::lcf::kernel<mmt>, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size));

    // Launch kernel
    dim3 grid(mmt::grid(M, N, K));
    dim3 block(kittens::prototype::detail::NUM_THREADS_v<mmt>);

    int num_warmups = 5;
    int num_iters = 10;

    // Warmup
    for(int i = 0; i < num_warmups; i++) {
        inner_run<mmt>(d_A, d_B, d_C, M, N, K, grid, block);
    }
    CUDACHECK(cudaDeviceSynchronize());

    // Benchmark TK kernel with per-iteration CUDA event timing
    float tk_total_ms = 0.0f;
    float tk_min_ms = 1e30f;
    for(int i = 0; i < num_iters; i++) {
        cudaEvent_t start, stop;
        CUDACHECK(cudaEventCreate(&start));
        CUDACHECK(cudaEventCreate(&stop));
        CUDACHECK(cudaEventRecord(start));
        inner_run<mmt>(d_A, d_B, d_C, M, N, K, grid, block);
        CUDACHECK(cudaEventRecord(stop));
        CUDACHECK(cudaEventSynchronize(stop));
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        tk_total_ms += ms;
        if (ms < tk_min_ms) tk_min_ms = ms;
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    float tk_avg_ms = tk_total_ms / num_iters;

    // Verify TK correctness against reference
    // Copy results to host for error checking
    __nv_fp8_e4m3 *h_C_fp8 = new __nv_fp8_e4m3[M * N];
    __nv_fp8_e4m3 *h_C_ref_fp8 = new __nv_fp8_e4m3[M * N];
    cudaMemcpy(h_C_fp8, d_C, M*N*sizeof(fp8e4m3), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C_ref_fp8, d_C_ref, M*N*sizeof(fp8e4m3), cudaMemcpyDeviceToHost);

    int error_count = 0;
    for (size_t i = 0; i < M * N; ++i) {
        float error = std::abs(float(h_C_fp8[i]) - float(h_C_ref_fp8[i]));
        if (error > 0.25f) error_count++;
    }
    std::cout << "TK error count (tol=0.25): " << error_count << " / " << M*N << std::endl;
    check_correctness(d_C, d_C_ref, M * N);
    std::cout << "Passed" << std::endl;

    std::cout << "Ref time: " << std::fixed << std::setprecision(4) << tk_avg_ms
              << " ms (avg over " << num_iters << " trials, min: " << tk_min_ms << " ms)" << std::endl;

#ifdef KH_TEST_SOLUTION
    // Allocate solution output buffer
    fp8e4m3 *d_C_solution;
    CUDACHECK(cudaMalloc(&d_C_solution, M*N*sizeof(fp8e4m3)));

    // Run solution
    Fp8Gemm(reinterpret_cast<__nv_fp8_e4m3*>(d_A),
             reinterpret_cast<__nv_fp8_e4m3*>(d_B),
             reinterpret_cast<__nv_fp8_e4m3*>(d_C_solution),
             M, N, K);
    CUDACHECK(cudaDeviceSynchronize());

    // Check correctness against reference
    std::cout << "Solution correctness:" << std::endl;
    check_correctness(d_C_solution, d_C_ref, M * N);
    {
      std::vector<__nv_fp8_e4m3> h_sol(M * N), h_ref(M * N);
      cudaMemcpy(h_sol.data(), d_C_solution, M * N * sizeof(__nv_fp8_e4m3), cudaMemcpyDeviceToHost);
      cudaMemcpy(h_ref.data(), d_C_ref, M * N * sizeof(__nv_fp8_e4m3), cudaMemcpyDeviceToHost);
      float max_diff = 0;
      for (size_t i = 0; i < M * N; ++i)
        max_diff = fmaxf(max_diff, fabsf(float(h_sol[i]) - float(h_ref[i])));
      if (max_diff > 0.5f) {
        fprintf(stderr, "Solution incorrect vs reference: max_diff=%.6f\n", max_diff);
        std::cout << "Incorrect" << std::endl;
        exit(-1);
      }
    }
    std::cout << "Passed" << std::endl;

    // Warmup solution
    for(int i = 0; i < num_warmups; i++) {
        Fp8Gemm(reinterpret_cast<__nv_fp8_e4m3*>(d_A),
                 reinterpret_cast<__nv_fp8_e4m3*>(d_B),
                 reinterpret_cast<__nv_fp8_e4m3*>(d_C_solution),
                 M, N, K);
    }
    CUDACHECK(cudaDeviceSynchronize());

    // Benchmark solution with per-iteration CUDA event timing
    float sol_total_ms = 0.0f;
    float sol_min_ms = 1e30f;
    for(int i = 0; i < num_iters; i++) {
        cudaEvent_t start, stop;
        CUDACHECK(cudaEventCreate(&start));
        CUDACHECK(cudaEventCreate(&stop));
        CUDACHECK(cudaEventRecord(start));
        Fp8Gemm(reinterpret_cast<__nv_fp8_e4m3*>(d_A),
                 reinterpret_cast<__nv_fp8_e4m3*>(d_B),
                 reinterpret_cast<__nv_fp8_e4m3*>(d_C_solution),
                 M, N, K);
        CUDACHECK(cudaEventRecord(stop));
        CUDACHECK(cudaEventSynchronize(stop));
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        sol_total_ms += ms;
        if (ms < sol_min_ms) sol_min_ms = ms;
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    float sol_avg_ms = sol_total_ms / num_iters;

    std::cout << "Kernel time: " << std::fixed << std::setprecision(4) << sol_avg_ms
              << " ms (avg over " << num_iters << " trials, min: " << sol_min_ms << " ms)" << std::endl;
    std::cout << "Speedup: " << std::fixed << std::setprecision(4) << (tk_min_ms / sol_min_ms)
              << "x (ref_min / kernel_min)" << std::endl;

    cudaFree(d_C_solution);
#endif

    // Clean up
    delete[] h_A;
    delete[] h_B;
    delete[] h_A_fp8;
    delete[] h_B_fp8;
    delete[] h_C_fp8;
    delete[] h_C_ref_fp8;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_C_ref);
}

int main() {
    run_benchmark<matmul_template<8>>("fp8_gemm_4096", 4096, 4096, 4096);
    run_benchmark<matmul_template<8>>("fp8_gemm_8192", 8192, 8192, 8192);
    return 0;
}
#endif
