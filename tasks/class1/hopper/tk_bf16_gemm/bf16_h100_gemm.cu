#include "kittens.cuh"
#include "prototype.cuh"
#include "common.cuh"

#ifdef KH_TEST_SOLUTION
#include "solution.h"
#endif

using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::lcf;
template<int M_BLOCK, int N_BLOCK>
struct matmul_layout {
    using  base_tile      = st_bf<64, 64>;
    using  global_layout  = gl<bf16, 1, 1, -1, -1, base_tile>;
    struct globals        { global_layout A, B, C; };
    struct input_block    { base_tile a[M_BLOCK], b[N_BLOCK]; };
    struct finish_block   { base_tile c[M_BLOCK][N_BLOCK]; };
    struct common_state   { int2 coord; };
    struct consumer_state { rt_fl<16, N_BLOCK*base_tile::cols> accum; };
};
template<int _M_BLOCK=2, int _N_BLOCK=4, int _SUPER_M=12>
struct matmul_template {
    static constexpr int M_BLOCK = _M_BLOCK, N_BLOCK = _N_BLOCK, SUPER_M = _SUPER_M;
    using layout    = matmul_layout<M_BLOCK, N_BLOCK>;
    using wide_tile = st_bf<64, 64*N_BLOCK>;
    static constexpr int NUM_CONSUMER_WARPS=M_BLOCK*4, INPUT_PIPE_STAGES=4, PRODUCER_BARRIER_ARRIVALS=1;
    // Helper functions
    template<bool PERISISTENT_GRID=true> __host__ static inline dim3 grid(int M, int N, int K) {
        return dim3(PERISISTENT_GRID ? 132 : M*N/(M_BLOCK*N_BLOCK*layout::base_tile::num_elements));
    }
    // ThunderKittens template functions
    __device__ static inline void common_setup(common_setup_args<layout> args) {
        int Rblocks = args.globals.C.rows() / (M_BLOCK*64), Cblocks = args.globals.C.cols() / (N_BLOCK*64);
        int super_rows = (Rblocks/SUPER_M)*SUPER_M,
            final_rows = Rblocks - super_rows,
            super_repeat = SUPER_M*Cblocks;
        int task_id = args.task_iter*gridDim.x + blockIdx.x;
        if (task_id < super_rows * Cblocks)
            args.common.coord = { SUPER_M*(task_id/super_repeat) + task_id%SUPER_M,
                           (task_id%super_repeat)/SUPER_M };
        else if (task_id < Rblocks*Cblocks) {
            int remainder_id = task_id - super_rows*Cblocks;
            args.common.coord = { super_rows + (remainder_id%final_rows), remainder_id/final_rows };
        }
        else { // Id is too high, no more work to do
            args.num_iters = -1;
            return;
        }
        args.num_iters = args.globals.A.cols()/64;
        int id = warpgroup::groupid() == NUM_CONSUMER_WARPS/4 ? 0 : warpgroup::groupid(); // producer sets as 0
        args.common.coord = { args.common.coord.x*M_BLOCK + id, args.common.coord.y*N_BLOCK };
    }
    struct producer {
        __device__ static void setup(producer_setup_args<layout> args) {
            warpgroup::decrease_registers<40>(); // decrease registers for producers
        }
        __device__ static void load(producer_load_args<layout> args) {
            if (warpgroup::laneid() == 0) {
                tma::expect(args.inputs_arrived, args.input);
                for(int i = 0; i < M_BLOCK; i++)
                    tma::load_async(args.input.a[i], args.globals.A,
                                    {args.common.coord.x+i, args.iter}, args.inputs_arrived);
                for(int i = 0; i < N_BLOCK; i++)
                    tma::load_async(args.input.b[i], args.globals.B,
                                    {args.iter, args.common.coord.y+i}, args.inputs_arrived);
            }
        }
    };
    struct consumer {
        __device__ static void setup(consumer_setup_args<layout> args) {
            warpgroup::increase_registers<232>(); // increase registers for consumers
            kittens::warp::zero(args.state.accum);
        }
        __device__ static void compute(consumer_compute_args<layout> args) {
            warpgroup::mma_AB(
                args.state.accum, // dest registers
                args.input.a[warpgroup::groupid()], // A matrix
                reinterpret_cast<wide_tile&>(args.input.b) // B matrix
            );
            warpgroup::mma_async_wait();
            if (warp::laneid() == 0) arrive(args.inputs_finished);
        }
        __device__ static void finish(consumer_finish_args<layout> args) {
            warpgroup::store(reinterpret_cast<wide_tile&>(args.finish.c[warpgroup::groupid()]), args.state.accum);
            warpgroup::sync(warpgroup::groupid()+4);
            if (warpgroup::laneid() == 0) for(int i = 0; i < N_BLOCK; i++) {
                tma::store_async(args.globals.C, args.finish.c[warpgroup::groupid()][i],
                                             {args.common.coord.x, args.common.coord.y+i});
                tma::store_async_read_wait(); // wait that store is finished before reusing finish memory
            }
            kittens::warp::zero(args.state.accum);
            if (warp::laneid() == 0) arrive(args.finish_finished);
        }
    };
};

#include <iostream>
#include <cuda_bf16.h>

#define CUDACHECK(err) do { \
    cudaError_t err_ = (err); \
    if (err_ != cudaSuccess) { \
        fprintf(stderr, "CUDA error %d at %s:%d: %s\n", err_, __FILE__, __LINE__, cudaGetErrorString(err_)); \
        exit(1); \
    } \
} while(0)

template<typename mmt>
void inner_run(bf16 *d_A, bf16 *d_B, bf16 *d_C, size_t M, size_t N, size_t K, dim3 grid, dim3 block) {
    using global_layout = typename mmt::layout::global_layout;
    using globals  = typename mmt::layout::globals;
    global_layout Ag{d_A, nullptr, nullptr, M, K};
    global_layout Bg{d_B, nullptr, nullptr, K, N};
    global_layout Cg{d_C, nullptr, nullptr, M, N};
    globals G{Ag, Bg, Cg};
    prototype::lcf::kernel<mmt><<<grid, block, MAX_SHARED_MEMORY-1024>>>(G);
}

template<typename mmt>
void run_benchmark(const char* label, size_t M, size_t N, size_t K) {
    std::cout << "\n=== " << label << ": M=" << M << ", N=" << N << ", K=" << K << " ===" << std::endl;

    // Cooldown between configurations
    sleep_ms(500);

    // L2 cache eviction - multiple buffer groups
    int l2_cache_size;
    cudaDeviceGetAttribute(&l2_cache_size, cudaDevAttrL2CacheSize, 0);
    const size_t arg_size = 2 * (size_t(M) * K + size_t(N) * K + size_t(M) * N);
    const size_t ideal_arg_size = size_t(l2_cache_size) * 3;
    const int arg_group_count = (arg_size > ideal_arg_size) ? 1 : int(ideal_arg_size / arg_size) + 1;

    // Allocate device memory
    std::vector<__nv_bfloat16*> d_A(arg_group_count);
    std::vector<__nv_bfloat16*> d_B(arg_group_count);
    std::vector<__nv_bfloat16*> d_C(arg_group_count);
    __nv_bfloat16* d_C_ref;
    for (int i = 0; i < arg_group_count; i++) {
        CUDACHECK(cudaMalloc(&d_A[i], M*K*sizeof(__nv_bfloat16)));
        CUDACHECK(cudaMalloc(&d_B[i], K*N*sizeof(__nv_bfloat16)));
        CUDACHECK(cudaMalloc(&d_C[i], M*N*sizeof(__nv_bfloat16)));
    }
    CUDACHECK(cudaMalloc(&d_C_ref, M*N*sizeof(__nv_bfloat16)));

    // Initialize matrices with random values on device
    uint64_t seed = 42;
    for (int i = 0; i < arg_group_count; i++) {
        fill<__nv_bfloat16, FillMode::RANDOM>(d_A[i], M*K, seed + i*100, -1.0f, 1.0f);
        fill<__nv_bfloat16, FillMode::RANDOM>(d_B[i], K*N, seed + i*100 + 1, -1.0f, 1.0f);
        fill<__nv_bfloat16, FillMode::CONSTANT>(d_C[i], M*N, 0.0f);
    }
    fill<__nv_bfloat16, FillMode::CONSTANT>(d_C_ref, M*N, 0.0f);
    CUDACHECK(cudaDeviceSynchronize());

    // Compute reference GEMM on device (transpose_b=false for RowMajor K*N B layout)
    reference_gemm<__nv_bfloat16, __nv_bfloat16, false>(d_C_ref, d_A[0], d_B[0], M, N, K);
    CUDACHECK(cudaDeviceSynchronize());
    std::cout << "Computed reference GEMM" << std::endl;

    // Set kernel attributes
    unsigned long mem_size = MAX_SHARED_MEMORY - 1024;
    CUDACHECK(cudaFuncSetAttribute(prototype::lcf::kernel<mmt>, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size));

    // Launch kernel
    dim3 grid(mmt::grid(M, N, K));
    dim3 block(kittens::prototype::detail::NUM_THREADS_v<mmt>);

    int num_warmups = 5;
    int num_iters = 10;

    // Warmup
    for(int i = 0; i < num_warmups; i++) {
        int idx = i % arg_group_count;
        inner_run<mmt>(d_A[idx], d_B[idx], d_C[idx], M, N, K, grid, block);
    }

    // Benchmark TK kernel with per-iteration CUDA event timing
    float tk_total_ms = 0.0f;
    float tk_min_ms = 1e30f;
    for(int i = 0; i < num_iters; i++) {
        int idx = i % arg_group_count;
        cudaEvent_t start, stop;
        CUDACHECK(cudaEventCreate(&start));
        CUDACHECK(cudaEventCreate(&stop));
        CUDACHECK(cudaEventRecord(start));
        inner_run<mmt>(d_A[idx], d_B[idx], d_C[idx], M, N, K, grid, block);
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

    // Verify TK results
    check_correctness(d_C[0], d_C_ref, M * N);
    std::cout << "Passed" << std::endl;

    std::cout << "Ref time: " << std::fixed << std::setprecision(4) << tk_avg_ms
              << " ms (avg over " << num_iters << " trials, min: " << tk_min_ms << " ms)" << std::endl;

#ifdef KH_TEST_SOLUTION
    // Allocate solution output buffer
    __nv_bfloat16 *d_C_solution;
    CUDACHECK(cudaMalloc(&d_C_solution, M*N*sizeof(__nv_bfloat16)));
    fill<__nv_bfloat16, FillMode::CONSTANT>(d_C_solution, M*N, 0.0f);
    CUDACHECK(cudaDeviceSynchronize());

    // Run solution
    Bf16Gemm(d_A[0], d_B[0], d_C_solution, M, N, K);
    CUDACHECK(cudaDeviceSynchronize());

    // Check correctness against reference
    std::cout << "Solution correctness:" << std::endl;
    check_correctness(d_C_solution, d_C_ref, M * N);
    {
      std::vector<__nv_bfloat16> h_sol(M * N), h_ref(M * N);
      cudaMemcpy(h_sol.data(), d_C_solution, M * N * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
      cudaMemcpy(h_ref.data(), d_C_ref, M * N * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
      float max_diff = 0;
      for (size_t i = 0; i < M * N; ++i)
        max_diff = fmaxf(max_diff, fabsf(__bfloat162float(h_sol[i]) - __bfloat162float(h_ref[i])));
      if (max_diff > 1e-1f) {
        fprintf(stderr, "Solution incorrect vs reference: max_diff=%.6f\n", max_diff);
        std::cout << "Incorrect" << std::endl;
        exit(-1);
      }
    }
    std::cout << "Passed" << std::endl;

    // Warmup solution
    for(int i = 0; i < num_warmups; i++) {
        Bf16Gemm(d_A[0], d_B[0], d_C_solution, M, N, K);
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
        Bf16Gemm(d_A[0], d_B[0], d_C_solution, M, N, K);
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
    for (int i = 0; i < arg_group_count; i++) {
        cudaFree(d_A[i]);
        cudaFree(d_B[i]);
        cudaFree(d_C[i]);
    }
    cudaFree(d_C_ref);
}

int main() {
    run_benchmark<matmul_template<2,4,8>>("bf16_gemm_4096", 4096, 4096, 4096);
    run_benchmark<matmul_template<2,4,8>>("bf16_gemm_8192", 8192, 8192, 8192);
    return 0;
}
