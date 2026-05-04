#include <iostream>
#include <iomanip>
#include <fstream>
#include <random>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include "kittens.cuh"
#include "prototype.cuh"
#include "common.cuh"

#ifdef KH_TEST_SOLUTION
#include "solution.h"
#endif

using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::lcf;

using c_dtype = float;

struct matmul_layout {
    // tiles for the quantized inputs
    using  a_tile   = st_fp8e4m3<64, 128>;
    using  b_tile   = st_fp8e4m3<128, 128>;
    using  c_tile   = st<c_dtype, 64, 128>;
    using  a_layout = gl<fp8e4m3, 1, 1, -1, -1, a_tile>;
    using  b_layout = gl<fp8e4m3, 1, 1, -1, -1, b_tile>;
    using  c_layout = gl<c_dtype, 1, 1, -1, -1, c_tile>;

    // tiles for the dequantized inputs
    using scale_a_layout = gl<c_dtype, 1, 1, 1, -1>;
    using scale_b_layout = gl<c_dtype, 1, 1, 1, -1>;

    template<typename T=float> using accum_tile = rt<T, 16, c_tile::cols>;

    struct globals        {
        a_layout A; b_layout B; c_layout C;
        scale_a_layout scale_a; scale_b_layout scale_b;
    };

    struct input_block    {
        a_tile a[2]; b_tile b;
    };
    struct finish_block   {
        c_tile c[2];
    };
    struct scratch_block  {
    };
    struct common_state   { int2 coord; };
    struct consumer_state {
        accum_tile<c_dtype> accum;      // Changed to single tall accumulator
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
        int id = warpgroup::groupid() == NUM_CONSUMER_WARPS/4 ? 0 : warpgroup::groupid();
        args.common.coord = { args.common.coord.x*2 + id, args.common.coord.y };
    }

    struct producer {
        __device__ static void setup(producer_setup_args<layout> args) {
            warpgroup::decrease_registers<40>(); // decrease registers for producers
        }
        __device__ static void load(producer_load_args<layout> args) {
            if(warpgroup::laneid() == 0) {
                tma::expect(args.inputs_arrived, args.input);
                #pragma unroll
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
            if(laneid() == 0) arrive(args.inputs_finished);
        }
        __device__ static void finish(consumer_finish_args<layout> args) {
            col_vec<rt<c_dtype, 16, 128>> scale_a_rv;
            row_vec<rt<c_dtype, 16, 128>> scale_b_rv;
            warpgroup::load(scale_a_rv, args.globals.scale_a, {args.common.coord.x});
            warp::load(scale_b_rv, args.globals.scale_b, {args.common.coord.y});
            warp::mul_col(args.state.accum, args.state.accum, scale_b_rv);
            warp::mul_row(args.state.accum, args.state.accum, scale_a_rv);
            warpgroup::store(args.finish.c[warpgroup::groupid()], args.state.accum);
            warpgroup::sync(warpgroup::groupid()+4);
            if(warpgroup::laneid() == 0) {
                tma::store_async(args.globals.C, args.finish.c[warpgroup::groupid()],
                                 {args.common.coord.x, args.common.coord.y});
                tma::store_async_read_wait();
            }
            if(laneid() == 0) arrive(args.finish_finished);
        }
    };
};

template<typename mmt>
void inner_run(
    fp8e4m3 *d_A, fp8e4m3 *d_B, c_dtype *d_C,
    c_dtype *d_scale_a, c_dtype *d_scale_b,
    size_t M, size_t N, size_t K,
    dim3 grid, dim3 block
) {
    using a_layout = typename mmt::layout::a_layout;
    using b_layout = typename mmt::layout::b_layout;
    using c_layout = typename mmt::layout::c_layout;
    using globals  = typename mmt::layout::globals;
    a_layout Ag{d_A, nullptr, nullptr, M, K};
    b_layout Bg{d_B, nullptr, nullptr, N, K};
    c_layout Cg{d_C, nullptr, nullptr, M, N};

    // scales
    using scale_a_layout = typename mmt::layout::scale_a_layout;
    using scale_b_layout = typename mmt::layout::scale_b_layout;
    scale_a_layout scale_a{d_scale_a, nullptr, nullptr, nullptr, M};
    scale_b_layout scale_b{d_scale_b, nullptr, nullptr, nullptr, N};

    globals G{Ag, Bg, Cg, scale_a, scale_b};
    prototype::lcf::kernel<mmt><<<grid, block, MAX_SHARED_MEMORY-1024>>>(G);
}

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
    float *h_C = new float[M * N];

    // Initialize random number generator
    std::mt19937 gen(42);
    std::normal_distribution dis(0.0f, 1.0f);

    for (size_t i = 0; i < M * K; ++i) h_A[i] = dis(gen) * 0.2f;
    for (size_t i = 0; i < K * N; ++i) h_B[i] = dis(gen) * 0.2f;

    // Allocate device memory
    fp8e4m3 *d_A, *d_B;
    c_dtype *d_C;
    CUDACHECK(cudaMalloc(&d_A, M*K*sizeof(fp8e4m3)));
    CUDACHECK(cudaMalloc(&d_B, K*N*sizeof(fp8e4m3)));
    CUDACHECK(cudaMalloc(&d_C, M*N*sizeof(c_dtype)));
    // scales
    c_dtype *d_scale_a, *d_scale_b;
    CUDACHECK(cudaMalloc(&d_scale_a, M*sizeof(c_dtype)));
    CUDACHECK(cudaMalloc(&d_scale_b, N*sizeof(c_dtype)));
    // float buffers for reference GEMM
    float *d_A_float, *d_B_float, *d_C_ref;
    CUDACHECK(cudaMalloc(&d_A_float, M*K*sizeof(float)));
    CUDACHECK(cudaMalloc(&d_B_float, K*N*sizeof(float)));
    CUDACHECK(cudaMalloc(&d_C_ref, M*N*sizeof(float)));

    // Copy float matrices to device and compute reference GEMM on GPU
    CUDACHECK(cudaMemcpy(d_A_float, h_A, M*K*sizeof(float), cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(d_B_float, h_B, K*N*sizeof(float), cudaMemcpyHostToDevice));
    reference_gemm<float, float, true>(d_C_ref, d_A_float, d_B_float, M, N, K);
    CUDACHECK(cudaDeviceSynchronize());
    std::cout << "Computed reference GEMM" << std::endl;

    // Obtain FP8 inputs with per-row/per-column scaling
    const float FP8_E4M3_MAX = 448.0f;
    c_dtype *h_scale_a = new c_dtype[M];
    c_dtype *h_scale_b = new c_dtype[N];
    __nv_fp8_e4m3 *h_A_fp8_scaled = new __nv_fp8_e4m3[M * K];
    __nv_fp8_e4m3 *h_B_fp8_scaled = new __nv_fp8_e4m3[K * N];

    // row-wise scaling for A
    for(size_t row = 0; row < M; row++) {
        float max_val = 0.0f;
        for(size_t col = 0; col < K; col++) {
            float abs_val = std::abs(h_A[row * K + col]);
            max_val = std::max(max_val, abs_val);
        }
        h_scale_a[row] = c_dtype(max_val / FP8_E4M3_MAX);
    }

    // fill h_A_fp8_scaled
    for(size_t i = 0; i < M; i++) {
        for(size_t j = 0; j < K; j++) {
            h_A_fp8_scaled[i * K + j] = __nv_fp8_e4m3(h_A[i * K + j] / float(h_scale_a[i]));
        }
    }

    // column-wise scaling for B (B is [N,K] layout)
    for(size_t col = 0; col < N; col++) {
        float max_val = 0.0f;
        for(size_t row = 0; row < K; row++) {
            float abs_val = std::abs(h_B[row + col*K]);
            max_val = std::max(max_val, abs_val);
        }
        h_scale_b[col] = c_dtype(max_val / FP8_E4M3_MAX);
    }

    // fill h_B_fp8_scaled
    for(size_t i = 0; i < N; i++) {
        for(size_t j = 0; j < K; j++) {
            h_B_fp8_scaled[j + i * K] = __nv_fp8_e4m3(h_B[j + i * K] / float(h_scale_b[i]));
        }
    }

    CUDACHECK(cudaMemcpy(d_A, h_A_fp8_scaled, M*K*sizeof(fp8e4m3), cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(d_B, h_B_fp8_scaled, K*N*sizeof(fp8e4m3), cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(d_scale_a, h_scale_a, M*sizeof(c_dtype), cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(d_scale_b, h_scale_b, N*sizeof(c_dtype), cudaMemcpyHostToDevice));

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
        inner_run<mmt>(d_A, d_B, d_C, d_scale_a, d_scale_b, M, N, K, grid, block);
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
        inner_run<mmt>(d_A, d_B, d_C, d_scale_a, d_scale_b, M, N, K, grid, block);
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

    // Verify TK correctness - compare d_C (float) vs d_C_ref (float)
    float *h_C_out = new float[M * N];
    float *h_C_ref = new float[M * N];
    CUDACHECK(cudaMemcpy(h_C_out, d_C, M*N*sizeof(c_dtype), cudaMemcpyDeviceToHost));
    CUDACHECK(cudaMemcpy(h_C_ref, d_C_ref, M*N*sizeof(float), cudaMemcpyDeviceToHost));

    int error_count = 0;
    for (size_t i = 0; i < M * N; ++i) {
        float error = std::abs(float(h_C_out[i]) - h_C_ref[i]);
        if (error > 0.7f) error_count++;
    }
    std::cout << "TK error count (tol=0.7): " << error_count << " / " << M*N << std::endl;

    // Also use check_correctness for detailed stats (cast d_C to float* for comparison)
    // check_correctness works on device pointers of the same type
    // d_C is c_dtype* (float*) and d_C_ref is float*, so compatible
    check_correctness(d_C, d_C_ref, M * N);
    std::cout << "Passed" << std::endl;

    std::cout << "Ref time: " << std::fixed << std::setprecision(4) << tk_avg_ms
              << " ms (avg over " << num_iters << " trials, min: " << tk_min_ms << " ms)" << std::endl;

#ifdef KH_TEST_SOLUTION
    // Allocate solution output buffer
    float *d_C_solution;
    CUDACHECK(cudaMalloc(&d_C_solution, M*N*sizeof(float)));

    // Run solution
    Fp8GemmScaled(reinterpret_cast<__nv_fp8_e4m3*>(d_A),
                   reinterpret_cast<__nv_fp8_e4m3*>(d_B),
                   d_C_solution,
                   d_scale_a, d_scale_b,
                   M, N, K);
    CUDACHECK(cudaDeviceSynchronize());

    // Check correctness against reference
    std::cout << "Solution correctness:" << std::endl;
    check_correctness(d_C_solution, d_C_ref, M * N);
    {
      std::vector<float> h_sol(M * N), h_ref(M * N);
      cudaMemcpy(h_sol.data(), d_C_solution, M * N * sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(h_ref.data(), d_C_ref, M * N * sizeof(float), cudaMemcpyDeviceToHost);
      float max_diff = 0;
      for (size_t i = 0; i < M * N; ++i)
        max_diff = fmaxf(max_diff, fabsf(h_sol[i] - h_ref[i]));
      if (max_diff > 1.0f) {
        fprintf(stderr, "Solution incorrect vs reference: max_diff=%.6f\n", max_diff);
        std::cout << "Incorrect" << std::endl;
        exit(-1);
      }
    }
    std::cout << "Passed" << std::endl;

    // Warmup solution
    for(int i = 0; i < num_warmups; i++) {
        Fp8GemmScaled(reinterpret_cast<__nv_fp8_e4m3*>(d_A),
                       reinterpret_cast<__nv_fp8_e4m3*>(d_B),
                       d_C_solution,
                       d_scale_a, d_scale_b,
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
        Fp8GemmScaled(reinterpret_cast<__nv_fp8_e4m3*>(d_A),
                       reinterpret_cast<__nv_fp8_e4m3*>(d_B),
                       d_C_solution,
                       d_scale_a, d_scale_b,
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
    delete[] h_C;
    delete[] h_C_out;
    delete[] h_C_ref;
    delete[] h_scale_a;
    delete[] h_scale_b;
    delete[] h_A_fp8_scaled;
    delete[] h_B_fp8_scaled;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_A_float);
    cudaFree(d_B_float);
    cudaFree(d_C_ref);
    cudaFree(d_scale_a);
    cudaFree(d_scale_b);
}

int main() {
    run_benchmark<matmul_template<8>>("fp8_gemm_scaled_4096", 4096, 4096, 4096);
    run_benchmark<matmul_template<8>>("fp8_gemm_scaled_8192", 8192, 8192, 8192);
    return 0;
}
