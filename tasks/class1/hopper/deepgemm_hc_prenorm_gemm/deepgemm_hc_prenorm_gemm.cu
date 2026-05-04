/*
 * DeepGEMM Hyperconnection Prenorm GEMM — CUDA-Hercules Class 1 harness
 *
 * Reference: DeepGEMM (DeepSeek), MIT license
 * Fused TF32 GEMM (BF16 A × FP32 B → FP32 D) + per-row L2 norm squared.
 *
 * Template instantiation: N=24, K=7168, BLOCK_M=64, BLOCK_N=32, BLOCK_K=64
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <curand_kernel.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cfloat>

#include "../../include/kh_benchmark.h"

// DeepGEMM kernel
#include <deep_gemm/impls/sm90_tf32_hc_prenorm_gemm.cuh>

#ifdef KH_TEST_SOLUTION
#include "solution.h"
#endif

// ─── Error checking ───────────────────────────────────────────────────

#define KH_CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

#define KH_DRIVER_CHECK(call) do { \
    CUresult err = (call); \
    if (err != CUDA_SUCCESS) { \
        const char* msg; cuGetErrorString(err, &msg); \
        fprintf(stderr, "CUDA driver error at %s:%d: %s\n", __FILE__, __LINE__, msg); \
        exit(1); \
    } \
} while(0)

// ─── Constants ────────────────────────────────────────────────────────

static constexpr int SHAPE_N = 24;
static constexpr int SHAPE_K = 7168;
static constexpr int BLOCK_M = 64;
static constexpr int BLOCK_N = 32;   // align(24, 16)
static constexpr int BLOCK_K = 64;
static constexpr int NUM_SPLITS = 1;
static constexpr int SWIZZLE_CD = 128;
static constexpr int NUM_MATH = 128;
static constexpr int NUM_TMA  = 128;
static constexpr int NUM_THREADS = NUM_MATH + NUM_TMA;
static constexpr int SMEM_CAPACITY = 232448;

struct TestConfig { int m; const char* label; };
static const TestConfig configs[] = {
    {1024,  "M=1024"},
    {4096,  "M=4096"},
    {8192,  "M=8192"}
};

// ─── TMA descriptor helpers ──────────────────────────────────────────

static inline int get_swizzle_mode(int block_size, int elem_size) {
    for (int mode : {128, 64, 32, 16}) {
        if ((block_size * elem_size) % mode == 0) return mode;
    }
    return 0;
}

static CUtensorMap make_tma_2d(void* data, CUtensorMapDataType dtype,
                                int gmem_inner, int gmem_outer,
                                int smem_inner, int smem_outer,
                                int stride_outer_bytes,
                                int swizzle_mode) {
    int elem_size;
    switch (dtype) {
        case CU_TENSOR_MAP_DATA_TYPE_BFLOAT16: elem_size = 2; break;
        case CU_TENSOR_MAP_DATA_TYPE_FLOAT32:  elem_size = 4; break;
        case CU_TENSOR_MAP_DATA_TYPE_TFLOAT32: elem_size = 4; break;
        default: elem_size = 1; break;
    }
    if (swizzle_mode != 0)
        smem_inner = swizzle_mode / elem_size;

    CUtensorMap tensor_map;
    cuuint64_t gmem_dims[2] = {(cuuint64_t)gmem_inner, (cuuint64_t)gmem_outer};
    cuuint32_t smem_dims[2] = {(cuuint32_t)smem_inner, (cuuint32_t)smem_outer};
    cuuint64_t gmem_strides[1] = {(cuuint64_t)stride_outer_bytes};
    cuuint32_t elem_strides[2] = {1, 1};

    CUtensorMapSwizzle sw;
    switch (swizzle_mode) {
        case 0:  case 16: sw = CU_TENSOR_MAP_SWIZZLE_NONE; break;
        case 32:  sw = CU_TENSOR_MAP_SWIZZLE_32B; break;
        case 64:  sw = CU_TENSOR_MAP_SWIZZLE_64B; break;
        case 128: sw = CU_TENSOR_MAP_SWIZZLE_128B; break;
        default:  sw = CU_TENSOR_MAP_SWIZZLE_NONE; break;
    }

    KH_DRIVER_CHECK(cuTensorMapEncodeTiled(
        &tensor_map, dtype, 2, data,
        gmem_dims, gmem_strides, smem_dims, elem_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, sw,
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    return tensor_map;
}

// ─── Random data generation ──────────────────────────────────────────

__global__ void fill_rand_bf16(nv_bfloat16* data, int n, unsigned seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    curandStatePhilox4_32_10_t state;
    curand_init(seed, idx, 0, &state);
    data[idx] = __float2bfloat16(curand_normal(&state) * 0.1f);
}

__global__ void fill_rand_f32(float* data, int n, unsigned seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    curandStatePhilox4_32_10_t state;
    curand_init(seed, idx, 0, &state);
    data[idx] = curand_normal(&state) * 0.1f;
}

// ─── Simple CPU reference for correctness checking ──────────────────

static void cpu_hc_prenorm_gemm(const nv_bfloat16* a_host, const float* b_host,
                                 float* d_host, float* s_host,
                                 int M, int N, int K) {
    for (int m = 0; m < M; m++) {
        float sq = 0;
        for (int ki = 0; ki < K; ki++) {
            float av = __bfloat162float(a_host[m * K + ki]);
            sq += av * av;
        }
        s_host[m] = sq;
        for (int n = 0; n < N; n++) {
            float acc = 0;
            for (int ki = 0; ki < K; ki++) {
                float av = __bfloat162float(a_host[m * K + ki]);
                acc += av * b_host[n * K + ki];
            }
            d_host[m * N + n] = acc;
        }
    }
}

// ─── Run DeepGEMM reference kernel ──────────────────────────────────

// Fixed number of pipeline stages (pre-verified to fit in smem)
static constexpr int NUM_STAGES = 12;

static void run_ref_kernel(nv_bfloat16* a, float* b, float* d, float* sqr_sum, int M) {
    // A: K-major → inner=K, outer=M
    int sw_a = get_swizzle_mode(BLOCK_K, (int)sizeof(nv_bfloat16));
    CUtensorMap tma_a = make_tma_2d(
        a, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        SHAPE_K, M, BLOCK_K, BLOCK_M,
        SHAPE_K * (int)sizeof(nv_bfloat16), sw_a);

    // B: K-major → inner=K, outer=N (allow_tf32 → TF32 dtype)
    int sw_b = get_swizzle_mode(BLOCK_K, (int)sizeof(float));
    CUtensorMap tma_b = make_tma_2d(
        b, CU_TENSOR_MAP_DATA_TYPE_TFLOAT32,
        SHAPE_K, SHAPE_N, BLOCK_K, BLOCK_N,
        SHAPE_K * (int)sizeof(float), sw_b);

    // D: inner=N, outer=M
    CUtensorMap tma_d = make_tma_2d(
        d, CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
        SHAPE_N, M, BLOCK_N, BLOCK_M,
        SHAPE_N * (int)sizeof(float), SWIZZLE_CD);

    int smem_size = SMEM_CAPACITY;
    auto kernel = deep_gemm::sm90_tf32_hc_prenorm_gemm_impl<
        SHAPE_N, SHAPE_K, BLOCK_M, BLOCK_N, BLOCK_K,
        NUM_SPLITS, SWIZZLE_CD, NUM_STAGES, NUM_MATH, NUM_TMA>;

    KH_CUDA_CHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    int grid = (M + BLOCK_M - 1) / BLOCK_M;
    kernel<<<grid, NUM_THREADS, smem_size>>>(
        (uint32_t)M, tma_a, tma_b, tma_d, sqr_sum);
    KH_CUDA_CHECK(cudaGetLastError());
}

// ─── main ────────────────────────────────────────────────────────────

int main() {
    // Initialize CUDA driver API
    KH_DRIVER_CHECK(cuInit(0));

    int benchmark_trials = kh_benchmark_trials();

    // Allocate B (constant across configs): [N, K] float
    float* b_dev;
    KH_CUDA_CHECK(cudaMalloc(&b_dev, SHAPE_N * SHAPE_K * sizeof(float)));
    fill_rand_f32<<<(SHAPE_N * SHAPE_K + 255) / 256, 256>>>(b_dev, SHAPE_N * SHAPE_K, 42);
    KH_CUDA_CHECK(cudaDeviceSynchronize());

    bool all_passed = true;

    for (auto& cfg : configs) {
        printf("\n=== %s: N=%d, K=%d ===\n", cfg.label, SHAPE_N, SHAPE_K);
        int M = cfg.m;

        // Allocate A: [M, K] bf16
        nv_bfloat16* a_dev;
        KH_CUDA_CHECK(cudaMalloc(&a_dev, (size_t)M * SHAPE_K * sizeof(nv_bfloat16)));
        fill_rand_bf16<<<((size_t)M * SHAPE_K + 255) / 256, 256>>>(
            a_dev, M * SHAPE_K, 123 + M);

        // Reference output
        float *d_ref, *s_ref;
        KH_CUDA_CHECK(cudaMalloc(&d_ref, (size_t)M * SHAPE_N * sizeof(float)));
        KH_CUDA_CHECK(cudaMalloc(&s_ref, M * sizeof(float)));
        KH_CUDA_CHECK(cudaMemset(d_ref, 0, (size_t)M * SHAPE_N * sizeof(float)));
        KH_CUDA_CHECK(cudaMemset(s_ref, 0, M * sizeof(float)));
        KH_CUDA_CHECK(cudaDeviceSynchronize());

        run_ref_kernel(a_dev, b_dev, d_ref, s_ref, M);
        KH_CUDA_CHECK(cudaDeviceSynchronize());

#ifdef KH_TEST_SOLUTION
        // Solution output
        float *d_sol, *s_sol;
        KH_CUDA_CHECK(cudaMalloc(&d_sol, (size_t)M * SHAPE_N * sizeof(float)));
        KH_CUDA_CHECK(cudaMalloc(&s_sol, M * sizeof(float)));
        KH_CUDA_CHECK(cudaMemset(d_sol, 0, (size_t)M * SHAPE_N * sizeof(float)));
        KH_CUDA_CHECK(cudaMemset(s_sol, 0, M * sizeof(float)));
        KH_CUDA_CHECK(cudaDeviceSynchronize());

        solution_hc_prenorm_gemm(a_dev, b_dev, d_sol, s_sol, M, SHAPE_N, SHAPE_K, 0);
        KH_CUDA_CHECK(cudaDeviceSynchronize());

        // Compare D
        {
            size_t sz = (size_t)M * SHAPE_N;
            float *h_ref = new float[sz], *h_sol = new float[sz];
            KH_CUDA_CHECK(cudaMemcpy(h_ref, d_ref, sz * sizeof(float), cudaMemcpyDeviceToHost));
            KH_CUDA_CHECK(cudaMemcpy(h_sol, d_sol, sz * sizeof(float), cudaMemcpyDeviceToHost));
            float max_diff = 0;
            for (size_t i = 0; i < sz; i++) {
                float diff = fabsf(h_ref[i] - h_sol[i]);
                float denom = fmaxf(fabsf(h_ref[i]), 1e-6f);
                max_diff = fmaxf(max_diff, diff / denom);
            }
            printf("D max relative diff: %.6f\n", max_diff);
            if (max_diff > 0.01f) { printf("FAILED (D)\n"); printf("Incorrect\n"); all_passed = false; }
            delete[] h_ref; delete[] h_sol;
        }

        // Compare sqr_sum
        {
            float *h_ref = new float[M], *h_sol = new float[M];
            KH_CUDA_CHECK(cudaMemcpy(h_ref, s_ref, M * sizeof(float), cudaMemcpyDeviceToHost));
            KH_CUDA_CHECK(cudaMemcpy(h_sol, s_sol, M * sizeof(float), cudaMemcpyDeviceToHost));
            float max_diff = 0;
            for (int i = 0; i < M; i++) {
                float diff = fabsf(h_ref[i] - h_sol[i]);
                float denom = fmaxf(fabsf(h_ref[i]), 1e-6f);
                max_diff = fmaxf(max_diff, diff / denom);
            }
            printf("sqr_sum max relative diff: %.6f\n", max_diff);
            if (max_diff > 0.01f) { printf("FAILED (sqr_sum)\n"); printf("Incorrect\n"); all_passed = false; }
            delete[] h_ref; delete[] h_sol;
        }

        if (!all_passed) {
            KH_CUDA_CHECK(cudaFree(d_sol));
            KH_CUDA_CHECK(cudaFree(s_sol));
            KH_CUDA_CHECK(cudaFree(a_dev));
            KH_CUDA_CHECK(cudaFree(d_ref));
            KH_CUDA_CHECK(cudaFree(s_ref));
            KH_CUDA_CHECK(cudaFree(b_dev));
            return -1;
        }
        printf("Passed\n");

        // Benchmark
        if (benchmark_trials > 0) {
            kh_benchmark([&]() {
                solution_hc_prenorm_gemm(a_dev, b_dev, d_sol, s_sol, M, SHAPE_N, SHAPE_K, 0);
            }, benchmark_trials, "Kernel");

            kh_benchmark([&]() {
                run_ref_kernel(a_dev, b_dev, d_ref, s_ref, M);
            }, benchmark_trials, "Ref");
        }

        KH_CUDA_CHECK(cudaFree(d_sol));
        KH_CUDA_CHECK(cudaFree(s_sol));
#else
        printf("Passed\n");

        if (benchmark_trials > 0) {
            kh_benchmark([&]() {
                run_ref_kernel(a_dev, b_dev, d_ref, s_ref, M);
            }, benchmark_trials, "Ref");
        }
#endif
        KH_CUDA_CHECK(cudaFree(a_dev));
        KH_CUDA_CHECK(cudaFree(d_ref));
        KH_CUDA_CHECK(cudaFree(s_ref));
    }

    KH_CUDA_CHECK(cudaFree(b_dev));

    if (!all_passed) {
        fprintf(stderr, "Some tests FAILED\n");
        return 1;
    }
    return 0;
}
