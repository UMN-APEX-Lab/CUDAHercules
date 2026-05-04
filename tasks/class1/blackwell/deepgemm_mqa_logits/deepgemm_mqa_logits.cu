/*
 * DeepGEMM FP8 MQA Attention Scoring — CUDA-Hercules Class 1 harness
 *
 * Reference: DeepGEMM (DeepSeek), MIT license
 * Computes: logits[q,kv] = sum_h(relu(Q[q,h,:] @ K[kv,:]^T) * w[q,h]) * scale[kv]
 *
 * Template: num_heads=64, head_dim=128, block_q=2, block_kv=256
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <curand_kernel.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cfloat>

#include "../../include/kh_benchmark.h"

// DeepGEMM kernel
#include <deep_gemm/impls/sm100_fp8_mqa_logits.cuh>

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

static constexpr int NUM_HEADS = 64;
static constexpr int HEAD_DIM  = 128;
static constexpr int BLOCK_QH  = 128;  // block_q * num_heads
static constexpr int BLOCK_Q   = BLOCK_QH / NUM_HEADS;  // = 2
static constexpr int BLOCK_KV  = 256;
static constexpr int NUM_Q_STAGES  = 3;
static constexpr int NUM_KV_STAGES = 3;
static constexpr int NUM_TMA  = 128;
static constexpr int NUM_MATH = 256;
static constexpr int NUM_THREADS = NUM_TMA + NUM_MATH;

struct TestConfig { int seq_len; int seq_len_kv; const char* label; };
static const TestConfig configs[] = {
    {1024, 4096,  "S=1024,SKV=4096"},
    {2048, 4096,  "S=2048,SKV=4096"},
    {2048, 8192,  "S=2048,SKV=8192"}
};

// ─── TMA helpers ─────────────────────────────────────────────────────

static inline int tma_aligned_size(int size, int elem_size) {
    // Minimum 256-byte alignment for TMA descriptors
    int alignment = 256 / elem_size;
    return ((size + alignment - 1) / alignment) * alignment;
}

static CUtensorMap make_tma_2d(void* data, CUtensorMapDataType dtype,
                                int gmem_inner, int gmem_outer,
                                int smem_inner, int smem_outer,
                                int stride_outer_bytes,
                                CUtensorMapSwizzle swizzle) {
    CUtensorMap tensor_map;
    cuuint64_t gmem_dims[2] = {(cuuint64_t)gmem_inner, (cuuint64_t)gmem_outer};
    cuuint32_t smem_dims[2] = {(cuuint32_t)smem_inner, (cuuint32_t)smem_outer};
    cuuint64_t gmem_strides[1] = {(cuuint64_t)stride_outer_bytes};
    cuuint32_t elem_strides[2] = {1, 1};

    KH_DRIVER_CHECK(cuTensorMapEncodeTiled(
        &tensor_map, dtype, 2, data,
        gmem_dims, gmem_strides, smem_dims, elem_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
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

// Cast BF16 to FP8 E4M3 (simple truncation, no scaling for Q)
__global__ void cast_bf16_to_fp8(__nv_fp8_e4m3* out, const nv_bfloat16* in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = __nv_fp8_e4m3(__bfloat162float(in[idx]));
}

// Per-token quantize BF16 KV to FP8 E4M3 + scales
__global__ void quantize_kv_to_fp8(
    __nv_fp8_e4m3* kv_fp8, float* kv_scales,
    const nv_bfloat16* kv_bf16,
    int seq_len_kv, int head_dim) {
    int token = blockIdx.x;
    if (token >= seq_len_kv) return;

    // Find max abs in this token's row
    float max_abs = 0;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float v = fabsf(__bfloat162float(kv_bf16[token * head_dim + d]));
        max_abs = fmaxf(max_abs, v);
    }
    // Warp reduction
    for (int offset = 16; offset > 0; offset >>= 1)
        max_abs = fmaxf(max_abs, __shfl_down_sync(0xffffffff, max_abs, offset));
    max_abs = __shfl_sync(0xffffffff, max_abs, 0);

    float scale = fmaxf(max_abs, 1e-4f) / 448.0f;
    if (threadIdx.x == 0) kv_scales[token] = scale;

    float inv_scale = 1.0f / scale;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float v = __bfloat162float(kv_bf16[token * head_dim + d]) * inv_scale;
        kv_fp8[token * head_dim + d] = __nv_fp8_e4m3(v);
    }
}

// Fill logits with -inf
__global__ void fill_neginf(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] = -INFINITY;
}

// ─── Run DeepGEMM reference kernel ──────────────────────────────────

static void run_ref_kernel(__nv_fp8_e4m3* q, __nv_fp8_e4m3* kv,
                            float* kv_scales, float* weights,
                            int* ks, int* ke, float* logits,
                            int seq_len, int seq_len_kv, int num_sms) {
    // swizzle for head_dim=128, FP8 elem_size=1: (128*1)%128==0 → 128
    // smem_inner overridden: 128/1=128 (matches head_dim)

    // Q TMA: inner=head_dim, outer=seq_len*num_heads
    CUtensorMap tma_q = make_tma_2d(
        q, CU_TENSOR_MAP_DATA_TYPE_UINT8,
        HEAD_DIM, seq_len * NUM_HEADS,
        HEAD_DIM, BLOCK_QH,
        HEAD_DIM * 1,
        CU_TENSOR_MAP_SWIZZLE_128B);

    // KV TMA: inner=head_dim, outer=seq_len_kv
    CUtensorMap tma_kv = make_tma_2d(
        kv, CU_TENSOR_MAP_DATA_TYPE_UINT8,
        HEAD_DIM, seq_len_kv,
        HEAD_DIM, BLOCK_KV,
        HEAD_DIM * 1,
        CU_TENSOR_MAP_SWIZZLE_128B);

    // KV scales TMA: inner=aligned_seq_len_kv, outer=1
    int aligned_skv = tma_aligned_size(seq_len_kv, (int)sizeof(float));
    CUtensorMap tma_kv_scales = make_tma_2d(
        kv_scales, CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
        aligned_skv, 1,
        BLOCK_KV, 1,
        0,
        CU_TENSOR_MAP_SWIZZLE_NONE);

    // Weights TMA: inner=num_heads, outer=seq_len
    CUtensorMap tma_weights = make_tma_2d(
        weights, CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
        NUM_HEADS, seq_len,
        NUM_HEADS, BLOCK_Q,
        NUM_HEADS * (int)sizeof(float),
        CU_TENSOR_MAP_SWIZZLE_NONE);

    // Shared memory
    int smem_size = 0;
    smem_size += NUM_Q_STAGES * BLOCK_Q * NUM_HEADS * HEAD_DIM * 1;    // Q (FP8)
    smem_size += NUM_KV_STAGES * BLOCK_KV * HEAD_DIM * 1;              // KV (FP8)
    smem_size += NUM_Q_STAGES * BLOCK_Q * NUM_HEADS * (int)sizeof(float); // weights
    smem_size += NUM_KV_STAGES * BLOCK_KV * (int)sizeof(float);          // kv_scales
    smem_size += (NUM_Q_STAGES * 2 + NUM_KV_STAGES * 2 + (NUM_MATH / 128) * 2) * 8; // barriers
    smem_size += 4;

    auto kernel = deep_gemm::sm100_fp8_mqa_logits<
        NUM_HEADS, HEAD_DIM,
        false,  // not compressed
        BLOCK_Q, BLOCK_KV,
        NUM_Q_STAGES, NUM_KV_STAGES,
        NUM_TMA, NUM_MATH>;

    KH_CUDA_CHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    // Align seq_len to BLOCK_Q
    int aligned_seq_len = ((seq_len + BLOCK_Q - 1) / BLOCK_Q) * BLOCK_Q;

    kernel<<<num_sms, NUM_THREADS, smem_size>>>(
        (uint32_t)aligned_seq_len, (uint32_t)seq_len_kv,
        (uint32_t)0, (uint64_t)seq_len_kv,
        (uint32_t*)ks, (uint32_t*)ke,
        logits,
        tma_q, tma_kv, tma_kv_scales, tma_weights);
    KH_CUDA_CHECK(cudaGetLastError());
}

// ─── CPU reference for validation ───────────────────────────────────

static void cpu_mqa_logits(
    const __nv_fp8_e4m3* q, const __nv_fp8_e4m3* kv,
    const float* kv_scales, const float* weights,
    const int* ks, const int* ke,
    float* logits,
    int seq_len, int seq_len_kv) {

    for (int qi = 0; qi < seq_len; qi++) {
        for (int ki = 0; ki < seq_len_kv; ki++) {
            if (ki < ks[qi] || ki >= ke[qi]) {
                logits[qi * seq_len_kv + ki] = -INFINITY;
                continue;
            }
            float result = 0;
            for (int h = 0; h < NUM_HEADS; h++) {
                float dot = 0;
                for (int d = 0; d < HEAD_DIM; d++) {
                    float qv = float(q[(qi * NUM_HEADS + h) * HEAD_DIM + d]);
                    float kk = float(kv[ki * HEAD_DIM + d]);
                    dot += qv * kk;
                }
                float score = dot > 0 ? dot : 0;  // ReLU
                result += score * weights[qi * NUM_HEADS + h];
            }
            logits[qi * seq_len_kv + ki] = result * kv_scales[ki];
        }
    }
}

// ─── main ────────────────────────────────────────────────────────────

int main() {
    KH_DRIVER_CHECK(cuInit(0));

    // Get number of SMs
    int device;
    KH_CUDA_CHECK(cudaGetDevice(&device));
    int num_sms;
    KH_CUDA_CHECK(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device));
    printf("Device has %d SMs\n", num_sms);

    int benchmark_trials = kh_benchmark_trials();
    bool all_passed = true;

    for (auto& cfg : configs) {
        printf("\n=== %s ===\n", cfg.label);
        int S = cfg.seq_len, SKV = cfg.seq_len_kv;
        // Align S to BLOCK_Q
        int S_aligned = ((S + BLOCK_Q - 1) / BLOCK_Q) * BLOCK_Q;

        // Allocate BF16 source data
        size_t q_elems = (size_t)S_aligned * NUM_HEADS * HEAD_DIM;
        size_t kv_elems = (size_t)SKV * HEAD_DIM;

        nv_bfloat16 *q_bf16, *kv_bf16;
        KH_CUDA_CHECK(cudaMalloc(&q_bf16, q_elems * sizeof(nv_bfloat16)));
        KH_CUDA_CHECK(cudaMalloc(&kv_bf16, kv_elems * sizeof(nv_bfloat16)));
        fill_rand_bf16<<<(q_elems + 255) / 256, 256>>>(q_bf16, q_elems, 100 + S);
        fill_rand_bf16<<<(kv_elems + 255) / 256, 256>>>(kv_bf16, kv_elems, 200 + SKV);

        // Convert Q to FP8 (simple cast)
        __nv_fp8_e4m3* q_fp8;
        KH_CUDA_CHECK(cudaMalloc(&q_fp8, q_elems * sizeof(__nv_fp8_e4m3)));
        cast_bf16_to_fp8<<<(q_elems + 255) / 256, 256>>>(q_fp8, q_bf16, q_elems);

        // Quantize KV to FP8 with per-token scales
        __nv_fp8_e4m3* kv_fp8;
        float* kv_scales;
        KH_CUDA_CHECK(cudaMalloc(&kv_fp8, kv_elems * sizeof(__nv_fp8_e4m3)));
        KH_CUDA_CHECK(cudaMalloc(&kv_scales, SKV * sizeof(float)));
        quantize_kv_to_fp8<<<SKV, 32>>>(kv_fp8, kv_scales, kv_bf16, SKV, HEAD_DIM);

        // Weights: [S, num_heads]
        float* weights;
        KH_CUDA_CHECK(cudaMalloc(&weights, (size_t)S_aligned * NUM_HEADS * sizeof(float)));
        fill_rand_f32<<<((size_t)S_aligned * NUM_HEADS + 255) / 256, 256>>>(
            weights, S_aligned * NUM_HEADS, 300 + S);

        // Masking: simple causal — each query sees [0, q_idx + offset)
        int offset = SKV - S;
        int *h_ks = new int[S_aligned], *h_ke = new int[S_aligned];
        for (int i = 0; i < S_aligned; i++) {
            h_ks[i] = 0;
            h_ke[i] = (i < S) ? (i + offset) : 0;
        }
        int *d_ks, *d_ke;
        KH_CUDA_CHECK(cudaMalloc(&d_ks, S_aligned * sizeof(int)));
        KH_CUDA_CHECK(cudaMalloc(&d_ke, S_aligned * sizeof(int)));
        KH_CUDA_CHECK(cudaMemcpy(d_ks, h_ks, S_aligned * sizeof(int), cudaMemcpyHostToDevice));
        KH_CUDA_CHECK(cudaMemcpy(d_ke, h_ke, S_aligned * sizeof(int), cudaMemcpyHostToDevice));

        // Reference output
        float* logits_ref;
        KH_CUDA_CHECK(cudaMalloc(&logits_ref, (size_t)S_aligned * SKV * sizeof(float)));
        fill_neginf<<<((size_t)S_aligned * SKV + 255) / 256, 256>>>(logits_ref, S_aligned * SKV);
        KH_CUDA_CHECK(cudaDeviceSynchronize());

        run_ref_kernel(q_fp8, kv_fp8, kv_scales, weights,
                       d_ks, d_ke, logits_ref, S_aligned, SKV, num_sms);
        KH_CUDA_CHECK(cudaDeviceSynchronize());

#ifdef KH_TEST_SOLUTION
        float* logits_sol;
        KH_CUDA_CHECK(cudaMalloc(&logits_sol, (size_t)S_aligned * SKV * sizeof(float)));
        fill_neginf<<<((size_t)S_aligned * SKV + 255) / 256, 256>>>(logits_sol, S_aligned * SKV);
        KH_CUDA_CHECK(cudaDeviceSynchronize());

        solution_mqa_logits(q_fp8, kv_fp8, kv_scales, weights,
                            d_ks, d_ke, logits_sol,
                            S_aligned, SKV, NUM_HEADS, HEAD_DIM, 0);
        KH_CUDA_CHECK(cudaDeviceSynchronize());

        // Compare (skip -inf positions)
        {
            size_t sz = (size_t)S * SKV;
            float *h_ref = new float[sz], *h_sol = new float[sz];
            // Copy only the first S rows (not padded rows)
            for (int qi = 0; qi < S; qi++) {
                KH_CUDA_CHECK(cudaMemcpy(h_ref + qi * SKV,
                    logits_ref + qi * SKV, SKV * sizeof(float), cudaMemcpyDeviceToHost));
                KH_CUDA_CHECK(cudaMemcpy(h_sol + qi * SKV,
                    logits_sol + qi * SKV, SKV * sizeof(float), cudaMemcpyDeviceToHost));
            }

            int mismatches = 0;
            float max_rel_diff = 0;
            for (size_t i = 0; i < sz; i++) {
                if (isinf(h_ref[i]) && h_ref[i] < 0) {
                    if (!(isinf(h_sol[i]) && h_sol[i] < 0)) mismatches++;
                    continue;
                }
                float diff = fabsf(h_ref[i] - h_sol[i]);
                float denom = fmaxf(fabsf(h_ref[i]), 1e-6f);
                max_rel_diff = fmaxf(max_rel_diff, diff / denom);
            }
            printf("Max relative diff: %.6f, mask mismatches: %d\n", max_rel_diff, mismatches);
            if (max_rel_diff > 0.01f || mismatches > 0) {
                printf("FAILED\n");
                all_passed = false;
            }
            delete[] h_ref; delete[] h_sol;
        }

        if (all_passed) printf("Passed\n");

        if (benchmark_trials > 0) {
            kh_benchmark([&]() {
                solution_mqa_logits(q_fp8, kv_fp8, kv_scales, weights,
                                    d_ks, d_ke, logits_sol,
                                    S_aligned, SKV, NUM_HEADS, HEAD_DIM, 0);
            }, benchmark_trials, "Kernel");

            kh_benchmark([&]() {
                run_ref_kernel(q_fp8, kv_fp8, kv_scales, weights,
                               d_ks, d_ke, logits_ref, S_aligned, SKV, num_sms);
            }, benchmark_trials, "Ref");
        }

        KH_CUDA_CHECK(cudaFree(logits_sol));
#else
        printf("Passed\n");

        if (benchmark_trials > 0) {
            kh_benchmark([&]() {
                run_ref_kernel(q_fp8, kv_fp8, kv_scales, weights,
                               d_ks, d_ke, logits_ref, S_aligned, SKV, num_sms);
            }, benchmark_trials, "Ref");
        }
#endif

        KH_CUDA_CHECK(cudaFree(q_bf16));
        KH_CUDA_CHECK(cudaFree(kv_bf16));
        KH_CUDA_CHECK(cudaFree(q_fp8));
        KH_CUDA_CHECK(cudaFree(kv_fp8));
        KH_CUDA_CHECK(cudaFree(kv_scales));
        KH_CUDA_CHECK(cudaFree(weights));
        KH_CUDA_CHECK(cudaFree(d_ks));
        KH_CUDA_CHECK(cudaFree(d_ke));
        KH_CUDA_CHECK(cudaFree(logits_ref));
        delete[] h_ks; delete[] h_ke;
    }

    if (!all_passed) {
        fprintf(stderr, "Some tests FAILED\n");
        return 1;
    }
    return 0;
}
