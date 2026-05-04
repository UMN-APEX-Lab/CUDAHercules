/*
 * Standalone benchmark harness for ExaChem CCSD(T) fully-fused GPU kernel.
 *
 * Runs multiple test cases representing different molecular systems from the
 * ExaChem benchmark suite. Each test case is one kernel launch with dimensions
 * derived from real molecules (ccsdt_tilesize=40).
 *
 * Original kernel: ExaChem (Apache 2.0) - Pacific Northwest National Laboratory
 */

#include "standalone_common.cuh"
#include "kernel_interface.cuh"
#include <cmath>
#include <cstring>
#include <random>
#include <algorithm>

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

// Max constant memory limits (must match kernel)
#define MAX_NOAB 50
#define MAX_NVAB 140

// ============================================================================
// Test case definitions — real molecular systems, ccsdt_tilesize=40
// ============================================================================

struct TestCase {
    const char* name;
    int noab, nvab;           // number of occupied/virtual orbital blocks
    int h1, h2, h3;           // occupied tile dimensions for this task
    int p4, p5, p6;           // virtual tile dimensions for this task
    int h7_dim;               // occupied orbital block size for D1 inner loop
    int p7_dim;               // virtual orbital block size for D2 inner loop
};

// Each test case represents a single (h1b,h2b,h3b,p4b,p5b,p6b) task from a
// real molecule's CCSD(T) calculation.  Dimensions come from k_range[] which
// depends on the orbital count and tilesize=40.
static const TestCase TEST_CASES[] = {
    // ---- H2O / cc-pvdz (Oa=5, Va=19, tilesize=40) ----
    // noab=1, nvab=1 — smallest realistic system
    {"H2O_cc-pvdz",
     1, 1,           // noab, nvab
     5, 5, 5,        // h dims (occupied: 5 orbitals)
     19, 19, 19,     // p dims (virtual: 19 orbitals)
     5, 19},         // h7, p7

    // ---- Ubiquitin / 6-31g (Oa=146, Va=278, tilesize=40) ----
    // noab=4 (blocks: 40,40,40,26), nvab=7 (blocks: 40,40,40,40,40,40,38)
    // Task: all full blocks — medium system
    {"Ubiquitin_6-31g",
     4, 7,
     40, 40, 40,
     40, 40, 40,
     40, 40},

    // ---- Water-53 / cc-pvdz (Oa=235, Va=1007, tilesize=40) ----
    // noab=6, nvab=26 — most equations per kernel, partial edge blocks
    // Tests: high equation count (297 variants), non-aligned tile dimensions
    {"W53_cc-pvdz",
     6, 26,
     40, 40, 35,
     40, 40, 7,
     40, 40},
};

static const int NUM_TEST_CASES = sizeof(TEST_CASES) / sizeof(TEST_CASES[0]);

// ============================================================================
// Data generation helpers
// ============================================================================

static void generate_execution_flags(int noab, int nvab,
                                     int* exec_s1, int* exec_d1, int* exec_d2) {
    for (int i = 0; i < 9; i++)
        exec_s1[i] = i;

    int offset = 0;
    for (int n = 0; n < noab; n++)
        for (int i = 0; i < 9; i++)
            exec_d1[n * 9 + i] = offset++;

    offset = 0;
    for (int n = 0; n < nvab; n++)
        for (int i = 0; i < 9; i++)
            exec_d2[n * 9 + i] = offset++;
}

static void fill_random(double* data, size_t count, std::mt19937& rng) {
    std::normal_distribution<double> dist(0.0, 0.01);
    for (size_t i = 0; i < count; i++)
        data[i] = dist(rng);
}

static void fill_eigenvalues(double* evl, int size, double base, double step) {
    for (int i = 0; i < size; i++)
        evl[i] = base + i * step;
}

// ============================================================================
// Run a single test case
// ============================================================================

struct TestResult {
    double energy_1, energy_2;
    float kernel_ms;
    bool correct;
};

static TestResult run_test_case(const TestCase& tc, cudaStream_t stream) {
    TestResult result = {0, 0, 0, false};

    int max_hdim = std::max({tc.h1, tc.h2, tc.h3, tc.h7_dim});
    int max_pdim = std::max({tc.p4, tc.p5, tc.p6, tc.p7_dim});

    size_t max_dim_s1_t1 = (size_t)max_pdim * max_hdim;
    size_t max_dim_s1_v2 = (size_t)max_pdim * max_pdim * max_hdim * max_hdim;
    size_t max_dim_d1_t2 = (size_t)max_pdim * max_pdim * max_hdim * max_hdim;
    size_t max_dim_d1_v2 = (size_t)max_pdim * max_hdim * max_hdim * max_hdim;
    size_t max_dim_d2_t2 = (size_t)max_pdim * max_pdim * max_hdim * max_hdim;
    size_t max_dim_d2_v2 = (size_t)max_pdim * max_pdim * max_pdim * max_hdim;

    size_t size_s1_t1 = 9 * max_dim_s1_t1;
    size_t size_s1_v2 = 9 * max_dim_s1_v2;
    size_t size_d1_t2 = 9 * tc.noab * max_dim_d1_t2;
    size_t size_d1_v2 = 9 * tc.noab * max_dim_d1_v2;
    size_t size_d2_t2 = 9 * tc.nvab * max_dim_d2_t2;
    size_t size_d2_v2 = 9 * tc.nvab * max_dim_d2_v2;

    size_t numBlks = (size_t)CEIL_DIV(tc.h3, 4) * CEIL_DIV(tc.h2, 4) *
                     CEIL_DIV(tc.h1, 4) * CEIL_DIV(tc.p6, 4) *
                     CEIL_DIV(tc.p5, 4) * CEIL_DIV(tc.p4, 4);

    size_t total_bytes = (size_s1_t1 + size_s1_v2 + size_d1_t2 + size_d1_v2 +
                          size_d2_t2 + size_d2_v2) * sizeof(double);

    printf("  [%s] noab=%d nvab=%d h=(%d,%d,%d) p=(%d,%d,%d)\n",
           tc.name, tc.noab, tc.nvab, tc.h1, tc.h2, tc.h3, tc.p4, tc.p5, tc.p6);
    printf("  Grid blocks: %zu, Tensor memory: %.1f MB\n", numBlks, total_bytes / 1e6);

    // Generate host data with reproducible seed per test case
    std::mt19937 rng(42 + tc.noab * 1000 + tc.nvab);

    // Execution flags
    int exec_s1[9];
    int exec_d1[9 * MAX_NOAB];
    int exec_d2[9 * MAX_NVAB];
    memset(exec_d1, -1, sizeof(exec_d1));
    memset(exec_d2, -1, sizeof(exec_d2));
    generate_execution_flags(tc.noab, tc.nvab, exec_s1, exec_d1, exec_d2);

    // h7b/p7b dimensions
    int h7b_dims[MAX_NOAB];
    int p7b_dims[MAX_NVAB];
    for (int i = 0; i < tc.noab; i++) h7b_dims[i] = tc.h7_dim;
    for (int i = 0; i < tc.nvab; i++) p7b_dims[i] = tc.p7_dim;

    // Allocate and fill host tensors
    double* h_s1_t1 = new double[size_s1_t1];
    double* h_s1_v2 = new double[size_s1_v2];
    double* h_d1_t2 = new double[size_d1_t2];
    double* h_d1_v2 = new double[size_d1_v2];
    double* h_d2_t2 = new double[size_d2_t2];
    double* h_d2_v2 = new double[size_d2_v2];

    fill_random(h_s1_t1, size_s1_t1, rng);
    fill_random(h_s1_v2, size_s1_v2, rng);
    fill_random(h_d1_t2, size_d1_t2, rng);
    fill_random(h_d1_v2, size_d1_v2, rng);
    fill_random(h_d2_t2, size_d2_t2, rng);
    fill_random(h_d2_v2, size_d2_v2, rng);

    // Eigenvalues
    int max_evl_dim = std::max(max_hdim, max_pdim);
    double* h_evl_h1 = new double[max_evl_dim];
    double* h_evl_h2 = new double[max_evl_dim];
    double* h_evl_h3 = new double[max_evl_dim];
    double* h_evl_p4 = new double[max_evl_dim];
    double* h_evl_p5 = new double[max_evl_dim];
    double* h_evl_p6 = new double[max_evl_dim];
    fill_eigenvalues(h_evl_h1, tc.h1, -1.5, -0.1);
    fill_eigenvalues(h_evl_h2, tc.h2, -1.5, -0.1);
    fill_eigenvalues(h_evl_h3, tc.h3, -1.5, -0.1);
    fill_eigenvalues(h_evl_p4, tc.p4, 0.5, 0.1);
    fill_eigenvalues(h_evl_p5, tc.p5, 0.5, 0.1);
    fill_eigenvalues(h_evl_p6, tc.p6, 0.5, 0.1);

    // GPU allocations
    double *d_s1_t1, *d_s1_v2, *d_d1_t2, *d_d1_v2, *d_d2_t2, *d_d2_v2;
    double *d_energies;
    double *d_evl_h1, *d_evl_h2, *d_evl_h3, *d_evl_p4, *d_evl_p5, *d_evl_p6;

    CUDA_SAFE(cudaMalloc(&d_s1_t1, size_s1_t1 * sizeof(double)));
    CUDA_SAFE(cudaMalloc(&d_s1_v2, size_s1_v2 * sizeof(double)));
    CUDA_SAFE(cudaMalloc(&d_d1_t2, size_d1_t2 * sizeof(double)));
    CUDA_SAFE(cudaMalloc(&d_d1_v2, size_d1_v2 * sizeof(double)));
    CUDA_SAFE(cudaMalloc(&d_d2_t2, size_d2_t2 * sizeof(double)));
    CUDA_SAFE(cudaMalloc(&d_d2_v2, size_d2_v2 * sizeof(double)));
    CUDA_SAFE(cudaMalloc(&d_energies, numBlks * 2 * sizeof(double)));
    CUDA_SAFE(cudaMalloc(&d_evl_h1, max_evl_dim * sizeof(double)));
    CUDA_SAFE(cudaMalloc(&d_evl_h2, max_evl_dim * sizeof(double)));
    CUDA_SAFE(cudaMalloc(&d_evl_h3, max_evl_dim * sizeof(double)));
    CUDA_SAFE(cudaMalloc(&d_evl_p4, max_evl_dim * sizeof(double)));
    CUDA_SAFE(cudaMalloc(&d_evl_p5, max_evl_dim * sizeof(double)));
    CUDA_SAFE(cudaMalloc(&d_evl_p6, max_evl_dim * sizeof(double)));

    // Copy to GPU
    CUDA_SAFE(cudaMemcpyAsync(d_s1_t1, h_s1_t1, size_s1_t1 * sizeof(double), cudaMemcpyHostToDevice, stream));
    CUDA_SAFE(cudaMemcpyAsync(d_s1_v2, h_s1_v2, size_s1_v2 * sizeof(double), cudaMemcpyHostToDevice, stream));
    CUDA_SAFE(cudaMemcpyAsync(d_d1_t2, h_d1_t2, size_d1_t2 * sizeof(double), cudaMemcpyHostToDevice, stream));
    CUDA_SAFE(cudaMemcpyAsync(d_d1_v2, h_d1_v2, size_d1_v2 * sizeof(double), cudaMemcpyHostToDevice, stream));
    CUDA_SAFE(cudaMemcpyAsync(d_d2_t2, h_d2_t2, size_d2_t2 * sizeof(double), cudaMemcpyHostToDevice, stream));
    CUDA_SAFE(cudaMemcpyAsync(d_d2_v2, h_d2_v2, size_d2_v2 * sizeof(double), cudaMemcpyHostToDevice, stream));
    CUDA_SAFE(cudaMemcpyAsync(d_evl_h1, h_evl_h1, tc.h1 * sizeof(double), cudaMemcpyHostToDevice, stream));
    CUDA_SAFE(cudaMemcpyAsync(d_evl_h2, h_evl_h2, tc.h2 * sizeof(double), cudaMemcpyHostToDevice, stream));
    CUDA_SAFE(cudaMemcpyAsync(d_evl_h3, h_evl_h3, tc.h3 * sizeof(double), cudaMemcpyHostToDevice, stream));
    CUDA_SAFE(cudaMemcpyAsync(d_evl_p4, h_evl_p4, tc.p4 * sizeof(double), cudaMemcpyHostToDevice, stream));
    CUDA_SAFE(cudaMemcpyAsync(d_evl_p5, h_evl_p5, tc.p5 * sizeof(double), cudaMemcpyHostToDevice, stream));
    CUDA_SAFE(cudaMemcpyAsync(d_evl_p6, h_evl_p6, tc.p6 * sizeof(double), cudaMemcpyHostToDevice, stream));
    CUDA_SAFE(cudaStreamSynchronize(stream));

    // Warmup (1 iteration)
    CUDA_SAFE(cudaMemsetAsync(d_energies, 0, numBlks * 2 * sizeof(double), stream));
    launch_ccsd_t_kernel(
        stream, numBlks, tc.h3, tc.h2, tc.h1, tc.p6, tc.p5, tc.p4,
        d_s1_t1, d_s1_v2, d_d1_t2, d_d1_v2, d_d2_t2, d_d2_v2,
        h7b_dims, p7b_dims, exec_s1, exec_d1, exec_d2,
        tc.noab, tc.nvab,
        max_dim_s1_t1, max_dim_s1_v2, max_dim_d1_t2, max_dim_d1_v2,
        max_dim_d2_t2, max_dim_d2_v2,
        d_evl_h1, d_evl_h2, d_evl_h3, d_evl_p4, d_evl_p5, d_evl_p6,
        d_energies);
    CUDA_SAFE(cudaStreamSynchronize(stream));

    // Read energies for correctness check
    std::vector<double> host_energies(numBlks * 2);
    CUDA_SAFE(cudaMemcpy(host_energies.data(), d_energies, numBlks * 2 * sizeof(double),
                         cudaMemcpyDeviceToHost));

    result.energy_1 = 0.0;
    result.energy_2 = 0.0;
    for (size_t i = 0; i < numBlks; i++) {
        result.energy_1 += host_energies[i];
        result.energy_2 += host_energies[i + numBlks];
    }

    result.correct = std::isfinite(result.energy_1) && std::isfinite(result.energy_2) &&
                     !(result.energy_1 == 0.0 && result.energy_2 == 0.0);

    printf("  Energy_T:  %.15e\n", result.energy_1);
    printf("  Energy_T5: %.15e\n", result.energy_2);

    if (!result.correct) {
        fprintf(stderr, "  FAIL: invalid energy values\n");
    }

    // Timed run (1 iteration — each test case runs once, like real CCSD(T))
    cudaEvent_t ev_start, ev_stop;
    CUDA_SAFE(cudaEventCreate(&ev_start));
    CUDA_SAFE(cudaEventCreate(&ev_stop));

    CUDA_SAFE(cudaMemsetAsync(d_energies, 0, numBlks * 2 * sizeof(double), stream));
    CUDA_SAFE(cudaEventRecord(ev_start, stream));
    launch_ccsd_t_kernel(
        stream, numBlks, tc.h3, tc.h2, tc.h1, tc.p6, tc.p5, tc.p4,
        d_s1_t1, d_s1_v2, d_d1_t2, d_d1_v2, d_d2_t2, d_d2_v2,
        h7b_dims, p7b_dims, exec_s1, exec_d1, exec_d2,
        tc.noab, tc.nvab,
        max_dim_s1_t1, max_dim_s1_v2, max_dim_d1_t2, max_dim_d1_v2,
        max_dim_d2_t2, max_dim_d2_v2,
        d_evl_h1, d_evl_h2, d_evl_h3, d_evl_p4, d_evl_p5, d_evl_p6,
        d_energies);
    CUDA_SAFE(cudaEventRecord(ev_stop, stream));
    CUDA_SAFE(cudaEventSynchronize(ev_stop));

    CUDA_SAFE(cudaEventElapsedTime(&result.kernel_ms, ev_start, ev_stop));
    printf("  Time: %.3f ms\n", result.kernel_ms);

    // Cleanup
    CUDA_SAFE(cudaEventDestroy(ev_start));
    CUDA_SAFE(cudaEventDestroy(ev_stop));
    CUDA_SAFE(cudaFree(d_s1_t1)); CUDA_SAFE(cudaFree(d_s1_v2));
    CUDA_SAFE(cudaFree(d_d1_t2)); CUDA_SAFE(cudaFree(d_d1_v2));
    CUDA_SAFE(cudaFree(d_d2_t2)); CUDA_SAFE(cudaFree(d_d2_v2));
    CUDA_SAFE(cudaFree(d_energies));
    CUDA_SAFE(cudaFree(d_evl_h1)); CUDA_SAFE(cudaFree(d_evl_h2));
    CUDA_SAFE(cudaFree(d_evl_h3)); CUDA_SAFE(cudaFree(d_evl_p4));
    CUDA_SAFE(cudaFree(d_evl_p5)); CUDA_SAFE(cudaFree(d_evl_p6));

    delete[] h_s1_t1; delete[] h_s1_v2;
    delete[] h_d1_t2; delete[] h_d1_v2;
    delete[] h_d2_t2; delete[] h_d2_v2;
    delete[] h_evl_h1; delete[] h_evl_h2; delete[] h_evl_h3;
    delete[] h_evl_p4; delete[] h_evl_p5; delete[] h_evl_p6;

    return result;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    int device = 0;
    CUDA_SAFE(cudaSetDevice(device));

    cudaDeviceProp prop;
    CUDA_SAFE(cudaGetDeviceProperties(&prop, device));
    printf("Device: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);

    if (prop.major < 8) {
        fprintf(stderr, "Error: This benchmark requires SM 8.0+ for FP64 Tensor Cores\n");
        return 1;
    }

    cudaStream_t stream;
    CUDA_SAFE(cudaStreamCreate(&stream));

    bool all_correct = true;
    float total_kernel_ms = 0;

    printf("\nRunning %d test cases (CCSD(T) tasks from real molecules):\n\n", NUM_TEST_CASES);

    for (int t = 0; t < NUM_TEST_CASES; t++) {
        printf("Test %d/%d: %s\n", t + 1, NUM_TEST_CASES, TEST_CASES[t].name);
        TestResult r = run_test_case(TEST_CASES[t], stream);

        printf("  Energy_T[%d]:  %.15e\n", t, r.energy_1);
        printf("  Energy_T5[%d]: %.15e\n", t, r.energy_2);

        if (!r.correct) all_correct = false;
        total_kernel_ms += r.kernel_ms;
        printf("\n");
    }

    CUDA_SAFE(cudaStreamDestroy(stream));

    printf("========================================\n");
    printf("Total kernel time across all test cases:\n");
    printf("Kernel time: %.3f ms\n", total_kernel_ms);

    if (all_correct) {
        printf("Passed\n");
    } else {
        printf("FAILED\n");
    }

    return all_correct ? 0 : 1;
}
