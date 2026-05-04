#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

#include "include/cuSZp/cuSZp_entry_1D_f32.h"
#include "include/cuSZp/cuSZp_entry_1D_f64.h"
#include "include/cuSZp/cuSZp_entry_2D_f32.h"
#include "include/cuSZp/cuSZp_entry_2D_f64.h"
#include "include/cuSZp/cuSZp_entry_3D_f32.h"
#include "include/cuSZp/cuSZp_entry_3D_f64.h"
#include "include/cuSZp/cuSZp_timer.h"

// Data sizes: 2 GB per precision
static const size_t F32_NBELE = 512ULL * 1024 * 1024;   // 512M x 4B = 2GB
static const size_t F64_NBELE = 256ULL * 1024 * 1024;   // 256M x 8B = 2GB

// REL error bounds tested (following cuSZp SC'23/SC'24/SC'25 papers)
// Applied to value range [-20, 20] = 40 → abs bounds: {0.4, 0.04, 0.004}
static const int NUM_REL_EBS = 3;
static const double REL_EBS[3] = {1E-2, 1E-3, 1E-4};
static const char* REL_EB_NAMES[3] = {"1E-2", "1E-3", "1E-4"};
static const double VALUE_RANGE = 40.0;  // max - min = 20 - (-20)

struct ModeResult {
    const char* variant;     // e.g. "1D_f32"
    const char* mode;        // "fixed", "plain", "outlier"
    const char* rel_eb_name; // "1E-2", "1E-3", "1E-4"
    float cmpTimeMs;
    float decTimeMs;
    float cmpRatio;
    int   errorCount;
    double maxError;
    double errorBound;       // absolute error bound
    size_t nbEle;
    size_t dataSzBytes;
};

static int nResults = 0;
static ModeResult allResults[128]; // 6 variants x 3 modes x 3 eb = 54
static float totalKernelTime = 0.0f;

// Copy compressed data through CPU to verify round-trip integrity
static void round_trip(unsigned char* d_cmp, size_t cmpSize, size_t bufSize) {
    unsigned char* dup = (unsigned char*)malloc(cmpSize);
    cudaMemcpy(dup, d_cmp, cmpSize, cudaMemcpyDeviceToHost);
    cudaMemset(d_cmp, 0, bufSize);
    cudaMemcpy(d_cmp, dup, cmpSize, cudaMemcpyHostToDevice);
    free(dup);
}

// Check errors with FP tolerance (1e-6 relative, same spirit as cuSZp's 1.1x
// but much stricter — only allows IEEE 754 rounding, not algorithmic slack)
template<typename T>
static void check_errors(const T* ori, const T* dec, size_t n, T eb,
                         int& errorCount, double& maxError) {
    errorCount = 0;
    maxError = 0.0;
    T tol = eb * (T)(1.0 + 1e-6);
    for (size_t i = 0; i < n; i++) {
        T diff = ori[i] > dec[i] ? ori[i] - dec[i] : dec[i] - ori[i];
        double d = (double)diff;
        if (d > maxError) maxError = d;
        if (diff > tol) errorCount++;
    }
}

// Print structured per-kernel result line (machine-parseable)
static void report_kernel(const ModeResult& r) {
    double dataMB = (double)r.dataSzBytes / (1024.0 * 1024.0);
    float cmpGBs = (float)(dataMB / r.cmpTimeMs);
    float decGBs = (float)(dataMB / r.decTimeMs);
    int correct = (r.errorCount == 0) ? 1 : 0;
    double errRatio = (r.errorBound > 0) ? r.maxError / r.errorBound : 0;

    printf("KERNEL %s %s eb=%s: correct=%d errors=%d max_error=%.16e error_bound=%.16e "
           "err_ratio=%.10f cmp_ms=%.4f dec_ms=%.4f ratio=%.4f "
           "cmp_gbps=%.2f dec_gbps=%.2f nbEle=%zu\n",
           r.variant, r.mode, r.rel_eb_name, correct, r.errorCount,
           r.maxError, r.errorBound, errRatio,
           r.cmpTimeMs, r.decTimeMs, r.cmpRatio,
           cmpGBs, decGBs, r.nbEle);
}

// ============================================================
// Templated test runners
// ============================================================
template<typename T>
using cmp_1d_fn = void(*)(T*, unsigned char*, size_t, size_t*, T, cudaStream_t);
template<typename T>
using dec_1d_fn = void(*)(T*, unsigned char*, size_t, size_t, T, cudaStream_t);

template<typename T>
static ModeResult test_1d(const char* variant, const char* mode, const char* eb_name,
    cmp_1d_fn<T> cmp, dec_1d_fn<T> dec,
    T* d_ori, T* d_dec, unsigned char* d_cmp,
    const T* h_ori, T* h_dec,
    size_t nbEle, T eb, cudaStream_t stream)
{
    ModeResult r = {};
    r.variant = variant; r.mode = mode; r.rel_eb_name = eb_name;
    r.errorBound = (double)eb; r.nbEle = nbEle;
    r.dataSzBytes = nbEle * sizeof(T);

    size_t bufSz = r.dataSzBytes;
    size_t cmpSize = 0;
    TimingGPU timer;

    timer.StartCounter();
    cmp(d_ori, d_cmp, nbEle, &cmpSize, eb, stream);
    r.cmpTimeMs = timer.GetCounter();
    round_trip(d_cmp, cmpSize, bufSz);
    timer.StartCounter();
    dec(d_dec, d_cmp, nbEle, cmpSize, eb, stream);
    r.decTimeMs = timer.GetCounter();

    r.cmpRatio = (float)((double)bufSz / (double)cmpSize);
    cudaMemcpy(h_dec, d_dec, bufSz, cudaMemcpyDeviceToHost);
    check_errors(h_ori, h_dec, nbEle, eb, r.errorCount, r.maxError);

    cudaMemset(d_cmp, 0, bufSz);
    cudaMemset(d_dec, 0, bufSz);

    totalKernelTime += r.cmpTimeMs + r.decTimeMs;
    return r;
}

template<typename T>
using cmp_nd_fn = void(*)(T*, unsigned char*, size_t, size_t*, uint3, T, cudaStream_t);
template<typename T>
using dec_nd_fn = void(*)(T*, unsigned char*, size_t, size_t, uint3, T, cudaStream_t);

template<typename T>
static ModeResult test_nd(const char* variant, const char* mode, const char* eb_name,
    cmp_nd_fn<T> cmp, dec_nd_fn<T> dec,
    T* d_ori, T* d_dec, unsigned char* d_cmp,
    const T* h_ori, T* h_dec,
    size_t nbEle, uint3 dims, T eb, cudaStream_t stream)
{
    ModeResult r = {};
    r.variant = variant; r.mode = mode; r.rel_eb_name = eb_name;
    r.errorBound = (double)eb; r.nbEle = nbEle;
    r.dataSzBytes = nbEle * sizeof(T);

    size_t bufSz = r.dataSzBytes;
    size_t cmpSize = 0;
    TimingGPU timer;

    timer.StartCounter();
    cmp(d_ori, d_cmp, nbEle, &cmpSize, dims, eb, stream);
    r.cmpTimeMs = timer.GetCounter();
    round_trip(d_cmp, cmpSize, bufSz);
    timer.StartCounter();
    dec(d_dec, d_cmp, nbEle, cmpSize, dims, eb, stream);
    r.decTimeMs = timer.GetCounter();

    r.cmpRatio = (float)((double)bufSz / (double)cmpSize);
    cudaMemcpy(h_dec, d_dec, bufSz, cudaMemcpyDeviceToHost);
    check_errors(h_ori, h_dec, nbEle, eb, r.errorCount, r.maxError);

    cudaMemset(d_cmp, 0, bufSz);
    cudaMemset(d_dec, 0, bufSz);

    totalKernelTime += r.cmpTimeMs + r.decTimeMs;
    return r;
}

// ============================================================
// Float32 suite: 3 error bounds x 3 dims x 3 modes = 27 tests
// ============================================================
static void run_f32_tests(cudaStream_t stream) {
    size_t nbEle = F32_NBELE;
    size_t dataSz = nbEle * sizeof(float);

    printf("=== Float32 tests (%zu elements, %.1f GB) ===\n",
           nbEle, (double)dataSz / (1024.0 * 1024.0 * 1024.0));

    float* h_ori = (float*)malloc(dataSz);
    float* h_dec = (float*)malloc(dataSz);
    if (!h_ori || !h_dec) { fprintf(stderr, "CPU malloc failed\n"); exit(1); }

    float val = -20.0f, step = 0.1f;
    for (size_t i = 0; i < nbEle; i++) {
        h_ori[i] = val; val += step;
        if (val > 20.0f) val = -20.0f;
    }

    float *d_ori, *d_dec;
    unsigned char* d_cmp;
    cudaMalloc(&d_ori, dataSz);
    cudaMalloc(&d_dec, dataSz);
    cudaMalloc(&d_cmp, dataSz);
    cudaMemcpy(d_ori, h_ori, dataSz, cudaMemcpyHostToDevice);

    // Warmup
    float wb = (float)(VALUE_RANGE * REL_EBS[0]);
    size_t dummy;
    for (int i = 0; i < 3; i++) {
        cuSZp_compress_1D_fixed_f32(d_ori, d_cmp, nbEle, &dummy, wb, stream);
        cuSZp_decompress_1D_fixed_f32(d_dec, d_cmp, nbEle, dummy, wb, stream);
        cudaMemset(d_cmp, 0, dataSz);
    }

    uint3 dims_2d = {512, 1024, 1024};
    uint3 dims_3d = {512, 512, 2048};

    for (int ei = 0; ei < NUM_REL_EBS; ei++) {
        float eb = (float)(VALUE_RANGE * REL_EBS[ei]);
        const char* eb_name = REL_EB_NAMES[ei];

        printf("\n--- f32 REL %s (abs eb=%.6f) ---\n", eb_name, eb);

        // 1D
        allResults[nResults] = test_1d<float>("1D_f32", "fixed", eb_name,
            cuSZp_compress_1D_fixed_f32, cuSZp_decompress_1D_fixed_f32,
            d_ori, d_dec, d_cmp, h_ori, h_dec, nbEle, eb, stream);
        report_kernel(allResults[nResults]); nResults++;

        allResults[nResults] = test_1d<float>("1D_f32", "plain", eb_name,
            cuSZp_compress_1D_plain_f32, cuSZp_decompress_1D_plain_f32,
            d_ori, d_dec, d_cmp, h_ori, h_dec, nbEle, eb, stream);
        report_kernel(allResults[nResults]); nResults++;

        allResults[nResults] = test_1d<float>("1D_f32", "outlier", eb_name,
            cuSZp_compress_1D_outlier_f32, cuSZp_decompress_1D_outlier_f32,
            d_ori, d_dec, d_cmp, h_ori, h_dec, nbEle, eb, stream);
        report_kernel(allResults[nResults]); nResults++;

        // 2D
        allResults[nResults] = test_nd<float>("2D_f32", "fixed", eb_name,
            cuSZp_compress_2D_fixed_f32, cuSZp_decompress_2D_fixed_f32,
            d_ori, d_dec, d_cmp, h_ori, h_dec, nbEle, dims_2d, eb, stream);
        report_kernel(allResults[nResults]); nResults++;

        allResults[nResults] = test_nd<float>("2D_f32", "plain", eb_name,
            cuSZp_compress_2D_plain_f32, cuSZp_decompress_2D_plain_f32,
            d_ori, d_dec, d_cmp, h_ori, h_dec, nbEle, dims_2d, eb, stream);
        report_kernel(allResults[nResults]); nResults++;

        allResults[nResults] = test_nd<float>("2D_f32", "outlier", eb_name,
            cuSZp_compress_2D_outlier_f32, cuSZp_decompress_2D_outlier_f32,
            d_ori, d_dec, d_cmp, h_ori, h_dec, nbEle, dims_2d, eb, stream);
        report_kernel(allResults[nResults]); nResults++;

        // 3D
        allResults[nResults] = test_nd<float>("3D_f32", "fixed", eb_name,
            cuSZp_compress_3D_fixed_f32, cuSZp_decompress_3D_fixed_f32,
            d_ori, d_dec, d_cmp, h_ori, h_dec, nbEle, dims_3d, eb, stream);
        report_kernel(allResults[nResults]); nResults++;

        allResults[nResults] = test_nd<float>("3D_f32", "plain", eb_name,
            cuSZp_compress_3D_plain_f32, cuSZp_decompress_3D_plain_f32,
            d_ori, d_dec, d_cmp, h_ori, h_dec, nbEle, dims_3d, eb, stream);
        report_kernel(allResults[nResults]); nResults++;

        allResults[nResults] = test_nd<float>("3D_f32", "outlier", eb_name,
            cuSZp_compress_3D_outlier_f32, cuSZp_decompress_3D_outlier_f32,
            d_ori, d_dec, d_cmp, h_ori, h_dec, nbEle, dims_3d, eb, stream);
        report_kernel(allResults[nResults]); nResults++;
    }

    cudaFree(d_ori); cudaFree(d_dec); cudaFree(d_cmp);
    free(h_ori); free(h_dec);
}

// ============================================================
// Float64 suite: 3 error bounds x 3 dims x 3 modes = 27 tests
// ============================================================
static void run_f64_tests(cudaStream_t stream) {
    size_t nbEle = F64_NBELE;
    size_t dataSz = nbEle * sizeof(double);

    printf("\n=== Float64 tests (%zu elements, %.1f GB) ===\n",
           nbEle, (double)dataSz / (1024.0 * 1024.0 * 1024.0));

    double* h_ori = (double*)malloc(dataSz);
    double* h_dec = (double*)malloc(dataSz);
    if (!h_ori || !h_dec) { fprintf(stderr, "CPU malloc failed\n"); exit(1); }

    double val = -20.0, step = 0.1;
    for (size_t i = 0; i < nbEle; i++) {
        h_ori[i] = val; val += step;
        if (val > 20.0) val = -20.0;
    }

    double *d_ori, *d_dec;
    unsigned char* d_cmp;
    cudaMalloc(&d_ori, dataSz);
    cudaMalloc(&d_dec, dataSz);
    cudaMalloc(&d_cmp, dataSz);
    cudaMemcpy(d_ori, h_ori, dataSz, cudaMemcpyHostToDevice);

    // Warmup
    double wb = VALUE_RANGE * REL_EBS[0];
    size_t dummy;
    for (int i = 0; i < 3; i++) {
        cuSZp_compress_1D_fixed_f64(d_ori, d_cmp, nbEle, &dummy, wb, stream);
        cuSZp_decompress_1D_fixed_f64(d_dec, d_cmp, nbEle, dummy, wb, stream);
        cudaMemset(d_cmp, 0, dataSz);
    }

    uint3 dims_2d = {256, 1024, 1024};
    uint3 dims_3d = {256, 512, 2048};

    for (int ei = 0; ei < NUM_REL_EBS; ei++) {
        double eb = VALUE_RANGE * REL_EBS[ei];
        const char* eb_name = REL_EB_NAMES[ei];

        printf("\n--- f64 REL %s (abs eb=%.6f) ---\n", eb_name, eb);

        // 1D
        allResults[nResults] = test_1d<double>("1D_f64", "fixed", eb_name,
            cuSZp_compress_1D_fixed_f64, cuSZp_decompress_1D_fixed_f64,
            d_ori, d_dec, d_cmp, h_ori, h_dec, nbEle, eb, stream);
        report_kernel(allResults[nResults]); nResults++;

        allResults[nResults] = test_1d<double>("1D_f64", "plain", eb_name,
            cuSZp_compress_1D_plain_f64, cuSZp_decompress_1D_plain_f64,
            d_ori, d_dec, d_cmp, h_ori, h_dec, nbEle, eb, stream);
        report_kernel(allResults[nResults]); nResults++;

        allResults[nResults] = test_1d<double>("1D_f64", "outlier", eb_name,
            cuSZp_compress_1D_outlier_f64, cuSZp_decompress_1D_outlier_f64,
            d_ori, d_dec, d_cmp, h_ori, h_dec, nbEle, eb, stream);
        report_kernel(allResults[nResults]); nResults++;

        // 2D
        allResults[nResults] = test_nd<double>("2D_f64", "fixed", eb_name,
            cuSZp_compress_2D_fixed_f64, cuSZp_decompress_2D_fixed_f64,
            d_ori, d_dec, d_cmp, h_ori, h_dec, nbEle, dims_2d, eb, stream);
        report_kernel(allResults[nResults]); nResults++;

        allResults[nResults] = test_nd<double>("2D_f64", "plain", eb_name,
            cuSZp_compress_2D_plain_f64, cuSZp_decompress_2D_plain_f64,
            d_ori, d_dec, d_cmp, h_ori, h_dec, nbEle, dims_2d, eb, stream);
        report_kernel(allResults[nResults]); nResults++;

        allResults[nResults] = test_nd<double>("2D_f64", "outlier", eb_name,
            cuSZp_compress_2D_outlier_f64, cuSZp_decompress_2D_outlier_f64,
            d_ori, d_dec, d_cmp, h_ori, h_dec, nbEle, dims_2d, eb, stream);
        report_kernel(allResults[nResults]); nResults++;

        // 3D
        allResults[nResults] = test_nd<double>("3D_f64", "fixed", eb_name,
            cuSZp_compress_3D_fixed_f64, cuSZp_decompress_3D_fixed_f64,
            d_ori, d_dec, d_cmp, h_ori, h_dec, nbEle, dims_3d, eb, stream);
        report_kernel(allResults[nResults]); nResults++;

        allResults[nResults] = test_nd<double>("3D_f64", "plain", eb_name,
            cuSZp_compress_3D_plain_f64, cuSZp_decompress_3D_plain_f64,
            d_ori, d_dec, d_cmp, h_ori, h_dec, nbEle, dims_3d, eb, stream);
        report_kernel(allResults[nResults]); nResults++;

        allResults[nResults] = test_nd<double>("3D_f64", "outlier", eb_name,
            cuSZp_compress_3D_outlier_f64, cuSZp_decompress_3D_outlier_f64,
            d_ori, d_dec, d_cmp, h_ori, h_dec, nbEle, dims_3d, eb, stream);
        report_kernel(allResults[nResults]); nResults++;
    }

    cudaFree(d_ori); cudaFree(d_dec); cudaFree(d_cmp);
    free(h_ori); free(h_dec);
}

// ============================================================
// Main
// ============================================================
int main() {
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    run_f32_tests(stream);
    run_f64_tests(stream);

    // Summary
    printf("\n=== Summary ===\n");
    int allPassed = 1;
    int nPassed = 0;
    for (int i = 0; i < nResults; i++) {
        if (allResults[i].errorCount > 0) {
            printf("  FAIL: %s %s eb=%s (%d errors, max_error/eb=%.6f)\n",
                   allResults[i].variant, allResults[i].mode, allResults[i].rel_eb_name,
                   allResults[i].errorCount,
                   allResults[i].maxError / allResults[i].errorBound);
            allPassed = 0;
        } else {
            nPassed++;
        }
    }
    printf("Passed kernels: %d/%d\n", nPassed, nResults);

    if (allPassed)
        printf("Passed\n");
    else
        printf("FAILED\n");

    printf("Kernel time: %.4f ms\n", totalKernelTime);

    cudaStreamDestroy(stream);
    return allPassed ? 0 : 1;
}
