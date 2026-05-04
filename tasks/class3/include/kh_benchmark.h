#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// Check environment variable KH_BENCHMARK for number of timing trials.
// Usage: KH_BENCHMARK=20 ./test
// If not set or 0, no benchmarking is done.
static inline int kh_benchmark_trials() {
    const char* env = getenv("KH_BENCHMARK");
    if (env) return atoi(env);
    return 0;
}

// Time a kernel call using CUDA events.
// func: a callable (lambda) that launches the kernel.
// trials: number of timed iterations.
// label: "Kernel" for LLM solution, "Ref" for reference kernel.
// Prints: "<label> time: X.XXXX ms (avg over N trials, min: X.XXXX ms)"
template <typename Func>
static void kh_benchmark(Func&& func, int trials, const char* label = "Kernel") {
    // Warmup
    for (int i = 0; i < 3; i++) {
        func();
        cudaDeviceSynchronize();
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float total_ms = 0;
    float min_ms = 1e30f;

    for (int i = 0; i < trials; i++) {
        cudaEventRecord(start);
        func();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        total_ms += ms;
        if (ms < min_ms) min_ms = ms;
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    fprintf(stdout, "%s time: %.4f ms (avg over %d trials, min: %.4f ms)\n",
            label, total_ms / trials, trials, min_ms);
}
