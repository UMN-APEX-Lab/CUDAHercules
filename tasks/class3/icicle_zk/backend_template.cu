// Minimal custom backend skeleton for the fixed Icicle ZK ABI.
// The benchmark compiles this file into a .so and calls the exported symbols
// declared in custom_backend_api.h.

#include <cuda_runtime.h>
#include <cstdio>

#include "custom_backend_api.h"

namespace {

char g_last_error[256] = "backend not initialized";

__global__ void kh_warmup_kernel() {}

void set_last_error(const char* message) {
    std::snprintf(g_last_error, sizeof(g_last_error), "%s", message ? message : "unknown error");
}

}  // namespace

extern "C" int kh_custom_backend_init() {
    kh_warmup_kernel<<<1, 1>>>();
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        set_last_error(cudaGetErrorString(err));
        return 1;
    }
    set_last_error("ok");
    return 0;
}

extern "C" void kh_custom_backend_shutdown() {}

extern "C" const char* kh_custom_backend_last_error() { return g_last_error; }

extern "C" int kh_custom_ntt_forward_bn254(
    const bn254::scalar_t* input,
    int log_n,
    bn254::scalar_t* output) {
    (void)input;
    (void)log_n;
    (void)output;
    set_last_error("kh_custom_ntt_forward_bn254 is not implemented");
    return 1;
}

extern "C" int kh_custom_msm_bn254(
    const bn254::scalar_t* scalars,
    const bn254::affine_t* points,
    int log_n,
    bn254::projective_t* output) {
    (void)scalars;
    (void)points;
    (void)log_n;
    (void)output;
    set_last_error("kh_custom_msm_bn254 is not implemented");
    return 1;
}
