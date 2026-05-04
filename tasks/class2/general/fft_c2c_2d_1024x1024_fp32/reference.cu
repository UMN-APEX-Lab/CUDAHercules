/*
 * cuFFT reference: 2D Complex-to-Complex FFT, 1024x1024, fp32, batch=1
 */
#include <cuda_runtime.h>
#include <cufft.h>
#include <cstdint>

static cufftHandle cached_plan = 0;
static int cached_N = 0;
static int cached_batch = 0;

extern "C"
void launch_fft_c2c_2d_1024x1024_fp32(
    const void* input,
    void* output,
    int N,
    int batch_size,
    int inverse,
    cudaStream_t stream
) {
    if (cached_plan == 0 || cached_N != N || cached_batch != batch_size) {
        if (cached_plan) cufftDestroy(cached_plan);
        int dims[2] = { 1024, 1024 };
        cufftPlanMany(&cached_plan, 2, dims,
                      nullptr, 1, 0,
                      nullptr, 1, 0,
                      CUFFT_C2C, batch_size);
        cached_N = N;
        cached_batch = batch_size;
    }
    cufftSetStream(cached_plan, stream);

    cufftExecC2C(cached_plan, (cufftComplex*)input, (cufftComplex*)output, inverse ? CUFFT_INVERSE : CUFFT_FORWARD);
}
