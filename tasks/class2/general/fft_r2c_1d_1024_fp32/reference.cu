/*
 * cuFFT reference: 1D Real-to-Complex FFT, N=1024, fp32, batch=1024
 */
#include <cuda_runtime.h>
#include <cufft.h>
#include <cstdint>

static cufftHandle cached_plan = 0;
static int cached_N = 0;
static int cached_batch = 0;

extern "C"
void launch_fft_r2c_1d_1024_fp32(
    const void* input,
    void* output,
    int N,
    int batch_size,
    int inverse,
    cudaStream_t stream
) {
    if (cached_plan == 0 || cached_N != N || cached_batch != batch_size) {
        if (cached_plan) cufftDestroy(cached_plan);
        int dims[1] = { 1024 };
        cufftPlanMany(&cached_plan, 1, dims,
                      nullptr, 1, 0,
                      nullptr, 1, 0,
                      CUFFT_R2C, batch_size);
        cached_N = N;
        cached_batch = batch_size;
    }
    cufftSetStream(cached_plan, stream);

    cufftExecR2C(cached_plan, (cufftReal*)input, (cufftComplex*)output);
}
