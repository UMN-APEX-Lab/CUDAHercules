/*
 * cuFFT reference: 3D Complex-to-Complex FFT, 64^3, fp32
 */
#include <cuda_runtime.h>
#include <cufft.h>
#include <cstdint>

static cufftHandle cached_plan = 0;
static int cached_N = 0;
static int cached_batch = 0;

extern "C"
void launch_fft_c2c_3d_64x64x64_fp32(
    const void* input,
    void* output,
    int N,
    int batch_size,
    int inverse,
    cudaStream_t stream
) {
    if (cached_plan == 0 || cached_N != N || cached_batch != batch_size) {
        if (cached_plan) cufftDestroy(cached_plan);
        int dims[3] = { 64, 64, 64 };
        cufftPlanMany(&cached_plan, 3, dims,
                      nullptr, 1, 0,
                      nullptr, 1, 0,
                      CUFFT_C2C, batch_size);
        cached_N = N;
        cached_batch = batch_size;
    }
    cufftSetStream(cached_plan, stream);

    cufftExecC2C(cached_plan, (cufftComplex*)input, (cufftComplex*)output, inverse ? CUFFT_INVERSE : CUFFT_FORWARD);
}
