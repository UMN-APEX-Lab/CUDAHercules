/*
 * cuFFT reference: 1D Complex-to-Complex FFT, N=4096, fp64, batch=64
 */
#include <cuda_runtime.h>
#include <cufft.h>
#include <cstdint>

static cufftHandle cached_plan = 0;
static int cached_N = 0;
static int cached_batch = 0;

extern "C"
void launch_fft_c2c_1d_4096_fp64(
    const void* input,
    void* output,
    int N,
    int batch_size,
    int inverse,
    cudaStream_t stream
) {
    if (cached_plan == 0 || cached_N != N || cached_batch != batch_size) {
        if (cached_plan) cufftDestroy(cached_plan);
        int dims[1] = { 4096 };
        cufftPlanMany(&cached_plan, 1, dims,
                      nullptr, 1, 0,
                      nullptr, 1, 0,
                      CUFFT_Z2Z, batch_size);
        cached_N = N;
        cached_batch = batch_size;
    }
    cufftSetStream(cached_plan, stream);

    cufftExecZ2Z(cached_plan, (cufftDoubleComplex*)input, (cufftDoubleComplex*)output, inverse ? CUFFT_INVERSE : CUFFT_FORWARD);
}
