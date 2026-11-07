// Minimal stand-in for the PyTorch c10 header used by the FlashAttention-2 forward launch
// template, so that the FlashAttention-2 kernels can be compiled without PyTorch.
#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define C10_CUDA_CHECK(EXPR)                                                          \
  do {                                                                                \
    cudaError_t _kh_err = (EXPR);                                                     \
    if (_kh_err != cudaSuccess) {                                                     \
      fprintf(stderr, "CUDA error: %s at %s:%d\n", cudaGetErrorString(_kh_err),       \
              __FILE__, __LINE__);                                                    \
      std::exit(1);                                                                   \
    }                                                                                 \
  } while (0)

#define C10_CUDA_KERNEL_LAUNCH_CHECK() C10_CUDA_CHECK(cudaGetLastError())
