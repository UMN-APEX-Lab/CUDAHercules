/*
 * Standalone common header for ExaChem CCSD(T) benchmark.
 * Replaces TAMM-dependent ccsd_t_common.hpp for standalone compilation.
 *
 * Original: ExaChem (Apache 2.0) - Copyright 2023-2024 Pacific Northwest National Laboratory
 */

#pragma once

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// Error checking macros
#define CUDA_SAFE(call) do {                                                    \
    cudaError_t err = (call);                                                   \
    if (err != cudaSuccess) {                                                   \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",                           \
                __FILE__, __LINE__, cudaGetErrorString(err));                    \
        exit(EXIT_FAILURE);                                                     \
    }                                                                           \
} while(0)

// Type aliases (replaces TAMM types)
using Index = long;
