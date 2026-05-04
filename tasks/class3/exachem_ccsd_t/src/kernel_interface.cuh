/*
 * Interface for the CCSD(T) fully-fused kernel driver.
 * Both reference and solution kernels must implement this function.
 */

#pragma once
#include <cuda_runtime.h>

void launch_ccsd_t_kernel(
    cudaStream_t stream, size_t numBlks,
    size_t size_h3, size_t size_h2, size_t size_h1,
    size_t size_p6, size_t size_p5, size_t size_p4,
    //
    double* dev_s1_t1_all, double* dev_s1_v2_all,
    double* dev_d1_t2_all, double* dev_d1_v2_all,
    double* dev_d2_t2_all, double* dev_d2_v2_all,
    //
    int* host_size_d1_h7b, int* host_size_d2_p7b,
    int* host_exec_s1, int* host_exec_d1, int* host_exec_d2,
    //
    size_t size_noab, size_t size_nvab,
    size_t size_max_dim_s1_t1, size_t size_max_dim_s1_v2,
    size_t size_max_dim_d1_t2, size_t size_max_dim_d1_v2,
    size_t size_max_dim_d2_t2, size_t size_max_dim_d2_v2,
    //
    double* dev_evl_sorted_h1b, double* dev_evl_sorted_h2b, double* dev_evl_sorted_h3b,
    double* dev_evl_sorted_p4b, double* dev_evl_sorted_p5b, double* dev_evl_sorted_p6b,
    //
    double* dev_energies);
