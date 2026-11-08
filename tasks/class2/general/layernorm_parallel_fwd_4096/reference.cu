/*
 * LayerNorm Parallel Residual Forward — hidden_size=4096
 * BF16 activations, FP32 weights, FP32 compute
 *
 * Source: flash-attention/csrc/layer_norm/ln_parallel_residual_fwd_kernels.cuh,
 * instantiated as registered in ln_parallel_fwd_4096.cu for
 * (wtype=fp32, itype=bf16, rtype=bf16, otype=bf16, ctype=fp32).
 * Kernel source code in reference_sources/.
 *
 * Fused residual addition + layer normalization:
 *   x = x0 + residual   (FP32 sum; stored as bf16)
 *   z = LayerNorm(x, gamma, beta)   (statistics and z from the FP32 sum)
 */

#include "ln_parallel_residual_fwd_kernels.cuh"

namespace layer_norm {
    FwdRegistry FWD_FUNCS;
    FwdRegistry PARALLEL_FWD_FUNCS;
    BwdRegistry BWD_FUNCS;
    BwdRegistry PARALLEL_BWD_FUNCS;
}

extern "C" void launch_layernorm_parallel_fwd(
    const void* x0,
    const void* residual,
    const float* gamma,
    const float* beta,
    void* z,
    void* x,
    float* mu,
    float* rs,
    int rows,
    int cols,
    float eps,
    cudaStream_t stream
) {
    using namespace layer_norm;

    FwdParams params;
    params.rows = rows;
    params.cols = cols;
    params.x0 = const_cast<void*>(x0);
    params.residual = const_cast<void*>(residual);
    params.x = x;
    params.gamma = const_cast<float*>(gamma);
    params.beta = const_cast<float*>(beta);
    params.z = z;
    params.mu = mu;
    params.rs = rs;
    params.epsilon = eps;
    params.inverse_cols = 1.0f / static_cast<float>(cols);

    params.x1 = nullptr;
    params.gamma1 = nullptr;
    params.beta1 = nullptr;
    params.z1 = nullptr;
    params.rowscale = nullptr;
    params.colscale = nullptr;
    params.x0_subset = nullptr;
    params.z_subset = nullptr;
    params.dmask = nullptr;
    params.dmask1 = nullptr;
    params.dropout_keep_p = 1.0f;
    params.dropout_scale = 1.0f;
    params.is_rms_norm = false;
    params.workspace = nullptr;
    params.barrier = nullptr;

    LaunchParams<FwdParams> launch_params;
    launch_params.params = params;
    launch_params.stream = stream;

    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    launch_params.props = &prop;

    launch_parallel_residual_<fp32, bf16, bf16, bf16, fp32, uint32_t,
            4096, 1, 1, 4, 16
    >(launch_params, true);

    launch_parallel_residual_<fp32, bf16, bf16, bf16, fp32, uint32_t,
            4096, 1, 1, 4, 16
    >(launch_params, false);
}
