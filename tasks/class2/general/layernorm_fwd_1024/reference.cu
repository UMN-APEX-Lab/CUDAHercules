/*
 * LayerNorm Forward — hidden_size=1024
 * BF16 input/output, FP32 weights, FP32 compute
 *
 * Source: flash-attention/csrc/layer_norm/ln_fwd_kernels.cuh,
 * instantiated as registered in ln_fwd_1024.cu for
 * (wtype=fp32, itype=bf16, rtype=bf16, otype=bf16, ctype=fp32).
 * Kernel source code in reference_sources/.
 */

#include "ln_fwd_kernels.cuh"

namespace layer_norm {
    FwdRegistry FWD_FUNCS;
    FwdRegistry PARALLEL_FWD_FUNCS;
    BwdRegistry BWD_FUNCS;
    BwdRegistry PARALLEL_BWD_FUNCS;
}

extern "C" void launch_layernorm_forward(
    const void* x,
    const float* gamma,
    const float* beta,
    void* z,
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
    params.x0 = const_cast<void*>(x);
    params.x = const_cast<void*>(x);  // not written: input_t == residual_t and no residual
    params.gamma = const_cast<float*>(gamma);
    params.beta = const_cast<float*>(beta);
    params.z = z;
    params.mu = mu;
    params.rs = rs;
    params.epsilon = eps;
    params.inverse_cols = 1.0f / static_cast<float>(cols);

    params.residual = nullptr;
    params.x1 = nullptr;
    params.gamma1 = nullptr;
    params.beta1 = nullptr;
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

    launch_<fp32, bf16, bf16, bf16, fp32, uint32_t,
            1024, 1, 4, 1, 16
    >(launch_params, true);

    launch_<fp32, bf16, bf16, bf16, fp32, uint32_t,
            1024, 1, 4, 1, 16
    >(launch_params, false);
}
