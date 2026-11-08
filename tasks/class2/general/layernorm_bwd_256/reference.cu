/*
 * LayerNorm Backward — hidden_size=256
 * BF16 activations/gradients, FP32 weights, FP32 compute
 *
 * Source: flash-attention/csrc/layer_norm/ln_bwd_kernels.cuh,
 * instantiated as registered in ln_bwd_256.cu for
 * (wtype=fp32, itype=bf16, rtype=bf16, otype=bf16, ctype=fp32).
 * Kernel source code in reference_sources/.
 */

#include "ln_bwd_kernels.cuh"

namespace layer_norm {
    FwdRegistry FWD_FUNCS;
    FwdRegistry PARALLEL_FWD_FUNCS;
    BwdRegistry BWD_FUNCS;
    BwdRegistry PARALLEL_BWD_FUNCS;
}

extern "C" void launch_layernorm_backward(
    const void* dz,
    const void* x,
    const float* mu,
    const float* rs,
    const float* gamma,
    void* dx,
    float* dgamma,
    float* dbeta,
    int rows,
    int cols,
    cudaStream_t stream
) {
    using namespace layer_norm;

    BwdParams params;
    params.rows = rows;
    params.cols = cols;
    params.dz = const_cast<void*>(dz);
    params.x = const_cast<void*>(x);
    params.x0 = const_cast<void*>(x);
    params.mu = const_cast<float*>(mu);
    params.rs = const_cast<float*>(rs);
    params.gamma = const_cast<float*>(gamma);
    params.dx0 = dx;
    params.dgamma = dgamma;
    params.dbeta = dbeta;
    params.inverse_cols = 1.0f / static_cast<float>(cols);

    params.residual = nullptr;
    params.x1 = nullptr;
    params.dz1 = nullptr;
    params.dx = nullptr;
    params.dx1 = nullptr;
    params.dresidual = nullptr;
    params.gamma1 = nullptr;
    params.rowscale = nullptr;
    params.colscale = nullptr;
    params.x0_subset = nullptr;
    params.z_subset = nullptr;
    params.dmask = nullptr;
    params.dmask1 = nullptr;
    params.dbeta1 = nullptr;
    params.dgamma1 = nullptr;
    params.dbeta1_part = nullptr;
    params.dgamma1_part = nullptr;
    params.dcolscale = nullptr;
    params.dcolscale_part = nullptr;
    params.dropout_keep_p = 1.0f;
    params.dropout_scale = 1.0f;
    params.is_rms_norm = false;
    params.workspace = nullptr;
    params.barrier = nullptr;

    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    LaunchParams<BwdParams> launch_params;
    launch_params.params = params;
    launch_params.stream = stream;
    launch_params.props = &prop;

    launch_<fp32, bf16, bf16, bf16, fp32, uint32_t,
            256, 1, 4, 1, 16, 4
    >(launch_params, true);

    // FP32 per-CTA partial sums of dgamma/dbeta (reduced by the finalize kernel). The
    // buffers are cached across calls, so a call does no cudaMalloc/cudaFree and no
    // host synchronization (both kernels are stream-ordered).
    const size_t part_bytes = static_cast<size_t>(launch_params.params.ctas_per_col) * cols * sizeof(float);
    static void* dgamma_part = nullptr;
    static void* dbeta_part = nullptr;
    static size_t part_cached_bytes = 0;
    if (part_cached_bytes < part_bytes) {
        if (dgamma_part) cudaFree(dgamma_part);
        if (dbeta_part) cudaFree(dbeta_part);
        cudaMalloc(&dgamma_part, part_bytes);
        cudaMalloc(&dbeta_part, part_bytes);
        part_cached_bytes = part_bytes;
    }
    launch_params.params.dgamma_part = dgamma_part;
    launch_params.params.dbeta_part = dbeta_part;

    launch_<fp32, bf16, bf16, bf16, fp32, uint32_t,
            256, 1, 4, 1, 16, 4
    >(launch_params, false);
}
