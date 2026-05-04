/*
 * LayerNorm Backward — hidden_size=8192
 *
 * Source: flash-attention/csrc/layer_norm/ln_bwd_kernels.cuh
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
    const float* dz,
    const float* x,
    const float* mu,
    const float* rs,
    const float* gamma,
    float* dx,
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
    params.dz = const_cast<float*>(dz);
    params.x = const_cast<float*>(x);
    params.x0 = const_cast<float*>(x);
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

    launch_<float, float, float, float, float, uint32_t,
            8192, 1, 1, 8, 16, 4
    >(launch_params, true);

    int ctas_per_col = launch_params.params.ctas_per_col;
    static float* dgamma_part = nullptr; static size_t dgamma_part_cached_size = 0;
    static float* dbeta_part = nullptr; static size_t dbeta_part_cached_size = 0;
    { size_t _need = ctas_per_col * 8192 * sizeof(float);
        if (dgamma_part_cached_size < _need) { if (dgamma_part) cudaFree(dgamma_part); cudaMalloc(&dgamma_part, _need); dgamma_part_cached_size = _need; } }
    { size_t _need = ctas_per_col * 8192 * sizeof(float);
        if (dbeta_part_cached_size < _need) { if (dbeta_part) cudaFree(dbeta_part); cudaMalloc(&dbeta_part, _need); dbeta_part_cached_size = _need; } }
    launch_params.params.dgamma_part = dgamma_part;
    launch_params.params.dbeta_part = dbeta_part;

    launch_<float, float, float, float, float, uint32_t,
            8192, 1, 1, 8, 16, 4
    >(launch_params, false);

    cudaStreamSynchronize(stream);
    /* cudaFree(dgamma_part); removed — buffer cached across invocations */
    /* cudaFree(dbeta_part); removed — buffer cached across invocations */
}
