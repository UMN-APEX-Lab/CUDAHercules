/**
 * FlashAttention-4 Backward Pass - Hand-Written CUDA Implementation
 *
 * Target: NVIDIA Blackwell B200/GB200 (SM100, compute capability 10.0)
 *
 * Requirements:
 * - Implement the FA4 backward pass using raw CUDA/PTX (no CuTe DSL)
 * - Support BF16 precision, head_dim in {64, 128, 256}
 * - Support causal and non-causal masking
 * - Support GQA (num_heads_k != num_heads)
 *
 * The runner calls flash_attn4_bwd(Q, K, V, O, dO, LSE, is_causal) and expects
 * {dQ, dK, dV}. This skeleton only defines the ABI; replace it with a real
 * tiled FA4 backward implementation.
 */

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <vector>

__global__ void zero_bf16_kernel(__nv_bfloat16* ptr, int64_t n) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        ptr[idx] = __float2bfloat16(0.0f);
    }
}

static void zero_tensor(torch::Tensor t) {
    const int threads = 256;
    const int64_t n = t.numel();
    const int blocks = static_cast<int>((n + threads - 1) / threads);
    zero_bf16_kernel<<<blocks, threads>>>(
        reinterpret_cast<__nv_bfloat16*>(t.data_ptr<at::BFloat16>()), n);
}

std::vector<torch::Tensor> flash_attn4_bwd(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor O,
    torch::Tensor dO,
    torch::Tensor LSE,
    bool is_causal
) {
    (void)O;
    (void)dO;
    (void)LSE;
    (void)is_causal;

    auto dQ = torch::empty_like(Q);
    auto dK = torch::empty_like(K);
    auto dV = torch::empty_like(V);

    zero_tensor(dQ);
    zero_tensor(dK);
    zero_tensor(dV);

    return {dQ, dK, dV};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flash_attn4_bwd", &flash_attn4_bwd, "FlashAttention-4 Backward (CUDA)");
}
