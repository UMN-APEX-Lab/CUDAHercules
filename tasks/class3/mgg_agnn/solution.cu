#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

__global__ void naive_agnn_forward_kernel(
    float* output,
    float* edge_attention,
    const float* input,
    const int* row_pointers,
    const int* column_index,
    const int num_nodes,
    const int dim
) {
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= num_nodes) return;

    const int nb_begin = row_pointers[node];
    const int nb_end = row_pointers[node + 1];

    // Compute src norm
    float src_norm2 = 0.0f;
    for (int d = 0; d < dim; d++) {
        float v = input[node * dim + d];
        src_norm2 += v * v;
    }
    float src_norm = sqrtf(src_norm2 + 1e-8f);

    // Phase 1: compute attention weights
    float attn_sum = 0.0f;
    for (int e = nb_begin; e < nb_end; e++) {
        int nid = column_index[e];
        float dot = 0.0f, dst_norm2 = 0.0f;
        for (int d = 0; d < dim; d++) {
            float sv = input[node * dim + d];
            float dv = input[nid * dim + d];
            dot += sv * dv;
            dst_norm2 += dv * dv;
        }
        float dst_norm = sqrtf(dst_norm2 + 1e-8f);
        float cos_sim = dot / (src_norm * dst_norm);
        float attn = expf(cos_sim);
        edge_attention[e] = attn;
        attn_sum += attn;
    }

    // Phase 2: normalize + aggregate
    for (int e = nb_begin; e < nb_end; e++) {
        int nid = column_index[e];
        float attn = edge_attention[e] / (attn_sum + 1e-8f);
        edge_attention[e] = attn;
        for (int d = 0; d < dim; d++) {
            output[node * dim + d] += attn * input[nid * dim + d];
        }
    }
}

__global__ void naive_agnn_backward_kernel(
    float* grad_input,
    const float* grad_output,
    const float* edge_attention,
    const int* row_pointers,
    const int* column_index,
    const int num_nodes,
    const int dim
) {
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= num_nodes) return;

    const int nb_begin = row_pointers[node];
    const int nb_end = row_pointers[node + 1];

    for (int e = nb_begin; e < nb_end; e++) {
        int nid = column_index[e];
        float attn = edge_attention[e];
        for (int d = 0; d < dim; d++) {
            grad_input[node * dim + d] += attn * grad_output[nid * dim + d];
        }
    }
}

static std::vector<torch::Tensor> agnn_core_forward_cuda(
    torch::Tensor input,
    torch::Tensor row_pointers,
    torch::Tensor column_index,
    int num_nodes,
    int num_edges,
    int dim
) {
    auto output = torch::zeros_like(input);
    auto edge_attention = torch::zeros({num_edges}, input.options());

    int threads = 256;
    int blocks = (num_nodes + threads - 1) / threads;
    naive_agnn_forward_kernel<<<blocks, threads>>>(
        output.data_ptr<float>(),
        edge_attention.data_ptr<float>(),
        input.data_ptr<float>(),
        row_pointers.data_ptr<int>(),
        column_index.data_ptr<int>(),
        num_nodes, dim);

    return {output, edge_attention};
}

static std::vector<torch::Tensor> agnn_core_backward_cuda(
    torch::Tensor grad_output,
    torch::Tensor edge_attention,
    torch::Tensor row_pointers,
    torch::Tensor column_index,
    int num_nodes,
    int num_edges,
    int dim
) {
    auto grad_input = torch::zeros_like(grad_output);

    int threads = 256;
    int blocks = (num_nodes + threads - 1) / threads;
    naive_agnn_backward_kernel<<<blocks, threads>>>(
        grad_input.data_ptr<float>(),
        grad_output.data_ptr<float>(),
        edge_attention.data_ptr<float>(),
        row_pointers.data_ptr<int>(),
        column_index.data_ptr<int>(),
        num_nodes, dim);

    return {grad_input};
}

std::vector<torch::Tensor> agnn_forward_cuda(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor row_pointers,
    torch::Tensor column_index,
    int num_nodes,
    int num_edges,
    bool apply_relu
) {
    auto projected = torch::mm(input, weights).contiguous();
    auto result = agnn_core_forward_cuda(
        projected,
        row_pointers,
        column_index,
        num_nodes,
        num_edges,
        projected.size(1));

    auto output = result[0];
    if (apply_relu) {
        output = torch::relu(output);
    }
    return {output, result[1]};
}

std::vector<torch::Tensor> agnn_backward_cuda(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor edge_attention,
    torch::Tensor layer_output,
    torch::Tensor row_pointers,
    torch::Tensor column_index,
    int num_nodes,
    int num_edges,
    bool apply_relu
) {
    auto grad = grad_output.contiguous();
    if (apply_relu) {
        auto relu_mask = layer_output.gt(0).to(grad.scalar_type());
        grad = grad * relu_mask;
    }

    auto grad_input_prime = agnn_core_backward_cuda(
        grad,
        edge_attention,
        row_pointers,
        column_index,
        num_nodes,
        num_edges,
        weights.size(1))[0];
    auto grad_input = torch::mm(grad_input_prime, weights.transpose(0, 1));
    auto grad_weights = torch::mm(input.transpose(0, 1), grad_input_prime);

    return {grad_input, grad_weights};
}
