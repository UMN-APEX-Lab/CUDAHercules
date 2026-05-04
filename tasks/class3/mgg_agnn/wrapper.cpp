#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> agnn_forward_cuda(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor row_pointers,
    torch::Tensor column_index,
    int num_nodes,
    int num_edges,
    bool apply_relu
);

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
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &agnn_forward_cuda, "AGNN layer forward (GEMM + attention + aggregation + optional ReLU)");
    m.def("backward", &agnn_backward_cuda, "AGNN layer backward (optional ReLU + weighted SpMM + GEMM)");
}
