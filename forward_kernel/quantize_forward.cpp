#include <torch/extension.h>

// return output, time vector, q_input, q_weight
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> quantize_cuda(torch::Tensor input, torch::Tensor weight, float scale_input, float scale_weight);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> quantize(torch::Tensor input, torch::Tensor weight, float scale_input, float scale_weight){
    TORCH_CHECK(input.type().is_cuda(), "input must be a CUDA tensor!");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous!");
    TORCH_CHECK(input.dim() == 2, "input must be 2D!");

    TORCH_CHECK(weight.type().is_cuda(), "weight must be a CUDA tensor!");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous!");
    TORCH_CHECK(weight.dim() == 2, "weight must be 2D!");

    return quantize_cuda(input, weight, scale_input, scale_weight);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("quantize", &quantize);
}