#include <torch/extension.h>

std::tuple<torch::Tensor, torch::Tensor> quantize_cuda(torch::Tensor grad_output_flatten, torch::Tensor q_input_flatten, torch::Tensor q_weight, float scale_input, float scale_weight, float scale_grad);


std::tuple<torch::Tensor, torch::Tensor> quantize(torch::Tensor grad_output_flatten, torch::Tensor q_input_flatten, torch::Tensor q_weight, float scale_input, float scale_weight, float scale_grad){

    TORCH_CHECK(grad_output_flatten.type().is_cuda(), "grad_output must be a CUDA tensor!");
    TORCH_CHECK(grad_output_flatten.is_contiguous(), "grad_output must be contiguous!");
    TORCH_CHECK(grad_output_flatten.dim() == 2, "grad_output must be 2D!");

    TORCH_CHECK(q_input_flatten.type().is_cuda(), "qinput must be a CUDA tensor!");
    TORCH_CHECK(q_input_flatten.is_contiguous(), "qinput must be contiguous!");
    TORCH_CHECK(q_input_flatten.dim() == 2, "qinput must be 2D!");

    TORCH_CHECK(q_weight.type().is_cuda(), "qweight must be a CUDA tensor!");
    TORCH_CHECK(q_weight.is_contiguous(), "qweight must be contiguous!");
    TORCH_CHECK(q_weight.dim() == 2, "qweight must be 2D!");
    return quantize_cuda(grad_output_flatten, q_input_flatten, q_weight, scale_input, scale_weight, scale_grad);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("quantize", &quantize);
}