#include <torch/extension.h>

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> quantize_cuda(torch::Tensor x, int num_bins, torch::Tensor qy, float scaley, torch::Tensor hadamard_activation, torch::Tensor scale_activation);


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> quantize(torch::Tensor x, int num_bins, torch::Tensor qy, float scaley, torch::Tensor hadamard_activation, torch::Tensor scale_activation){

    return quantize_cuda(x, num_bins, qy, scaley, hadamard_activation, scale_activation);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("quantize", &quantize);
}