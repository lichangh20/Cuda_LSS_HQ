#include <torch/extension.h>

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> quantize_cuda(torch::Tensor x, int num_bins, torch::Tensor y, torch::Tensor qy, float scaley, torch::Tensor hadamard_weight, torch::Tensor scale_weight);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> quantize(torch::Tensor x, int num_bins, torch::Tensor y, torch::Tensor qy, float scaley, torch::Tensor hadamard_weight, torch::Tensor scale_weight){
    return quantize_cuda(x, num_bins, y, qy, scaley, hadamard_weight, scale_weight);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("quantize", &quantize);
}