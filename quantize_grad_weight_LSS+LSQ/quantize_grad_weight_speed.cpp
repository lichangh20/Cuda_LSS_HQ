#include <torch/extension.h>

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, std::vector<double>, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, float> quantize_cuda(torch::Tensor x, int num_bits, torch::Tensor qy, float scaley, torch::Tensor lsq_weight, torch::Tensor q_weight, int lsq_weight_size);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, std::vector<double>, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, float> quantize(torch::Tensor x, int num_bits, torch::Tensor qy, float scaley, torch::Tensor lsq_weight, torch::Tensor q_weight, int lsq_weight_size){
    return quantize_cuda(x, num_bits, qy, scaley, lsq_weight, q_weight, lsq_weight_size);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("quantize", &quantize);
}