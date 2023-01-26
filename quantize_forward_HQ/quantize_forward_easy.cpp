#include <torch/extension.h>

std::pair<torch::Tensor, std::vector<double>> quantize_cuda(torch::Tensor hx, torch::Tensor hy, float scale_x, float scale_y);

std::pair<torch::Tensor, std::vector<double>> quantize(torch::Tensor hx, torch::Tensor hy, float scale_x, float scale_y){
    TORCH_CHECK(hx.type().is_cuda(), "x must be a CUDA tensor!");
    TORCH_CHECK(hx.is_contiguous(), "x must be contiguous!");
    TORCH_CHECK(hx.dim() == 2, "x must be 2D!");

    TORCH_CHECK(hy.type().is_cuda(), "y must be a CUDA tensor!");
    TORCH_CHECK(hy.is_contiguous(), "y must be contiguous!");
    TORCH_CHECK(hy.dim() == 2, "y must be 2D!");

    return quantize_cuda(hx, hy, scale_x, scale_y);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("quantize", &quantize);
}