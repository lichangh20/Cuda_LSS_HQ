#include <torch/extension.h>
#include <tuple>

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int, int, int, float, float, float, float, int, int, int, torch::Tensor> first_quantize_cuda(torch::Tensor x, torch::Tensor y, int num_bins);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> second_quantize_cuda(torch::Tensor vec_norm, torch::Tensor first_transform, torch::Tensor second_transform, torch::Tensor first_quantize, torch::Tensor second_quantize, int len_norm, 
                                                                                      int small_num_, int large_num_, float scale1, float zero_point1, float scale2, float zero_point2, int nx, int ny, int nz,
                                                                                      torch::Tensor weight_phi_in, torch::Tensor qy, float scaley, torch::Tensor hadamard_weight, torch::Tensor scale_weight);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int, int, int, float, float, float, float, int, int, int, torch::Tensor> first_quantize(torch::Tensor x, torch::Tensor y, int num_bins){
    TORCH_CHECK(x.type().is_cuda(), "x must be a CUDA tensor!");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous!");
    TORCH_CHECK(x.dim() == 2, "x must be 2D!");

    TORCH_CHECK(y.type().is_cuda(), "y must be a CUDA tensor!");
    TORCH_CHECK(y.is_contiguous(), "y must be contiguous!");
    TORCH_CHECK(y.dim() == 2, "y must be 2D!");

    return first_quantize_cuda(x, y, num_bins);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> second_quantize(torch::Tensor vec_norm, torch::Tensor first_transform, torch::Tensor second_transform, torch::Tensor first_quantize, torch::Tensor second_quantize, int len_norm, 
                                                                                      int small_num_, int large_num_, float scale1, float zero_point1, float scale2, float zero_point2, int nx, int ny, int nz,
                                                                                      torch::Tensor weight_phi_in, torch::Tensor qy, float scaley, torch::Tensor hadamard_weight, torch::Tensor scale_weight){

    return second_quantize_cuda(vec_norm, first_transform, second_transform, first_quantize, second_quantize, len_norm, small_num_, large_num_, scale1, zero_point1, scale2, zero_point2, nx, ny, nz, weight_phi_in, qy, scaley, hadamard_weight, scale_weight);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("first_quantize", &first_quantize);
  m.def("second_quantize", &second_quantize);
}