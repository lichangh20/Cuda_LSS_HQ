#include <torch/extension.h>
#include <torch/torch.h>
#include <ATen/ATen.h>
#include <cmath>
#include <iostream>
#include <sstream>
#include <vector>
#include <cuda.h>
#include <ctime>
#include "cuda_fp16.hpp"
#include "cuda_fp16.h"

#include "cuda_runtime.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include <curand_kernel.h>
#include <tuple>
#include <bits/stdc++.h>
// #include <torch/distributions/gumbel.h>

#include "torch/script.h"
using namespace torch::indexing;

template<typename scalar_t>
__global__ void quantize_cuda_kernel(const scalar_t * __restrict__  MatI, int8_t * first_transform, scalar_t * __restrict__  first_quantize, int8_t * second_transform, scalar_t * __restrict__  second_quantize, 
                                    const int num_bins_half, const int num_bins_clamp, const float scale, int size, unsigned long seed){
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    if (x<size){
        // set random value
        curandStatePhilox4_32_10_t state;
        curand_init(seed, x, 0, &state);
        const float noise = curand_uniform(&state);

        float trans_input = MatI[x] * scale;

        float tmp1 = round(trans_input / num_bins_half);
        int firstTransform = std::clamp((int)(tmp1), -num_bins_clamp, num_bins_clamp);
        first_transform[x] = firstTransform;
        // float quantize = (transform + 8) / scale + zero_point;
        first_quantize[x] = firstTransform * num_bins_half / scale;

        float tmp2 = round(trans_input - firstTransform * num_bins_half + noise - 0.5);
        int secondTransform = std::clamp((int)(tmp2), -num_bins_clamp, num_bins_clamp);
        second_transform[x] = secondTransform;
        second_quantize[x] = secondTransform / scale;
    }
}

// template<typename scalar_t>
// __global__ void second_quantize_cuda_kernel(const scalar_t * __restrict__  MatI, int8_t * MatO_transform, scalar_t * __restrict__  MatO_quantize, const float scale, const float  zero_point, int size, unsigned long seed){
//     int x = threadIdx.x + blockIdx.x * blockDim.x;
//     if (x<size){
//         // set random value
//         curandStatePhilox4_32_10_t state;
//         curand_init(seed, x, 0, &state);
//         const float noise = curand_uniform(&state);

//         float input = MatI[x];
//         // scalar_t tmp1 = (input - zero_point) * scale + noise - 8.5;
//         // // scalar_t tmp1 = (MatI[x] - zero_point) * scale - 8;
//         // int tmp2 = tmp1;
//         // int bias = (tmp1 - tmp2) * 2;
//         // MatO_transform[x] = std::clamp(tmp2+bias, -8, 7);
//         float tmp1 = round((input - zero_point) * scale + noise - 8.5);
//         MatO_transform[x] = std::clamp((int)(tmp1), -8, 7);
//         MatO_quantize[x] = (MatO_transform[x] + 8) / scale + zero_point;
//     }
// }

__global__ void pack_cuda_kernel(int8_t * in, int8_t * out, int size){
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    if (x<size){
        out[x] = (in[(x<<1)+1] << 4) | (in[x<<1] & 15);
    }
}

cudaError_t CutlassSgemmNN_fp16(
  const int M,
  const int N,
  const int K,
  const cutlass::half_t *A,
  int lda,
  const cutlass::half_t *B,
  int ldb,
  cutlass::half_t *C,
  int ldc) {

// The code section below describes datatype for input, output matrices and computation between
// elements in input matrices.
using ElementAccumulator = float;                 // <- data type of accumulator
using ElementComputeEpilogue = ElementAccumulator;  // <- data type of epilogue operations
using ElementInputA = cutlass::half_t;                       // <- data type of elements in input matrix A
using ElementInputB = cutlass::half_t;                       // <- data type of elements in input matrix B
using ElementOutput = cutlass::half_t;                      // <- data type of elements in output matrix D

// The code section below describes matrix layout of input and output matrices. Column Major for
// Matrix A, Row Major for Matrix B and Row Major for Matrix C
using LayoutInputA = cutlass::layout::RowMajor;
using LayoutInputB = cutlass::layout::ColumnMajor;
using LayoutOutput = cutlass::layout::RowMajor;

// This code section describes whether you want to use tensor cores or regular SIMT cores on GPU SM
using MMAOp = cutlass::arch::OpClassTensorOp;

// This code section describes CUDA SM architecture number
using SmArch = cutlass::arch::Sm70;

// This code section describes the tile size a thread block will compute
using ShapeMMAThreadBlock =
    cutlass::gemm::GemmShape<256, 128, 32>;  // <- threadblock tile M = 128, N = 256, K = 64
// This code section describes tile size a warp will compute
using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 32>;  // <- warp tile M = 64, N = 64, K = 64 
// This code section describes the size of MMA op
using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;  // <- MMA Op tile M = 8, N = 8, K = 16

// This code section describes how threadblocks are scheduled on GPU
using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>;  // <- ??

// This code section describes the epilogue part of the kernel
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,                                     // <- data type of output matrix
    4,  // <- the number of elements per vectorized
                                                       // memory access. For a byte, it's 16
                                                       // elements. This becomes the vector width of
                                                       // math instructions in the epilogue too
    ElementAccumulator,                                // <- data type of accumulator
    ElementComputeEpilogue>;  // <- data type for alpha/beta in linear combination function

// Number of pipelines you want to use
constexpr int NumStages = 2;

using Gemm = cutlass::gemm::device::Gemm<ElementInputA,
                                         LayoutInputA,
                                         ElementInputB,
                                         LayoutInputB,
                                         ElementOutput,
                                         LayoutOutput,
                                         ElementAccumulator,
                                         MMAOp,
                                         SmArch,
                                         ShapeMMAThreadBlock,
                                         ShapeMMAWarp,
                                         ShapeMMAOp,
                                         EpilogueOp,
                                         SwizzleThreadBlock,
                                         NumStages>;
  // Create a tuple of problem size for matrix multiplication
  cutlass::gemm::GemmCoord problem_size(M, N, K);

  // Initialize alpha and beta for dot product computation
  ElementComputeEpilogue alpha = ElementComputeEpilogue(1);
  ElementComputeEpilogue beta = ElementComputeEpilogue(0);
    
  // Split K dimension into 1 partitions
  int split_k_slices = 1;

  // Create a tuple of gemm kernel arguments. This is later passed as arguments to launch
  // instantiated CUTLASS kernel
  typename Gemm::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                     {A, lda},  // <- reference to matrix A on device
                                     {B, ldb},  // <- reference to matrix B on device
                                     {C, ldc},  // <- reference to matrix C on device
                                     {C, ldc},  // <- reference to matrix D on device
                                     {alpha, beta},          // <- tuple of alpha and beta
                                     split_k_slices};        // <- k-dimension split factor
    Gemm gemm_op;

    // Launch initialized CUTLASS kernel
    cutlass::Status status = gemm_op(arguments);

  if (status != cutlass::Status::kSuccess) {
    return cudaErrorUnknown;
  }

  // Return success, if no errors were encountered.
  return cudaSuccess;
}

/// Define a CUTLASS GEMM template and launch a GEMM kernel.
cudaError_t CutlassSgemmNN(
  const int M,
  const int N,
  const int K,
  const cutlass::int4b_t *A,
  int lda,
  const cutlass::int4b_t *B,
  int ldb,
  int32_t *C,
  int ldc) {

// The code section below describes datatype for input, output matrices and computation between
// elements in input matrices.
using ElementAccumulator = int32_t;                 // <- data type of accumulator
using ElementComputeEpilogue = ElementAccumulator;  // <- data type of epilogue operations
using ElementInputA = cutlass::int4b_t;                       // <- data type of elements in input matrix A
using ElementInputB = cutlass::int4b_t;                       // <- data type of elements in input matrix B
using ElementOutput = int32_t;                      // <- data type of elements in output matrix D

// The code section below describes matrix layout of input and output matrices. Column Major for
// Matrix A, Row Major for Matrix B and Row Major for Matrix C
using LayoutInputA = cutlass::layout::RowMajor;
using LayoutInputB = cutlass::layout::ColumnMajor;
using LayoutOutput = cutlass::layout::RowMajor;

// This code section describes whether you want to use tensor cores or regular SIMT cores on GPU SM
using MMAOp = cutlass::arch::OpClassTensorOp;

// This code section describes CUDA SM architecture number
using SmArch = cutlass::arch::Sm75;

// This code section describes the tile size a thread block will compute
using ShapeMMAThreadBlock =
    cutlass::gemm::GemmShape<256, 128, 128>;  // <- threadblock tile M = 128, N = 256, K = 64
// This code section describes tile size a warp will compute
using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 128>;  // <- warp tile M = 64, N = 64, K = 64 
// This code section describes the size of MMA op
using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 32>;  // <- MMA Op tile M = 8, N = 8, K = 16

// This code section describes how threadblocks are scheduled on GPU
using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>;  // <- ??

// This code section describes the epilogue part of the kernel
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,                                     // <- data type of output matrix
    4,  // <- the number of elements per vectorized
                                                       // memory access. For a byte, it's 16
                                                       // elements. This becomes the vector width of
                                                       // math instructions in the epilogue too
    ElementAccumulator,                                // <- data type of accumulator
    ElementComputeEpilogue>;  // <- data type for alpha/beta in linear combination function

// Number of pipelines you want to use
constexpr int NumStages = 2;

using Gemm = cutlass::gemm::device::Gemm<ElementInputA,
                                         LayoutInputA,
                                         ElementInputB,
                                         LayoutInputB,
                                         ElementOutput,
                                         LayoutOutput,
                                         ElementAccumulator,
                                         MMAOp,
                                         SmArch,
                                         ShapeMMAThreadBlock,
                                         ShapeMMAWarp,
                                         ShapeMMAOp,
                                         EpilogueOp,
                                         SwizzleThreadBlock,
                                         NumStages>;
  // Create a tuple of problem size for matrix multiplication
  cutlass::gemm::GemmCoord problem_size(M, N, K);

  // Initialize alpha and beta for dot product computation
  ElementComputeEpilogue alpha = ElementComputeEpilogue(1);
  ElementComputeEpilogue beta = ElementComputeEpilogue(0);
    
  // Split K dimension into 1 partitions
  int split_k_slices = 1;

  // Create a tuple of gemm kernel arguments. This is later passed as arguments to launch
  // instantiated CUTLASS kernel
  typename Gemm::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                     {A, lda},  // <- reference to matrix A on device
                                     {B, ldb},  // <- reference to matrix B on device
                                     {C, ldc},  // <- reference to matrix C on device
                                     {C, ldc},  // <- reference to matrix D on device
                                     {alpha, beta},          // <- tuple of alpha and beta
                                     split_k_slices};        // <- k-dimension split factor
  
    Gemm gemm_op;
    cutlass::Status status = gemm_op(arguments);

  if (status != cutlass::Status::kSuccess) {
    return cudaErrorUnknown;
  }

  // Return success, if no errors were encountered.
  return cudaSuccess;
}

#define N_THREADS 256

template<typename scalar_t>
__global__ void dequantize_cuda_kernel(const int32_t * gemm1, const int32_t * gemm2, const scalar_t * __restrict__ gemm3, 
                                        const scalar_t * __restrict__ gemm4, scalar_t * __restrict__ output, 
                                        const float scale_gemm1, const float scale_gemm2, int size){  
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    // int row = x / ny, col = x - row * ny;

    if (x<size){
       output[x] = gemm1[x] * scale_gemm1 + gemm2[x] * scale_gemm2 + gemm3[x] + gemm4[x];
        // output[x] = 0;
    }
}

// template<typename scalar_t>
// __global__ void dequantize_cuda_kernel_fp16(const scalar_t * __restrict__ gemm1, const scalar_t * __restrict__ gemm2, scalar_t * __restrict__ output, 
//                                         int size){  
//     int x = threadIdx.x + blockIdx.x * blockDim.x;

//     if (x<size){
//        output[x] = gemm1[x] + gemm2[x];
//     }
// }


__device__ __inline__ c10::Half __shfl_down_sync(const unsigned mask, const c10::Half var,
                                                 const unsigned int delta, const int width) {
  __half var_ = var;
  return __shfl_down_sync(mask, var_, delta, width);
}

//TODO: N means rows, D means cols
template<typename scalar_t>
__global__ void linalg_norm_cuda_kernel(const scalar_t * __restrict__ in, scalar_t * __restrict__ linalg, int N, int D, int stride_D){
  float sum_val = 0;

  for (int64_t k1_outer = 0; k1_outer < stride_D; ++k1_outer) {
    float temp = in[blockIdx.x * D + (k1_outer << 5) + threadIdx.x];
    sum_val += temp * temp;
  }

  unsigned int mask;
  float sum_val_t;
  mask = __activemask();

  sum_val_t = __shfl_down_sync(mask, sum_val, 16, 32);
  sum_val += sum_val_t;
  sum_val_t = __shfl_down_sync(mask, sum_val, 8, 32);
  sum_val += sum_val_t;
  sum_val_t = __shfl_down_sync(mask, sum_val, 4, 32);
  sum_val += sum_val_t;
  sum_val_t = __shfl_down_sync(mask, sum_val, 2, 32);
  sum_val += sum_val_t;
  sum_val_t = __shfl_down_sync(mask, sum_val, 1, 32);
  sum_val += sum_val_t;
  linalg[blockIdx.x] = sqrt(sum_val);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> quantize_cuda(torch::Tensor x, int num_bits, torch::Tensor y, torch::Tensor qy, float scaley, torch::Tensor hadamard_weight, torch::Tensor scale_weight){
    // std::vector<double> time_vector;
    int nz = x.size(0);
    int nx = x.size(1);
    int ny = y.size(1);

    // cudaDeviceSynchronize();
    // clock_t time_quantize1_start = clock();

    auto option_transform = torch::TensorOptions().dtype(torch::kInt8).device(x.device());
    auto option_quantize = torch::TensorOptions().dtype(x.dtype()).device(x.device());
    torch::Tensor first_transform = torch::empty({nz, nx}, option_transform);
    torch::Tensor first_quantize = torch::empty({nz, nx}, option_quantize);
    torch::Tensor second_transform = torch::empty({nz, nx}, option_transform);
    torch::Tensor second_quantize = torch::empty({nz, nx}, option_quantize);

    dim3 block(N_THREADS);
    dim3 grid1((nx*nz-1)/block.x+1);
    int size_quantize = nz * nx ;
    // process of first quantize
    float mn = std::min(x.min().item<float>() - 1e-8, 0.);
    float mx = std::max(x.max().item<float>() + 1e-8, 0.);

    int num_bins_half = pow(2, num_bits) - 2;
    int num_bins = num_bins_half * num_bins_half;
    int num_bins_clamp = num_bins_half / 2 - 1;

    float scale1 = num_bins / (2 * max(fabs(mn), fabs(mx)));

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "quantize_cuda", ([&] {
    quantize_cuda_kernel<scalar_t><<<grid1, block>>>(
        x.data_ptr<scalar_t>(),
        first_transform.data_ptr<int8_t>(),
        first_quantize.data_ptr<scalar_t>(),
        second_transform.data_ptr<int8_t>(),
        second_quantize.data_ptr<scalar_t>(),
        num_bins_half, num_bins_clamp,
        scale1, size_quantize,rand());
    }));

    // cudaDeviceSynchronize();
    // clock_t time_quantize2_end = clock();

    // leverage score
    // TODO: use dim=0 because torch.linalg only supports dim=1
    int threads = 32;
    int blocks = nz;

    auto x1_len = torch::empty({nz,}, option_quantize);
    auto x2_len = torch::empty({nz,}, option_quantize);
    auto y_len = torch::empty({nz,}, option_quantize);

    int stride_x = nx / 32;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x1_len.scalar_type(), "linalg_cuda", ([&] {
    linalg_norm_cuda_kernel<scalar_t><<<blocks, threads>>>(
        first_quantize.data_ptr<scalar_t>(), 
        x1_len.data_ptr<scalar_t>(),
        nz,nx,stride_x);
    }));
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x2_len.scalar_type(), "linalg_cuda", ([&] {
    linalg_norm_cuda_kernel<scalar_t><<<blocks, threads>>>(
        second_quantize.data_ptr<scalar_t>(), 
        x2_len.data_ptr<scalar_t>(),
        nz,nx,stride_x);
    }));
    int stride_y = ny / 32;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(y.scalar_type(), "linalg_cuda", ([&] {
    linalg_norm_cuda_kernel<scalar_t><<<blocks, threads>>>(
        y.data_ptr<scalar_t>(), 
        y_len.data_ptr<scalar_t>(),
        nz,ny,stride_y);
    }));

    // TODO: whether need to change dtype from half into float? It depends 
    auto vec_norm = torch::cat({torch::mul(x1_len, y_len), torch::mul(x2_len, y_len)}).to(torch::kFloat32);
    // auto vec_norm = torch::cat({torch::mul(x1_len, y_len), torch::mul(x2_len, y_len)});
    int len_norm = vec_norm.numel();

    int cnt = 0;
    auto norm_weight_loop = vec_norm * len_norm / (2 * vec_norm.sum());
    bool whileloop = (norm_weight_loop.max() > 1).item<bool>();
    while (whileloop && cnt < len_norm / 2){
        auto small_index = torch::nonzero((norm_weight_loop < 1)).squeeze();
        auto small_value = norm_weight_loop.index({small_index});
        cnt = len_norm - small_index.numel();
        norm_weight_loop = torch::clamp(norm_weight_loop, 0, 1);
        bool breakloop = (small_value.max() == 0).item<bool>() && (small_value.min() == 0).item<bool>();
        if (breakloop)
            break;
        small_value = small_value * (len_norm / 2 - cnt) / small_value.sum();
        // norm_weight_loop[small_index] = small_value;
        norm_weight_loop.index_put_({small_index}, small_value);
        whileloop = (norm_weight_loop.max() > 1).item<bool>();
    } 
    auto sample_index = torch::bernoulli(norm_weight_loop);
    auto small_indices = torch::nonzero(sample_index.index({Slice({None, len_norm/2})}) == 1).squeeze(1);
    auto small_num = (norm_weight_loop.index({small_indices}) == 1).sum();
    // TODO: test if .int() can work in libtorch
    small_num = ((small_num / 32).floor() * 32).to(torch::kInt32);
    int small_num_ = small_num.item<int>();
    auto small_int_indices = small_indices.index({torch::nonzero(norm_weight_loop.index({small_indices}) == 1).squeeze(1)}).index({Slice({None, small_num_})});
    auto small_left_indices = small_indices.index({~torch::isin(small_indices, small_int_indices)});
    auto large_indices = torch::nonzero(sample_index.index({Slice(len_norm/2)}) == 1).squeeze(1);
    auto large_num = (norm_weight_loop.index({large_indices + len_norm / 2}) == 1).sum();
    // TODO: test if .int() can work in libtorch
    large_num = ((large_num / 32).floor() * 32).to(torch::kInt32);
    int large_num_ = large_num.item<int>();
    auto large_int_indices = large_indices.index({torch::nonzero(norm_weight_loop.index({large_indices + len_norm / 2}) == 1).squeeze(1)}).index({Slice({None, large_num_})});
    auto large_left_indices = large_indices.index({~torch::isin(large_indices, large_int_indices)});
    // auto _index = torch::nonzero((sample_index == 1)).squeeze();

    norm_weight_loop.index_put_({norm_weight_loop == 0}, 1e-10);

    
    // auto small_num_ = (_index < len_norm / 2).sum();
    // auto large_num_ = _index.numel() - small_num_;
    // auto small_indices = _index.index({Slice({None, small_num_.item<int>()})});
    // auto large_indices = _index.index({Slice(small_num_.item<int>())}) - int(len_norm / 2);
    // auto norm_weight_small = norm_weight_loop.index({small_indices});
    // auto norm_weight_large = norm_weight_loop.index({large_indices + len_norm / 2});
    // auto output = torch::cat({first_quantize, second_quantize});
    // output = output / norm_weight_loop.unsqueeze(1);

    // cudaDeviceSynchronize();
    // clock_t time_leverage_end = clock();

    //TODO: suppose an easy situation so that it can be faster
    auto sample_x1 = first_transform.index({small_int_indices}).t().contiguous();
    auto sample_x2 = second_transform.index({large_int_indices}).t().contiguous();
    auto sample_y1 = qy.index({small_int_indices}).t().contiguous();
    auto sample_y2 = qy.index({large_int_indices}).t().contiguous();
    auto sample_x3 = (first_quantize.index({small_left_indices}).t() / norm_weight_loop.index({small_left_indices})).to(torch::kFloat16);
    auto sample_x4 = (second_quantize.index({large_left_indices}).t() / norm_weight_loop.index({large_left_indices + len_norm / 2})).to(torch::kFloat16);
    //todo:currently multiply a scaley to convert it into fp16
    auto sample_y3 = (qy.index({small_left_indices}) * scaley).to(torch::kFloat16);
    auto sample_y4 = (qy.index({large_left_indices}) * scaley).to(torch::kFloat16);
    // auto sample_x3 = (first_quantize.index({small_left_indices}).t().contiguous() / norm_weight_loop.index({small_left_indices})).to(torch::kFloat16);
    // auto sample_x4 = (second_quantize.index({large_left_indices}).t().contiguous() / norm_weight_loop.index({large_left_indices + len_norm / 2})).to(torch::kFloat16);
    // //todo:currently multiply a scaley to convert it into fp16
    // auto sample_y3 = (qy.index({small_left_indices}) * scaley).to(torch::kFloat16).t().contiguous();
    // auto sample_y4 = (qy.index({large_left_indices}) * scaley).to(torch::kFloat16).t().contiguous();

    // auto dequantize_sample_y = torch::cat({sample_y1, sample_y2}, 1);

    // cudaDeviceSynchronize();
    // clock_t time_sample_end = clock();

    // pack process
    // auto option_transform = torch::TensorOptions().dtype(torch::kInt8).device(x.device());
    auto sample_x1_int4 = torch::empty({nx, small_num_>>1}, option_transform);
    auto sample_x2_int4 = torch::empty({nx, large_num_>>1}, option_transform);
    auto sample_y1_int4 = torch::empty({ny, small_num_>>1}, option_transform);
    auto sample_y2_int4 = torch::empty({ny, large_num_>>1}, option_transform);
    int grid_size_x1 = nx*small_num_/2;
    int grid_size_x2 = nx*large_num_/2;
    int grid_size_y1 = ny*small_num_/2;
    int grid_size_y2 = ny*large_num_/2;
    dim3 grid_pack_x1((grid_size_x1-1)/block.x+1);
    dim3 grid_pack_x2((grid_size_x2-1)/block.x+1);
    dim3 grid_pack_y1((grid_size_y1-1)/block.x+1);
    dim3 grid_pack_y2((grid_size_y2-1)/block.x+1);
    if (small_num_ > 0) {
        pack_cuda_kernel<<<grid_pack_x1,block>>>(sample_x1.data_ptr<int8_t>(), sample_x1_int4.data_ptr<int8_t>(), grid_size_x1);
        pack_cuda_kernel<<<grid_pack_y1,block>>>(sample_y1.data_ptr<int8_t>(), sample_y1_int4.data_ptr<int8_t>(), grid_size_y1);
    }
    if (large_num_ > 0) {
        pack_cuda_kernel<<<grid_pack_x2,block>>>(sample_x2.data_ptr<int8_t>(), sample_x2_int4.data_ptr<int8_t>(), grid_size_x2);
        pack_cuda_kernel<<<grid_pack_y2,block>>>(sample_y2.data_ptr<int8_t>(), sample_y2_int4.data_ptr<int8_t>(), grid_size_y2);
    }

    // cudaDeviceSynchronize();
    // clock_t time_pack_end = clock();

    // gemm process
    cudaError_t result;
    int lda_first = small_num_;
    int ldb_first = small_num_;
    int ldc = ny;
    // Chunked matrix multiplication
    auto gemm1 = torch::empty({nx,ny}, at::device(at::kCUDA).dtype(torch::kInt32));
    if (small_num_ > 0) {
        result = CutlassSgemmNN(nx, ny, small_num_, reinterpret_cast<cutlass::int4b_t *>(sample_x1_int4.data_ptr<int8_t>()), lda_first, reinterpret_cast<cutlass::int4b_t *>(sample_y1_int4.data_ptr<int8_t>()), ldb_first, gemm1.data_ptr<int32_t>(), ldc);
    } else {
        gemm1 = torch::zeros({nx,ny}, at::device(at::kCUDA).dtype(torch::kInt32));
    }
    // result = CutlassSgemmNN(nx, ny, small_num_, reinterpret_cast<cutlass::int4b_t *>(sample_x1_int4.data_ptr<int8_t>()), lda_first, reinterpret_cast<cutlass::int4b_t *>(sample_y1_int4.data_ptr<int8_t>()), ldb_first, gemm1.data_ptr<int32_t>(), ldc);

    int lda_second = large_num_;
    int ldb_second = large_num_;
    auto gemm2 = torch::empty({nx,ny}, at::device(at::kCUDA).dtype(torch::kInt32));
    if (large_num_ > 0) {
        result = CutlassSgemmNN(nx, ny, large_num_, reinterpret_cast<cutlass::int4b_t *>(sample_x2_int4.data_ptr<int8_t>()), lda_second, reinterpret_cast<cutlass::int4b_t *>(sample_y2_int4.data_ptr<int8_t>()), ldb_second, gemm2.data_ptr<int32_t>(), ldc);
    } else {
        gemm2 = torch::zeros({nx,ny}, at::device(at::kCUDA).dtype(torch::kInt32));
    }
    // result = CutlassSgemmNN(nx, ny, large_num_, reinterpret_cast<cutlass::int4b_t *>(sample_x2_int4.data_ptr<int8_t>()), lda_second, reinterpret_cast<cutlass::int4b_t *>(sample_y2_int4.data_ptr<int8_t>()), ldb_second, gemm2.data_ptr<int32_t>(), ldc);

    auto gemm3 = torch::matmul(sample_x3, sample_y3);
    auto gemm4 = torch::matmul(sample_x4, sample_y4);
    // auto gemm3 = torch::empty({nx, ny}, at::device(at::kCUDA).dtype(torch::kFloat16));
    // auto gemm4 = torch::empty({nx, ny}, at::device(at::kCUDA).dtype(torch::kFloat16));


    // auto gemm3 = torch::empty({nx,ny}, at::device(at::kCUDA).dtype(torch::kFloat16));

    // cudaDeviceSynchronize();
    // clock_t time_gemm1_end = clock();

    // auto gemm2 = torch::empty({nx,ny}, at::device(at::kCUDA).dtype(torch::kFloat16));

    // cudaDeviceSynchronize();
    // clock_t time_gemm2_end = clock();

    // dequantize process
    dim3 grid2((nx*ny-1)/block.x+1);
    // First dequantize higher 4 bits
    auto option_output = torch::TensorOptions().dtype(x.dtype()).device(x.device());
    // auto sum_y1_column = torch::sum(qy.index({small_int_indices}), 0);
    // auto sum_y2_column = torch::sum(qy.index({large_int_indices}), 0);
    auto grad_output = torch::empty({nx,ny}, option_output);

    // float const_x1 = (8.0 / scale1 + zero_point1) * scaley;
    // float const_x2 = (8.0 / scale2 + zero_point2) * scaley;
    // float const_x = const_x1 + const_x2;
    float scale_gemm1 = scaley * num_bins_half / (scale1);
    float scale_gemm2 = scaley / (scale1);
    int size = nx*ny;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad_output.scalar_type(), "dequantize_cuda", ([&] {
    dequantize_cuda_kernel<scalar_t><<<grid2, block>>>(
        gemm1.data_ptr<int32_t>(), 
        gemm2.data_ptr<int32_t>(),
        gemm3.data_ptr<scalar_t>(), 
        gemm4.data_ptr<scalar_t>(),
        grad_output.data_ptr<scalar_t>(),
        // sum_y1_column.data_ptr<int64_t>(),
        // sum_y2_column.data_ptr<int64_t>(),
        // const_x1, const_x2, 
        scale_gemm1, scale_gemm2,
        size);
    }));



    // AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad_output.scalar_type(), "dequantize_cuda", ([&] {
    // dequantize_cuda_kernel_fp16<scalar_t><<<grid2, block>>>(
    //     gemm1.data_ptr<scalar_t>(), 
    //     gemm2.data_ptr<scalar_t>(),
    //     grad_output.data_ptr<scalar_t>(),
    //     size);
    // }));

    auto q_w = hadamard_weight / scale_weight;
    auto indicate_small = (q_w < -8).to(torch::kFloat16);
    auto indicate_big = (q_w > 7).to(torch::kFloat16);
    auto indicate_middle = 1.0 - indicate_small - indicate_big;
    auto grad_scale = 1.0 / sqrt(hadamard_weight.numel() * 7);
    auto grad_alpha = ((indicate_small * -8 + indicate_big * 7 + indicate_middle * (
                    -q_w + q_w.round())) * grad_output * grad_scale).sum().unsqueeze(0);
    auto grad_input = indicate_middle * grad_output; 
    //TODO:to test use this way, later change into Gumble

    // cudaDeviceSynchronize();
    // clock_t time_dequantize_end = clock();

    // double quantize1_time = (double)(time_quantize1_end - time_quantize1_start) / CLOCKS_PER_SEC;
    // double quantize2_time = (double)(time_quantize2_end - time_quantize1_end) / CLOCKS_PER_SEC;
    // double leverage_time = (double)(time_leverage_end - time_quantize2_end) / CLOCKS_PER_SEC;
    // double sample_time = (double)(time_sample_end - time_leverage_end) / CLOCKS_PER_SEC;
    // double pack_time = (double)(time_pack_end - time_sample_end) / CLOCKS_PER_SEC;
    // double gemm1_time = (double)(time_gemm1_end - time_pack_end) / CLOCKS_PER_SEC;
    // double gemm2_time = (double)(time_gemm2_end - time_gemm1_end) / CLOCKS_PER_SEC;
    // double dequantize_time = (double)(time_dequantize_end - time_gemm2_end) / CLOCKS_PER_SEC;
    // // time_leverage_end

    // time_vector.push_back(quantize1_time);
    // time_vector.push_back(quantize2_time);
    // time_vector.push_back(leverage_time);
    // time_vector.push_back(sample_time);
    // time_vector.push_back(pack_time);
    // time_vector.push_back(gemm1_time);
    // time_vector.push_back(gemm2_time);
    // time_vector.push_back(dequantize_time);
    // auto sample_x = torch::cat({sample_x1, sample_x2}, 0);

    return std::make_tuple(grad_input, grad_alpha, grad_output);
    // return std::make_tuple(gemm1, gemm2, gemm3, gemm4, sum_y1_column, sum_y2_column);
}
