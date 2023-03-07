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
#include "torch/script.h"
#include "cuda_runtime.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include <curand_kernel.h>

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

//Todo:output high corresponds to scale1 and zero1, low corresponds to scale2 and zero2
template<typename scalar_t>
__global__ void dequantize_cuda_kernel(const int32_t * gemm1, const int32_t * gemm2, const float * norm_small, const float * norm_large,
                                        scalar_t * __restrict__ output_low, scalar_t * __restrict__ output_high, 
                                        const float scale_gemm1, const float scale_gemm2, int size, int ny){
    // extern __shared__ float s[];
    // float * y_col = s;  // N_THREADS float
    
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int row = x / ny;
    // , col = x - row * ny
    float norm1 = norm_small[row];
    float norm2 = norm_large[row];
    // int64_t sumY = sum_y_column[col];

    // y_col[threadIdx.x] = sum_y_column[col];
    // __syncthreads();

    if (x<size){
    //    output[x] = (gemm1[x] * scale_gemm1 + const_x1 * sumY) / norm1 + (gemm2[x] * scale_gemm2 + const_x2 * sumY) / norm2;
        // float tmp1 = (gemm1[x] * scale_gemm1 + const_x1 * sumY) / norm1;
        // output_low[x] = tmp1;
        // float tmp2 = (gemm2[x] * scale_gemm2 + const_x2 * sumY) / norm2;
        // output_high[x] = tmp2;
        output_low[x] = (gemm1[x] * scale_gemm1) / norm1;
        output_high[x] = (gemm2[x] * scale_gemm2) / norm2;
    }
}

template<typename scalar_t>
__global__ void norm_cuda_kernel(const float * norm_small, const float * norm_large, const scalar_t * __restrict__ output_low, const scalar_t * __restrict__ output_high, 
                                scalar_t * __restrict__ grad_output, int size, int ny){
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int row = x / ny;
    float norm1 = norm_small[row];
    float norm2 = norm_large[row];

    if (x<size){
    //    output[x] = (gemm1[x] * scale_gemm1 + const_x1 * sumY) / norm1 + (gemm2[x] * scale_gemm2 + const_x2 * sumY) / norm2;
        grad_output[x] = output_low[x] / norm1 + output_high[x] / norm2;
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

template<typename scalar_t>
__global__ void LSQ_cuda_kernel(const scalar_t * __restrict__ hadamard_activation, const scalar_t * __restrict__ grad_output, scalar_t * __restrict__ grad_alpha_out, 
                                scalar_t * __restrict__ grad_input, const float scale_activation, const float grad_scale, const int size){  
    int x = threadIdx.x + blockIdx.x * blockDim.x;

    if (x<size){
       float q_w = hadamard_activation[x] / scale_activation;
       scalar_t indicate_small = (q_w < -8);
       scalar_t indicate_big = (q_w > 7);
       scalar_t indicate_middle = 1.0 - indicate_small - indicate_big;
       scalar_t grad_out = grad_output[x];
       grad_alpha_out[x] = (indicate_small * -8 + indicate_big * 7 + indicate_middle * (-q_w + round(q_w))) * grad_out * grad_scale;
       grad_input[x] = indicate_middle * grad_out;
    }
}

__device__ __inline__ c10::Half __shfl_down_sync(const unsigned mask, const c10::Half var,
                                                 const unsigned int delta, const int width) {
  __half var_ = var;
  return __shfl_down_sync(mask, var_, delta, width);
}

//TODO: N means rows, D means cols
template<typename scalar_t>
__global__ void linalg_norm_cuda_kernel(const scalar_t * __restrict__ in, float * linalg, int N, int D, int stride_D){
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

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, std::vector<double>, int> quantize_cuda(torch::Tensor x, int num_bits, torch::Tensor qy, float scaley, torch::Tensor hadamard_activation, torch::Tensor scale_activation){
    std::vector<double> time_vector;
    int nx = x.size(0);
    int nz = x.size(1);
    int ny = qy.size(1);

    cudaDeviceSynchronize();
    clock_t time_quantize_start = clock();

    auto option_transform = torch::TensorOptions().dtype(torch::kInt8).device(x.device());
    auto option_quantize = torch::TensorOptions().dtype(x.dtype()).device(x.device());
    auto option_float = torch::TensorOptions().dtype(torch::kFloat32).device(x.device());
    torch::Tensor first_transform = torch::empty({nx, nz}, option_transform);
    torch::Tensor first_quantize = torch::empty({nx, nz}, option_quantize);
    torch::Tensor second_transform = torch::empty({nx, nz}, option_transform);
    torch::Tensor second_quantize = torch::empty({nx, nz}, option_quantize);
    
    dim3 block(N_THREADS);
    dim3 grid1((nx*nz-1)/block.x+1);
    int size_quantize = nx * nz ;
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

    cudaDeviceSynchronize();
    clock_t time_quantize_end = clock();

    // leverage score
    // TODO: use dim=0 because torch.linalg only supports dim=1
    int threads = 32;
    int blocks = nx;
    // auto I = torch::eye({nx}, option_quantize);
    // auto I2 = torch::cat({I, I}, 0);
    // auto x_sample = torch::cat({first_quantize, second_quantize}, 0);
    auto x1_len = torch::empty({nx,}, option_float);
    auto x2_len = torch::empty({nx,}, option_float);
    // auto I_len = torch::empty({nx,}, option_quantize);

    int stride_x = nz / 32;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(first_quantize.scalar_type(), "linalg_cuda", ([&] {
    linalg_norm_cuda_kernel<scalar_t><<<blocks, threads>>>(
        first_quantize.data_ptr<scalar_t>(), 
        x1_len.data_ptr<float>(),
        nx,nz,stride_x);
    }));
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(second_quantize.scalar_type(), "linalg_cuda", ([&] {
    linalg_norm_cuda_kernel<scalar_t><<<blocks, threads>>>(
        second_quantize.data_ptr<scalar_t>(), 
        x2_len.data_ptr<float>(),
        nx,nz,stride_x);
    }));
    // int stride_I = nx / 32;
    // AT_DISPATCH_FLOATING_TYPES_AND_HALF(I.scalar_type(), "linalg_cuda", ([&] {
    // linalg_norm_cuda_kernel<scalar_t><<<blocks, threads>>>(
    //     I.data_ptr<scalar_t>(), 
    //     I_len.data_ptr<scalar_t>(),
    //     nx,nx,stride_I);
    // }));
    auto vec_norm = torch::cat({x1_len, x2_len});
    int len_norm = vec_norm.numel();

    cudaDeviceSynchronize();
    clock_t time_leverage_end = clock();
    
    int cnt = 0;
    int whilenum = 0;
    auto norm_activation_loop = vec_norm * len_norm / (2 * vec_norm.sum());
    // cudaDeviceSynchronize();
    // clock_t time_sample1_end = clock();
    if ((norm_activation_loop > 0).sum().item<int>() < len_norm / 2){
        norm_activation_loop.index_put_({norm_activation_loop > 0}, 1);
    } else {
        bool whileloop = (norm_activation_loop.max() > 1).item<bool>();
        while (whileloop && cnt < len_norm / 2){
            auto small_index = torch::nonzero((norm_activation_loop < 1)).squeeze();
            auto small_value = norm_activation_loop.index({small_index});
            cnt = len_norm - small_index.numel();
            norm_activation_loop = torch::clamp(norm_activation_loop, 0, 1);
            bool breakloop = (small_value.max() == 0).item<bool>();
            if (breakloop)
                break;
            small_value = small_value * (len_norm / 2 - cnt) / small_value.sum();
            // norm_activation_loop[small_index] = small_value;
            norm_activation_loop.index_put_({small_index}, small_value);
            whileloop = (norm_activation_loop.max() > 1).item<bool>();
            whilenum++;
        } 
    }
    // cudaDeviceSynchronize();
    // clock_t time_sample2_end = clock();
    auto sample_index = torch::bernoulli(norm_activation_loop);
    // cudaDeviceSynchronize();
    // clock_t time_sample3_end = clock();
    // auto small_indices = torch::where(sample_index.index({Slice({None, len_norm/2})}) == 1)[0];
    auto small_left_indices = torch::nonzero(sample_index.index({Slice({None, len_norm/2})}) != 1).squeeze();
    // cudaDeviceSynchronize();
    // clock_t time_sample4_end = clock();
    // auto large_indices = torch::where(sample_index.index({Slice(len_norm/2)}) == 1)[0];
    auto large_left_indices = torch::nonzero(sample_index.index({Slice(len_norm/2)}) != 1).squeeze();
    // cudaDeviceSynchronize();
    // clock_t time_sample5_end = clock();
    // auto _index = torch::nonzero((sample_index == 1)).squeeze();

    norm_activation_loop.index_put_({norm_activation_loop == 0}, 1);
    // cudaDeviceSynchronize();
    // clock_t time_sample6_end = clock();
    // auto index_norm = vec_norm * len(vec_norm) / (2 * vec_norm.sum());

    // sample process
    auto sample_x1 = first_transform;
    auto sample_x2 = second_transform;
    // auto sample_x1 = (first_quantize / norm_activation_loop.index({Slice({None, len_norm/2})}).unsqueeze(1)).to(torch::kFloat16);
    // auto sample_x2 = (second_quantize / norm_activation_loop.index({Slice(len_norm/2)}).unsqueeze(1)).to(torch::kFloat16);
    // auto Ind_small = torch::zeros_like(sample_x1);
    // Ind_small.index_put_({small_indices,"..."}, 1);
    // sample_x1 = sample_x1.mul(Ind_small);
    // auto Ind_large = torch::zeros_like(sample_x2);
    // Ind_large.index_put_({large_indices,"..."}, 1);
    // sample_x2 = sample_x2.mul(Ind_large);
    // auto sample_y = (qy * scaley).to(torch::kFloat16).t().contiguous();
    auto sample_y = qy.t().contiguous();

    cudaDeviceSynchronize();
    clock_t time_sample_end = clock();

    // pack process
    auto sample_x1_int4 = torch::empty({nx, nz>>1}, option_transform);
    auto sample_x2_int4 = torch::empty({nx, nz>>1}, option_transform);
    auto sample_y_int4 = torch::empty({ny, nz>>1}, option_transform);
    int grid_size_x = nx*nz/2;
    int grid_size_y = nz*ny/2;
    dim3 grid_pack_x((grid_size_x-1)/block.x+1);
    dim3 grid_pack_y((grid_size_y-1)/block.x+1);
    pack_cuda_kernel<<<grid_pack_x,block>>>(sample_x1.data_ptr<int8_t>(), sample_x1_int4.data_ptr<int8_t>(), grid_size_x);
    pack_cuda_kernel<<<grid_pack_x,block>>>(sample_x2.data_ptr<int8_t>(), sample_x2_int4.data_ptr<int8_t>(), grid_size_x);
    pack_cuda_kernel<<<grid_pack_y,block>>>(sample_y.data_ptr<int8_t>(), sample_y_int4.data_ptr<int8_t>(), grid_size_y);

    cudaDeviceSynchronize();
    clock_t time_pack_end = clock();

    // gemm process
    cudaError_t result;
    int lda = nz;
    int ldb = nz;
    int ldc = ny;
    // Chunked matrix multiplication
    auto gemm1 = torch::empty({nx,ny}, at::device(at::kCUDA).dtype(torch::kInt32));
    result = CutlassSgemmNN(nx, ny, nz, reinterpret_cast<cutlass::int4b_t *>(sample_x1_int4.data_ptr<int8_t>()), lda, reinterpret_cast<cutlass::int4b_t *>(sample_y_int4.data_ptr<int8_t>()), ldb, gemm1.data_ptr<int32_t>(), ldc);
    // auto gemm1 = torch::empty({nx,ny}, at::device(at::kCUDA).dtype(torch::kFloat16));
    // result = CutlassSgemmNN_fp16(nx, ny, nz, reinterpret_cast<cutlass::half_t *>(sample_x1.data_ptr()), lda, reinterpret_cast<cutlass::half_t *>(sample_y.data_ptr()), ldb, reinterpret_cast<cutlass::half_t *>(gemm1.data_ptr()), ldc);

    // cudaDeviceSynchronize();
    // clock_t time_gemm1_end = clock();

    auto gemm2 = torch::empty({nx,ny}, at::device(at::kCUDA).dtype(torch::kInt32));
    result = CutlassSgemmNN(nx, ny, nz, reinterpret_cast<cutlass::int4b_t *>(sample_x2_int4.data_ptr<int8_t>()), lda, reinterpret_cast<cutlass::int4b_t *>(sample_y_int4.data_ptr<int8_t>()), ldb, gemm2.data_ptr<int32_t>(), ldc);
    // auto gemm2 = torch::empty({nx,ny}, at::device(at::kCUDA).dtype(torch::kFloat16));
    // result = CutlassSgemmNN_fp16(nx, ny, nz, reinterpret_cast<cutlass::half_t *>(sample_x2.data_ptr()), lda, reinterpret_cast<cutlass::half_t *>(sample_y.data_ptr()), ldb, reinterpret_cast<cutlass::half_t *>(gemm2.data_ptr()), ldc);
    cudaDeviceSynchronize();
    clock_t time_gemm_end = clock();

    // dequantize process
    dim3 grid2((nx*ny-1)/block.x+1);
    // First dequantize higher 4 bits
    auto option_output = torch::TensorOptions().dtype(torch::kFloat16).device(gemm1.device());
    // auto sum_y_column = torch::sum(qy, 0);
    auto output_low = torch::empty({nx,ny}, option_output);
    auto output_high = torch::empty({nx,ny}, option_output);
    // auto grad_output = torch::empty({nx,ny}, option_output);

    // float const_x1 = (8.0 / scale1 + zero_point1) * scaley;
    // float const_x2 = (8.0 / scale2 + zero_point2) * scaley;
    float scale_gemm1 = scaley * num_bins_half / (scale1);
    float scale_gemm2 = scaley / (scale1);
    auto norm_small = norm_activation_loop.index({Slice({None, len_norm/2})});
    auto norm_large = norm_activation_loop.index({Slice(len_norm/2)});
    int size = nx*ny;

    // AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad_output.scalar_type(), "dequantize_cuda", ([&] {
    // dequantize_cuda_kernel_fp16<scalar_t><<<grid2, block>>>(
    //     gemm1.data_ptr<scalar_t>(), 
    //     gemm2.data_ptr<scalar_t>(),
    //     grad_output.data_ptr<scalar_t>(),
    //     size);
    // }));


    AT_DISPATCH_FLOATING_TYPES_AND_HALF(output_low.scalar_type(), "dequantize_cuda", ([&] {
    // dequantize_cuda_kernel<scalar_t><<<grid2, block, N_THREADS * sizeof(float)>>>(
    dequantize_cuda_kernel<scalar_t><<<grid2, block>>>(
        gemm1.data_ptr<int32_t>(), 
        gemm2.data_ptr<int32_t>(),
        norm_small.data_ptr<float>(),
        norm_large.data_ptr<float>(),
        output_low.data_ptr<scalar_t>(),
        output_high.data_ptr<scalar_t>(),
        // sum_y_column.data_ptr<int64_t>(),
        // const_x1, const_x2, 
        scale_gemm1, scale_gemm2,
        size, ny);
    }));

    cudaDeviceSynchronize();
    clock_t time_dequantize1_end = clock();

    output_low.index_fill_(0, small_left_indices, 0);
    output_high.index_fill_(0, large_left_indices, 0);
    cudaDeviceSynchronize();
    clock_t time_dequantize2_end = clock();
    auto grad_output = torch::empty({nx,ny}, option_output);
    grad_output = output_low + output_high;

    // auto Ind_small = torch::zeros_like(output_low);
    // Ind_small.index_put_({small_indices,"..."}, 1);
    // // auto Index1 = torch::nonzero(norm_small == 1e-10).squeeze();
    // // Ind_small.index_put_({Index1,"..."}, 0);
    // output_low = output_low.mul(Ind_small);
    // auto Ind_large = torch::zeros_like(output_high);
    // Ind_large.index_put_({large_indices,"..."}, 1);
    // // auto Index2 = torch::nonzero(norm_large == 1e-10).squeeze();
    // // Ind_large.index_put_({Index2,"..."}, 0);
    // output_high = output_high.mul(Ind_large);

    // AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad_output.scalar_type(), "norm_cuda", ([&] {
    // norm_cuda_kernel<scalar_t><<<grid2, block>>>(
    //     norm_small.data_ptr<float>(),
    //     norm_large.data_ptr<float>(),
    //     output_low.data_ptr<scalar_t>(),
    //     output_high.data_ptr<scalar_t>(),
    //     grad_output.data_ptr<scalar_t>(),
    //     size, ny);
    // }));

    cudaDeviceSynchronize();
    clock_t time_dequantize_end = clock();

    // auto grad_output = output_low;
    float grad_scale = 1.0 / sqrt(hadamard_activation.numel() * 7);
    auto grad_alpha_out = torch::empty({nx,ny}, option_output);
    auto grad_input = torch::empty({nx,ny}, option_output);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad_output.scalar_type(), "LSQ_cuda", ([&] {
    LSQ_cuda_kernel<scalar_t><<<grid2, block>>>(
        hadamard_activation.data_ptr<scalar_t>(), 
        grad_output.data_ptr<scalar_t>(),
        grad_alpha_out.data_ptr<scalar_t>(),
        grad_input.data_ptr<scalar_t>(),
        scale_activation.item<float>(), grad_scale, size);
    }));

    // auto q_w = hadamard_activation / scale_activation;
    // auto indicate_small = (q_w < -8).to(torch::kFloat16);
    // auto indicate_big = (q_w > 7).to(torch::kFloat16);
    // auto indicate_middle = 1.0 - indicate_small - indicate_big;
    // auto grad_alpha = ((indicate_small * -8 + indicate_big * 7 + indicate_middle * (
    //                 -q_w + q_w.round())) * grad_output * grad_scale).sum().unsqueeze(0);
    // auto grad_input = indicate_middle * grad_output;
    auto grad_alpha = grad_alpha_out.sum().unsqueeze(0);
    cudaDeviceSynchronize();
    clock_t time_LSQ_end = clock();

    // using namespace torch;
    // torch::distributions::Gumbel gumbel_distribution(norm_activation, torch::ones_like(norm_activation));
    // auto activation_phi = gumbel_distribution.rsample();

    // auto activation_phi = norm_activation;
    

    // cudaDeviceSynchronize();
    // clock_t time_LSS_end = clock();


    double quantize_time = (double)(time_quantize_end - time_quantize_start) / CLOCKS_PER_SEC;
    double leverage_time = (double)(time_leverage_end - time_quantize_end) / CLOCKS_PER_SEC;
    double sample_time = (double)((time_sample_end - time_leverage_end)) / CLOCKS_PER_SEC;
    // double sample1_time = (double)((time_sample1_end - time_leverage_end)) / CLOCKS_PER_SEC;
    // double sample2_time = (double)((time_sample2_end - time_sample1_end)) / CLOCKS_PER_SEC;
    // double sample3_time = (double)((time_sample3_end - time_sample2_end)) / CLOCKS_PER_SEC;
    // double sample4_time = (double)((time_sample4_end - time_sample3_end)) / CLOCKS_PER_SEC;
    // double sample5_time = (double)((time_sample5_end - time_sample4_end)) / CLOCKS_PER_SEC;
    // double sample6_time = (double)((time_sample6_end - time_sample5_end)) / CLOCKS_PER_SEC;
    // double sample7_time = (double)((time_sample_end - time_sample6_end)) / CLOCKS_PER_SEC;
    double pack_time = (double)(time_pack_end - time_sample_end) / CLOCKS_PER_SEC;
    double gemm_time = (double)(time_gemm_end - time_pack_end) / CLOCKS_PER_SEC;
    // double gemm2_time = (double)(time_gemm2_end - time_gemm1_end) / CLOCKS_PER_SEC;
    double dequantize1_time = (double)(time_dequantize1_end - time_gemm_end) / CLOCKS_PER_SEC;
    double dequantize2_time = (double)(time_dequantize2_end - time_dequantize1_end) / CLOCKS_PER_SEC;
    double dequantize3_time = (double)(time_dequantize_end - time_dequantize2_end) / CLOCKS_PER_SEC;
    double LSQ_time = (double)(time_LSQ_end - time_dequantize_end) / CLOCKS_PER_SEC;
    // double LSS_time = (double)(time_LSS_end - time_dequantize_end) / CLOCKS_PER_SEC;
    // // time_leverage_end

    time_vector.push_back(quantize_time);
    time_vector.push_back(leverage_time);
    time_vector.push_back(sample_time);
    // time_vector.push_back(sample1_time);
    // time_vector.push_back(sample2_time);
    // time_vector.push_back(sample3_time);
    // time_vector.push_back(sample4_time);
    // time_vector.push_back(sample5_time);
    // time_vector.push_back(sample6_time);
    // time_vector.push_back(sample7_time);
    time_vector.push_back(pack_time);
    time_vector.push_back(gemm_time);
    // time_vector.push_back(gemm2_time);
    time_vector.push_back(dequantize1_time);
    time_vector.push_back(dequantize2_time);
    time_vector.push_back(dequantize3_time);
    // time_vector.push_back(LSS_time);
    time_vector.push_back(LSQ_time);

    return std::make_tuple(grad_input, grad_alpha, output_low, output_high, time_vector, whilenum);
}
