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

#define N_THREADS 256

template<typename scalar_t>
__global__ void quantize_cuda_kernel(const scalar_t * __restrict__  MatI, int8_t * MatO, const float scale, long long int size){
    long long int x = threadIdx.x + blockIdx.x * blockDim.x;
    if (x<size){
        MatO[x] = (int)(round(MatI[x] * scale));
    }
}

cudaError_t CutlassSgemmNN(
  const int M,
  const int N,
  const int K,
  const int8_t *A,
  int lda,
  const int8_t *B,
  int ldb,
  int32_t *C,
  int ldc) {

// The code section below describes datatype for input, output matrices and computation between
// elements in input matrices.
using ElementAccumulator = int32_t;                 // <- data type of accumulator
using ElementComputeEpilogue = ElementAccumulator;  // <- data type of epilogue operations
using ElementInputA = int8_t;                       // <- data type of elements in input matrix A
using ElementInputB = int8_t;                       // <- data type of elements in input matrix B
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
    cutlass::gemm::GemmShape<256, 128, 64>;  // <- threadblock tile M = 128, N = 256, K = 64
// This code section describes tile size a warp will compute
using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 64>;  // <- warp tile M = 64, N = 64, K = 64 
// This code section describes the size of MMA op
using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 16>;  // <- MMA Op tile M = 8, N = 8, K = 16

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
  
    // Using the arguments, query for extra workspace required for matrix multiplication computation
    // size_t workspace_size = Gemm::get_workspace_size(arguments);

    // Allocate workspace memory
    // cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    // Instantiate CUTLASS kernel depending on templates
    Gemm gemm_op;

    // Initialize CUTLASS kernel with arguments and workspace pointer
    // cutlass::Status status = gemm_op.initialize(arguments, workspace.get());

    // Launch initialized CUTLASS kernel
    cutlass::Status status = gemm_op(arguments);

  if (status != cutlass::Status::kSuccess) {
    return cudaErrorUnknown;
  }

  // Return success, if no errors were encountered.
  return cudaSuccess;
}

std::tuple<torch::Tensor, torch::Tensor> quantize_cuda(torch::Tensor grad_output_flatten, torch::Tensor q_input_flatten, torch::Tensor q_weight, float scale_input, float scale_weight, float scale_grad){
    // grad_output is nx * nz, weight is nz * ny, input is nx * ny
    // weight ~ q_weight * scale_weight, input ~ q_input * scale_input
    // grad_weight ~ grad_out.t().mm(input), grad_input ~ grad_out.mm(weight)
    cudaError_t result;
    long long int nx = grad_output_flatten.size(0);
    long long int nz = grad_output_flatten.size(1);
    long long int ny = q_weight.size(1);

    auto option_quantize = torch::TensorOptions().dtype(torch::kInt8).device(grad_output_flatten.device());
    auto option_gemm = torch::TensorOptions().dtype(torch::kInt32).device(grad_output_flatten.device());
    
    dim3 block(N_THREADS);

    dim3 grid1((nx*nz-1)/(block.x)+1);
    torch::Tensor q_grad = torch::empty({nx,nz}, option_quantize);
    long long int grad_size = (nx*nz);
    // process of quantize
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad_output_flatten.scalar_type(), "quantize_cuda", ([&] {
    quantize_cuda_kernel<scalar_t><<<grid1, block>>>(
        grad_output_flatten.data_ptr<scalar_t>(),
        q_grad.data_ptr<int8_t>(),
        scale_grad,grad_size);
    }));

    //TODO: then int8 gemm
    int lda = nz;
    int ldb = nz;
    int ldc = ny;
    torch::Tensor gemm_input = torch::empty({nx, ny}, option_gemm);
    result = CutlassSgemmNN(nx, ny, nz, q_grad.data_ptr<int8_t>(), lda, 
            q_weight.t().contiguous().data_ptr<int8_t>(), ldb, gemm_input.data_ptr<int32_t>(), ldc);

    lda = nx;
    ldb = nx;
    torch::Tensor gemm_weight = torch::empty({nz, ny}, option_gemm);
    result = CutlassSgemmNN(nz, ny, nx, q_grad.t().contiguous().data_ptr<int8_t>(), lda, 
            q_input_flatten.t().contiguous().data_ptr<int8_t>(), ldb, gemm_weight.data_ptr<int32_t>(), ldc);
    
    float scale_grad_input = 1. / (scale_grad * scale_weight);
    torch::Tensor grad_input = gemm_input * scale_grad_input;
    float scale_grad_weight = 1. / (scale_grad * scale_input);
    torch::Tensor grad_weight = gemm_weight * scale_grad_weight;


    if (result != cudaSuccess) {
        std::cerr << "cudaGetDeviceProperties() returned an error during int8 gemm: " << cudaGetErrorString(result) << std::endl;
    }

    return std::make_tuple(grad_input, grad_weight);
}