/***************************************************************************************************
 * Copyright (c) 2017 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/**

This example adopts example 16 to use 3xTF32 to bring FP32 accuracy with 2x performance
compared with CUDA Cores.  See example 27 for the trick of 3xTF32. 
*/

#include <iostream>
#include <fstream>
#include <sstream>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/conv/kernel/default_conv2d_fprop.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/device/convolution.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/convolution.h"
#include "cutlass/util/reference/host/error_metrics.h"
#include "cutlass/util/tensor_view_io.h"

#include "helper.h"

#ifdef KH_TEST_SOLUTION
#include "solution.h"
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////

// The code section below describes datatype for input, output tensors and computation between
// elements 
using ElementAccumulator = float;                  // Data type of accumulator
using ElementComputeEpilogue = float;              // Data type of epilogue computation (alpha, beta)
using ElementInputA = float;                       // Data type of elements in input tensor
using ElementInputB = float;                       // Data type of elements in input tensor
using ElementOutput = float;                       // Data type of elements in output tensor

using LayoutInputA = cutlass::layout::TensorNHWC;
using LayoutInputB = cutlass::layout::TensorNHWC;
using LayoutOutput = cutlass::layout::TensorNHWC;

// This code section describes whether you want to use tensor cores or regular SIMT cores on GPU SM
using MMAOp = cutlass::arch::OpClassTensorOp;

// This code section describes CUDA SM architecture number
using SmArch = cutlass::arch::Sm80;

// This code section describes the tile size a thread block will compute
using ThreadblockShape = cutlass::gemm::GemmShape<128, 64, 16>;  // Threadblock tile shape

// This code section describes tile size a warp will compute
using WarpShape = cutlass::gemm::GemmShape<64, 32, 16>;         // Warp tile shape

// This code section describes the size of MMA op
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;    // TensorCore instruction shape

// This code section describes how threadblocks are scheduled on GPU
using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

// Number of pipelines you want to use
constexpr int NumStages = 3;

// This code section describe iterator algorithm selected is Analytic or Optimized
static cutlass::conv::IteratorAlgorithm const IteratorAlgorithm = cutlass::conv::IteratorAlgorithm::kOptimized;

// This code section describes the epilogue part of the kernel, we use default value
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,                                     // Data type of output matrix.
    128 / cutlass::sizeof_bits<ElementOutput>::value,  // The number of elements per vectorized.
                                                       // memory access. This becomes the vector width of
                                                       // math instructions in the epilogue too.
    ElementAccumulator,                                // Data type of accumulator
    ElementComputeEpilogue>;                           // Data type for alpha/beta in linear combination

// 3xTF32 Fprop
using Conv2dFpropKernel_3xTF32 = typename cutlass::conv::kernel::DefaultConv2dFprop<
  ElementInputA, LayoutInputA,
  ElementInputB, LayoutInputB,
  ElementOutput, LayoutOutput,
  ElementAccumulator,
  MMAOp,
  SmArch,
  ThreadblockShape,
  WarpShape,
  InstructionShape,
  EpilogueOp,
  SwizzleThreadBlock,
  NumStages,
  // Only thing needs to be changed from normal Fprop
  cutlass::arch::OpMultiplyAddFastF32,
  IteratorAlgorithm
>::Kernel;

// 1xTF32 Fprop
using Conv2dFpropKernel_1xTF32 = typename cutlass::conv::kernel::DefaultConv2dFprop<
  ElementInputA, LayoutInputA,
  ElementInputB, LayoutInputB,
  ElementOutput, LayoutOutput,
  ElementAccumulator,
  MMAOp,
  SmArch,
  ThreadblockShape,
  WarpShape,
  InstructionShape,
  EpilogueOp,
  SwizzleThreadBlock,
  NumStages,
  cutlass::arch::OpMultiplyAdd,
  IteratorAlgorithm
>::Kernel;

using ImplicitGemm_3xTF32 = cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel_3xTF32>;
using ImplicitGemm_1xTF32 = cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel_1xTF32>;

/////////////////////////////////////////////////////////////////////////////////////////////////

// Command line options parsing
struct Options {

  bool help;
  cutlass::Tensor4DCoord input_size;
  cutlass::Tensor4DCoord filter_size;
  cutlass::Tensor4DCoord padding;
  cutlass::MatrixCoord conv_stride;
  cutlass::MatrixCoord dilation;
  int iterations;
  bool save_workspace;
  ElementComputeEpilogue alpha;
  ElementComputeEpilogue beta;
  bool benchmark;
  std::string tag;

  Options():
    help(false),
    input_size(1, 32, 32, 32),
    filter_size(32, 3, 3, 32),
    padding(1, 1, 1, 1),
    conv_stride(1, 1),
    dilation(1, 1),
    iterations(20),
    save_workspace(false),
    alpha(1),
    beta(0),
    benchmark(false) { }

  // Verify the problem size is compatible with the CUTLASS Convolution implementation.
  bool valid() {

    //
    // CUTLASS attempts to load 128b vectors of cutlass::half_t (F16) elements. Consequently,
    // all pointers, strides, and tensor extents must be divisible by 8 elements.
    //
    int const kAlignment = 4;

    if ((input_size.c() % kAlignment) ||
      (filter_size.n() % kAlignment)) {

      // misaligned tensors
      return false;
    }

    // Invalid padding
    if ((padding.h() != filter_size.h() / 2) ||
      (padding.w() != filter_size.w() / 2)) {

      return false;
    }

    return true;
  }

  /// Updates input and filter sizes
  void update(
    cutlass::Tensor4DCoord input_size,
    cutlass::Tensor4DCoord filter_size) {

    this->input_size = input_size;
    this->filter_size = filter_size;

    padding.n() = filter_size.h() / 2;
    padding.h() = filter_size.h() / 2;
    padding.w() = filter_size.w() / 2;
    padding.c() = filter_size.w() / 2;
  }

  // Parses the command line
  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
    }

    if (cmd.check_cmd_line_flag("save-workspace")) {
      save_workspace = true;
    }

    if (cmd.check_cmd_line_flag("benchmark")) {
      benchmark = true;
    }

    cmd.get_cmd_line_argument("n", input_size.n());
    cmd.get_cmd_line_argument("h", input_size.h());
    cmd.get_cmd_line_argument("w", input_size.w());
    cmd.get_cmd_line_argument("c", input_size.c());

    cmd.get_cmd_line_argument("k", filter_size.n());
    cmd.get_cmd_line_argument("r", filter_size.h());
    cmd.get_cmd_line_argument("s", filter_size.w());
    filter_size.c() = input_size.c(); 

    cmd.get_cmd_line_argument("alpha", alpha);
    cmd.get_cmd_line_argument("beta", beta);
    
    cmd.get_cmd_line_argument("iterations", iterations);
    cmd.get_cmd_line_argument("tag", tag);

    if (filter_size.h() == 3 && filter_size.w() == 3) {
      padding = {1, 1, 1, 1};
    }
    else {
      filter_size.h() = 1;
      filter_size.w() = 1;
      padding = {0, 0, 0, 0};
    }
  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "28_ampere_3xtf32_fast_accurate_tensorop_fprop example\n\n"
      << "  This example uses Ampere's Tensor Core operators on F16 data types to compute\n"
      << "  forward convolution on tensors of layout NHWC.\n\n"
      << "Options:\n\n"
      << "  --help               If specified, displays this usage statement.\n\n"
      << "  --n=<int>            Input tensor extent N\n"
      << "  --h=<int>            Input tensor extent H\n"
      << "  --w=<int>            Input tensor extent W\n"
      << "  --c=<int>            Input tensor extent C\n"
      << "  --k=<int>            Filter extent K\n"
      << "  --r=<int>            Filter extent R\n"
      << "  --s=<int>            Filter extent S\n\n"
      << "  --alpha=<float>      Epilogue scalar alpha\n"
      << "  --beta=<float>       Epilogue scalar beta\n\n"
      << "  --benchmark          If set (true), performance benchmarking on several layers and batch-size.\n"
      << "  --iterations=<int>   Number of profiling iterations to perform.\n"
      << "  --save-workspace     If set, workspace is written to a text file.\n"
      << "  --tag=<string>       String to replicate across the first column in the results table\n";

    out << "\n\nExamples:\n\n"
      << "$ ./examples/28_ampere_3xtf32_fast_accurate_tensorop_fprop/28_ampere_3xtf32_fast_accurate_tensorop_fprop  --n=32 --h=224 --w=224 --c=128 --k=256 --r=1 --s=1\n\n"
      << "$ ./examples/28_ampere_3xtf32_fast_accurate_tensorop_fprop/28_ampere_3xtf32_fast_accurate_tensorop_fprop  --n=1 --h=224 --w=224 --c=32 --k=32 --r=3 --s=3 --ref-check\n\n";

    return out;
  }
  
  /// Computes the output tensor size (NPQK)
  cutlass::Tensor4DCoord output_size() const {
    return cutlass::Tensor4DCoord(
      input_size.n(),
      (input_size.h() + padding.n() + padding.h() - filter_size.h()) / conv_stride.row() + 1,
      (input_size.w() + padding.w() + padding.c() - filter_size.w()) / conv_stride.column() + 1,
      filter_size.n());
  }

  /// Compute performance in GFLOP/s
  double gflops(double runtime_s) const {

    // Number of multiply-adds = NPQK * CRS
    int64_t fmas = output_size().product() * int64_t(filter_size.h() * filter_size.w() * filter_size.c());
    
    // Two flops per multiply-add
    return 2.0 * double(fmas) / double(1.0e9) / runtime_s;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

struct Result {
  double runtime_ms;
  double gflops;
  cutlass::Status status;
  cudaError_t error;

  double l2_norm_3xtf32_vs_fp64;
  double l2_norm_1xtf32_vs_fp64;
  double l2_norm_fp32_vs_fp64;

  Result(): 
    runtime_ms(0), 
    gflops(0),
    status(cutlass::Status::kSuccess),
    error(cudaSuccess),
    l2_norm_3xtf32_vs_fp64(0),
    l2_norm_1xtf32_vs_fp64(0),
    l2_norm_fp32_vs_fp64(0) { }

  static std::ostream & print_header(std::ostream &out, Options const &options) {

    if (!options.tag.empty()) {
      out << "Name,";
    }

    out << "Layer,N,H,W,C,K,R,S,Runtime,GFLOPs,3xTF32_vs_FP64,1xTF32_vs_FP64,FP32_vs_FP64";

    return out;
  }

  std::ostream & print(std::ostream &out, int idx, Options const &options) {

    if (!options.tag.empty()) {
      out << options.tag << ",";
    }

    out 
      << "conv_" << idx << ","
      << options.input_size.n() << ","
      << options.input_size.h() << ","
      << options.input_size.w() << ","
      << options.input_size.c() << ","
      << options.filter_size.n() << ","
      << options.filter_size.h() << ","
      << options.filter_size.w() << ","
      << runtime_ms << ","
      << gflops << ","
      << l2_norm_3xtf32_vs_fp64 << ","
      << l2_norm_1xtf32_vs_fp64 << ","
      << l2_norm_fp32_vs_fp64;

    return out;
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Run CUTLASS 3xTF32 Conv2d Fprop on given device buffers
cudaError_t CutlassConv2dFprop3xTF32(
    float const *input, float const *filter, float *output,
    int N, int C, int H, int W, int K, int R, int S,
    int pad_h, int pad_w, int stride_h, int stride_w) {

  cutlass::Tensor4DCoord input_size(N, H, W, C);
  cutlass::Tensor4DCoord filter_size(K, R, S, C);
  cutlass::Tensor4DCoord padding(pad_h, pad_h, pad_w, pad_w);
  cutlass::MatrixCoord conv_stride(stride_h, stride_w);
  cutlass::MatrixCoord dilation(1, 1);

  int P = (H + 2 * pad_h - R) / stride_h + 1;
  int Q = (W + 2 * pad_w - S) / stride_w + 1;
  cutlass::Tensor4DCoord output_size(N, P, Q, K);

  cutlass::conv::Conv2dProblemSize problem_size(
      input_size, filter_size, padding, conv_stride, dilation, output_size,
      cutlass::conv::Mode::kCrossCorrelation, 1);

  using TensorRefA = cutlass::TensorRef<float, LayoutInputA>;
  using TensorRefB = cutlass::TensorRef<float, LayoutInputB>;
  using TensorRefC = cutlass::TensorRef<float, LayoutOutput>;

  TensorRefA ref_input(const_cast<float*>(input), LayoutInputA::packed(input_size));
  TensorRefB ref_filter(const_cast<float*>(filter), LayoutInputB::packed(filter_size));
  TensorRefC ref_output(output, LayoutOutput::packed(output_size));

  typename ImplicitGemm_3xTF32::Arguments arguments{
    problem_size, ref_input, ref_filter, ref_output, ref_output,
    {1.0f, 0.0f}
  };

  ImplicitGemm_3xTF32 op;
  size_t ws_size = op.get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(ws_size);

  cutlass::Status status = op.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) return cudaErrorUnknown;

  status = op.initialize(arguments, workspace.get());
  if (status != cutlass::Status::kSuccess) return cudaErrorUnknown;

  status = op();
  if (status != cutlass::Status::kSuccess) return cudaErrorUnknown;

  return cudaSuccess;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Naive GPU reference for conv2d fprop, NHWC layout
__global__ void ReferenceConv2dFprop_kernel(
    float const *input, float const *filter, float *output,
    int N, int C, int H, int W, int K, int R, int S,
    int P, int Q, int pad_h, int pad_w, int stride_h, int stride_w) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = N * P * Q * K;
  if (idx >= total) return;

  int k = idx % K; idx /= K;
  int q = idx % Q; idx /= Q;
  int p = idx % P; idx /= P;
  int n = idx;

  float acc = 0.0f;
  for (int r = 0; r < R; ++r)
    for (int s = 0; s < S; ++s)
      for (int c = 0; c < C; ++c) {
        int h = p * stride_h + r - pad_h;
        int w = q * stride_w + s - pad_w;
        if (h >= 0 && h < H && w >= 0 && w < W) {
          acc += input[((n * H + h) * W + w) * C + c]
               * filter[((k * R + r) * S + s) * C + c];
        }
      }
  output[((n * P + p) * Q + q) * K + k] = acc;
}

cudaError_t ReferenceConv2dFprop(
    float const *input, float const *filter, float *output,
    int N, int C, int H, int W, int K, int R, int S,
    int pad_h, int pad_w, int stride_h, int stride_w) {
  int P = (H + 2 * pad_h - R) / stride_h + 1;
  int Q = (W + 2 * pad_w - S) / stride_w + 1;
  int total = N * P * Q * K;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  ReferenceConv2dFprop_kernel<<<blocks, threads>>>(
      input, filter, output, N, C, H, W, K, R, S,
      P, Q, pad_h, pad_w, stride_h, stride_w);
  return cudaGetLastError();
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Kernel to initialize a tensor with small integers.
__global__ void InitializeTensor_kernel(float *data, int count, int seed) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) {
    int const k = 16807;
    int const m = 16;
    float value = float(((i + seed) * k % m) - m / 2);
    data[i] = value;
  }
}

cudaError_t InitializeTensor(float *data, int count, int seed) {
  int threads = 256;
  int blocks = (count + threads - 1) / threads;
  InitializeTensor_kernel<<<blocks, threads>>>(data, count, seed);
  return cudaGetLastError();
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Test correctness: CUTLASS ref vs naive, and (if KH_TEST_SOLUTION) solution vs CUTLASS.
cudaError_t TestCorrectness(int N, int C, int H, int W, int K, int R, int S,
                            int pad_h, int pad_w, int stride_h, int stride_w) {
  int P = (H + 2 * pad_h - R) / stride_h + 1;
  int Q = (W + 2 * pad_w - S) / stride_w + 1;

  int input_count = N * H * W * C;
  int filter_count = K * R * S * C;
  int output_count = N * P * Q * K;

  float *d_input, *d_filter, *d_out_cutlass, *d_out_naive;
  cudaMalloc(&d_input, input_count * sizeof(float));
  cudaMalloc(&d_filter, filter_count * sizeof(float));
  cudaMalloc(&d_out_cutlass, output_count * sizeof(float));
  cudaMalloc(&d_out_naive, output_count * sizeof(float));

  InitializeTensor(d_input, input_count, 0);
  InitializeTensor(d_filter, filter_count, 17);
  cudaMemset(d_out_cutlass, 0, output_count * sizeof(float));
  cudaMemset(d_out_naive, 0, output_count * sizeof(float));

  // Run CUTLASS reference
  cudaError_t result = CutlassConv2dFprop3xTF32(d_input, d_filter, d_out_cutlass,
      N, C, H, W, K, R, S, pad_h, pad_w, stride_h, stride_w);
  if (result != cudaSuccess) {
    std::cerr << "CUTLASS Conv2d failed: " << cudaGetErrorString(result) << std::endl;
    cudaFree(d_input); cudaFree(d_filter); cudaFree(d_out_cutlass); cudaFree(d_out_naive);
    return result;
  }
  cudaDeviceSynchronize();

  // Run naive reference
  result = ReferenceConv2dFprop(d_input, d_filter, d_out_naive,
      N, C, H, W, K, R, S, pad_h, pad_w, stride_h, stride_w);
  if (result != cudaSuccess) {
    std::cerr << "Reference Conv2d failed: " << cudaGetErrorString(result) << std::endl;
    cudaFree(d_input); cudaFree(d_filter); cudaFree(d_out_cutlass); cudaFree(d_out_naive);
    return result;
  }
  cudaDeviceSynchronize();

  // Compare CUTLASS vs naive
  std::vector<float> h_cutlass(output_count), h_naive(output_count);
  cudaMemcpy(h_cutlass.data(), d_out_cutlass, output_count * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_naive.data(), d_out_naive, output_count * sizeof(float), cudaMemcpyDeviceToHost);

  float max_diff = 0;
  for (int i = 0; i < output_count; ++i)
    max_diff = fmaxf(max_diff, fabsf(h_cutlass[i] - h_naive[i]));
  if (max_diff > 1e-2f) {
    std::cerr << "CUTLASS ref incorrect vs naive. Max diff: " << max_diff << std::endl;
    cudaFree(d_input); cudaFree(d_filter); cudaFree(d_out_cutlass); cudaFree(d_out_naive);
    return cudaErrorUnknown;
  }

#ifdef KH_TEST_SOLUTION
  float *d_out_solution;
  cudaMalloc(&d_out_solution, output_count * sizeof(float));
  cudaMemset(d_out_solution, 0, output_count * sizeof(float));

  result = Conv2dFprop3xTF32(d_input, d_filter, d_out_solution,
      N, C, H, W, K, R, S, pad_h, pad_w, stride_h, stride_w);
  if (result != cudaSuccess) {
    std::cerr << "Solution kernel failed: " << cudaGetErrorString(result) << std::endl;
    cudaFree(d_out_solution); cudaFree(d_input); cudaFree(d_filter); cudaFree(d_out_cutlass); cudaFree(d_out_naive);
    return result;
  }
  cudaDeviceSynchronize();

  std::vector<float> h_solution(output_count);
  cudaMemcpy(h_solution.data(), d_out_solution, output_count * sizeof(float), cudaMemcpyDeviceToHost);

  max_diff = 0;
  for (int i = 0; i < output_count; ++i)
    max_diff = fmaxf(max_diff, fabsf(h_solution[i] - h_cutlass[i]));
  if (max_diff > 1e-2f) {
    std::cerr << "Solution incorrect. Max diff vs reference: " << max_diff << std::endl;
    cudaFree(d_out_solution); cudaFree(d_input); cudaFree(d_filter); cudaFree(d_out_cutlass); cudaFree(d_out_naive);
    return cudaErrorUnknown;
  }

  cudaFree(d_out_solution);
#endif

  cudaFree(d_out_naive);
  cudaFree(d_out_cutlass);
  cudaFree(d_filter);
  cudaFree(d_input);
  return cudaSuccess;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Profile reference (and solution if KH_TEST_SOLUTION) with CUDA events.
void Profile(int N, int C, int H, int W, int K, int R, int S,
             int pad_h, int pad_w, int stride_h, int stride_w, int iterations) {
  int P = (H + 2 * pad_h - R) / stride_h + 1;
  int Q = (W + 2 * pad_w - S) / stride_w + 1;

  int input_count = N * H * W * C;
  int filter_count = K * R * S * C;
  int output_count = N * P * Q * K;

  float *d_input, *d_filter, *d_output;
  cudaMalloc(&d_input, input_count * sizeof(float));
  cudaMalloc(&d_filter, filter_count * sizeof(float));
  cudaMalloc(&d_output, output_count * sizeof(float));

  InitializeTensor(d_input, input_count, 0);
  InitializeTensor(d_filter, filter_count, 17);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Warmup reference
  for (int i = 0; i < 3; i++) {
    cudaMemset(d_output, 0, output_count * sizeof(float));
    CutlassConv2dFprop3xTF32(d_input, d_filter, d_output,
        N, C, H, W, K, R, S, pad_h, pad_w, stride_h, stride_w);
  }
  cudaDeviceSynchronize();

  // Profile reference
  float ref_total_ms = 0, ref_min_ms = 1e30f;
  for (int iter = 0; iter < iterations; ++iter) {
    cudaMemset(d_output, 0, output_count * sizeof(float));
    cudaEventRecord(start);
    CutlassConv2dFprop3xTF32(d_input, d_filter, d_output,
        N, C, H, W, K, R, S, pad_h, pad_w, stride_h, stride_w);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    ref_total_ms += ms;
    if (ms < ref_min_ms) ref_min_ms = ms;
  }
  fprintf(stdout, "Ref time: %.4f ms (avg over %d trials, min: %.4f ms)\n",
          ref_total_ms / iterations, iterations, ref_min_ms);

#ifdef KH_TEST_SOLUTION
  // Warmup solution
  for (int i = 0; i < 3; i++) {
    cudaMemset(d_output, 0, output_count * sizeof(float));
    Conv2dFprop3xTF32(d_input, d_filter, d_output,
        N, C, H, W, K, R, S, pad_h, pad_w, stride_h, stride_w);
  }
  cudaDeviceSynchronize();

  // Profile solution
  float sol_total_ms = 0, sol_min_ms = 1e30f;
  for (int iter = 0; iter < iterations; ++iter) {
    cudaMemset(d_output, 0, output_count * sizeof(float));
    cudaEventRecord(start);
    Conv2dFprop3xTF32(d_input, d_filter, d_output,
        N, C, H, W, K, R, S, pad_h, pad_w, stride_h, stride_w);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    sol_total_ms += ms;
    if (ms < sol_min_ms) sol_min_ms = ms;
  }
  fprintf(stdout, "Kernel time: %.4f ms (avg over %d trials, min: %.4f ms)\n",
          sol_total_ms / iterations, iterations, sol_min_ms);
  fprintf(stdout, "Speedup: %.4fx (ref_min / kernel_min)\n", ref_min_ms / sol_min_ms);
#endif

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_output);
  cudaFree(d_filter);
  cudaFree(d_input);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, const char *arg[]) {

  // Conv2D configs: N, H, W, C, K, R, S, pad, stride
  struct TestConfig {
    const char* label;
    int N, H, W, C, K, R, S, pad_h, pad_w, stride_h, stride_w;
  };

  int iterations = 20;
  std::vector<TestConfig> configs = {
    {"small",   1,  56,  56,   64,   64, 3, 3, 1, 1, 1, 1},
    {"medium",  1, 112, 112,  128,  128, 3, 3, 1, 1, 1, 1},
    {"large",   4,  56,  56,  256,  256, 3, 3, 1, 1, 1, 1},
  };

  for (const auto& cfg : configs) {
    fprintf(stdout, "\n=== %s: N=%d, H=%d, W=%d, C=%d, K=%d, R=%d, S=%d ===\n",
            cfg.label, cfg.N, cfg.H, cfg.W, cfg.C, cfg.K, cfg.R, cfg.S);

    cudaError_t result = TestCorrectness(cfg.N, cfg.C, cfg.H, cfg.W, cfg.K, cfg.R, cfg.S,
                                         cfg.pad_h, cfg.pad_w, cfg.stride_h, cfg.stride_w);
    if (result != cudaSuccess) {
      std::cout << "Incorrect" << std::endl;
      return -1;
    }
    std::cout << "Passed" << std::endl;

    if (iterations > 0) {
      Profile(cfg.N, cfg.C, cfg.H, cfg.W, cfg.K, cfg.R, cfg.S,
              cfg.pad_h, cfg.pad_w, cfg.stride_h, cfg.stride_w, iterations);
    }
  }

  return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
