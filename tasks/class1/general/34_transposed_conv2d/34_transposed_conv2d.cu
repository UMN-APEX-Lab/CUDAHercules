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

/*
This example shows how to compute 2d transposed convolution, also known as deconvolution, using CUTLASS
conv2d Dgrad kernels. Although two operations are computationaly equivalent, some care is needed to correctly
set up a problem size for CUTLASS.
In deep learning, transposed convolution is sometimes used for upscaling feature maps. This example
demonstrates the 2x upscaling case using the strided Dgrad kernel.
*/

#include <iostream>
#include <sstream>

#include "cutlass/cutlass.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/conv/kernel/default_conv2d_dgrad.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/device/convolution.h"
#include "cutlass/util/tensor_view_io.h"

#include "helper.h"

#ifdef KH_TEST_SOLUTION
#include "solution.h"
#endif

// The code section below describes datatype for input, output tensors and computation between
// elements
using cutlass::layout::TensorNHWC;
using cutlass::TensorRef;

using ElementAccumulator = cutlass::half_t;                  // Data type of accumulator
using ElementComputeEpilogue = cutlass::half_t;              // Data type of epilogue computation (alpha, beta)
using ElementInputA = cutlass::half_t;             // Data type of elements in input tensor
using ElementInputB = cutlass::half_t;             // Data type of elements in input tensor
using ElementOutput = cutlass::half_t;                       // Data type of elements in output tensor
using ElementC = ElementOutput;
using ElementCompute = ElementComputeEpilogue;
using LayoutInputA = TensorNHWC;
using LayoutInputB = TensorNHWC;
using LayoutOutput = TensorNHWC;

// This code section describes whether you want to use tensor cores or regular SIMT cores on GPU SM
using MMAOp = cutlass::arch::OpClassTensorOp;

// This code section describes CUDA SM architecture number
using SmArch = cutlass::arch::Sm80;

// This code section describes the tile size a thread block will compute
using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>; // Threadblock tile shape

// This code section describes tile size a warp will compute
using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;          // Warp tile shape

// This code section describes the size of MMA op
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;    // TensorCore instruction shape

// This code section describes how threadblocks are scheduled on GPU
using SwizzleThreadBlock = cutlass::conv::threadblock::StridedDgradIdentityThreadblockSwizzle<1>;

// Number of pipelines you want to use
constexpr int NumStages = 3;

// This code section describe iterator algorithm selected is Analytic or Optimized
static cutlass::conv::IteratorAlgorithm const IteratorAlgorithm = cutlass::conv::IteratorAlgorithm::kOptimized;

// This code section describes the epilogue part of the kernel, we use default value
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementCompute,                                     // Data type of output matrix.
    128 / cutlass::sizeof_bits<ElementCompute>::value,  // The number of elements per vectorized.
    // memory access. This becomes the vector width of
    // math instructions in the epilogue too.
    ElementAccumulator,                                // Data type of accumulator
    ElementComputeEpilogue>;                           // Data type for alpha/beta in linear combination

using Conv2dDgradKernel = typename cutlass::conv::kernel::DefaultConv2dDgrad<
    ElementInputA, LayoutInputA,
    ElementInputB, LayoutInputB,
    ElementAccumulator, LayoutOutput,
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
    IteratorAlgorithm,
    cutlass::conv::StrideSupport::kStrided  // Use the strided Dgrad specialization
    >::Kernel;

using ImplicitGemm = cutlass::conv::device::ImplicitGemmConvolution<Conv2dDgradKernel>;

/////////////////////////////////////////////////////////////////////////////////////////////////

// Command line options parsing
struct Options {

  bool help;
  cutlass::Tensor4DCoord input_size;
  cutlass::Tensor4DCoord filter_size;
  cutlass::Tensor4DCoord padding;
  cutlass::MatrixCoord conv_stride;
  cutlass::MatrixCoord dilation;
  bool reference_check;
  bool measure_performance;
  int iterations;
  ElementComputeEpilogue alpha;
  ElementComputeEpilogue beta;
  std::string tag;

  Options():
    help(false),
    input_size(1, 32, 32, 32),
    filter_size(32, 3, 3, 16),
    padding(1, 1, 1, 1),
    conv_stride(2, 2),
    dilation(1, 1),
    reference_check(true),
    measure_performance(false),
    iterations(20),
    alpha(1),
    beta(0) {}

  // Verify the problem size is compatible with the CUTLASS Convolution implementation.
  bool valid() {

    //
    // CUTLASS attempts to load 128b vectors of cutlass::half_t (F16) elements. Consequently,
    // all pointers, strides, and tensor extents must be divisible by 8 elements.
    //
    int const kAlignment = 8;

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

  // Parses the command line
  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
    }

    if (cmd.check_cmd_line_flag("skip-ref-check")) {
      reference_check = false;
    }

    if (cmd.check_cmd_line_flag("perf-check")) {
      measure_performance = true;
    }

    cmd.get_cmd_line_argument("n", input_size.n());
    cmd.get_cmd_line_argument("h", input_size.h());
    cmd.get_cmd_line_argument("w", input_size.w());
    cmd.get_cmd_line_argument("c", input_size.c());

    // Filter layout is CRSK
    cmd.get_cmd_line_argument("k", filter_size.c());
    cmd.get_cmd_line_argument("r", filter_size.h());
    cmd.get_cmd_line_argument("s", filter_size.w());
    filter_size.n() = input_size.c();

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

    out << "34_transposed_conv2d example\n\n"
	<< "  This example shows how to compute 2d transposed convolution, also known as\n"
	<< "  deconvolution, using CUTLASS conv2d Dgrad kernels. Although two operations are\n"
	<< "  computationaly equivalent, some care is needed to correctly set up a problem size.\n\n"
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
	<< "  --skip-ref-check     If set (true), skip reference check on the host\n"
	<< "  --perf-check         If set (true), performance is measured.\n"
	<< "  --iterations=<int>   Number of profiling iterations to perform.\n"
	<< "  --tag=<string>       String to replicate across the first column in the results table\n";

    out << "\n\nExamples:\n\n"
	<< "$ ./examples/34_transposed_conv2d/34_transposed_conv2d --n=8 --h=32 --w=32 --c=16 --k=32 --r=3 --s=3\n\n";

    return out;
  }

  /// Computes the output tensor size (NPQK)
  cutlass::Tensor4DCoord output_size() const {
    // Here, out_pad corresponds to "output_padding" of conv2d_transpose op in deep learning frameworks.
    // See for example https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html
    int out_pad_h = conv_stride.row() > 1 ? 1 : 0;
    int out_pad_w = conv_stride.column() > 1 ? 1 : 0;
    int out_h = (input_size.h() - 1) * conv_stride.row() - 2 * padding.n() + (((filter_size.h() - 1) * dilation.row() + 1)) + out_pad_h;
    int out_w = (input_size.w() - 1) * conv_stride.column() - 2 * padding.w() + (((filter_size.w() - 1) * dilation.column() + 1)) + out_pad_w;
    return cutlass::Tensor4DCoord(input_size.n(), out_h, out_w, filter_size.c());
  }

  /// Compute performance in GFLOP/s
  double gflops(double runtime_s) const {

    // Number of multiply-adds = NHWC * KRS
    // Note that the input with the layout NHWC corresponds to the output from the perspective of dgrad,
    // and that the filter layout is CRSK.
    int64_t fmas = input_size.product() * int64_t(filter_size.h() * filter_size.w() * filter_size.n());

    // Two flops per multiply-add
    return 2.0 * double(fmas) / double(1.0e9) / runtime_s;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

struct Result {
  double runtime_ms;
  double gflops;
  cutlass::Status status;
  cutlass::Status reference_check;
  cudaError_t error;

  Result():
    runtime_ms(0),
    gflops(0),
    status(cutlass::Status::kSuccess),
    reference_check(cutlass::Status::kInvalid),
    error(cudaSuccess) { }

  static std::ostream & print_header(std::ostream &out, Options const &options) {

    if (!options.tag.empty()) {
      out << "Name,";
    }

    out << "Layer,N,H,W,C,K,R,S,Stride_H,Stride_W,Runtime,GFLOPs";

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
      << options.filter_size.c() << ","
      << options.filter_size.h() << ","
      << options.filter_size.w() << ","
      << options.conv_stride.row() << ","
      << options.conv_stride.column() << ","
      << runtime_ms << ","
      << gflops;

    return out;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Run CUTLASS transposed conv2d (via dgrad) on device buffers
cudaError_t CutlassTransposedConv2d(
    cutlass::half_t const *input, cutlass::half_t const *filter, cutlass::half_t *output,
    int N, int C, int H, int W, int K, int R, int S,
    int pad_h, int pad_w, int stride_h, int stride_w) {

  // Transposed conv output size
  int out_pad_h = stride_h > 1 ? 1 : 0;
  int out_pad_w = stride_w > 1 ? 1 : 0;
  int H_out = (H - 1) * stride_h - 2 * pad_h + R + out_pad_h;
  int W_out = (W - 1) * stride_w - 2 * pad_w + S + out_pad_w;

  // Input: NHWC, Filter: CRSK
  cutlass::Tensor4DCoord input_size(N, H, W, C);
  cutlass::Tensor4DCoord filter_size(C, R, S, K);
  cutlass::Tensor4DCoord output_size_coord(N, H_out, W_out, K);
  cutlass::Tensor4DCoord padding(pad_h, pad_h, pad_w, pad_w);
  cutlass::MatrixCoord conv_stride_coord(stride_h, stride_w);
  cutlass::MatrixCoord dilation(1, 1);

  // For transposed conv: swap input/output in dgrad problem setup
  cutlass::conv::Conv2dProblemSize problem_size(
      output_size_coord,  // "input" for dgrad = our output
      filter_size,
      padding,
      conv_stride_coord,
      dilation,
      input_size,        // "output" for dgrad = our input
      cutlass::conv::Mode::kCrossCorrelation
  );

  // tensor_a = input (from dgrad perspective, this is the "output gradient" = our input)
  cutlass::TensorRef<ElementInputB, LayoutInputB> ref_input(
      const_cast<cutlass::half_t*>(input), LayoutInputB::packed(input_size));
  cutlass::TensorRef<ElementOutput, LayoutOutput> ref_filter(
      const_cast<cutlass::half_t*>(filter), LayoutOutput::packed(filter_size));

  // C and D tensors are the transposed conv output
  cutlass::TensorRef<ElementInputA, LayoutInputA> ref_c(
      output, LayoutInputA::packed(output_size_coord));
  cutlass::TensorRef<ElementInputA, LayoutInputA> ref_d(
      output, LayoutInputA::packed(output_size_coord));

  typename ImplicitGemm::Arguments arguments{
    problem_size,
    ref_input,
    ref_filter,
    ref_c,
    ref_d,
    {ElementCompute(1), ElementCompute(0)}
  };

  ImplicitGemm op;
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

/// Naive GPU reference for transposed conv2d, NHWC, half precision
__global__ void ReferenceTransposedConv2d_kernel(
    cutlass::half_t const *input, cutlass::half_t const *filter, float *output_fp32,
    int N, int C, int H, int W, int K, int R, int S,
    int H_out, int W_out, int pad_h, int pad_w, int stride_h, int stride_w) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = N * H_out * W_out * K;
  if (idx >= total) return;

  int k = idx % K; idx /= K;
  int w_out = idx % W_out; idx /= W_out;
  int h_out = idx % H_out; idx /= H_out;
  int n = idx;

  float acc = 0.0f;
  for (int c = 0; c < C; ++c)
    for (int r = 0; r < R; ++r)
      for (int s = 0; s < S; ++s) {
        int h_val = h_out - r + pad_h;
        int w_val = w_out - s + pad_w;
        if (h_val >= 0 && (h_val % stride_h) == 0 &&
            w_val >= 0 && (w_val % stride_w) == 0) {
          int h = h_val / stride_h;
          int w = w_val / stride_w;
          if (h < H && w < W) {
            acc += float(input[((n * H + h) * W + w) * C + c])
                 * float(filter[((c * R + r) * S + s) * K + k]);
          }
        }
      }
  output_fp32[((n * H_out + h_out) * W_out + w_out) * K + k] = acc;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Kernel to initialize a tensor with small integers (half precision).
__global__ void InitializeHalfTensor_kernel(cutlass::half_t *data, int count, int seed) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) {
    int const k = 16807;
    int const m = 16;
    float value = float(((i + seed) * k % m) - m / 2) * 0.01f;
    data[i] = cutlass::half_t(value);
  }
}

cudaError_t InitializeHalfTensor(cutlass::half_t *data, int count, int seed) {
  int threads = 256;
  int blocks = (count + threads - 1) / threads;
  InitializeHalfTensor_kernel<<<blocks, threads>>>(data, count, seed);
  return cudaGetLastError();
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Test correctness: CUTLASS ref vs naive, and (if KH_TEST_SOLUTION) solution vs CUTLASS.
cudaError_t TestCorrectness(int N, int C, int H, int W, int K, int R, int S,
                            int pad_h, int pad_w, int stride_h, int stride_w) {
  int out_pad_h = stride_h > 1 ? 1 : 0;
  int out_pad_w = stride_w > 1 ? 1 : 0;
  int H_out = (H - 1) * stride_h - 2 * pad_h + R + out_pad_h;
  int W_out = (W - 1) * stride_w - 2 * pad_w + S + out_pad_w;

  int input_count = N * H * W * C;
  int filter_count = C * R * S * K;  // CRSK layout
  int output_count = N * H_out * W_out * K;

  cutlass::half_t *d_input, *d_filter, *d_out_cutlass;
  cudaMalloc(&d_input, input_count * sizeof(cutlass::half_t));
  cudaMalloc(&d_filter, filter_count * sizeof(cutlass::half_t));
  cudaMalloc(&d_out_cutlass, output_count * sizeof(cutlass::half_t));

  InitializeHalfTensor(d_input, input_count, 0);
  InitializeHalfTensor(d_filter, filter_count, 17);
  cudaMemset(d_out_cutlass, 0, output_count * sizeof(cutlass::half_t));

  // Run CUTLASS reference
  cudaError_t result = CutlassTransposedConv2d(d_input, d_filter, d_out_cutlass,
      N, C, H, W, K, R, S, pad_h, pad_w, stride_h, stride_w);
  if (result != cudaSuccess) {
    std::cerr << "CUTLASS transposed conv2d failed: " << cudaGetErrorString(result) << std::endl;
    cudaFree(d_input); cudaFree(d_filter); cudaFree(d_out_cutlass);
    return result;
  }
  cudaDeviceSynchronize();

  // Run naive reference (in fp32)
  float *d_out_naive_fp32;
  cudaMalloc(&d_out_naive_fp32, output_count * sizeof(float));
  {
    int threads = 256;
    int blocks = (output_count + threads - 1) / threads;
    ReferenceTransposedConv2d_kernel<<<blocks, threads>>>(
        d_input, d_filter, d_out_naive_fp32,
        N, C, H, W, K, R, S, H_out, W_out,
        pad_h, pad_w, stride_h, stride_w);
  }
  cudaDeviceSynchronize();

  // Compare CUTLASS vs naive
  std::vector<cutlass::half_t> h_cutlass(output_count);
  std::vector<float> h_naive(output_count);
  cudaMemcpy(h_cutlass.data(), d_out_cutlass, output_count * sizeof(cutlass::half_t), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_naive.data(), d_out_naive_fp32, output_count * sizeof(float), cudaMemcpyDeviceToHost);

  float max_diff = 0;
  for (int i = 0; i < output_count; ++i)
    max_diff = fmaxf(max_diff, fabsf(float(h_cutlass[i]) - h_naive[i]));
  if (max_diff > 1.0f) {
    std::cerr << "CUTLASS ref incorrect vs naive. Max diff: " << max_diff << std::endl;
    cudaFree(d_out_naive_fp32); cudaFree(d_input); cudaFree(d_filter); cudaFree(d_out_cutlass);
    return cudaErrorUnknown;
  }

#ifdef KH_TEST_SOLUTION
  __half *d_out_solution;
  cudaMalloc(&d_out_solution, output_count * sizeof(__half));
  cudaMemset(d_out_solution, 0, output_count * sizeof(__half));

  result = TransposedConv2d(reinterpret_cast<__half const*>(d_input),
                            reinterpret_cast<__half const*>(d_filter),
                            d_out_solution,
                            N, C, H, W, K, R, S, pad_h, pad_w, stride_h, stride_w);
  if (result != cudaSuccess) {
    std::cerr << "Solution kernel failed: " << cudaGetErrorString(result) << std::endl;
    cudaFree(d_out_solution); cudaFree(d_out_naive_fp32);
    cudaFree(d_input); cudaFree(d_filter); cudaFree(d_out_cutlass);
    return result;
  }
  cudaDeviceSynchronize();

  std::vector<__half> h_solution(output_count);
  cudaMemcpy(h_solution.data(), d_out_solution, output_count * sizeof(__half), cudaMemcpyDeviceToHost);

  max_diff = 0;
  for (int i = 0; i < output_count; ++i)
    max_diff = fmaxf(max_diff, fabsf(__half2float(h_solution[i]) - float(h_cutlass[i])));
  if (max_diff > 1.0f) {
    std::cerr << "Solution incorrect. Max diff vs reference: " << max_diff << std::endl;
    cudaFree(d_out_solution); cudaFree(d_out_naive_fp32);
    cudaFree(d_input); cudaFree(d_filter); cudaFree(d_out_cutlass);
    return cudaErrorUnknown;
  }

  cudaFree(d_out_solution);
#endif

  cudaFree(d_out_naive_fp32);
  cudaFree(d_out_cutlass);
  cudaFree(d_filter);
  cudaFree(d_input);
  return cudaSuccess;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Profile reference (and solution if KH_TEST_SOLUTION) with CUDA events.
void Profile(int N, int C, int H, int W, int K, int R, int S,
             int pad_h, int pad_w, int stride_h, int stride_w, int iterations) {
  int out_pad_h = stride_h > 1 ? 1 : 0;
  int out_pad_w = stride_w > 1 ? 1 : 0;
  int H_out = (H - 1) * stride_h - 2 * pad_h + R + out_pad_h;
  int W_out = (W - 1) * stride_w - 2 * pad_w + S + out_pad_w;

  int input_count = N * H * W * C;
  int filter_count = C * R * S * K;
  int output_count = N * H_out * W_out * K;

  cutlass::half_t *d_input, *d_filter, *d_output;
  cudaMalloc(&d_input, input_count * sizeof(cutlass::half_t));
  cudaMalloc(&d_filter, filter_count * sizeof(cutlass::half_t));
  cudaMalloc(&d_output, output_count * sizeof(cutlass::half_t));

  InitializeHalfTensor(d_input, input_count, 0);
  InitializeHalfTensor(d_filter, filter_count, 17);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Warmup reference
  for (int i = 0; i < 3; i++) {
    cudaMemset(d_output, 0, output_count * sizeof(cutlass::half_t));
    CutlassTransposedConv2d(d_input, d_filter, d_output,
        N, C, H, W, K, R, S, pad_h, pad_w, stride_h, stride_w);
  }
  cudaDeviceSynchronize();

  // Profile reference
  float ref_total_ms = 0, ref_min_ms = 1e30f;
  for (int iter = 0; iter < iterations; ++iter) {
    cudaMemset(d_output, 0, output_count * sizeof(cutlass::half_t));
    cudaEventRecord(start);
    CutlassTransposedConv2d(d_input, d_filter, d_output,
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
  __half *d_out_sol;
  cudaMalloc(&d_out_sol, output_count * sizeof(__half));

  // Warmup solution
  for (int i = 0; i < 3; i++) {
    cudaMemset(d_out_sol, 0, output_count * sizeof(__half));
    TransposedConv2d(reinterpret_cast<__half const*>(d_input),
                     reinterpret_cast<__half const*>(d_filter),
                     d_out_sol,
                     N, C, H, W, K, R, S, pad_h, pad_w, stride_h, stride_w);
  }
  cudaDeviceSynchronize();

  // Profile solution
  float sol_total_ms = 0, sol_min_ms = 1e30f;
  for (int iter = 0; iter < iterations; ++iter) {
    cudaMemset(d_out_sol, 0, output_count * sizeof(__half));
    cudaEventRecord(start);
    TransposedConv2d(reinterpret_cast<__half const*>(d_input),
                     reinterpret_cast<__half const*>(d_filter),
                     d_out_sol,
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

  cudaFree(d_out_sol);
#endif

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_output);
  cudaFree(d_filter);
  cudaFree(d_input);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, const char *arg[]) {

  struct TestConfig {
    const char* label;
    int N, H, W, C, K, R, S, pad_h, pad_w, stride_h, stride_w;
  };

  int iterations = 20;
  // Transposed conv: stride=2 for upscaling, matching the original example usage
  std::vector<TestConfig> configs = {
    {"small",   1,  56,  56,   64,   64, 3, 3, 1, 1, 1, 1},
    {"medium",  1, 112, 112,  128,  128, 3, 3, 1, 1, 1, 1},
    {"large",   4,  56,  56,  256,  256, 3, 3, 1, 1, 1, 1},
  };

  for (const auto& cfg : configs) {
    fprintf(stdout, "\n=== %s: N=%d, H=%d, W=%d, C=%d, K=%d, R=%d, S=%d, stride=%d ===\n",
            cfg.label, cfg.N, cfg.H, cfg.W, cfg.C, cfg.K, cfg.R, cfg.S, cfg.stride_h);

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
