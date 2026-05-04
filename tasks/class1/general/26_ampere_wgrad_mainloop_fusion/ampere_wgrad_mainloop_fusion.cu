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

This example shows how to fuse activation's per channel scale+bias+relu
into the wgrad mainloop.

Compared with original fprop kernel, this example has two more vectors, one for
the scale and one for the bias.  The length of the vectors are the same as the
activation channel number.  This kernels loads the vectors when the associated
activation channels are loaded in the mainloop.  Between reading the
activations and scale/bias data from the shared memory and calling tensor core
instructions, scale+bias+relu is computed in the register file.

This example is customized for Ampere 16816 fp16 tensor core instruction.
Changing to different data types or different tensor core instruction require
source code changing.  See
include/cutlass/conv/threadblock/implicit_gemm_wgrad_fusion_multistage.h for more
technical details.
*/

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/conv/kernel/default_conv2d_wgrad_fusion.h"
#include "cutlass/conv/device/implicit_gemm_convolution_fusion.h"

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
#include <cuda_fp16.h>
#include "solution.h"
#endif

// The code section below describes datatype for input, output tensors and computation between
// elements
using ElementAccumulator = float;                  // Data type of accumulator
using ElementComputeEpilogue = float;              // Data type of epilogue computation (alpha, beta)
using ElementInputA = cutlass::half_t;             // Data type of elements in input tensor
using ElementInputB = cutlass::half_t;             // Data type of elements in input tensor
using ElementInputScaleBias = cutlass::half_t;     // Data type of elements in input sclae and bias vectors
using ElementOutput = float;                       // Data type of elements in output tensor

using LayoutInputA = cutlass::layout::TensorNHWC;
using LayoutInputB = cutlass::layout::TensorNHWC;
using LayoutInputScaleBias = cutlass::layout::RowMajor;
using LayoutOutput = cutlass::layout::TensorNHWC;

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
using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

// Number of pipelines you want to use
constexpr int NumStages = 5;

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

using Conv2dWgradFusionKernel = typename cutlass::conv::kernel::DefaultConv2dWgradFusion<
  ElementInputA, LayoutInputA,
  ElementInputB, LayoutInputB,
  ElementInputScaleBias, LayoutInputScaleBias,
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

using ImplicitGemmFusion = cutlass::conv::device::ImplicitGemmConvolutionFusion<Conv2dWgradFusionKernel>;

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Run CUTLASS wgrad with fused scale+bias+relu on given device buffers.
/// In wgrad: tensor_a = grad_output (NPQK), tensor_b = input (NHWC).
/// Scale/bias are applied to the input (activation) before wgrad.
/// Output grad_weight is float, NHWC layout (K,R,S,C).
cudaError_t CutlassConv2dWgradFusion(
    cutlass::half_t const *grad_output, cutlass::half_t const *input,
    cutlass::half_t const *scale, cutlass::half_t const *bias,
    float *grad_weight,
    int N, int C, int H, int W, int K, int R, int S,
    int pad_h, int pad_w, int stride_h, int stride_w,
    float alpha, float beta) {

  cutlass::Tensor4DCoord input_size(N, H, W, C);
  cutlass::Tensor4DCoord filter_size(K, R, S, C);
  cutlass::Tensor4DCoord padding(pad_h, pad_h, pad_w, pad_w);
  cutlass::MatrixCoord conv_stride_coord(stride_h, stride_w);
  cutlass::MatrixCoord dilation(1, 1);

  int P = (H + 2 * pad_h - R) / stride_h + 1;
  int Q = (W + 2 * pad_w - S) / stride_w + 1;
  cutlass::Tensor4DCoord output_size(N, P, Q, K);

  cutlass::conv::Conv2dProblemSize problem_size(
      input_size, filter_size, padding, conv_stride_coord, dilation, output_size,
      cutlass::conv::Mode::kCrossCorrelation, 1);

  using RowMajor = cutlass::layout::RowMajor;

  // In wgrad: tensor_a = grad_output (NPQK), tensor_b = input (NHWC)
  cutlass::TensorRef<ElementInputA, LayoutInputA> ref_grad_output(
      const_cast<cutlass::half_t*>(grad_output), LayoutInputA::packed(output_size));
  cutlass::TensorRef<ElementInputB, LayoutInputB> ref_input(
      const_cast<cutlass::half_t*>(input), LayoutInputB::packed(input_size));
  cutlass::TensorRef<ElementInputScaleBias, LayoutInputScaleBias> ref_scale(
      const_cast<cutlass::half_t*>(scale), RowMajor(C));
  cutlass::TensorRef<ElementInputScaleBias, LayoutInputScaleBias> ref_bias(
      const_cast<cutlass::half_t*>(bias), RowMajor(C));

  // C and D tensors (grad_weight, KRSC)
  cutlass::TensorRef<ElementOutput, LayoutOutput> ref_c(
      grad_weight, LayoutOutput::packed(filter_size));
  cutlass::TensorRef<ElementOutput, LayoutOutput> ref_d(
      grad_weight, LayoutOutput::packed(filter_size));

  typename ImplicitGemmFusion::Arguments arguments{
    problem_size,
    ref_grad_output,
    ref_input,
    ref_scale,
    ref_bias,
    ref_c,
    ref_d,
    {alpha, beta},
  };

  ImplicitGemmFusion implicit_gemm_fusion_op;

  size_t workspace_size = implicit_gemm_fusion_op.get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  cutlass::Status status = implicit_gemm_fusion_op.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) return cudaErrorUnknown;

  status = implicit_gemm_fusion_op.initialize(arguments, workspace.get());
  if (status != cutlass::Status::kSuccess) return cudaErrorUnknown;

  status = implicit_gemm_fusion_op();
  if (status != cutlass::Status::kSuccess) return cudaErrorUnknown;

  return cudaSuccess;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

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

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Naive GPU reference for conv2d wgrad with fused scale+bias+relu, NHWC, half->float.
/// grad_weight[k][r][s][c] = sum_{n,p,q} grad_output[n][p][q][k] * relu(scale[c]*input[n][h][w][c]+bias[c])
__global__ void ReferenceConv2dWgradFusion_kernel(
    cutlass::half_t const *grad_output, cutlass::half_t const *input,
    cutlass::half_t const *scale, cutlass::half_t const *bias,
    float *grad_weight,
    int N, int C, int H, int W, int K, int R, int S,
    int P, int Q, int pad_h, int pad_w, int stride_h, int stride_w) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = K * R * S * C;
  if (idx >= total) return;

  int c = idx % C; idx /= C;
  int s = idx % S; idx /= S;
  int r = idx % R; idx /= R;
  int k = idx;

  float acc = 0.0f;
  for (int n = 0; n < N; ++n)
    for (int p = 0; p < P; ++p)
      for (int q = 0; q < Q; ++q) {
        int h = p * stride_h + r - pad_h;
        int w = q * stride_w + s - pad_w;
        if (h >= 0 && h < H && w >= 0 && w < W) {
          float x = float(input[((n * H + h) * W + w) * C + c]);
          float sc = float(scale[c]);
          float bi = float(bias[c]);
          float activated = fmaxf(0.0f, sc * x + bi);  // scale + bias + relu
          acc += float(grad_output[((n * P + p) * Q + q) * K + k]) * activated;
        }
      }
  grad_weight[((k * R + r) * S + s) * C + c] = acc;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Test correctness: CUTLASS ref vs naive, and (if KH_TEST_SOLUTION) solution vs CUTLASS.
cudaError_t TestCorrectness(int N, int C, int H, int W, int K, int R, int S,
                            int pad_h, int pad_w, int stride_h, int stride_w) {
  int P = (H + 2 * pad_h - R) / stride_h + 1;
  int Q = (W + 2 * pad_w - S) / stride_w + 1;

  int input_count = N * H * W * C;
  int grad_out_count = N * P * Q * K;
  int weight_count = K * R * S * C;

  cutlass::half_t *d_input, *d_grad_output, *d_scale, *d_bias;
  float *d_wgrad_cutlass, *d_wgrad_naive;

  cudaMalloc(&d_input, input_count * sizeof(cutlass::half_t));
  cudaMalloc(&d_grad_output, grad_out_count * sizeof(cutlass::half_t));
  cudaMalloc(&d_scale, C * sizeof(cutlass::half_t));
  cudaMalloc(&d_bias, C * sizeof(cutlass::half_t));
  cudaMalloc(&d_wgrad_cutlass, weight_count * sizeof(float));
  cudaMalloc(&d_wgrad_naive, weight_count * sizeof(float));

  InitializeHalfTensor(d_input, input_count, 0);
  InitializeHalfTensor(d_grad_output, grad_out_count, 17);
  InitializeHalfTensor(d_scale, C, 42);
  InitializeHalfTensor(d_bias, C, 99);
  cudaMemset(d_wgrad_cutlass, 0, weight_count * sizeof(float));
  cudaMemset(d_wgrad_naive, 0, weight_count * sizeof(float));

  // Run CUTLASS reference
  cudaError_t result = CutlassConv2dWgradFusion(d_grad_output, d_input, d_scale, d_bias,
      d_wgrad_cutlass, N, C, H, W, K, R, S, pad_h, pad_w, stride_h, stride_w, 1.0f, 0.0f);
  if (result != cudaSuccess) {
    std::cerr << "CUTLASS wgrad fusion failed: " << cudaGetErrorString(result) << std::endl;
    cudaFree(d_input); cudaFree(d_grad_output); cudaFree(d_scale); cudaFree(d_bias);
    cudaFree(d_wgrad_cutlass); cudaFree(d_wgrad_naive);
    return result;
  }
  cudaDeviceSynchronize();

  // Run naive reference
  {
    int threads = 256;
    int blocks = (weight_count + threads - 1) / threads;
    ReferenceConv2dWgradFusion_kernel<<<blocks, threads>>>(
        d_grad_output, d_input, d_scale, d_bias, d_wgrad_naive,
        N, C, H, W, K, R, S, P, Q, pad_h, pad_w, stride_h, stride_w);
  }
  cudaDeviceSynchronize();

  // Compare CUTLASS vs naive
  std::vector<float> h_cutlass(weight_count), h_naive(weight_count);
  cudaMemcpy(h_cutlass.data(), d_wgrad_cutlass, weight_count * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_naive.data(), d_wgrad_naive, weight_count * sizeof(float), cudaMemcpyDeviceToHost);

  float max_diff = 0;
  for (int i = 0; i < weight_count; ++i)
    max_diff = fmaxf(max_diff, fabsf(h_cutlass[i] - h_naive[i]));
  if (max_diff > 1e-1f) {
    std::cerr << "CUTLASS ref incorrect vs naive. Max diff: " << max_diff << std::endl;
    cudaFree(d_input); cudaFree(d_grad_output); cudaFree(d_scale); cudaFree(d_bias);
    cudaFree(d_wgrad_cutlass); cudaFree(d_wgrad_naive);
    return cudaErrorUnknown;
  }

#ifdef KH_TEST_SOLUTION
  float *d_wgrad_solution;
  cudaMalloc(&d_wgrad_solution, weight_count * sizeof(float));
  cudaMemset(d_wgrad_solution, 0, weight_count * sizeof(float));

  result = Conv2dWgradScaleBias(
      reinterpret_cast<__half const*>(d_input),
      reinterpret_cast<__half const*>(d_grad_output),
      reinterpret_cast<__half const*>(d_scale),
      reinterpret_cast<__half const*>(d_bias),
      d_wgrad_solution,
      N, C, H, W, K, R, S, pad_h, pad_w, stride_h, stride_w);
  if (result != cudaSuccess) {
    std::cerr << "Solution kernel failed: " << cudaGetErrorString(result) << std::endl;
    cudaFree(d_wgrad_solution); cudaFree(d_input); cudaFree(d_grad_output);
    cudaFree(d_scale); cudaFree(d_bias); cudaFree(d_wgrad_cutlass); cudaFree(d_wgrad_naive);
    return result;
  }
  cudaDeviceSynchronize();

  std::vector<float> h_solution(weight_count);
  cudaMemcpy(h_solution.data(), d_wgrad_solution, weight_count * sizeof(float), cudaMemcpyDeviceToHost);

  max_diff = 0;
  for (int i = 0; i < weight_count; ++i)
    max_diff = fmaxf(max_diff, fabsf(h_solution[i] - h_cutlass[i]));
  if (max_diff > 1e-1f) {
    std::cerr << "Solution incorrect. Max diff vs reference: " << max_diff << std::endl;
    cudaFree(d_wgrad_solution); cudaFree(d_input); cudaFree(d_grad_output);
    cudaFree(d_scale); cudaFree(d_bias); cudaFree(d_wgrad_cutlass); cudaFree(d_wgrad_naive);
    return cudaErrorUnknown;
  }

  cudaFree(d_wgrad_solution);
#endif

  cudaFree(d_wgrad_naive);
  cudaFree(d_wgrad_cutlass);
  cudaFree(d_bias);
  cudaFree(d_scale);
  cudaFree(d_grad_output);
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
  int grad_out_count = N * P * Q * K;
  int weight_count = K * R * S * C;

  cutlass::half_t *d_input, *d_grad_output, *d_scale, *d_bias;
  float *d_wgrad;

  cudaMalloc(&d_input, input_count * sizeof(cutlass::half_t));
  cudaMalloc(&d_grad_output, grad_out_count * sizeof(cutlass::half_t));
  cudaMalloc(&d_scale, C * sizeof(cutlass::half_t));
  cudaMalloc(&d_bias, C * sizeof(cutlass::half_t));
  cudaMalloc(&d_wgrad, weight_count * sizeof(float));

  InitializeHalfTensor(d_input, input_count, 0);
  InitializeHalfTensor(d_grad_output, grad_out_count, 17);
  InitializeHalfTensor(d_scale, C, 42);
  InitializeHalfTensor(d_bias, C, 99);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Warmup reference
  for (int i = 0; i < 3; i++) {
    cudaMemset(d_wgrad, 0, weight_count * sizeof(float));
    CutlassConv2dWgradFusion(d_grad_output, d_input, d_scale, d_bias, d_wgrad,
        N, C, H, W, K, R, S, pad_h, pad_w, stride_h, stride_w, 1.0f, 0.0f);
  }
  cudaDeviceSynchronize();

  // Profile reference
  float ref_total_ms = 0, ref_min_ms = 1e30f;
  for (int iter = 0; iter < iterations; ++iter) {
    cudaMemset(d_wgrad, 0, weight_count * sizeof(float));
    cudaEventRecord(start);
    CutlassConv2dWgradFusion(d_grad_output, d_input, d_scale, d_bias, d_wgrad,
        N, C, H, W, K, R, S, pad_h, pad_w, stride_h, stride_w, 1.0f, 0.0f);
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
  float *d_wgrad_sol;
  cudaMalloc(&d_wgrad_sol, weight_count * sizeof(float));

  // Warmup solution
  for (int i = 0; i < 3; i++) {
    cudaMemset(d_wgrad_sol, 0, weight_count * sizeof(float));
    Conv2dWgradScaleBias(
        reinterpret_cast<__half const*>(d_input),
        reinterpret_cast<__half const*>(d_grad_output),
        reinterpret_cast<__half const*>(d_scale),
        reinterpret_cast<__half const*>(d_bias),
        d_wgrad_sol,
        N, C, H, W, K, R, S, pad_h, pad_w, stride_h, stride_w);
  }
  cudaDeviceSynchronize();

  // Profile solution
  float sol_total_ms = 0, sol_min_ms = 1e30f;
  for (int iter = 0; iter < iterations; ++iter) {
    cudaMemset(d_wgrad_sol, 0, weight_count * sizeof(float));
    cudaEventRecord(start);
    Conv2dWgradScaleBias(
        reinterpret_cast<__half const*>(d_input),
        reinterpret_cast<__half const*>(d_grad_output),
        reinterpret_cast<__half const*>(d_scale),
        reinterpret_cast<__half const*>(d_bias),
        d_wgrad_sol,
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

  cudaFree(d_wgrad_sol);
#endif

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_wgrad);
  cudaFree(d_bias);
  cudaFree(d_scale);
  cudaFree(d_grad_output);
  cudaFree(d_input);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, const char *arg[]) {

  bool notSupported = false;

  if (!(__CUDACC_VER_MAJOR__ > 11 || (__CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ >= 0))) {
    std::cerr << "Ampere Tensor Core operations must be compiled with CUDA 11.0 Toolkit or later." << std::endl;
    notSupported = true;
  }

  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, 0);

  if (!(props.major >= 8)) {
    std::cerr << "This test must run on SM80 or above.\n";
    notSupported = true;
  }

  if (notSupported) {
    return 0;
  }

  struct TestConfig {
    const char* label;
    int N, H, W, C, K, R, S, pad_h, pad_w, stride_h, stride_w;
  };

  bool explicit_size = (argc >= 8);
  std::vector<TestConfig> configs;
  int iterations = 20;

  if (explicit_size) {
    int N=1, H=56, W=56, C=64, K=64, R=3, S=3;
    std::stringstream(arg[1]) >> N;
    std::stringstream(arg[2]) >> H;
    std::stringstream(arg[3]) >> W;
    std::stringstream(arg[4]) >> C;
    std::stringstream(arg[5]) >> K;
    std::stringstream(arg[6]) >> R;
    std::stringstream(arg[7]) >> S;
    if (argc > 8) std::stringstream(arg[8]) >> iterations;
    configs.push_back({"custom", N, H, W, C, K, R, S, R/2, S/2, 1, 1});
  } else {
    configs = {
      {"small",   1, 56, 56,  64,  64, 3, 3, 1, 1, 1, 1},
      {"medium",  1, 112, 112, 128, 128, 3, 3, 1, 1, 1, 1},
      {"large",   4, 56, 56, 256, 256, 3, 3, 1, 1, 1, 1},
    };
  }

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
