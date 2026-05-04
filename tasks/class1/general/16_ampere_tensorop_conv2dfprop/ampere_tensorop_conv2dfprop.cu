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
This example shows how to run CUTLASS's convolution kernels
based on the Implicit GEMM algorithm, that use the Tensor Cores
on an NVIDIA Ampere GPU.

Input tensors use cutlass::half_t (F16) in NHWC layout.
Output tensor uses float (F32) in NHWC layout.
Accumulation is done in float.

C = alpha * Conv2dFprop(A, B) + beta * C
*/

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/conv/kernel/default_conv2d_fprop.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/convolution.h"
#include "cutlass/util/tensor_view_io.h"

#include "helper.h"

#ifdef KH_TEST_SOLUTION
#include "solution.h"
#endif

// Data types for input and output tensors
// and computation between elements
using ElementAccumulator = float;                  // Data type of accumulator
using ElementComputeEpilogue = float;              // Data type of epilogue computation (alpha, beta)
using ElementInputA = cutlass::half_t;             // Data type of elements in input tensor
using ElementInputB = cutlass::half_t;             // Data type of elements in input tensor
using ElementOutput = float;                       // Data type of elements in output tensor

using LayoutInputA = cutlass::layout::TensorNHWC;
using LayoutInputB = cutlass::layout::TensorNHWC;
using LayoutOutput = cutlass::layout::TensorNHWC;

// Whether to use tensor cores or regular SIMT cores on GPU SM
using MMAOp = cutlass::arch::OpClassTensorOp;

// SM architecture number
using SmArch = cutlass::arch::Sm80;

// Threadblock tile shape
using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 64>;

// Warp tile shape
using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;

// MMA (Tensor Core instruction, in this case) tile shape
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

// How the kernel schedules threadblocks
using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

// Number of pipeline stages to use
constexpr int NumStages = 3;

// Which iterator algorithm to use: Analytic or Optimized
static cutlass::conv::IteratorAlgorithm const IteratorAlgorithm = cutlass::conv::IteratorAlgorithm::kOptimized;

// Is the output packed or strided
static cutlass::conv::StrideSupport const OutputStride = cutlass::conv::StrideSupport::kUnity;

// The epilogue part of the kernel
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,                                     // Data type of output matrix.
    128 / cutlass::sizeof_bits<ElementOutput>::value,  // The number of elements per vectorized
                                                       // memory access.
    ElementAccumulator,                                // Data type of accumulator
    ElementComputeEpilogue>;                           // Data type for alpha/beta in linear combination

// Kernel properties type
using Conv2dFpropKernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
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
  IteratorAlgorithm,
  OutputStride
>::Kernel;

// Type of the actual kernel
using ImplicitGemm = cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel>;

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Run the CUTLASS conv2d fprop using HostTensor utilities
cudaError_t CutlassConv2dFprop(
  cutlass::conv::Conv2dProblemSize const &problem_size,
  cutlass::HostTensor<ElementInputA, LayoutInputA> &tensor_a,
  cutlass::HostTensor<ElementInputB, LayoutInputB> &tensor_b,
  cutlass::HostTensor<ElementOutput, LayoutOutput> &tensor_c,
  cutlass::HostTensor<ElementOutput, LayoutOutput> &tensor_d,
  float alpha, float beta) {

  typename ImplicitGemm::Arguments arguments{
    problem_size,
    tensor_a.device_ref(),
    tensor_b.device_ref(),
    tensor_c.device_ref(),
    tensor_d.device_ref(),
    {alpha, beta},
  };

  ImplicitGemm implicit_gemm_op;
  size_t workspace_size = implicit_gemm_op.get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  cutlass::Status status = implicit_gemm_op.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) return cudaErrorUnknown;

  status = implicit_gemm_op.initialize(arguments, workspace.get());
  if (status != cutlass::Status::kSuccess) return cudaErrorUnknown;

  status = implicit_gemm_op();
  if (status != cutlass::Status::kSuccess) return cudaErrorUnknown;

  return cudaSuccess;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Naive reference conv2d forward: NHWC layout, half input, float output
__global__ void ReferenceConv2d_kernel(
  cutlass::half_t const *input, cutlass::half_t const *filter,
  float const *C_in, float *output,
  int N, int C, int H, int W, int K, int R, int S,
  int P, int Q, int pad_h, int pad_w, int stride_h, int stride_w,
  float alpha, float beta) {

  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int total = N * P * Q * K;
  if (idx >= total) return;

  int k_idx = idx % K; int tmp = idx / K;
  int q_idx = tmp % Q; tmp /= Q;
  int p_idx = tmp % P; int n_idx = tmp / P;

  float acc = 0.0f;
  for (int c = 0; c < C; ++c)
    for (int r = 0; r < R; ++r)
      for (int s = 0; s < S; ++s) {
        int h = p_idx * stride_h + r - pad_h;
        int w = q_idx * stride_w + s - pad_w;
        if (h >= 0 && h < H && w >= 0 && w < W) {
          float inp = float(input[((n_idx * H + h) * W + w) * C + c]);
          float flt = float(filter[((k_idx * R + r) * S + s) * C + c]);
          acc += inp * flt;
        }
      }
  int out_idx = ((n_idx * P + p_idx) * Q + q_idx) * K + k_idx;
  output[out_idx] = alpha * acc + beta * C_in[out_idx];
}

cudaError_t ReferenceConv2d(
  cutlass::half_t const *input, cutlass::half_t const *filter,
  float const *C_in, float *output,
  int N, int C, int H, int W, int K, int R, int S,
  int P, int Q, int pad_h, int pad_w, int stride_h, int stride_w,
  float alpha, float beta) {

  int total = N * P * Q * K;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  ReferenceConv2d_kernel<<<blocks, threads>>>(
    input, filter, C_in, output,
    N, C, H, W, K, R, S, P, Q,
    pad_h, pad_w, stride_h, stride_w, alpha, beta);
  return cudaGetLastError();
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Test correctness: CUTLASS reference vs naive, and (if KH_TEST_SOLUTION) solution vs CUTLASS
cudaError_t TestCorrectness(
  int N, int H, int W, int C, int K, int R, int S,
  int pad_h, int pad_w, int stride_h, int stride_w,
  float alpha, float beta) {

  int P = (H + 2 * pad_h - R) / stride_h + 1;
  int Q = (W + 2 * pad_w - S) / stride_w + 1;

  cutlass::Tensor4DCoord input_size(N, H, W, C);
  cutlass::Tensor4DCoord filter_size(K, R, S, C);
  cutlass::Tensor4DCoord output_size(N, P, Q, K);
  cutlass::Tensor4DCoord padding(pad_h, pad_h, pad_w, pad_w);
  cutlass::MatrixCoord conv_stride(stride_h, stride_w);
  cutlass::MatrixCoord dilation(1, 1);

  cutlass::conv::Conv2dProblemSize problem_size(
      input_size, filter_size, padding, conv_stride, dilation,
      output_size, cutlass::conv::Mode::kCrossCorrelation, 1);

  // Allocate host-device tensors
  cutlass::HostTensor<ElementInputA, LayoutInputA> tensor_a(input_size);
  cutlass::HostTensor<ElementInputB, LayoutInputB> tensor_b(filter_size);
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_c(output_size);
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_d_cutlass(output_size);
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_d_ref(output_size);

  // Fill with random data
  cutlass::reference::host::TensorFillRandomUniform(tensor_a.host_view(), 1,
      ElementInputA(7), ElementInputA(-8), 0);
  cutlass::reference::host::TensorFillRandomUniform(tensor_b.host_view(), 1,
      ElementInputB(7), ElementInputB(-8), 0);
  cutlass::reference::host::TensorFillRandomUniform(tensor_c.host_view(), 1,
      ElementOutput(7), ElementOutput(-8), 0);
  cutlass::reference::host::TensorFill(tensor_d_cutlass.host_view());
  cutlass::reference::host::TensorFill(tensor_d_ref.host_view());

  tensor_a.sync_device();
  tensor_b.sync_device();
  tensor_c.sync_device();
  tensor_d_cutlass.sync_device();
  tensor_d_ref.sync_device();

  // Run CUTLASS reference
  cudaError_t result = CutlassConv2dFprop(
    problem_size, tensor_a, tensor_b, tensor_c, tensor_d_cutlass, alpha, beta);
  if (result != cudaSuccess) {
    std::cerr << "CUTLASS conv2d failed: " << cudaGetErrorString(result) << std::endl;
    return result;
  }
  cudaDeviceSynchronize();

  // Run naive reference
  result = ReferenceConv2d(
    tensor_a.device_data(), tensor_b.device_data(),
    tensor_c.device_data(), tensor_d_ref.device_data(),
    N, C, H, W, K, R, S, P, Q,
    pad_h, pad_w, stride_h, stride_w, alpha, beta);
  if (result != cudaSuccess) {
    std::cerr << "Reference conv2d failed: " << cudaGetErrorString(result) << std::endl;
    return result;
  }
  cudaDeviceSynchronize();

  // Verify CUTLASS vs naive
  tensor_d_cutlass.sync_host();
  tensor_d_ref.sync_host();

  float max_diff = 0;
  for (int n = 0; n < N; ++n)
    for (int p = 0; p < P; ++p)
      for (int q = 0; q < Q; ++q)
        for (int k = 0; k < K; ++k) {
          float cutlass_val = tensor_d_cutlass.at({n, p, q, k});
          float ref_val = tensor_d_ref.at({n, p, q, k});
          max_diff = fmaxf(max_diff, fabsf(cutlass_val - ref_val));
        }

  if (max_diff > 1e-2f) {
    std::cerr << "CUTLASS reference incorrect vs naive. Max diff: " << max_diff << std::endl;
    return cudaErrorUnknown;
  }

#ifdef KH_TEST_SOLUTION
  {
  // Allocate solution output (copy initial C values)
  int output_count = N * P * Q * K;
  float *d_solution;
  cudaMalloc(&d_solution, output_count * sizeof(float));
  cudaMemcpy(d_solution, tensor_c.device_data(), output_count * sizeof(float), cudaMemcpyDeviceToDevice);

  // Run solution
  result = Conv2dFpropF16(
    reinterpret_cast<__half const *>(tensor_a.device_data()),
    reinterpret_cast<__half const *>(tensor_b.device_data()),
    d_solution,
    N, C, H, W, K, R, S,
    pad_h, pad_w, stride_h, stride_w,
    alpha, beta);
  if (result != cudaSuccess) {
    std::cerr << "Solution kernel failed: " << cudaGetErrorString(result) << std::endl;
    cudaFree(d_solution);
    return result;
  }
  cudaDeviceSynchronize();

  // Verify solution vs CUTLASS reference
  std::vector<float> host_solution(output_count);
  cudaMemcpy(host_solution.data(), d_solution, output_count * sizeof(float), cudaMemcpyDeviceToHost);

  float sol_max_diff = 0;
  int sol_idx = 0;
  for (int n = 0; n < N; ++n)
    for (int p = 0; p < P; ++p)
      for (int q = 0; q < Q; ++q)
        for (int k = 0; k < K; ++k) {
          float cutlass_val = tensor_d_cutlass.at({n, p, q, k});
          sol_max_diff = fmaxf(sol_max_diff, fabsf(host_solution[sol_idx] - cutlass_val));
          sol_idx++;
        }

  if (sol_max_diff > 1e-1f) {
    std::cerr << "Solution incorrect. Max diff vs reference: " << sol_max_diff << std::endl;
    cudaFree(d_solution);
    return cudaErrorUnknown;
  }

  cudaFree(d_solution);
  }
#endif

  return cudaSuccess;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Profile reference (and solution if KH_TEST_SOLUTION) with CUDA events.
void Profile(
  int N, int H, int W, int C, int K, int R, int S,
  int pad_h, int pad_w, int stride_h, int stride_w,
  float alpha, float beta, int iterations) {

  int P = (H + 2 * pad_h - R) / stride_h + 1;
  int Q = (W + 2 * pad_w - S) / stride_w + 1;

  cutlass::Tensor4DCoord input_size(N, H, W, C);
  cutlass::Tensor4DCoord filter_size(K, R, S, C);
  cutlass::Tensor4DCoord output_size(N, P, Q, K);
  cutlass::Tensor4DCoord padding(pad_h, pad_h, pad_w, pad_w);
  cutlass::MatrixCoord conv_stride(stride_h, stride_w);
  cutlass::MatrixCoord dilation(1, 1);

  cutlass::conv::Conv2dProblemSize problem_size(
      input_size, filter_size, padding, conv_stride, dilation,
      output_size, cutlass::conv::Mode::kCrossCorrelation, 1);

  cutlass::HostTensor<ElementInputA, LayoutInputA> tensor_a(input_size);
  cutlass::HostTensor<ElementInputB, LayoutInputB> tensor_b(filter_size);
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_c(output_size);
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_d(output_size);

  cutlass::reference::host::TensorFillRandomUniform(tensor_a.host_view(), 1,
      ElementInputA(7), ElementInputA(-8), 0);
  cutlass::reference::host::TensorFillRandomUniform(tensor_b.host_view(), 1,
      ElementInputB(7), ElementInputB(-8), 0);
  cutlass::reference::host::TensorFillRandomUniform(tensor_c.host_view(), 1,
      ElementOutput(7), ElementOutput(-8), 0);
  cutlass::reference::host::TensorFill(tensor_d.host_view());

  tensor_a.sync_device();
  tensor_b.sync_device();
  tensor_c.sync_device();
  tensor_d.sync_device();

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Warmup reference
  for (int i = 0; i < 3; i++) {
    CutlassConv2dFprop(problem_size, tensor_a, tensor_b, tensor_c, tensor_d, alpha, beta);
  }
  cudaDeviceSynchronize();

  // Profile reference
  float ref_total_ms = 0, ref_min_ms = 1e30f;
  for (int iter = 0; iter < iterations; ++iter) {
    cudaEventRecord(start);
    CutlassConv2dFprop(problem_size, tensor_a, tensor_b, tensor_c, tensor_d, alpha, beta);
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
  {
  int output_count = N * P * Q * K;
  // Warmup solution
  float *d_sol_output;
  cudaMalloc(&d_sol_output, output_count * sizeof(float));
  for (int i = 0; i < 3; i++) {
    cudaMemcpy(d_sol_output, tensor_c.device_data(), output_count * sizeof(float), cudaMemcpyDeviceToDevice);
    Conv2dFpropF16(
      reinterpret_cast<__half const *>(tensor_a.device_data()),
      reinterpret_cast<__half const *>(tensor_b.device_data()),
      d_sol_output, N, C, H, W, K, R, S,
      pad_h, pad_w, stride_h, stride_w, alpha, beta);
  }
  cudaDeviceSynchronize();

  // Profile solution
  float sol_total_ms = 0, sol_min_ms = 1e30f;
  for (int iter = 0; iter < iterations; ++iter) {
    cudaMemcpy(d_sol_output, tensor_c.device_data(), output_count * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaEventRecord(start);
    Conv2dFpropF16(
      reinterpret_cast<__half const *>(tensor_a.device_data()),
      reinterpret_cast<__half const *>(tensor_b.device_data()),
      d_sol_output, N, C, H, W, K, R, S,
      pad_h, pad_w, stride_h, stride_w, alpha, beta);
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
  cudaFree(d_sol_output);
  }
#endif

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, const char *arg[]) {

  // Ampere Tensor Core operations require SM80+
  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, 0);
  if (props.major < 8) {
    std::cerr << "Ampere Tensor Ops require compute capability >= 80." << std::endl;
    return 0;
  }

  struct TestConfig {
    const char* label;
    int N, H, W, C, K, R, S, pad_h, pad_w, stride_h, stride_w;
  };

  bool explicit_size = (argc >= 8);
  std::vector<TestConfig> configs;
  float alpha = 1.0f, beta = 0.0f;
  int iterations = 20;

  if (explicit_size) {
    int N = 1, H = 56, W = 56, C = 64, K = 64, R = 3, S = 3;
    std::stringstream(arg[1]) >> N;
    std::stringstream(arg[2]) >> H;
    std::stringstream(arg[3]) >> W;
    std::stringstream(arg[4]) >> C;
    std::stringstream(arg[5]) >> K;
    std::stringstream(arg[6]) >> R;
    std::stringstream(arg[7]) >> S;
    int pad_h = R / 2, pad_w = S / 2;
    if (argc > 8) std::stringstream(arg[8]) >> alpha;
    if (argc > 9) std::stringstream(arg[9]) >> beta;
    if (argc > 10) std::stringstream(arg[10]) >> iterations;
    configs.push_back({"custom", N, H, W, C, K, R, S, pad_h, pad_w, 1, 1});
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

    cudaError_t result = TestCorrectness(
      cfg.N, cfg.H, cfg.W, cfg.C, cfg.K, cfg.R, cfg.S,
      cfg.pad_h, cfg.pad_w, cfg.stride_h, cfg.stride_w,
      alpha, beta);
    if (result != cudaSuccess) {
      std::cout << "Incorrect" << std::endl;
      return -1;
    }
    std::cout << "Passed" << std::endl;

    if (iterations > 0) {
      Profile(cfg.N, cfg.H, cfg.W, cfg.C, cfg.K, cfg.R, cfg.S,
              cfg.pad_h, cfg.pad_w, cfg.stride_h, cfg.stride_w,
              alpha, beta, iterations);
    }
  }

  return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
