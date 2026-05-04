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
The convolution version of 12_gemm_bias_relu. The bias vector is placed in Operand C
and the epilogue computes: output = relu(alpha * Conv2dFprop(A, B) + bias[k]).

Input/filter: cutlass::half_t (FP16) in NHWC layout.
Output: float (FP32) in NHWC layout.
Bias: float, per output channel (length K), broadcast over N, P, Q.
*/

#include <iostream>
#include <sstream>
#include <vector>
#include <cmath>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/conv/kernel/default_conv2d_fprop.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/host_reorder.h"
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

// Data types
using ElementAccumulator = float;
using ElementComputeEpilogue = ElementAccumulator;
using ElementInputA = cutlass::half_t;
using ElementInputB = cutlass::half_t;
using ElementOutput = float;

using LayoutInputA = cutlass::layout::TensorNHWC;
using LayoutInputB = cutlass::layout::TensorNHWC;
using LayoutOutput = cutlass::layout::TensorNHWC;

using MMAOp = cutlass::arch::OpClassTensorOp;
using SmArch = cutlass::arch::Sm80;

using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;
using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
constexpr int NumStages = 4;

static cutlass::conv::IteratorAlgorithm const IteratorAlgorithm = cutlass::conv::IteratorAlgorithm::kOptimized;

// Epilogue: alpha * X + per_channel_bias, then ReLU
using EpilogueOp = cutlass::epilogue::thread::LinearCombinationRelu<
    ElementOutput,
    128 / cutlass::sizeof_bits<ElementOutput>::value,
    ElementAccumulator,
    ElementComputeEpilogue,
    cutlass::epilogue::thread::ScaleType::NoBetaScaling>;

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
  IteratorAlgorithm
>::Kernel;

using ImplicitGemm = cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel>;

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Run the CUTLASS conv2d fprop with per-channel bias + ReLU
cudaError_t CutlassConv2dFpropBias(
  cutlass::conv::Conv2dProblemSize const &problem_size,
  cutlass::HostTensor<ElementInputA, LayoutInputA> &tensor_a,
  cutlass::HostTensor<ElementInputB, LayoutInputB> &tensor_b,
  cutlass::HostTensor<ElementOutput, LayoutOutput> &tensor_c_bias,
  cutlass::HostTensor<ElementOutput, LayoutOutput> &tensor_d,
  float alpha) {

  typename ImplicitGemm::Arguments arguments{
    problem_size,
    tensor_a.device_ref(),
    tensor_b.device_ref(),
    // tensor_c_bias has shape 1x1x1xK, stride=0 broadcasts across N,P,Q
    {tensor_c_bias.device_data(), LayoutOutput::Stride(0)},
    tensor_d.device_ref(),
    {alpha}
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

/// Naive reference: conv2d fprop + per-channel bias + ReLU, NHWC layout
/// output[n][p][q][k] = max(0, alpha * sum_{c,r,s} input[n][h][w][c] * filter[k][r][s][c] + bias[k])
__global__ void ReferenceConv2dBias_kernel(
  cutlass::half_t const *input, cutlass::half_t const *filter,
  float const *bias, float *output,
  int N, int C, int H, int W, int K, int R, int S,
  int P, int Q, int pad_h, int pad_w, int stride_h, int stride_w,
  float alpha) {

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
  float result = alpha * acc + bias[k_idx];
  result = fmaxf(0.0f, result);  // ReLU
  int out_idx = ((n_idx * P + p_idx) * Q + q_idx) * K + k_idx;
  output[out_idx] = result;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Test correctness
cudaError_t TestCorrectness(
  int N, int H, int W, int C, int K, int R, int S,
  int pad_h, int pad_w, int stride_h, int stride_w,
  float alpha) {

  int P = (H + 2 * pad_h - R) / stride_h + 1;
  int Q = (W + 2 * pad_w - S) / stride_w + 1;

  cutlass::Tensor4DCoord input_size(N, H, W, C);
  cutlass::Tensor4DCoord filter_size(K, R, S, C);
  cutlass::Tensor4DCoord output_size(N, P, Q, K);
  cutlass::Tensor4DCoord padding(pad_h, pad_h, pad_w, pad_w);
  cutlass::MatrixCoord conv_stride_coord(stride_h, stride_w);
  cutlass::MatrixCoord dilation(1, 1);

  cutlass::conv::Conv2dProblemSize problem_size(
      input_size, filter_size, padding, conv_stride_coord, dilation,
      output_size, cutlass::conv::Mode::kCrossCorrelation, 1);

  cutlass::HostTensor<ElementInputA, LayoutInputA> tensor_a(input_size);
  cutlass::HostTensor<ElementInputB, LayoutInputB> tensor_b(filter_size);
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_c_bias({1, 1, 1, K});
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_d_cutlass(output_size);
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_d_ref(output_size);

  cutlass::reference::host::TensorFillRandomUniform(tensor_a.host_view(), 1,
      ElementInputA(4), ElementInputA(-4), 0);
  cutlass::reference::host::TensorFillRandomUniform(tensor_b.host_view(), 1,
      ElementInputB(4), ElementInputB(-4), 0);
  cutlass::reference::host::TensorFillRandomUniform(tensor_c_bias.host_view(), 1,
      ElementOutput(4), ElementOutput(-4), 0);
  cutlass::reference::host::TensorFill(tensor_d_cutlass.host_view());
  cutlass::reference::host::TensorFill(tensor_d_ref.host_view());

  tensor_a.sync_device();
  tensor_b.sync_device();
  tensor_c_bias.sync_device();
  tensor_d_cutlass.sync_device();
  tensor_d_ref.sync_device();

  // Run CUTLASS reference
  cudaError_t result = CutlassConv2dFpropBias(
    problem_size, tensor_a, tensor_b, tensor_c_bias, tensor_d_cutlass, alpha);
  if (result != cudaSuccess) {
    std::cerr << "CUTLASS conv2d failed: " << cudaGetErrorString(result) << std::endl;
    return result;
  }
  cudaDeviceSynchronize();

  // Run naive reference
  int output_count = N * P * Q * K;
  int threads = 256;
  int blocks = (output_count + threads - 1) / threads;
  ReferenceConv2dBias_kernel<<<blocks, threads>>>(
    tensor_a.device_data(), tensor_b.device_data(),
    tensor_c_bias.device_data(), tensor_d_ref.device_data(),
    N, C, H, W, K, R, S, P, Q,
    pad_h, pad_w, stride_h, stride_w, alpha);
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

  if (max_diff > 1e-1f) {
    std::cerr << "CUTLASS reference incorrect vs naive. Max diff: " << max_diff << std::endl;
    return cudaErrorUnknown;
  }

#ifdef KH_TEST_SOLUTION
  {
  // Run solution
  float *d_solution;
  cudaMalloc(&d_solution, output_count * sizeof(float));
  cudaMemset(d_solution, 0, output_count * sizeof(float));

  // Copy bias to raw float array
  float *d_bias;
  cudaMalloc(&d_bias, K * sizeof(float));
  cudaMemcpy(d_bias, tensor_c_bias.device_data(), K * sizeof(float), cudaMemcpyDeviceToDevice);

  result = Conv2dFpropBiasRelu(
    reinterpret_cast<__half const *>(tensor_a.device_data()),
    reinterpret_cast<__half const *>(tensor_b.device_data()),
    d_bias, d_solution,
    N, C, H, W, K, R, S,
    pad_h, pad_w, stride_h, stride_w, alpha);
  if (result != cudaSuccess) {
    std::cerr << "Solution kernel failed: " << cudaGetErrorString(result) << std::endl;
    cudaFree(d_solution); cudaFree(d_bias);
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
    cudaFree(d_solution); cudaFree(d_bias);
    return cudaErrorUnknown;
  }

  cudaFree(d_solution);
  cudaFree(d_bias);
  }
#endif

  return cudaSuccess;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Profile reference (and solution if KH_TEST_SOLUTION) with CUDA events.
void Profile(
  int N, int H, int W, int C, int K, int R, int S,
  int pad_h, int pad_w, int stride_h, int stride_w,
  float alpha, int iterations) {

  int P = (H + 2 * pad_h - R) / stride_h + 1;
  int Q = (W + 2 * pad_w - S) / stride_w + 1;

  cutlass::Tensor4DCoord input_size(N, H, W, C);
  cutlass::Tensor4DCoord filter_size(K, R, S, C);
  cutlass::Tensor4DCoord output_size(N, P, Q, K);
  cutlass::Tensor4DCoord padding(pad_h, pad_h, pad_w, pad_w);
  cutlass::MatrixCoord conv_stride_coord(stride_h, stride_w);
  cutlass::MatrixCoord dilation(1, 1);

  cutlass::conv::Conv2dProblemSize problem_size(
      input_size, filter_size, padding, conv_stride_coord, dilation,
      output_size, cutlass::conv::Mode::kCrossCorrelation, 1);

  cutlass::HostTensor<ElementInputA, LayoutInputA> tensor_a(input_size);
  cutlass::HostTensor<ElementInputB, LayoutInputB> tensor_b(filter_size);
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_c_bias({1, 1, 1, K});
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_d(output_size);

  cutlass::reference::host::TensorFillRandomUniform(tensor_a.host_view(), 1,
      ElementInputA(4), ElementInputA(-4), 0);
  cutlass::reference::host::TensorFillRandomUniform(tensor_b.host_view(), 1,
      ElementInputB(4), ElementInputB(-4), 0);
  cutlass::reference::host::TensorFillRandomUniform(tensor_c_bias.host_view(), 1,
      ElementOutput(4), ElementOutput(-4), 0);
  cutlass::reference::host::TensorFill(tensor_d.host_view());

  tensor_a.sync_device();
  tensor_b.sync_device();
  tensor_c_bias.sync_device();
  tensor_d.sync_device();

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Warmup reference
  for (int i = 0; i < 3; i++) {
    CutlassConv2dFpropBias(problem_size, tensor_a, tensor_b, tensor_c_bias, tensor_d, alpha);
  }
  cudaDeviceSynchronize();

  // Profile reference
  float ref_total_ms = 0, ref_min_ms = 1e30f;
  for (int iter = 0; iter < iterations; ++iter) {
    cudaEventRecord(start);
    CutlassConv2dFpropBias(problem_size, tensor_a, tensor_b, tensor_c_bias, tensor_d, alpha);
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

  float *d_sol_output;
  cudaMalloc(&d_sol_output, output_count * sizeof(float));

  float *d_bias;
  cudaMalloc(&d_bias, K * sizeof(float));
  cudaMemcpy(d_bias, tensor_c_bias.device_data(), K * sizeof(float), cudaMemcpyDeviceToDevice);

  // Warmup solution
  for (int i = 0; i < 3; i++) {
    cudaMemset(d_sol_output, 0, output_count * sizeof(float));
    Conv2dFpropBiasRelu(
      reinterpret_cast<__half const *>(tensor_a.device_data()),
      reinterpret_cast<__half const *>(tensor_b.device_data()),
      d_bias, d_sol_output,
      N, C, H, W, K, R, S,
      pad_h, pad_w, stride_h, stride_w, alpha);
  }
  cudaDeviceSynchronize();

  // Profile solution
  float sol_total_ms = 0, sol_min_ms = 1e30f;
  for (int iter = 0; iter < iterations; ++iter) {
    cudaMemset(d_sol_output, 0, output_count * sizeof(float));
    cudaEventRecord(start);
    Conv2dFpropBiasRelu(
      reinterpret_cast<__half const *>(tensor_a.device_data()),
      reinterpret_cast<__half const *>(tensor_b.device_data()),
      d_bias, d_sol_output,
      N, C, H, W, K, R, S,
      pad_h, pad_w, stride_h, stride_w, alpha);
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
  cudaFree(d_bias);
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
  float alpha = 1.0f;
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
    if (argc > 9) std::stringstream(arg[9]) >> iterations;
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
      cfg.pad_h, cfg.pad_w, cfg.stride_h, cfg.stride_w, alpha);
    if (result != cudaSuccess) {
      std::cout << "Incorrect" << std::endl;
      return -1;
    }
    std::cout << "Passed" << std::endl;

    if (iterations > 0) {
      Profile(cfg.N, cfg.H, cfg.W, cfg.C, cfg.K, cfg.R, cfg.S,
              cfg.pad_h, cfg.pad_w, cfg.stride_h, cfg.stride_w,
              alpha, iterations);
    }
  }

  return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
