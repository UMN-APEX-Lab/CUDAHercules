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
  This example requires NVIDIA Maxwell GPU or beyond.
*/

// Standard Library includes
#include <iostream>
#include <sstream>
#include <vector>
#include <cmath>

// CUTLASS Includes
#include "cutlass/cutlass.h"
#include "cutlass/core_io.h"
#include "cutlass/functional.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/warp/mma_simt.h"
#include "cutlass/epilogue/warp/fragment_iterator_simt.h"
#include "cutlass/epilogue/warp/tile_iterator_simt.h"

// CUTLASS Utility Includes
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/tensor_view_io.h"

#include "cutlass/util/reference/host/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/gemm_complex.h"

#include "helper.h"

#ifdef KH_TEST_SOLUTION
#include "solution.h"
#endif

///////////////////////////////////////////////////////////////////////////////////////////////////

// Define the overall warp-level problem shape
int const kM = 14;
int const kN = 27;
int const kK = 17;

///////////////////////////////////////////////////////////////////////////////////////////////////

// Define a warp-level GEMM operator.
//
// This template could be part of the CUTLASS Template Library or implemented internally. This
// wraps the matrix multiply operation and epilogue with a GEMM-like interface that can be
// instantiated in device code.

namespace cutlass {
namespace gemm {
namespace warp {

template <
  typename Shape,
  typename ElementA,
  typename LayoutA,
  typename ElementB,
  typename LayoutB,
  typename ElementC,
  typename LayoutC,
  typename ElementScalar
>
class GemmSimt {
public:


  using Policy = cutlass::gemm::warp::MmaSimtPolicy<
    cutlass::MatrixShape<4, 8>,
    cutlass::layout::RowMajorInterleaved<2>,
    cutlass::gemm::GemmShape<4, 4, 1>
  >;

  using MmaWarp = cutlass::gemm::warp::MmaSimt<
    cutlass::gemm::GemmShape<16, 32, 8>,
    float,
    cutlass::layout::RowMajor,
    float,
    cutlass::layout::ColumnMajor,
    float,
    cutlass::layout::RowMajor,
    Policy
  >;

  // Number of 'K groups'
  int const kKgroups = Shape::kK;

  using FragmentIterator = cutlass::epilogue::warp::FragmentIteratorSimt<
    typename MmaWarp::Shape,
    typename MmaWarp::ThreadMma,
    layout::RowMajor,                // SMEM layout
    typename MmaWarp::Policy
  >;

  using AccumulatorTileIterator = cutlass::epilogue::warp::TileIteratorSimtCanonical<
    typename MmaWarp::Shape,
    typename MmaWarp::ThreadMma,
    float,                             // ElementAccumulator
    layout::RowMajor,                  // SMEM layout
    typename MmaWarp::Policy
  >;

  using TensorRefA = typename MmaWarp::IteratorA::TensorRef;
  using TensorRefB = typename MmaWarp::IteratorB::TensorRef;
  using TensorRefC = typename AccumulatorTileIterator::TensorRef;

public:
  CUTLASS_HOST_DEVICE
  GemmSimt() { }

  CUTLASS_DEVICE
  void operator()(
    ElementScalar alpha,
    TensorRefA ref_A,
    TensorRefB ref_B,
    ElementScalar beta,
    TensorRefC ref_C,
    TensorRefC ref_D,
    int lane_id) const {

    // Instantiate iterators pointing to slices of the A and B matrices in shared memory
    typename MmaWarp::IteratorA iter_A(ref_A, {Shape::kM, Shape::kK}, lane_id);
    typename MmaWarp::IteratorB iter_B(ref_B, {Shape::kK, Shape::kN}, lane_id);

    // Instantiate and clear accumulator tile holding the C matrix
    typename MmaWarp::FragmentC accum;
    accum.clear();

    // Instantiate the warp-level matrix multiply operator
    MmaWarp mma_op;

    // Instantiate fragments holding the slice of the matrix held by each warp
    typename MmaWarp::FragmentA frag_A[2];
    typename MmaWarp::FragmentB frag_B[2];

    // Load fragments from shared memory
    iter_A.load(frag_A[0]);
    iter_B.load(frag_B[0]);

    ++iter_A;
    ++iter_B;

    // Load fragments from shared memory
    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < kKgroups; ++k) {

      // Load fragments from shared memory
      iter_A.load(frag_A[(k + 1) % 2]);
      iter_B.load(frag_B[(k + 1) % 2]);

      ++iter_A;
      ++iter_B;

      // Compute the matrix multiply
      mma_op(accum, frag_A[k % 2], frag_B[k % 2], accum);
    }

    // Instantiate iterators
    FragmentIterator accum_frag_it(accum);
    AccumulatorTileIterator source_tile_it(ref_C, {Shape::kM, Shape::kN}, lane_id);
    AccumulatorTileIterator dest_tile_it(ref_D, {Shape::kM, Shape::kN}, lane_id);

    // Define function objects for linear scaling operation
    cutlass::multiplies<typename FragmentIterator::Fragment> mul_source;
    cutlass::multiply_add<typename FragmentIterator::Fragment> mul_add_accumulator;

    // Iterate over the epilogue components
    CUTLASS_PRAGMA_UNROLL
    for (int idx = 0; idx < FragmentIterator::kIterations; ++idx) {

      // Define storage for slices of the accumulators
      typename FragmentIterator::Fragment accum_fragment;
      typename FragmentIterator::Fragment source_fragment;

      // Select a slice of accumulators from the accumulator tile
      accum_frag_it.load(accum_fragment);
      ++accum_frag_it;

      // Load a corresponding slice from Shared memory
      source_tile_it.load(source_fragment);
      ++source_tile_it;

      // Compute linear scaling - alpha * AB + beta * C
      source_fragment = mul_source(beta, source_fragment);
      accum_fragment = mul_add_accumulator(alpha, accum_fragment, source_fragment);

      // Store the result to shared memory
      dest_tile_it.store(accum_fragment);
      ++dest_tile_it;
    }

  }

};

} // namespace warp
} // namespace gemm
} // namespace cutlass
///////////////////////////////////////////////////////////////////////////////////////////////////

// Sample kernel demonstrating a collective GEMM operation by a warp on arbitrary matrices held
// in Shared Memory.
__global__ void kernel(
  float *D_gmem,
  float alpha,
  float const *A_gmem,
  float const *B_gmem,
  float beta,
  float const *C_gmem) {

  // Define several matrices in shared memory
  __shared__ float A[kM][kK];
  __shared__ float B[kN][kK];
  __shared__ float C[kM][kN];

  // Copy data into SMEM
  if (threadIdx.x == 0) {
    CUTLASS_PRAGMA_NO_UNROLL
    for (int m = 0; m < kM; ++m) {
      for (int k = 0; k < kK; ++k) {
        A[m][k] = A_gmem[m * kK + k];
      }
    }
    CUTLASS_PRAGMA_NO_UNROLL
    for (int n = 0; n < kN; ++n) {
      for (int k = 0; k < kK; ++k) {
        B[n][k] = B_gmem[n * kK + k];
      }
    }
    CUTLASS_PRAGMA_NO_UNROLL
    for (int m = 0; m < kM; ++m) {
      CUTLASS_PRAGMA_NO_UNROLL
      for (int n = 0; n < kN; ++n) {
        C[m][n] = C_gmem[m * kN + n];
      }
    }
  }

  __syncthreads();

  //
  // Instantiate a warp-level matrix multiply operator given the fundamental instruction shape (8x8x4),
  // overall shape, data type of each operand, and layout of each operand.
  //

  using GemmSimt = cutlass::gemm::warp::GemmSimt<
    cutlass::gemm::GemmShape<kM, kN, kK>,
    float,                             // Data type of A elements
    cutlass::layout::RowMajor,          // Layout of A matrix
    float,                             // Data type of B elements
    cutlass::layout::ColumnMajor,       // Layout of B matrix
    float,                             // Data type of C elements
    cutlass::layout::RowMajor,          // Layout of C matrix
    float                              // Scalar type of alpha and beta
  >;

  // Instantiate the GEMM operator
  GemmSimt gemm;

  // Execute the warp-level GEMM operation
  gemm(
    alpha,
    {&A[0][0], kK},
    {&B[0][0], kK},
    beta,
    {&C[0][0], kN},
    {&C[0][0], kN},
    threadIdx.x);

  __syncthreads();

  // Copy data into SMEM
  if (threadIdx.x == 0) {
    CUTLASS_PRAGMA_NO_UNROLL
    for (int m = 0; m < kM; ++m) {
      CUTLASS_PRAGMA_NO_UNROLL
      for (int n = 0; n < kN; ++n) {
        D_gmem[m * kN + n] = C[m][n];
      }
    }
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

// CUTLASS device-level SGEMM reference for arbitrary sizes:
// A: row-major, B: column-major, C/D: row-major
using CutlassSgemmRef = cutlass::gemm::device::Gemm<
  float,                             // A type
  cutlass::layout::RowMajor,         // A layout
  float,                             // B type
  cutlass::layout::ColumnMajor,      // B layout
  float,                             // C type
  cutlass::layout::RowMajor,         // C layout
  float,                             // accumulator
  cutlass::arch::OpClassSimt,
  cutlass::arch::Sm80
>;

cudaError_t CutlassSgemmRef_run(
  int M, int N, int K, float alpha,
  float const *A, int lda,
  float const *B, int ldb,
  float beta,
  float const *C, int ldc,
  float *D, int ldd) {

  CutlassSgemmRef gemm_op;
  CutlassSgemmRef::Arguments args(
    {M, N, K},
    {A, lda},
    {B, ldb},
    {C, ldc},
    {D, ldd},
    {alpha, beta}
  );

  cutlass::Status status = gemm_op(args);
  if (status != cutlass::Status::kSuccess) {
    return cudaErrorUnknown;
  }
  return cudaSuccess;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Kernel to initialize a row-major matrix with small integers.
__global__ void InitializeMatrixRowMajor_kernel(
  float *matrix, int rows, int columns, int ld, int seed) {

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  if (i < rows && j < columns) {
    int offset = i * ld + j;
    int const k = 16807;
    int const m = 16;
    float value = float(((offset + seed) * k % m) - m / 2);
    matrix[offset] = value;
  }
}

cudaError_t InitializeMatrixRowMajor(float *matrix, int rows, int columns, int ld, int seed = 0) {
  dim3 block(16, 16);
  dim3 grid(
    (rows + block.x - 1) / block.x,
    (columns + block.y - 1) / block.y
  );
  InitializeMatrixRowMajor_kernel<<< grid, block >>>(matrix, rows, columns, ld, seed);
  return cudaGetLastError();
}

/// Kernel to initialize a column-major matrix with small integers.
__global__ void InitializeMatrixColMajor_kernel(
  float *matrix, int rows, int columns, int ld, int seed) {

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  if (i < rows && j < columns) {
    int offset = i + j * ld;
    int const k = 16807;
    int const m = 16;
    float value = float(((offset + seed) * k % m) - m / 2);
    matrix[offset] = value;
  }
}

cudaError_t InitializeMatrixColMajor(float *matrix, int rows, int columns, int ld, int seed = 0) {
  dim3 block(16, 16);
  dim3 grid(
    (rows + block.x - 1) / block.x,
    (columns + block.y - 1) / block.y
  );
  InitializeMatrixColMajor_kernel<<< grid, block >>>(matrix, rows, columns, ld, seed);
  return cudaGetLastError();
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Allocate row-major matrix
cudaError_t AllocateMatrixRowMajor(float **matrix, int rows, int columns, int ld, int seed = 0) {
  cudaError_t result;
  size_t sz = sizeof(float) * rows * ld;
  result = cudaMalloc(reinterpret_cast<void **>(matrix), sz);
  if (result != cudaSuccess) return result;
  result = cudaMemset(*matrix, 0, sz);
  if (result != cudaSuccess) return result;
  return InitializeMatrixRowMajor(*matrix, rows, columns, ld, seed);
}

/// Allocate column-major matrix
cudaError_t AllocateMatrixColMajor(float **matrix, int rows, int columns, int ld, int seed = 0) {
  cudaError_t result;
  size_t sz = sizeof(float) * ld * columns;
  result = cudaMalloc(reinterpret_cast<void **>(matrix), sz);
  if (result != cudaSuccess) return result;
  result = cudaMemset(*matrix, 0, sz);
  if (result != cudaSuccess) return result;
  return InitializeMatrixColMajor(*matrix, rows, columns, ld, seed);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Naive reference GEMM: D = alpha * A * B + beta * C
/// A: row-major (M x K, lda), B: column-major (K x N, ldb), C/D: row-major (M x N, ldc/ldd)
__global__ void ReferenceGemm_kernel(
  int M, int N, int K,
  float alpha,
  float const *A, int lda,
  float const *B, int ldb,
  float beta,
  float const *C, int ldc,
  float *D, int ldd) {

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  if (i < M && j < N) {
    float accumulator = 0;
    for (int k = 0; k < K; ++k) {
      accumulator += A[i * lda + k] * B[k + j * ldb];  // A row-major, B col-major
    }
    D[i * ldd + j] = alpha * accumulator + beta * C[i * ldc + j];
  }
}

cudaError_t ReferenceGemm(
  int M, int N, int K,
  float alpha,
  float const *A, int lda,
  float const *B, int ldb,
  float beta,
  float const *C, int ldc,
  float *D, int ldd) {

  dim3 block(16, 16);
  dim3 grid(
    (M + block.x - 1) / block.x,
    (N + block.y - 1) / block.y
  );

  ReferenceGemm_kernel<<< grid, block >>>(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, D, ldd);

  return cudaGetLastError();
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Test correctness: CUTLASS reference vs naive, and (if KH_TEST_SOLUTION) solution vs CUTLASS.
cudaError_t TestCorrectness(int M, int N, int K, float alpha, float beta) {
  cudaError_t result;

  // A: row-major M x K, lda = K
  // B: column-major K x N, ldb = K
  // C, D: row-major M x N, ldc = ldd = N
  int lda = K;
  int ldb = K;
  int ldc = N;
  int ldd = N;

  size_t sizeof_CD = sizeof(float) * M * ldd;

  float *A, *B, *C, *D_cutlass, *D_reference;

  // A is row-major
  result = AllocateMatrixRowMajor(&A, M, K, lda, 0);
  if (result != cudaSuccess) return result;

  // B is column-major
  result = AllocateMatrixColMajor(&B, K, N, ldb, 17);
  if (result != cudaSuccess) { cudaFree(A); return result; }

  // C is row-major
  result = AllocateMatrixRowMajor(&C, M, N, ldc, 101);
  if (result != cudaSuccess) { cudaFree(A); cudaFree(B); return result; }

  // D_cutlass
  result = cudaMalloc(&D_cutlass, sizeof_CD);
  if (result != cudaSuccess) { cudaFree(A); cudaFree(B); cudaFree(C); return result; }
  cudaMemset(D_cutlass, 0, sizeof_CD);

  // D_reference
  result = cudaMalloc(&D_reference, sizeof_CD);
  if (result != cudaSuccess) { cudaFree(A); cudaFree(B); cudaFree(C); cudaFree(D_cutlass); return result; }
  cudaMemset(D_reference, 0, sizeof_CD);

  // Run CUTLASS reference
  result = CutlassSgemmRef_run(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, D_cutlass, ldd);
  if (result != cudaSuccess) {
    std::cerr << "CUTLASS GEMM failed: " << cudaGetErrorString(result) << std::endl;
    cudaFree(D_reference); cudaFree(D_cutlass); cudaFree(C); cudaFree(B); cudaFree(A);
    return result;
  }
  cudaDeviceSynchronize();

  // Run naive reference
  result = ReferenceGemm(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, D_reference, ldd);
  if (result != cudaSuccess) {
    std::cerr << "Reference GEMM failed: " << cudaGetErrorString(result) << std::endl;
    cudaFree(D_reference); cudaFree(D_cutlass); cudaFree(C); cudaFree(B); cudaFree(A);
    return result;
  }
  cudaDeviceSynchronize();

  // Verify CUTLASS vs naive
  int count_D = M * ldd;
  std::vector<float> host_cutlass(count_D);
  std::vector<float> host_reference(count_D);

  cudaMemcpy(host_cutlass.data(), D_cutlass, sizeof_CD, cudaMemcpyDeviceToHost);
  cudaMemcpy(host_reference.data(), D_reference, sizeof_CD, cudaMemcpyDeviceToHost);

  float max_diff = 0;
  for (int i = 0; i < count_D; ++i) {
    max_diff = fmaxf(max_diff, fabsf(host_cutlass[i] - host_reference[i]));
  }
  if (max_diff > 1e-3f) {
    std::cerr << "CUTLASS reference incorrect vs naive. Max diff: " << max_diff << std::endl;
    cudaFree(D_reference); cudaFree(D_cutlass); cudaFree(C); cudaFree(B); cudaFree(A);
    return cudaErrorUnknown;
  }

#ifdef KH_TEST_SOLUTION
  // Allocate solution output
  float *D_solution;
  result = cudaMalloc(&D_solution, sizeof_CD);
  if (result != cudaSuccess) {
    cudaFree(D_reference); cudaFree(D_cutlass); cudaFree(C); cudaFree(B); cudaFree(A);
    return result;
  }
  cudaMemset(D_solution, 0, sizeof_CD);

  // Run solution
  result = SimtCanonicalGemm(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, D_solution, ldd);
  if (result != cudaSuccess) {
    std::cerr << "Solution kernel failed: " << cudaGetErrorString(result) << std::endl;
    cudaFree(D_solution); cudaFree(D_reference); cudaFree(D_cutlass); cudaFree(C); cudaFree(B); cudaFree(A);
    return result;
  }
  cudaDeviceSynchronize();

  // Verify solution vs CUTLASS reference
  std::vector<float> host_solution(count_D);
  cudaMemcpy(host_solution.data(), D_solution, sizeof_CD, cudaMemcpyDeviceToHost);

  float sol_max_diff = 0;
  for (int i = 0; i < count_D; ++i) {
    sol_max_diff = fmaxf(sol_max_diff, fabsf(host_solution[i] - host_cutlass[i]));
  }
  if (sol_max_diff > 1e-3f) {
    std::cerr << "Solution incorrect. Max diff vs reference: " << sol_max_diff << std::endl;
    cudaFree(D_solution); cudaFree(D_reference); cudaFree(D_cutlass); cudaFree(C); cudaFree(B); cudaFree(A);
    return cudaErrorUnknown;
  }

  cudaFree(D_solution);
#endif

  cudaFree(D_reference);
  cudaFree(D_cutlass);
  cudaFree(C);
  cudaFree(B);
  cudaFree(A);

  return cudaSuccess;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Profile reference (and solution if KH_TEST_SOLUTION) with CUDA events.
void Profile(int M, int N, int K, float alpha, float beta, int iterations) {
  int lda = K, ldb = K, ldc = N, ldd = N;

  float *A, *B, *C, *D;
  AllocateMatrixRowMajor(&A, M, K, lda, 0);
  AllocateMatrixColMajor(&B, K, N, ldb, 17);
  AllocateMatrixRowMajor(&C, M, N, ldc, 101);

  size_t sizeof_D = sizeof(float) * M * ldd;
  cudaMalloc(&D, sizeof_D);
  cudaMemset(D, 0, sizeof_D);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Warmup reference
  for (int i = 0; i < 3; i++) {
    CutlassSgemmRef_run(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, D, ldd);
  }
  cudaDeviceSynchronize();

  // Profile reference
  float ref_total_ms = 0, ref_min_ms = 1e30f;
  for (int iter = 0; iter < iterations; ++iter) {
    cudaEventRecord(start);
    CutlassSgemmRef_run(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, D, ldd);
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
  cudaMemset(D, 0, sizeof_D);
  for (int i = 0; i < 3; i++) {
    SimtCanonicalGemm(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, D, ldd);
  }
  cudaDeviceSynchronize();

  // Profile solution
  float sol_total_ms = 0, sol_min_ms = 1e30f;
  for (int iter = 0; iter < iterations; ++iter) {
    cudaEventRecord(start);
    SimtCanonicalGemm(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, D, ldd);
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
  cudaFree(D);
  cudaFree(C);
  cudaFree(B);
  cudaFree(A);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, const char *arg[]) {

  struct TestConfig {
    const char* label;
    int m, n, k;
  };

  bool explicit_size = (argc >= 4);

  std::vector<TestConfig> configs;
  float alpha = 1.0f, beta = 0.0f;
  int iterations = 20;

  if (explicit_size) {
    int M = 128, N = 128, K = 128;
    std::stringstream(arg[1]) >> M;
    std::stringstream(arg[2]) >> N;
    std::stringstream(arg[3]) >> K;
    if (argc > 4) std::stringstream(arg[4]) >> alpha;
    if (argc > 5) std::stringstream(arg[5]) >> beta;
    if (argc > 6) std::stringstream(arg[6]) >> iterations;
    configs.push_back({"custom", M, N, K});
  } else {
    configs = {
      {"small",   1024, 1024, 1024},
      {"medium",  4096, 4096, 4096},
      {"large",   8192, 8192, 8192},
    };
  }

  for (const auto& cfg : configs) {
    fprintf(stdout, "\n=== %s: M=%d, N=%d, K=%d ===\n", cfg.label, cfg.m, cfg.n, cfg.k);

    cudaError_t result = TestCorrectness(cfg.m, cfg.n, cfg.k, alpha, beta);
    if (result != cudaSuccess) {
      std::cout << "Incorrect" << std::endl;
      return -1;
    }
    std::cout << "Passed" << std::endl;

    if (iterations > 0) {
      Profile(cfg.m, cfg.n, cfg.k, alpha, beta, iterations);
    }
  }

  return 0;
}
///////////////////////////////////////////////////////////////////////////////////////////////////
