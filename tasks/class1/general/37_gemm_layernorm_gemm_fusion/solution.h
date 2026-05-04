#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Naive GEMM + LayerNorm + GEMM fusion for testing the harness.
//
// The CUTLASS example uses ColumnMajor output, so the memory layout of matrices
// is "transposed" relative to the logical row-major interpretation.
//
// When is_column_major=true (which matches the CUTLASS example):
//   GEMM0: C0 = alpha * A0 * B0 (A0: RowMajor [N0,K0], B0: ColMajor [K0,M])
//   LayerNorm: per-column of C0 (ColMajor [N0, M]) = per-"row" in transposed sense
//   GEMM1: C1 = normalized * A1^T (producing ColMajor [N1, M])
//
// Parameters (column-major case, transposed to row-major interpretation):
//   M_sol = valid_word_num, N0_sol = hidden_dim, K0_sol = hidden_dim, N1_sol = hidden_dim*4
//   A0: raw pointer to RowMajor [hidden_dim, hidden_dim] data (stride=lda0)
//   B0: raw pointer to ColMajor [hidden_dim, valid_word_num] data (stride=ldb0)
//   A1: raw pointer to the second GEMM weight matrix
//   C1: output pointer
//   gamma, beta_ln: layernorm scale/bias
//   shifted_K: shifted K for variance computation (unused in naive impl)

// Simple naive kernels for each step

__global__ void naive_gemm_kernel_37(
    int M, int N, int K,
    float alpha, float beta_val,
    __half const *A, int lda,
    __half const *B, int ldb,
    __half *C, int ldc,
    bool a_row_major, bool b_col_major) {

  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= M || col >= N) return;

  float acc = 0.0f;
  for (int k = 0; k < K; ++k) {
    float a_val, b_val;
    if (a_row_major) {
      a_val = __half2float(A[row * lda + k]);
    } else {
      a_val = __half2float(A[k * lda + row]);
    }
    if (b_col_major) {
      b_val = __half2float(B[k + col * ldb]);
    } else {
      b_val = __half2float(B[k * ldb + col]);
    }
    acc += a_val * b_val;
  }

  float result = alpha * acc;
  if (beta_val != 0.0f) {
    result += beta_val * __half2float(C[row * ldc + col]);
  }
  // Store in column-major: element (row, col) at offset row + col * ldc
  C[row + col * ldc] = __float2half(result);
}

// LayerNorm per-column of a column-major matrix [M_rows, N_cols]
// In column-major, column n has elements at offsets [0 + n*ldc, 1 + n*ldc, ..., (M-1) + n*ldc]
// We normalize along M dimension for each column n.
__global__ void naive_layernorm_col_kernel(
    int M, int N,
    __half *data, int ldc,
    __half const *gamma, __half const *beta_ln) {

  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= N) return;

  // Compute mean
  float sum = 0.0f;
  for (int m = 0; m < M; ++m) {
    sum += __half2float(data[m + col * ldc]);
  }
  float mean = sum / float(M);

  // Compute variance
  float var_sum = 0.0f;
  for (int m = 0; m < M; ++m) {
    float diff = __half2float(data[m + col * ldc]) - mean;
    var_sum += diff * diff;
  }
  float variance = var_sum / float(M);
  float inv_std = 1.0f / sqrtf(variance + 1e-6f);

  // Normalize with gamma and beta
  for (int m = 0; m < M; ++m) {
    float val = (__half2float(data[m + col * ldc]) - mean) * inv_std;
    float g = __half2float(gamma[m]);
    float b = __half2float(beta_ln[m]);
    data[m + col * ldc] = __float2half(val * g + b);
  }
}

// LayerNorm per-row of a row-major matrix
__global__ void naive_layernorm_row_kernel(
    int M, int N,
    __half *data, int ldc,
    __half const *gamma, __half const *beta_ln) {

  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= M) return;

  float sum = 0.0f;
  for (int n = 0; n < N; ++n) {
    sum += __half2float(data[row + n * ldc]);
  }
  float mean = sum / float(N);

  float var_sum = 0.0f;
  for (int n = 0; n < N; ++n) {
    float diff = __half2float(data[row + n * ldc]) - mean;
    var_sum += diff * diff;
  }
  float variance = var_sum / float(N);
  float inv_std = 1.0f / sqrtf(variance + 1e-6f);

  for (int n = 0; n < N; ++n) {
    float val = (__half2float(data[row + n * ldc]) - mean) * inv_std;
    float g = __half2float(gamma[n]);
    float b = __half2float(beta_ln[n]);
    data[row + n * ldc] = __float2half(val * g + b);
  }
}

// GEMM for column-major output:
// C1[M1, N1] = A1[M1, K1] (ColumnMajor) * normalized[K1, N1] (ColumnMajor)
// When is_column_major, the second GEMM is: C1 = A1 * normed_C0
//   A1 is LayoutInputA1, normed_C0 is LayoutOutputC0
//   In CUTLASS ColumnMajor case:
//     GEMM1 problem_size1 = {N1=hidden_dim*4, M_words=valid_word_num, K1=hidden_dim}
//     A1: [N1, K1] with LayoutInputA1
//     normed_C0: [K1, M_words] ColumnMajor
//     C1: [N1, M_words] ColumnMajor
//
//   The ref uses: gemm_device1(problem_size1, alpha, A1, normed_C0, beta, C1, C1)
//   where A1 is the left and normed_C0 is the right.
__global__ void naive_gemm2_colmajor_kernel(
    int M, int N, int K,
    float alpha,
    __half const *A, int lda,  // left matrix
    __half const *B, int ldb,  // right matrix (column-major)
    __half *C, int ldc) {      // output (column-major)

  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= M || col >= N) return;

  float acc = 0.0f;
  for (int k = 0; k < K; ++k) {
    // A: element (row, k) at A[row + k * lda] (column-major) or A[row * lda + k] (row-major)
    // We need to match LayoutInputA1. For the column-major case, A1's layout is ColumnMajor.
    float a_val = __half2float(A[row + k * lda]);
    float b_val = __half2float(B[k + col * ldb]);
    acc += a_val * b_val;
  }
  C[row + col * ldc] = __float2half(alpha * acc);
}

cudaError_t gemm_layernorm_gemm_solution(
    int M_sol, int N0_sol, int K0_sol, int N1_sol,
    float alpha, float beta,
    __half const *A0, int lda0,
    __half const *B0, int ldb0,
    __half const *A1, int lda1,
    __half *C1, int ldc1,
    __half const *gamma,
    __half const *beta_ln,
    __half const *shifted_K,
    bool is_column_major) {

  cudaError_t err;

  if (is_column_major) {
    // Column-major case:
    // GEMM0 produces C0[N0_sol, M_sol] in column-major with stride N0_sol
    // A0 is RowMajor [N0_sol, K0_sol], B0 is ColMajor [K0_sol, M_sol]
    int M0 = N0_sol;  // rows of C0 = hidden_dim
    int N0_out = M_sol;  // cols of C0 = valid_word_num
    int K0 = K0_sol;

    // Allocate intermediate buffer for C0
    __half *d_C0;
    err = cudaMalloc(&d_C0, (int64_t)M0 * N0_out * sizeof(__half));
    if (err != cudaSuccess) return err;
    cudaMemset(d_C0, 0, (int64_t)M0 * N0_out * sizeof(__half));

    // GEMM0: C0[M0, N0_out] = alpha * A0 * B0
    // A0 is RowMajor [M0, K0], B0 is ColMajor [K0, N0_out]
    // Output C0 is ColMajor [M0, N0_out] with stride M0
    {
      dim3 block(16, 16);
      dim3 grid((N0_out + 15) / 16, (M0 + 15) / 16);
      naive_gemm_kernel_37<<<grid, block>>>(
        M0, N0_out, K0,
        alpha, beta,
        A0, lda0,
        B0, ldb0,
        d_C0, M0,
        true, true);
    }
    err = cudaGetLastError();
    if (err != cudaSuccess) { cudaFree(d_C0); return err; }

    // LayerNorm per-column (normalize along M0 dimension for each column)
    {
      int threads = 256;
      int blocks = (N0_out + threads - 1) / threads;
      naive_layernorm_col_kernel<<<blocks, threads>>>(
        M0, N0_out, d_C0, M0, gamma, beta_ln);
    }
    err = cudaGetLastError();
    if (err != cudaSuccess) { cudaFree(d_C0); return err; }

    // GEMM1: C1[N1_sol, M_sol] = alpha * A1 * normed_C0
    // A1 is ColumnMajor [N1_sol, K1=N0_sol=M0], stride = lda1
    // normed_C0 is ColMajor [M0, N0_out=M_sol], stride = M0
    // C1 is ColMajor [N1_sol, M_sol], stride = ldc1
    {
      int M1 = N1_sol;
      int N1_out = M_sol;
      int K1 = M0;  // = N0_sol
      dim3 block(16, 16);
      dim3 grid((N1_out + 15) / 16, (M1 + 15) / 16);
      naive_gemm2_colmajor_kernel<<<grid, block>>>(
        M1, N1_out, K1,
        alpha,
        A1, lda1,
        d_C0, M0,
        C1, ldc1);
    }
    err = cudaGetLastError();
    cudaFree(d_C0);
    return err;
  } else {
    // Row-major case (not used in the CUTLASS example default, but supported)
    // Similar logic but with row-major layouts
    // For now, return an error since the CUTLASS example uses ColumnMajor
    return cudaErrorNotSupported;
  }
}
