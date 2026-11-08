#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <vector>

// Naive grouped GEMM (correctness baseline): one tiled FP32-accumulating kernel launch per
// non-empty group. The group tables live on the device, so this copies them to the host first.
//
//   D_g[M,N] = A_g[M,K] * B_g[K,N]
//   A_g row-major (lda = K), B_g column-major (ldb = K), D_g row-major (ldd = N); BF16 in / out.

#define NG_TILE 64
#define NG_KT 16

__global__ void naive_grouped_gemm_kernel(
    int M, int N, int K,
    __nv_bfloat16 const *A, int lda, __nv_bfloat16 const *B, int ldb,
    __nv_bfloat16 *D, int ldd) {
  __shared__ float As[NG_KT][NG_TILE + 1];
  __shared__ float Bs[NG_KT][NG_TILE + 1];
  int tx = threadIdx.x % 16, ty = threadIdx.x / 16;     // 16 x 16 threads, 4 x 4 outputs each
  int row0 = blockIdx.x * NG_TILE, col0 = blockIdx.y * NG_TILE;
  float acc[4][4] = {};
  for (int k0 = 0; k0 < K; k0 += NG_KT) {
    for (int idx = threadIdx.x; idx < NG_TILE * NG_KT; idx += 256) {
      int r = idx / NG_KT, kk = idx % NG_KT;
      int gr = row0 + r, gc = col0 + r, gk = k0 + kk;
      As[kk][r] = (gr < M && gk < K) ? __bfloat162float(A[size_t(gr) * lda + gk]) : 0.f;
      Bs[kk][r] = (gc < N && gk < K) ? __bfloat162float(B[size_t(gc) * ldb + gk]) : 0.f;
    }
    __syncthreads();
#pragma unroll
    for (int kk = 0; kk < NG_KT; ++kk)
#pragma unroll
      for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] += As[kk][ty * 4 + i] * Bs[kk][tx * 4 + j];
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      int r = row0 + ty * 4 + i, c = col0 + tx * 4 + j;
      if (r < M && c < N) D[size_t(r) * ldd + c] = __float2bfloat16_rn(acc[i][j]);
    }
}

cudaError_t BlackwellGroupedGemm(
    int num_groups,
    int const *Ms, int const *Ns, int const *Ks,
    __nv_bfloat16 const *const *A_ptrs, int const *ldas,
    __nv_bfloat16 const *const *B_ptrs, int const *ldbs,
    __nv_bfloat16 *const *D_ptrs, int const *ldds) {
  std::vector<int> hM(num_groups), hN(num_groups), hK(num_groups), hla(num_groups),
      hlb(num_groups), hld(num_groups);
  std::vector<__nv_bfloat16 const *> hA(num_groups), hB(num_groups);
  std::vector<__nv_bfloat16 *> hD(num_groups);
  size_t n4 = sizeof(int) * num_groups, np = sizeof(void *) * num_groups;
  cudaMemcpy(hM.data(), Ms, n4, cudaMemcpyDeviceToHost);
  cudaMemcpy(hN.data(), Ns, n4, cudaMemcpyDeviceToHost);
  cudaMemcpy(hK.data(), Ks, n4, cudaMemcpyDeviceToHost);
  cudaMemcpy(hla.data(), ldas, n4, cudaMemcpyDeviceToHost);
  cudaMemcpy(hlb.data(), ldbs, n4, cudaMemcpyDeviceToHost);
  cudaMemcpy(hld.data(), ldds, n4, cudaMemcpyDeviceToHost);
  cudaMemcpy(hA.data(), A_ptrs, np, cudaMemcpyDeviceToHost);
  cudaMemcpy(hB.data(), B_ptrs, np, cudaMemcpyDeviceToHost);
  cudaMemcpy(hD.data(), D_ptrs, np, cudaMemcpyDeviceToHost);
  for (int g = 0; g < num_groups; ++g) {
    if (hM[g] <= 0) continue;
    dim3 grid((hM[g] + NG_TILE - 1) / NG_TILE, (hN[g] + NG_TILE - 1) / NG_TILE);
    naive_grouped_gemm_kernel<<<grid, 256>>>(hM[g], hN[g], hK[g], hA[g], hla[g], hB[g], hlb[g],
                                             hD[g], hld[g]);
  }
  return cudaGetLastError();
}
