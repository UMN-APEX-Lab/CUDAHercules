#pragma once
// Reference #2 for the W4A16 task: the vLLM Marlin kernel (Apache-2.0, derived from IST-DASLab/Marlin),
// the BF16 x INT4 (group 128) GEMM that serves W4A16 models on sm80+ GPUs including B200. The weights
// are repacked into Marlin's tile layout in the constructor (what a deployment does once, at load time),
// so run() is only the kernel launch(es) with vLLM's own launch heuristics.
#include <cuda_bf16.h>
#include <cstdint>
#include <memory>

struct MarlinRefImpl;

/// True if the Marlin kernels instantiated in this build cover problem (M, N, K) with vLLM's
/// default launch heuristics (bf16 activations, uint4b8 weights, group size 128).
bool marlin_ref_supported(int M, int N, int K);

class MarlinRef {
 public:
  /// B: packed int4 [N][K/2] bytes (two's complement nibbles, low nibble = even k); S: bf16 scales
  /// [N][K/128]; A: bf16 [M][K]; D: bf16 [M][N]. All are device pointers; B and S are repacked here.
  MarlinRef(uint8_t const *B, __nv_bfloat16 const *S, __nv_bfloat16 const *A, __nv_bfloat16 *D,
            int M, int N, int K);
  ~MarlinRef();
  void run();
 private:
  std::unique_ptr<MarlinRefImpl> impl_;
};
