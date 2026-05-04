# Icicle ZK Build Contract

You generate exactly one source file:

- `custom_cuda_backend/backend.cu`

The benchmark owns the build. You do not need to generate `build.sh` or
`CMakeLists.txt`.

## Fixed ABI

The benchmark compiles `backend.cu` into a shared library and loads it with
`dlopen()`. Your source must implement the symbols declared in
`custom_backend_api.h`.

The required entrypoints are:

- `kh_custom_backend_init`
- `kh_custom_backend_shutdown`
- `kh_custom_backend_last_error`
- `kh_custom_ntt_forward_bn254`
- `kh_custom_msm_bn254`

## Input / Output Semantics

- `kh_custom_ntt_forward_bn254` receives host pointers to BN254 scalar-field
  inputs and must write host outputs for a forward NTT of size `1 << log_n`.
- `kh_custom_msm_bn254` receives host pointers to BN254 scalars and affine
  points and must write one BN254 projective output.
- Return `0` on success and non-zero on failure.
- On failure, expose a short message via `kh_custom_backend_last_error()`.

The CPU reference, GPU reference, and custom backend all use the same logical
inputs and the same correctness reference files.

## Evaluation Semantics

- The benchmark generates or loads deterministic inputs itself
- Correctness is checked against the CPU reference
- Performance is compared against the provided GPU reference backend
- You may use public Icicle headers and types, but you do not need to implement
  Icicle backend registration

## Guidance

- Keep the implementation self-contained in `backend.cu`
- Use CUDA kernels directly; the benchmark rejects solutions without a real
  kernel launch
- Handle allocation, copies, and synchronization inside your own code
