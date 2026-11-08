Vendored, unmodified sources of the vLLM Marlin W4A16 GEMM (Apache-2.0, see LICENSE).

  marlin.cuh, marlin_dtypes.cuh, dequant.h, marlin_mma.h, marlin_template.h, kernel.h
      <- https://github.com/vllm-project/vllm  csrc/quantization/marlin/   (commit 5efb24d5420b1f28737e4e8d1b62b354ddc18069)
  core/scalar_type.hpp
      <- csrc/core/scalar_type.hpp (same commit)
  stub/torch/headeronly/util/Exception.h
      minimal replacement for the torch header that scalar_type.hpp includes (STD_TORCH_CHECK only);
      this directory is NOT from vLLM.

Marlin itself: Copyright (C) Marlin.2024 Elias Frantar (https://github.com/IST-DASLab/marlin), modified by Neural Magic.
The launcher (../marlin_ref.cu) adapts vLLM's marlin.cu / gptq_marlin_repack.cu / marlin_utils.py to plain CUDA.
