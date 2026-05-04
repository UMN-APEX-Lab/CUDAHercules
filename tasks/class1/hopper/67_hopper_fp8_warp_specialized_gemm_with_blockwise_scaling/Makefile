CUTLASS_DIR ?= ../../../../reference_sources/cutlass
CUTLASS_INC = -I$(CUTLASS_DIR)/include -I$(CUTLASS_DIR)/tools/util/include -I$(CUTLASS_DIR)/examples/common
NVCC       ?= nvcc
NVCC_FLAGS  = -O3 --use_fast_math -std=c++17 --expt-relaxed-constexpr
ARCH        = -gencode arch=compute_90a,code=sm_90a

# Reference: compile original CUTLASS example (needs CUTLASS headers)
ref: 67_hopper_fp8_warp_specialized_gemm_with_blockwise_scaling.cu
	$(NVCC) $(NVCC_FLAGS) $(ARCH) $(CUTLASS_INC) $< -o $@

ref_67_hopper_fp8_warp_specialized_gemm_with_groupwise_scaling: 67_hopper_fp8_warp_specialized_gemm_with_groupwise_scaling.cu
	$(NVCC) $(NVCC_FLAGS) $(ARCH) $(CUTLASS_INC) $< -o $@

# Test LLM solution: compile with solution.h included (needs CUTLASS headers)
test: 67_hopper_fp8_warp_specialized_gemm_with_blockwise_scaling.cu solution.h
	$(NVCC) $(NVCC_FLAGS) $(ARCH) -gencode arch=compute_90,code=compute_90 $(CUTLASS_INC) -DKH_TEST_SOLUTION $< -o $@

clean:
	rm -f ref ref_* test

.PHONY: clean
