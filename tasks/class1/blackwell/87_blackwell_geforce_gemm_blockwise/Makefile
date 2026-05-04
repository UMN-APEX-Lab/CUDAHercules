CUTLASS_DIR ?= ../../../../reference_sources/cutlass
CUTLASS_INC = -I$(CUTLASS_DIR)/include -I$(CUTLASS_DIR)/tools/util/include -I$(CUTLASS_DIR)/examples/common
NVCC       ?= nvcc
NVCC_FLAGS  = -O3 --use_fast_math -std=c++17 --expt-relaxed-constexpr
ARCH        = -gencode arch=compute_120a,code=sm_120a

# Reference only: compile CUTLASS example (no solution needed)
ref: 87a_blackwell_geforce_fp8_bf16_gemm_blockwise.cu
	$(NVCC) $(NVCC_FLAGS) $(ARCH) $(CUTLASS_INC) $< -o $@

ref_87b_blackwell_geforce_fp8_bf16_gemm_groupwise: 87b_blackwell_geforce_fp8_bf16_gemm_groupwise.cu
	$(NVCC) $(NVCC_FLAGS) $(ARCH) $(CUTLASS_INC) $< -o $@

ref_87c_blackwell_geforce_fp8_bf16_grouped_gemm_groupwise: 87c_blackwell_geforce_fp8_bf16_grouped_gemm_groupwise.cu
	$(NVCC) $(NVCC_FLAGS) $(ARCH) $(CUTLASS_INC) $< -o $@

# Test LLM solution: compile with solution.h included (needs CUTLASS headers)
test: 87a_blackwell_geforce_fp8_bf16_gemm_blockwise.cu solution.h
	$(NVCC) $(NVCC_FLAGS) $(ARCH) $(CUTLASS_INC) -DKH_TEST_SOLUTION $< -o $@

clean:
	rm -f ref ref_* test

.PHONY: clean
