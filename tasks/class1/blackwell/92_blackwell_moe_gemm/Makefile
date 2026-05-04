CUTLASS_DIR ?= ../../../../reference_sources/cutlass
CUTLASS_INC = -I$(CUTLASS_DIR)/include -I$(CUTLASS_DIR)/tools/util/include -I$(CUTLASS_DIR)/examples/common
NVCC       ?= nvcc
NVCC_FLAGS  = -O3 --use_fast_math -std=c++17 --expt-relaxed-constexpr
ARCH        = -gencode arch=compute_100a,code=sm_100a

# Reference: compile original CUTLASS example (needs CUTLASS headers)
ref: 92_blackwell_moe_gemm_blockscaled_rcgrouped.cu
	$(NVCC) $(NVCC_FLAGS) $(ARCH) $(CUTLASS_INC) $< -o $@

ref_92_blackwell_moe_gemm_fp4_grouped: 92_blackwell_moe_gemm_fp4_grouped.cu
	$(NVCC) $(NVCC_FLAGS) $(ARCH) $(CUTLASS_INC) $< -o $@

ref_92_blackwell_moe_gemm_fp4_regular: 92_blackwell_moe_gemm_fp4_regular.cu
	$(NVCC) $(NVCC_FLAGS) $(ARCH) $(CUTLASS_INC) $< -o $@

ref_92_blackwell_moe_gemm_grouped: 92_blackwell_moe_gemm_grouped.cu
	$(NVCC) $(NVCC_FLAGS) $(ARCH) $(CUTLASS_INC) $< -o $@

ref_92_blackwell_moe_gemm_rcgrouped: 92_blackwell_moe_gemm_rcgrouped.cu
	$(NVCC) $(NVCC_FLAGS) $(ARCH) $(CUTLASS_INC) $< -o $@

ref_92_blackwell_moe_gemm_regular: 92_blackwell_moe_gemm_regular.cu
	$(NVCC) $(NVCC_FLAGS) $(ARCH) $(CUTLASS_INC) $< -o $@

# Test LLM solution: compile with solution.h included (needs CUTLASS headers)
test: 92_blackwell_moe_gemm_grouped.cu solution.h
	$(NVCC) $(NVCC_FLAGS) $(ARCH) $(CUTLASS_INC) -DKH_TEST_SOLUTION $< -o $@

clean:
	rm -f ref ref_* test

.PHONY: clean
