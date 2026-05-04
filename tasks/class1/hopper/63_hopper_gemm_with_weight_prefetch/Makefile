CUTLASS_DIR ?= ../../../../reference_sources/cutlass
CUTLASS_INC = -I$(CUTLASS_DIR)/include -I$(CUTLASS_DIR)/tools/util/include -I$(CUTLASS_DIR)/examples/common -I$(CUTLASS_DIR)/examples/63_hopper_gemm_with_weight_prefetch
NVCC       ?= nvcc
NVCC_FLAGS  = -O3 --use_fast_math -std=c++17 --expt-relaxed-constexpr
ARCH        = -gencode arch=compute_90a,code=sm_90a

# Reference only: compile CUTLASS example (no solution needed)
ref: 63_hopper_gemm_with_weight_prefetch.cu
	$(NVCC) $(NVCC_FLAGS) $(ARCH) $(CUTLASS_INC) $< -o $@

# Test LLM solution: compile with solution.h included (needs CUTLASS headers)
test: 63_hopper_gemm_with_weight_prefetch.cu solution.h
	$(NVCC) $(NVCC_FLAGS) $(ARCH) -gencode arch=compute_90,code=compute_90 $(CUTLASS_INC) -DKH_TEST_SOLUTION $< -o $@

clean:
	rm -f ref test

.PHONY: clean
