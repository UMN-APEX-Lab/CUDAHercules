CUTLASS_DIR ?= ../../../../reference_sources/cutlass
CUTLASS_INC = -I$(CUTLASS_DIR)/include -I$(CUTLASS_DIR)/tools/util/include -I$(CUTLASS_DIR)/examples/common
NVCC       ?= nvcc
NVCC_FLAGS  = -O3 --use_fast_math -std=c++17 --expt-relaxed-constexpr
ARCH        = -gencode arch=compute_90a,code=sm_90a
ARCH_PTX    = -gencode arch=compute_90,code=compute_90

ref: 50_hopper_gemm_with_epilogue_swizzle.cu
	$(NVCC) $(NVCC_FLAGS) $(ARCH) $(CUTLASS_INC) $< -o $@

test: 50_hopper_gemm_with_epilogue_swizzle.cu solution.h
	$(NVCC) $(NVCC_FLAGS) $(ARCH) $(ARCH_PTX) $(CUTLASS_INC) -DKH_TEST_SOLUTION $< -o $@

clean:
	rm -f ref test

.PHONY: clean
