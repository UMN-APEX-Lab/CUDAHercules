CUTLASS_DIR ?= ../../../../reference_sources/cutlass
CUTLASS_INC = -I$(CUTLASS_DIR)/include -I$(CUTLASS_DIR)/tools/util/include -I$(CUTLASS_DIR)/examples/common
NVCC       ?= nvcc
NVCC_FLAGS  = -O3 --use_fast_math -std=c++17 --expt-relaxed-constexpr -DNDEBUG
ARCH        = -gencode arch=compute_90a,code=sm_90a
ARCH_PTX    = -gencode arch=compute_90,code=compute_90

ref: 52_hopper_gather_scatter_fusion.cu hopper_cublaslt_reference.cuh
	$(NVCC) $(NVCC_FLAGS) $(ARCH) $(CUTLASS_INC) $< -o $@ -lcublasLt

test: 52_hopper_gather_scatter_fusion.cu solution.h hopper_cublaslt_reference.cuh
	$(NVCC) $(NVCC_FLAGS) $(ARCH) $(ARCH_PTX) $(CUTLASS_INC) -DKH_TEST_SOLUTION $< -o $@ -lcublasLt

clean:
	rm -f ref test

.PHONY: clean
