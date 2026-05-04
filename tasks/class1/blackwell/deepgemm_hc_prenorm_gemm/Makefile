CUTLASS_DIR  ?= ../../../../reference_sources/cutlass
DEEPGEMM_DIR ?= ../../../../reference_sources/deep_gemm
CUTLASS_INC  = -I$(CUTLASS_DIR)/include -I$(CUTLASS_DIR)/tools/util/include
DEEPGEMM_INC = -I$(DEEPGEMM_DIR)
NVCC         ?= nvcc
NVCC_FLAGS   = -O3 --use_fast_math -std=c++17 --expt-relaxed-constexpr
ARCH         = -gencode arch=compute_100a,code=sm_100a

ref: deepgemm_hc_prenorm_gemm.cu
	$(NVCC) $(NVCC_FLAGS) $(ARCH) $(CUTLASS_INC) $(DEEPGEMM_INC) -lcuda $< -o $@

test: deepgemm_hc_prenorm_gemm.cu solution.h
	$(NVCC) $(NVCC_FLAGS) $(ARCH) $(CUTLASS_INC) $(DEEPGEMM_INC) -lcuda -DKH_TEST_SOLUTION $< -o $@

clean:
	rm -f ref test

.PHONY: clean
