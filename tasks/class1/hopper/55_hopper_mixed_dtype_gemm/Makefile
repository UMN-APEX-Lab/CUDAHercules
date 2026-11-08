CUTLASS_DIR ?= ../../../../reference_sources/cutlass
CUTLASS_INC = -I$(CUTLASS_DIR)/include -I$(CUTLASS_DIR)/tools/util/include -I$(CUTLASS_DIR)/examples/common
NVCC       ?= nvcc
NVCC_FLAGS  = -O3 --use_fast_math -std=c++17 --expt-relaxed-constexpr -DNDEBUG
ARCH        = -gencode arch=compute_90a,code=sm_90a

# Reference only: compile CUTLASS example (no solution needed)
ref: 55_hopper_mixed_dtype_gemm.cu hopper_cublaslt_reference.cuh
	$(NVCC) $(NVCC_FLAGS) $(ARCH) $(CUTLASS_INC) $< -o $@ -lcublasLt

ref_55_hopper_int4_bf16_gemm: 55_hopper_int4_bf16_gemm.cu
	$(NVCC) $(NVCC_FLAGS) $(ARCH) $(CUTLASS_INC) $< -o $@ -lcublasLt

ref_55_hopper_int4_fp8_gemm: 55_hopper_int4_fp8_gemm.cu
	$(NVCC) $(NVCC_FLAGS) $(ARCH) $(CUTLASS_INC) $< -o $@ -lcublasLt

# Test LLM solution: compile with solution.h included (needs CUTLASS headers)
test: 55_hopper_mixed_dtype_gemm.cu solution.h hopper_cublaslt_reference.cuh
	$(NVCC) $(NVCC_FLAGS) $(ARCH) $(CUTLASS_INC) -DKH_TEST_SOLUTION $< -o $@ -lcublasLt

clean:
	rm -f ref ref_* test

.PHONY: clean
