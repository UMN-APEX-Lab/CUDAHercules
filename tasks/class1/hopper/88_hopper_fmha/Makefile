CUTLASS_DIR ?= ../../../../reference_sources/cutlass
CUTLASS_INC = -I$(CUTLASS_DIR)/include -I$(CUTLASS_DIR)/tools/util/include -I$(CUTLASS_DIR)/examples/common -I$(CUTLASS_DIR)/examples/88_hopper_fmha
FA3_DIR ?= $(CUTLASS_DIR)/../flash_attn_v3
FA3_CUTLASS ?= $(CUTLASS_DIR)/../flash-attention/csrc/cutlass
NVCC       ?= nvcc
NVCC_FLAGS  = -O3 --use_fast_math -std=c++17 --expt-relaxed-constexpr -DNDEBUG -DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED
ARCH        = -gencode arch=compute_90a,code=sm_90a

# Reference: compile original CUTLASS example (no solution needed)
ref: 88_hopper_fmha.cu fa3_reference.o hopper_cublaslt_reference.cuh
	$(NVCC) $(NVCC_FLAGS) $(ARCH) $(CUTLASS_INC) $< fa3_reference.o -lcublasLt -o $@

# Test LLM solution: compile with solution.h included (needs CUTLASS headers)
test: 88_hopper_fmha.cu solution.h fa3_reference.o hopper_cublaslt_reference.cuh
	$(NVCC) $(NVCC_FLAGS) $(ARCH) $(CUTLASS_INC) -DKH_TEST_SOLUTION $< fa3_reference.o -lcublasLt -o $@

clean:
	rm -f ref test fa3_reference.o

.PHONY: clean

fa3_reference.o: fa3_reference.cu
	$(NVCC) $(NVCC_FLAGS) $(ARCH) -I$(FA3_DIR) -I$(FA3_CUTLASS)/include -c $< -o $@
