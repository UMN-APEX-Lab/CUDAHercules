CUTLASS_DIR ?= ../../../../reference_sources/cutlass
CUTLASS_INC = -I$(CUTLASS_DIR)/include -I$(CUTLASS_DIR)/tools/util/include -I$(CUTLASS_DIR)/examples/common
NVCC       ?= nvcc
NVCC_FLAGS  = -O3 --use_fast_math -std=c++17 --expt-relaxed-constexpr
ARCH        = -gencode arch=compute_100a,code=sm_100a

# Reference only: compile the CUTLASS conv harness (no solution needed)
ref: 76_blackwell_conv.cu kh_verify.h
	$(NVCC) $(NVCC_FLAGS) $(ARCH) $(CUTLASS_INC) $< -o $@

# Test LLM solution: compile with solution.h included (needs CUTLASS headers)
test: 76_blackwell_conv.cu solution.h kh_verify.h
	$(NVCC) $(NVCC_FLAGS) $(ARCH) $(CUTLASS_INC) -DKH_TEST_SOLUTION $< -o $@

# Development aid: time every reference configuration on every layer
sweep: 76_blackwell_conv.cu kh_verify.h
	$(NVCC) $(NVCC_FLAGS) $(ARCH) $(CUTLASS_INC) -DKH_SWEEP $< -o $@

clean:
	rm -f ref test sweep

.PHONY: clean
