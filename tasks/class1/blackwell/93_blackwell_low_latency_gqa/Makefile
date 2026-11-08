CUTLASS_DIR ?= ../../../../reference_sources/cutlass
CUTLASS_INC = -I$(CUTLASS_DIR)/include -I$(CUTLASS_DIR)/tools/util/include -I$(CUTLASS_DIR)/examples/common
# The CUTLASS example 93 (TGV GQA) kernel headers, vendored from CUTLASS main (v4.8.1).
GQA_INC     = -I$(CUTLASS_DIR)/examples/93_blackwell_low_latency_gqa
NVCC       ?= nvcc
NVCC_FLAGS  = -O3 --use_fast_math -std=c++17 --expt-relaxed-constexpr
ARCH        = -gencode arch=compute_100a,code=sm_100a
SRC         = 93_blackwell_low_latency_gqa.cu

# Reference only: compile the CUTLASS reference (no solution needed)
ref: $(SRC) kh_verify.h
	$(NVCC) $(NVCC_FLAGS) $(ARCH) $(CUTLASS_INC) $(GQA_INC) $< -o $@

# Test LLM solution: compile with solution.h included
test: $(SRC) solution.h kh_verify.h
	$(NVCC) $(NVCC_FLAGS) $(ARCH) $(CUTLASS_INC) $(GQA_INC) -DKH_TEST_SOLUTION $< -o $@

# Development aids: time every reference configuration on every case / dump data for offline validation
sweep: $(SRC) kh_verify.h
	$(NVCC) $(NVCC_FLAGS) $(ARCH) $(CUTLASS_INC) $(GQA_INC) -DKH_SWEEP $< -o $@

dump: $(SRC) kh_verify.h
	$(NVCC) $(NVCC_FLAGS) $(ARCH) $(CUTLASS_INC) $(GQA_INC) -DKH_DUMP $< -o $@

clean:
	rm -f ref test sweep dump

.PHONY: clean
