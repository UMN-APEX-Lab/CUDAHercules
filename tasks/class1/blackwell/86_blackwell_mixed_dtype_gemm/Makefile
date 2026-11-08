CUTLASS_DIR ?= ../../../../reference_sources/cutlass
CUTLASS_INC = -I$(CUTLASS_DIR)/include -I$(CUTLASS_DIR)/tools/util/include -I$(CUTLASS_DIR)/examples/common
MARLIN_INC  = -Imarlin -Imarlin/stub
NVCC       ?= nvcc
NVCC_FLAGS  = -O3 --use_fast_math -std=c++17 --expt-relaxed-constexpr
ARCH        = -gencode arch=compute_100a,code=sm_100a

# The vLLM Marlin reference lives in its own translation unit (it compiles in parallel with the CUTLASS one
# under `make -j`, and it must not see solution.h).
marlin_ref.o: marlin_ref.cu marlin_ref.h marlin_repack_kernel.inc marlin/marlin_template.h
	$(NVCC) $(NVCC_FLAGS) $(ARCH) $(MARLIN_INC) -c $< -o $@

# Reference only: compile the references (no solution needed)
ref: 86_blackwell_mixed_dtype.cu marlin_ref.o kh_verify.h marlin_ref.h
	$(NVCC) $(NVCC_FLAGS) $(ARCH) $(CUTLASS_INC) $< marlin_ref.o -o $@

# Test LLM solution: compile with solution.h included (needs CUTLASS headers)
test: 86_blackwell_mixed_dtype.cu marlin_ref.o solution.h kh_verify.h marlin_ref.h
	$(NVCC) $(NVCC_FLAGS) $(ARCH) $(CUTLASS_INC) -DKH_TEST_SOLUTION $< marlin_ref.o -o $@

# Development aid: time every reference configuration on every shape
sweep: 86_blackwell_mixed_dtype.cu marlin_ref.o kh_verify.h marlin_ref.h
	$(NVCC) $(NVCC_FLAGS) $(ARCH) $(CUTLASS_INC) -DKH_SWEEP $< marlin_ref.o -o $@

clean:
	rm -f ref test sweep marlin_ref.o

.PHONY: clean
