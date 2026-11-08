CUTLASS_DIR  ?= ../../../../reference_sources/cutlass
DEEPGEMM_DIR ?= ../../../../reference_sources/deep_gemm
CUTLASS_INC  = -I$(CUTLASS_DIR)/include -I$(CUTLASS_DIR)/tools/util/include
# DeepGEMM kernel headers (upstream deepseek-ai/DeepGEMM, deep_gemm/include), vendored under reference_sources.
DEEPGEMM_INC = -I$(DEEPGEMM_DIR)/deep_gemm/include
NVCC         ?= nvcc
NVCC_FLAGS   = -O3 --use_fast_math -std=c++20 --expt-relaxed-constexpr -Xptxas --register-usage-level=10
ARCH         = -gencode arch=compute_100a,code=sm_100a
SRC          = deepgemm_mqa_logits.cu

# Reference only: compile the DeepGEMM reference (no solution needed)
ref: $(SRC) kh_verify.h
	$(NVCC) $(NVCC_FLAGS) $(ARCH) $(CUTLASS_INC) $(DEEPGEMM_INC) $< -lcuda -o $@

# Test LLM solution: compile with solution.h included
test: $(SRC) solution.h kh_verify.h
	$(NVCC) $(NVCC_FLAGS) $(ARCH) $(CUTLASS_INC) $(DEEPGEMM_INC) -DKH_TEST_SOLUTION $< -lcuda -o $@

# Development aids: time both reference schedulers / dump data for offline validation
sweep: $(SRC) kh_verify.h
	$(NVCC) $(NVCC_FLAGS) $(ARCH) $(CUTLASS_INC) $(DEEPGEMM_INC) -DKH_SWEEP $< -lcuda -o $@

dump: $(SRC) kh_verify.h
	$(NVCC) $(NVCC_FLAGS) $(ARCH) $(CUTLASS_INC) $(DEEPGEMM_INC) -DKH_DUMP $< -lcuda -o $@

clean:
	rm -f ref test sweep dump

.PHONY: clean
