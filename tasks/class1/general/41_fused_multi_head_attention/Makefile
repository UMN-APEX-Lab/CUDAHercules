CUTLASS_DIR ?= ../../../../reference_sources/cutlass
# FlashAttention-2 forward kernels (reference only; never on the solution's include path)
FA2_DIR     ?= $(CUTLASS_DIR)/../flash_attn
CUTLASS_INC = -I$(CUTLASS_DIR)/include -I$(CUTLASS_DIR)/tools/util/include -I$(CUTLASS_DIR)/examples/common
NVCC       ?= nvcc
NVCC_FLAGS  = -O3 --use_fast_math -std=c++17 --expt-relaxed-constexpr
ARCH        = -gencode arch=compute_80,code=compute_80
FA2_FLAGS   = --expt-extended-lambda -I$(FA2_DIR) -Ifa2_compat -I$(CUTLASS_DIR)/include \
              -DFLASHATTENTION_DISABLE_DROPOUT -DFLASHATTENTION_DISABLE_ALIBI \
              -DFLASHATTENTION_DISABLE_SOFTCAP -DFLASHATTENTION_DISABLE_LOCAL \
              -DFLASHATTENTION_DISABLE_UNEVEN_K

# Reference FMHA (FlashAttention-2), separate translation unit
fa2_ref.o: fa2_ref.cu fa2_ref.h
	$(NVCC) $(NVCC_FLAGS) $(ARCH) $(FA2_FLAGS) -c $< -o $@

# Reference only (no solution needed)
ref: fused_multihead_attention_fixed_seqlen.cu fa2_ref.h fa2_ref.o
	$(NVCC) $(NVCC_FLAGS) $(ARCH) $(CUTLASS_INC) $< fa2_ref.o -o $@

# Test LLM solution: compile with solution.h included (needs CUTLASS headers)
test: fused_multihead_attention_fixed_seqlen.cu fa2_ref.h solution.h fa2_ref.o
	$(NVCC) $(NVCC_FLAGS) $(ARCH) $(CUTLASS_INC) -DKH_TEST_SOLUTION $< fa2_ref.o -o $@

clean:
	rm -f ref test fa2_ref.o

.PHONY: clean
