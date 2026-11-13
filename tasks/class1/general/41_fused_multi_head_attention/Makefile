CUTLASS_DIR ?= ../../../../reference_sources/cutlass
# FlashAttention-2 forward kernels (reference only; never on the solution's include path)
FA2_DIR     ?= $(CUTLASS_DIR)/../flash_attn
CUTLASS_INC = -I$(CUTLASS_DIR)/include -I$(CUTLASS_DIR)/tools/util/include -I$(CUTLASS_DIR)/examples/common
NVCC       ?= nvcc
NVCC_FLAGS  = -O3 --use_fast_math -std=c++17 --expt-relaxed-constexpr
# general task: build for the GPU of the machine running the build (sm_XY, or the arch-specific
# sm_XYa where one exists, e.g. sm_90a on H100/H200, sm_120a on RTX PRO 6000). The evaluator
# passes ARCH explicitly; this default only applies to standalone builds.
comma := ,
ifeq ($(origin ARCH),undefined)
GPU_SM := $(shell nvidia-smi -i $(or $(firstword $(subst $(comma), ,$(CUDA_VISIBLE_DEVICES))),0) \
            --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d ' .')
GPU_SM_SUFFIX := $(if $(filter 90 100 101 103 110 120 121,$(GPU_SM)),a)
ARCH := $(if $(GPU_SM),-gencode arch=compute_$(GPU_SM)$(GPU_SM_SUFFIX)$(comma)code=sm_$(GPU_SM)$(GPU_SM_SUFFIX),\
          $(error cannot detect the GPU architecture; pass ARCH="-gencode arch=compute_XY,code=sm_XY"))
endif
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
