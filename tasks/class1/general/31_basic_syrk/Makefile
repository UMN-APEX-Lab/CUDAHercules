CUTLASS_DIR ?= ../../../../reference_sources/cutlass
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

ref: basic_syrk.cu
	$(NVCC) $(NVCC_FLAGS) $(ARCH) $(CUTLASS_INC) $< -o $@

test: basic_syrk.cu solution.h
	$(NVCC) $(NVCC_FLAGS) $(ARCH) $(CUTLASS_INC) -DKH_TEST_SOLUTION $< -o $@

clean:
	rm -f ref test

.PHONY: clean
