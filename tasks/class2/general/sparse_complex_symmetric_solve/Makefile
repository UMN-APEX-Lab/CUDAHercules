SPARSE_SOLVER_DIR ?= ../../../../reference_sources/sparse_solver
NVCC       ?= nvcc
NVCC_FLAGS  = -O3 -std=c++17 --expt-relaxed-constexpr
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

REF_DIR    := $(abspath $(SPARSE_SOLVER_DIR))
HOST_FLAGS  = -Xcompiler -fopenmp -lgomp -lz

# Reference only: times the reference solver, no solution
ref: reference solve_bench.cu
	$(NVCC) $(NVCC_FLAGS) $(ARCH) -DKH_REF_BIN_DIR=\"$(REF_DIR)/build/$(REF_ARCH)\" solve_bench.cu -o $@ $(HOST_FLAGS)

# Test LLM solution: harness linked with solution.cu
test: reference solve_bench.cu solution.cu
	$(NVCC) $(NVCC_FLAGS) $(ARCH) -DKH_REF_BIN_DIR=\"$(REF_DIR)/build/$(REF_ARCH)\" -DKH_TEST_SOLUTION solve_bench.cu solution.cu \
	    -o $@ $(HOST_FLAGS)

# The reference solver (two programs) is built once, next to its sources, for the same GPU target
# (its ARCH is sm_XY[a], taken from the code= field of ours).
REF_ARCH = $(patsubst code=%,%,$(filter code=sm_%,$(subst $(comma), ,$(ARCH))))
reference:
	@$(MAKE) --no-print-directory -C $(REF_DIR) ARCH=$(REF_ARCH)

clean:
	rm -f ref test

.PHONY: clean reference
