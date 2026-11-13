SPARSE_KERNELS_DIR ?= ../../../../reference_sources/sparse_kernels
NVCC       ?= nvcc
NVCC_FLAGS  = -O3 --use_fast_math -std=c++17 --expt-relaxed-constexpr -cudart shared
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

OP       = spmm
MATRICES = af_shell10 CoupCons3D delaunay_n24 asia_osm human_gene1 mouse_gene gupta3 in-2004 \
           ogbn-proteins reddit case39 TSOPF_FS_b300_c2
REF_SRC   := $(abspath $(SPARSE_KERNELS_DIR))
# Reference libraries are cached per target architecture next to their sources.
REF_BUILD := $(REF_SRC)/build/$(shell printf '%s' '$(ARCH)' | md5sum | cut -c1-12)
REF_LIBS   = $(foreach m,$(MATRICES),$(foreach k,8 32 128 256,$(REF_BUILD)/$(OP)_$(m)_k$(k).so))
REF_FLAGS  = $(NVCC_FLAGS) $(ARCH) -I$(REF_SRC) -Xcompiler -fPIC,-fvisibility=hidden

# Reference only: per-matrix reference kernels, no solution
ref: refs spmm.cu kh_sparse.h
	$(NVCC) $(NVCC_FLAGS) $(ARCH) -DKH_REF_DIR=\"$(REF_BUILD)\" spmm.cu -o $@ -ldl

# Test LLM solution: harness with solution.h included
test: refs spmm.cu kh_sparse.h solution.h
	$(NVCC) $(NVCC_FLAGS) $(ARCH) -DKH_REF_DIR=\"$(REF_BUILD)\" -DKH_TEST_SOLUTION spmm.cu -o $@ -ldl

refs:
	@$(MAKE) --no-print-directory -j16 ref_libs

ref_libs: $(REF_LIBS)

$(REF_BUILD)/sparse_formats.o: $(REF_SRC)/sparse_formats.cu $(REF_SRC)/sparse_formats.cuh
	@mkdir -p $(REF_BUILD)
	$(NVCC) $(REF_FLAGS) -c $< -o $@.$$$$.tmp && mv -f $@.$$$$.tmp $@

$(REF_BUILD)/$(OP)_%.so: $(REF_SRC)/$(OP)/%.cu $(REF_SRC)/export_$(OP).cu $(REF_BUILD)/sparse_formats.o
	$(NVCC) $(REF_FLAGS) -shared $^ -o $@.$$$$.tmp && mv -f $@.$$$$.tmp $@

clean:
	rm -f ref test

.PHONY: clean refs ref_libs
