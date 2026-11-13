# Builds ref_order (GPU nested-dissection ordering) and ref_solve (supernodal multifrontal LDL^T + FGMRES).
#
# SOLVER_ENV  conda env (or prefix) providing METIS, SuiteSparse (CHOLMOD) and MKL: include/ and lib/
#             (see environment.yml)
# CUDA  CUDA toolkit root
# ARCH  GPU target, sm_XY (sm_XYa for compute capability 90/100/101/103/110/120/121); defaults to the GPU
#       of the build machine (nvidia-smi), e.g.  make ARCH=sm_90a; outputs go to build/$(ARCH)/
SOLVER_ENV ?= $(firstword $(wildcard $(HOME)/.conda/envs/ref-solver $(HOME)/miniconda3/envs/ref-solver \
                                     $(HOME)/anaconda3/envs/ref-solver))
ifeq ($(strip $(SOLVER_ENV)),)
$(error no ref-solver conda env found: conda env create -f environment.yml, or pass SOLVER_ENV=<prefix>)
endif
CUDA ?= /usr/local/cuda
CXX  ?= g++
NVCC ?= $(CUDA)/bin/nvcc
CXXFLAGS ?= -O2 -std=c++17

ifndef ARCH
CC_RAW := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1 | tr -d '. \r')
ifeq ($(CC_RAW),)
$(error cannot detect the GPU architecture, pass ARCH=sm_XY)
endif
ifneq ($(filter 90 100 101 103 110 120 121,$(CC_RAW)),)
ARCH := sm_$(CC_RAW)a
else
ARCH := sm_$(CC_RAW)
endif
endif

INCLUDES = -I. -I$(SOLVER_ENV)/include
RPATH    = -Wl,-rpath,$(SOLVER_ENV)/lib
COMMON   = bench_common.h report.h COOBinaryIO.h
# One build directory per GPU target; outputs are written to a temporary name and renamed, so
# concurrent builds never see a half-written file.
OUT      = build/$(ARCH)
TMP      = $@.$$$$.tmp

all: $(OUT)/ref_order $(OUT)/ref_solve

$(OUT)/bench_kernels.o: bench_kernels.cu bench_kernels.h
	@mkdir -p $(OUT)
	$(NVCC) -O2 -std=c++17 -arch=$(ARCH) -c $< -o $(TMP) && mv -f $(TMP) $@

$(OUT)/gpu_nd.o: gpu_nd.cu gpu_nd.h
	@mkdir -p $(OUT)
	$(NVCC) -O2 -std=c++17 -arch=$(ARCH) --default-stream per-thread -I$(SOLVER_ENV)/include -c $< -o $(TMP) && mv -f $(TMP) $@

$(OUT)/ref_order: ref_order.cpp COOBinaryIO.cpp $(COMMON) gpu_nd.h $(OUT)/gpu_nd.o
	$(CXX) $(CXXFLAGS) -fopenmp $(INCLUDES) $(filter %.cpp %.o,$^) -o $(TMP) -L$(SOLVER_ENV)/lib -L$(CUDA)/lib64 $(RPATH) \
	    -Wl,-rpath,$(CUDA)/lib64 -lcholmod -lmetis -lsuitesparseconfig -lz -lcudart && mv -f $(TMP) $@

$(OUT)/gpu_symbolic.o: gpu_symbolic.cu gpu_symbolic.h
	@mkdir -p $(OUT)
	$(NVCC) -O2 -std=c++17 -arch=$(ARCH) -Xcompiler -fopenmp -c $< -o $(TMP) && mv -f $(TMP) $@

$(OUT)/ref_solve: ref_solve.cu front_kernels.cuh front_solve.cuh gpu_symbolic.h $(OUT)/gpu_symbolic.o COOBinaryIO.cpp \
                  $(COMMON) dd_refine.h mixed_solve.h $(OUT)/bench_kernels.o
	$(NVCC) -O2 -std=c++17 -arch=$(ARCH) -Xcompiler -fopenmp $(INCLUDES) $(filter %.cu %.cpp %.o,$^) -o $(TMP) \
	    -L$(SOLVER_ENV)/lib -Xlinker -rpath,$(SOLVER_ENV)/lib -lcholmod -lsuitesparseconfig -lcublas -lz -lgomp \
	    -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl && mv -f $(TMP) $@

clean:
	rm -rf build

.PHONY: all clean
