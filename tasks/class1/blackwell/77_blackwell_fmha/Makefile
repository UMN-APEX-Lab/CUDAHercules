# Reference: cuDNN SDPA through the cudnn-frontend graph API (ref_cudnn.cpp, built as plain C++ so
# that no cuDNN header reaches the solution translation unit).
#   CUDNN_FE_DIR : cudnn-frontend checkout / vendored copy, header-only (include/cudnn_frontend.h)
#   CUDNN_DIR    : cuDNN install with include/cudnn.h and lib/libcudnn.so.9 (default: the
#                  nvidia-cudnn-cu13 pip wheel of the active Python, else /usr/local/cuda)
# (the runner exports CUTLASS_DIR = <repo>/reference_sources/cutlass; cudnn_frontend is its sibling)
CUTLASS_DIR  ?= ../../../../reference_sources/cutlass
CUDNN_FE_DIR ?= $(CUTLASS_DIR)/../cudnn_frontend
CUDNN_DIR    ?= $(firstword $(shell python3 -c "import nvidia.cudnn as m; print(list(m.__path__)[0])" 2>/dev/null) \
                                 $(wildcard $(VIRTUAL_ENV)/lib/python*/site-packages/nvidia/cudnn) \
                                 /usr/local/cuda)
CUDA_HOME    ?= /usr/local/cuda
NVCC         ?= nvcc
CXX          ?= g++
NVCC_FLAGS    = -O3 --use_fast_math -std=c++17
ARCH          = -gencode arch=compute_100a,code=sm_100a
CUDNN_INC     = -I$(CUDNN_FE_DIR)/include -I$(CUDNN_DIR)/include -I$(CUDA_HOME)/include
CUDNN_LIBS    = -L$(CUDNN_DIR)/lib -l:libcudnn.so.9 -L$(CUDA_HOME)/lib64 -lnvrtc -lcuda -lcudart \
                -Xlinker -rpath=$(CUDNN_DIR)/lib -Xlinker -rpath=$(CUDA_HOME)/lib64
SRC           = 77_blackwell_fmha.cu

ref_cudnn.o: ref_cudnn.cpp ref_cudnn.h
	$(CXX) -O1 -std=c++17 -fPIC -c $(CUDNN_INC) $< -o $@

# Reference only (no solution needed)
ref: $(SRC) ref_cudnn.o kh_verify.h
	$(NVCC) $(NVCC_FLAGS) $(ARCH) $(SRC) ref_cudnn.o $(CUDNN_LIBS) -o $@

# Test LLM solution: compile with solution.h included
test: $(SRC) solution.h ref_cudnn.o kh_verify.h
	$(NVCC) $(NVCC_FLAGS) $(ARCH) -DKH_TEST_SOLUTION $(SRC) ref_cudnn.o $(CUDNN_LIBS) -o $@

# Development aid: dump data for offline validation
dump: $(SRC) ref_cudnn.o kh_verify.h
	$(NVCC) $(NVCC_FLAGS) $(ARCH) -DKH_DUMP $(SRC) ref_cudnn.o $(CUDNN_LIBS) -o $@

clean:
	rm -f ref test dump ref_cudnn.o

.PHONY: clean
