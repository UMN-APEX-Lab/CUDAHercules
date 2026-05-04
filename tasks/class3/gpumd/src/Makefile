NVCC ?= nvcc
GPU_ARCH := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')

# Allow CUDA_GENCODE env var to override architecture flags
ifdef CUDA_GENCODE
  NVCC_GENCODE = $(CUDA_GENCODE)
else
  NVCC_GENCODE = -gencode arch=compute_$(GPU_ARCH),code=sm_$(GPU_ARCH)
endif

NVCC_FLAGS = -O3 -std=c++17 $(NVCC_GENCODE)
all: md_bench
md_bench: main.cu md_kernels.cuh
	$(NVCC) $(NVCC_FLAGS) main.cu -o md_bench
clean:
	rm -f md_bench
