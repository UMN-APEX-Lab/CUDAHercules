NVCC ?= nvcc

# GPU architecture: use CUDA_GENCODE env var if set, otherwise auto-detect
ifndef CUDA_GENCODE
GPU_COMPUTE_CAPABILITY := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
NVCC_GENCODE := -gencode arch=compute_$(GPU_COMPUTE_CAPABILITY),code=[compute_$(GPU_COMPUTE_CAPABILITY),sm_$(GPU_COMPUTE_CAPABILITY)]
else
NVCC_GENCODE := $(CUDA_GENCODE)
endif

NVCC_FLAGS = --threads=0 --use_fast_math -std=c++17 -O3 -DENABLE_BF16 $(NVCC_GENCODE)
NVCC_LIBS = -lcublas -lcublasLt -lnvidia-ml

# Check for MPI
ifneq ($(shell dpkg -l libopenmpi-dev 2>/dev/null | grep -c "^ii"),0)
    MPI_INCLUDE = -I/usr/lib/x86_64-linux-gnu/openmpi/include/
    MPI_LINK = -L/usr/lib/x86_64-linux-gnu/openmpi/lib/ -lmpi
    MPI_FLAG = -DUSE_MPI
endif

all: train_gpt2cu

train_gpt2cu: train_gpt2.cu llmc/*.cuh llmc/*.h
	$(NVCC) $(NVCC_FLAGS) $(MPI_FLAG) train_gpt2.cu $(NVCC_LIBS) $(MPI_INCLUDE) $(MPI_LINK) -o $@

clean:
	rm -f train_gpt2cu
