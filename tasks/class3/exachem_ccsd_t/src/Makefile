NVCC ?= nvcc

# Detect GPU architecture
GPU_ARCH := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
ifeq ($(GPU_ARCH),)
  GPU_ARCH := 80
endif

# Allow CUDA_GENCODE env var to override architecture flags
ifdef CUDA_GENCODE
  ARCH_FLAGS = $(CUDA_GENCODE)
else
  # Architecture flags — compile native SASS + PTX fallback
  # Native SASS avoids JIT overhead; PTX provides forward compatibility
  ifeq ($(shell test $(GPU_ARCH) -ge 120; echo $$?),0)
    ARCH_FLAGS = -gencode arch=compute_$(GPU_ARCH),code=sm_$(GPU_ARCH) \
                 -gencode arch=compute_80,code=compute_80
  else ifeq ($(shell test $(GPU_ARCH) -ge 90; echo $$?),0)
    ARCH_FLAGS = -gencode arch=compute_$(GPU_ARCH),code=sm_$(GPU_ARCH) \
                 -gencode arch=compute_80,code=compute_80
  else
    ARCH_FLAGS = -gencode arch=compute_80,code=sm_80
  endif
endif

NVCC_FLAGS = -O3 --use_fast_math $(ARCH_FLAGS) -DUSE_NV_TC \
             --expt-relaxed-constexpr -rdc=true -std=c++17

TASK_DIR = $(shell cd .. && pwd)

.PHONY: ref test clean

ref: main.cu ccsd_t_all_fused_gpu.cu ccsd_t_g2s_device_functions.cu tensor_core_helper.cuh \
     standalone_common.cuh kernel_interface.cuh
	$(NVCC) $(NVCC_FLAGS) -I. main.cu ccsd_t_all_fused_gpu.cu -o ../ref_benchmark -lcudadevrt

test: main.cu $(TASK_DIR)/solution.cu ccsd_t_g2s_device_functions.cu tensor_core_helper.cuh \
      standalone_common.cuh kernel_interface.cuh
	$(NVCC) $(NVCC_FLAGS) -I. -I$(TASK_DIR) main.cu $(TASK_DIR)/solution.cu -o ../sol_benchmark -lcudadevrt

clean:
	rm -f ../ref_benchmark ../sol_benchmark *.o
