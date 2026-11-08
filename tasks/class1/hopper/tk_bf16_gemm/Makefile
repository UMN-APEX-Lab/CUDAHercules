TK_DIR  ?= ../../../../reference_sources/ThunderKittens
TK_INC   = -I$(TK_DIR)/include -I$(TK_DIR)/prototype
NVCC     ?= nvcc
NVCC_FLAGS = -O3 --use_fast_math -std=c++20 --expt-extended-lambda --expt-relaxed-constexpr -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing -DNDEBUG
ARCH     = -gencode arch=compute_90a,code=sm_90a
LIBS     = -lcuda -lcudadevrt -lcudart_static -lrt -lpthread -ldl

ref: bf16_h100_gemm.cu bf16_reference.o hopper_cublaslt_reference.cuh
	$(NVCC) $(NVCC_FLAGS) $(ARCH) $(TK_INC) $(LIBS) -DKITTENS_HOPPER $< bf16_reference.o -o $@ -lcublas -lcublasLt

test: bf16_h100_gemm.cu solution.h bf16_reference.o hopper_cublaslt_reference.cuh
	$(NVCC) $(NVCC_FLAGS) $(ARCH) $(TK_INC) $(LIBS) -DKITTENS_HOPPER -DKH_TEST_SOLUTION $< bf16_reference.o -o $@ -lcublas -lcublasLt

clean:
	rm -f ref test bf16_reference.o

.PHONY: clean

bf16_reference.o: bf16_reference.cu hopper_cublaslt_reference.cuh hopper_validation.cuh
	$(NVCC) $(NVCC_FLAGS) $(ARCH) -c $< -o $@
