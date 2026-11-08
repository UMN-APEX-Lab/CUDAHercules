TK_DIR  ?= ../../../../reference_sources/ThunderKittens
TK_INC   = -I$(TK_DIR)/include -I$(TK_DIR)/prototype
NVCC     ?= nvcc
NVCC_FLAGS = -O3 --use_fast_math -std=c++20 --expt-extended-lambda --expt-relaxed-constexpr -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing -DNDEBUG
ARCH     = -gencode arch=compute_90a,code=sm_90a
LIBS     = -lcuda -lcudadevrt -lcudart_static -lrt -lpthread -ldl

ref: fp8_h100_gemm.cu hopper_cublaslt_reference.cuh
	$(NVCC) $(NVCC_FLAGS) $(ARCH) $(TK_INC) $(LIBS) -DKITTENS_HOPPER $< -o $@ -lcublas -lcublasLt

test: fp8_h100_gemm.cu solution.h hopper_cublaslt_reference.cuh
	$(NVCC) $(NVCC_FLAGS) $(ARCH) $(TK_INC) $(LIBS) -DKITTENS_HOPPER -DKH_TEST_SOLUTION $< -o $@ -lcublas -lcublasLt

clean:
	rm -f ref test

.PHONY: clean
