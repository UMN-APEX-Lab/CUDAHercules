TK_DIR  ?= ../../../../reference_sources/ThunderKittens
TK_INC   = -I$(TK_DIR)/include -I$(TK_DIR)/prototype
NVCC     ?= nvcc
NVCC_FLAGS = -O3 --use_fast_math -std=c++20 --expt-extended-lambda --expt-relaxed-constexpr -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing -DNDEBUG
ARCH     = -gencode arch=compute_100a,code=sm_100a
LIBS     = -lcuda -lcudadevrt -lcudart_static -lrt -lpthread -ldl

ref: hedgehog.cu harness.impl
	$(NVCC) $(NVCC_FLAGS) $(ARCH) $(TK_INC) $(LIBS) -DKITTENS_BLACKWELL $< -o $@

test: hedgehog.cu harness.impl solution.h
	$(NVCC) $(NVCC_FLAGS) $(ARCH) $(TK_INC) $(LIBS) -DKITTENS_BLACKWELL -DKH_TEST_SOLUTION $< -o $@

clean:
	rm -f ref test

.PHONY: clean
