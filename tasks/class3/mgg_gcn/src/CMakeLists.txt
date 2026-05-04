cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(MGG_GCN LANGUAGES CXX CUDA)

# CUDA / NVSHMEM / MPI paths
set(CUDA_HOME "/usr/local/cuda")
set(NVSHMEM_INCLUDE "/usr/include/nvshmem_13")
set(MPI_HOME "/usr/lib/x86_64-linux-gnu/openmpi")

# Allow CUDA_GENCODE env var to override architecture flags
if(DEFINED ENV{CUDA_GENCODE})
    set(CMAKE_CUDA_ARCHITECTURES OFF)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} $ENV{CUDA_GENCODE} -O3 --use_fast_math -std=c++17 -rdc=true --expt-relaxed-constexpr --expt-extended-lambda -Xcompiler=-Wno-psabi")
else()
    # Architecture: SM120 for RTX PRO 6000 Blackwell
    # NVSHMEM 3.6.5 device lib includes sm_120 cubin
    set(CMAKE_CUDA_ARCHITECTURES "120")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math -std=c++17 -rdc=true --expt-relaxed-constexpr --expt-extended-lambda -Xcompiler=-Wno-psabi")
endif()

# Include paths
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${NVSHMEM_INCLUDE})
include_directories(${MPI_HOME}/include)

# Link paths
link_directories(/usr/lib/x86_64-linux-gnu)
link_directories(${MPI_HOME}/lib)

# NVSHMEM device library (static, must be linked with -rdc=true)
set(NVSHMEM_DEVICE_LIB "/usr/lib/x86_64-linux-gnu/libnvshmem_device.a")

# Main executable
add_executable(mgg_gcn
    main.cu
    include/loss.cu
    include/layer.cu
)

target_link_libraries(mgg_gcn
    ${NVSHMEM_DEVICE_LIB}
    nvshmem_host
    cuda
    cublas
    cublasLt
    cudnn
    mpi
    nvidia-ml
)

set_target_properties(mgg_gcn PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
