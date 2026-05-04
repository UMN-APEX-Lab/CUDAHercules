cmake_minimum_required(VERSION 3.17)
project(Liberator CUDA)

set(CMAKE_CUDA_STANDARD 17)
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")
add_executable(Liberator main.cu gpu_kernels.cu gpu_kernels.cuh range.cuh bfs-sync.cuh tools.cuh common.cuh common.cu GraphMeta.cu GraphMeta.cuh NewCalculateOpt.cu CalculateOpt.cu CalculateOpt.cuh constants.cuh TimeRecord.cu TimeRecord.cuh ArgumentParser.cu ArgumentParser.cuh)

# Allow CUDA_GENCODE env var to override architecture flags
if(DEFINED ENV{CUDA_GENCODE})
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} $ENV{CUDA_GENCODE}")
    set_target_properties(
            Liberator
            PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_ARCHITECTURES OFF)
else()
    set_target_properties(
            Liberator
            PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_ARCHITECTURES "120")
endif()

target_link_libraries(Liberator -lcublas -lcurand)
