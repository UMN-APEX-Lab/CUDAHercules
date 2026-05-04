cmake_minimum_required(VERSION 3.17)
project(cuSZp_bench CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

add_executable(cuSZp_bench
    main.cu
    cuSZp_kernels_1D_f32.cu
    cuSZp_kernels_1D_f64.cu
    cuSZp_kernels_2D_f32.cu
    cuSZp_kernels_2D_f64.cu
    cuSZp_kernels_3D_f32.cu
    cuSZp_kernels_3D_f64.cu
    cuSZp_entry_1D_f32.cu
    cuSZp_entry_1D_f64.cu
    cuSZp_entry_2D_f32.cu
    cuSZp_entry_2D_f64.cu
    cuSZp_entry_3D_f32.cu
    cuSZp_entry_3D_f64.cu
    cuSZp_timer.cu
)

target_include_directories(cuSZp_bench PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/cuSZp
)

# Allow CUDA_GENCODE env var to override architecture flags
if(DEFINED ENV{CUDA_GENCODE})
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} $ENV{CUDA_GENCODE}")
    set_target_properties(cuSZp_bench PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES OFF
    )
else()
    set_target_properties(cuSZp_bench PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "120"
    )
endif()
