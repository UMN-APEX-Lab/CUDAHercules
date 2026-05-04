cmake_minimum_required(VERSION 3.18)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
project(icicle_zk_bench)

find_package(CUDAToolkit REQUIRED)

set(ICICLE_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../reference_sources/icicle")
set(ICICLE_BUILD_DIR "${ICICLE_SRC_DIR}/build")

add_executable(bench bench.cpp)
target_include_directories(bench PRIVATE "${ICICLE_SRC_DIR}/include")
target_link_directories(bench PRIVATE "${ICICLE_BUILD_DIR}")
target_link_libraries(
  bench PRIVATE
  icicle_curve_bn254
  icicle_field_bn254
  icicle_device
  CUDA::cudart
  dl
)
set_target_properties(bench PROPERTIES
  BUILD_RPATH "${ICICLE_BUILD_DIR}"
  INSTALL_RPATH "${ICICLE_BUILD_DIR}"
)
