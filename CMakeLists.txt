cmake_minimum_required(VERSION 3.18)
project(CUDA_MatMul LANGUAGES CUDA CXX)

# Require CUDA
find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Optimization flags
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -lineinfo")
set(CMAKE_CUDA_ARCHITECTURES "61")  # Pascal (GTX 1050 Ti)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Kernel source files
set(KERNEL_SOURCES
    src/kernel_v1_naive.cu
    src/kernel_v2_tiled.cu
    src/kernel_v3_register.cu
    src/kernel_cublas.cu
)

# ============================================================================
# Benchmark executable
# ============================================================================
add_executable(matmul_bench
    src/benchmark.cu
    ${KERNEL_SOURCES}
)

target_link_libraries(matmul_bench PRIVATE CUDA::cublas)

# ============================================================================
# Test executable
# ============================================================================
add_executable(test_correctness
    tests/test_correctness.cu
    ${KERNEL_SOURCES}
)

target_link_libraries(test_correctness PRIVATE CUDA::cublas)

# ============================================================================
# Custom targets
# ============================================================================
add_custom_target(bench
    COMMAND matmul_bench > ${CMAKE_CURRENT_SOURCE_DIR}/results/benchmark_results.json
    DEPENDS matmul_bench
    COMMENT "Running benchmarks..."
)

add_custom_target(run_tests
    COMMAND test_correctness
    DEPENDS test_correctness
    COMMENT "Running correctness tests..."
)
