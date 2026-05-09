#pragma once

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <chrono>

// ============================================================================
// CUDA Error Checking Macro
// ============================================================================
#define CUDA_CHECK(err) \
  do { \
    cudaError_t err_code = (err); \
    if (err_code != cudaSuccess) { \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(err_code)); \
      exit(EXIT_FAILURE); \
    } \
  } while (0)

// ============================================================================
// GPU Device Timer (using CUDA events)
// ============================================================================
class GpuTimer {
private:
    cudaEvent_t start_, stop_;
    
public:
    GpuTimer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }
    
    ~GpuTimer() {
        CUDA_CHECK(cudaEventDestroy(start_));
        CUDA_CHECK(cudaEventDestroy(stop_));
    }
    
    void start() {
        CUDA_CHECK(cudaEventRecord(start_, 0));
    }
    
    void stop() {
        CUDA_CHECK(cudaEventRecord(stop_, 0));
        CUDA_CHECK(cudaEventSynchronize(stop_));
    }
    
    float elapsed_ms() const {
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }
};

// ============================================================================
// CPU Timer (for reference)
// ============================================================================
class CpuTimer {
private:
    std::chrono::high_resolution_clock::time_point start_;
    
public:
    void start() {
        start_ = std::chrono::high_resolution_clock::now();
    }
    
    float elapsed_ms() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<float, std::milli>(end - start_).count();
    }
};

// ============================================================================
// Matrix Utilities
// ============================================================================

// CPU matrix multiplication (reference implementation)
inline void cpu_matmul(const float* A, const float* B, float* C, 
                      int N, int K, int M) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * M + j];
            }
            C[i * M + j] = sum;
        }
    }
}

// Verify correctness of GPU result against CPU
inline bool verify_result(const float* C_gpu, const float* C_cpu, int N, int M, 
                         float eps = 1e-4f) {
    for (int i = 0; i < N * M; i++) {
        float diff = fabs(C_gpu[i] - C_cpu[i]);
        // Relative error tolerance
        float rel_err = diff / (fabs(C_cpu[i]) + 1e-6f);
        if (rel_err > eps) {
            fprintf(stderr, "Mismatch at index %d: GPU=%.6f, CPU=%.6f, rel_err=%.6f\n",
                    i, C_gpu[i], C_cpu[i], rel_err);
            return false;
        }
    }
    return true;
}

// Initialize matrix with random values
inline void init_matrix(float* mat, int rows, int cols, unsigned seed = 42) {
    srand(seed);
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = (float)rand() / RAND_MAX;
    }
}
