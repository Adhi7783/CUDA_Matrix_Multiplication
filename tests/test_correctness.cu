// ============================================================================
// Correctness Test Suite for CUDA Matrix Multiplication
// ============================================================================

#include "kernels.h"
#include "utils.h"
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <vector>
#ifdef _WIN32
#include <windows.h>
#endif

extern void init_cublas();
extern void cleanup_cublas();

// ============================================================================
// Test Framework
// ============================================================================

struct TestCase {
    int N;
    const char* name;
};

bool test_kernel(const TestCase& test, 
                void (*kernel_func)(const float*, const float*, float*, int),
                const char* kernel_name) {
    int N = test.N;
    size_t bytes = N * N * sizeof(float);
    
    // Allocate host memory
    float* h_A = (float*)malloc(bytes);
    float* h_B = (float*)malloc(bytes);
    float* h_C_cpu = (float*)malloc(bytes);
    float* h_C_gpu = (float*)malloc(bytes);
    
    // Initialize with deterministic values
    init_matrix(h_A, N, N, 123);
    init_matrix(h_B, N, N, 456);
    
    // CPU reference
    cpu_matmul(h_A, h_B, h_C_cpu, N, N, N);
    
    // Allocate device memory
    float* d_A, * d_B, * d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));
    
    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_C, 0, bytes));
    
    // Run kernel
    kernel_func(d_A, d_B, d_C, N);
    
    // Copy back
    CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, bytes, cudaMemcpyDeviceToHost));
    
    // Verify
    bool correct = verify_result(h_C_gpu, h_C_cpu, N, N, 1e-4f);
    
    if (correct) {
        printf("✓ PASS: %s - %s (N=%d)\n", test.name, kernel_name, N);
    } else {
        printf("✗ FAIL: %s - %s (N=%d)\n", test.name, kernel_name, N);
        // Print first few mismatches for debugging
        for (int i = 0; i < std::min(5, N * N); i++) {
            if (fabs(h_C_gpu[i] - h_C_cpu[i]) > 1e-4f * (fabs(h_C_cpu[i]) + 1e-6f)) {
                printf("  Mismatch at [%d]: GPU=%.6f, CPU=%.6f\n", i, h_C_gpu[i], h_C_cpu[i]);
            }
        }
    }
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C_cpu);
    free(h_C_gpu);
    
    return correct;
}

// ============================================================================
// Main Test Suite
// ============================================================================
int main() {
#ifdef _WIN32
    /* Ensure Windows console uses UTF-8 so Unicode symbols render correctly */
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif

    printf("CUDA Matrix Multiplication - Correctness Test Suite\n");
    printf("====================================================\n\n");
    
    init_cublas();
    
    std::vector<TestCase> test_cases = {
        {64, "Small Matrix"},
        {128, "Medium Matrix"},
        {256, "Large Matrix"},
    };
    
    int total_tests = 0;
    int passed_tests = 0;
    
    for (const auto& test : test_cases) {
        printf("\nRunning tests for %s (N=%d):\n", test.name, test.N);
        printf("─────────────────────────────────────\n");
        
        total_tests++;
        if (test_kernel(test, kernel_v1_naive, "Naive CUDA")) passed_tests++;
        
        total_tests++;
        if (test_kernel(test, kernel_v2_tiled_16, "Tiled (TILE=16)")) passed_tests++;
        
        total_tests++;
        if (test_kernel(test, kernel_v2_tiled_32, "Tiled (TILE=32)")) passed_tests++;
        
        total_tests++;
        if (test_kernel(test, kernel_v3_register_blocked, "Register-Blocked")) passed_tests++;
    }
    
    cleanup_cublas();
    
    printf("\n");
    printf("====================================================\n");
    printf("SUMMARY: %d/%d tests passed\n", passed_tests, total_tests);
    printf("====================================================\n");
    
    return (passed_tests == total_tests) ? 0 : 1;
}
