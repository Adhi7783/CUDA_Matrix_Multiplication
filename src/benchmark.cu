// ============================================================================
// Benchmarking Harness for CUDA Matrix Multiplication
// ============================================================================
// Tests all implementations across multiple matrix sizes.
// Outputs JSON results for plotting and analysis.

#include "kernels.h"
#include "utils.h"
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cmath>
#include <algorithm>

// Forward declarations
extern void init_cublas();
extern void cleanup_cublas();

// ============================================================================
// Benchmark Structure
// ============================================================================
struct BenchmarkResult {
    int N;
    float cpu_ms;
    float naive_ms;
    float tiled_16_ms;
    float tiled_32_ms;
    float register_blocked_ms;
    float cublas_ms;
};

// Calculate GFLOP/s: 2*N^3 / (time_ms * 1e-3 * 1e9)
float calculate_gflops(int N, float time_ms) {
    if (time_ms <= 0.0f) return 0.0f;
    return (2.0f * N * N * N) / (time_ms * 1e6f);
}

// Compute median of a sorted array
float compute_median(float* values, int count) {
    if (count == 0) return 0.0f;
    if (count == 1) return values[0];
    // Assumes values is already sorted; if not, sort first
    std::sort(values, values + count);
    return (count % 2 == 0) ? (values[count/2 - 1] + values[count/2]) * 0.5f : values[count/2];
}

// ============================================================================
// Benchmark a single implementation
// Warm-up iterations (discarded) + timed iterations (median taken)
// ============================================================================
float benchmark_kernel(const float* d_A, const float* d_B, float* d_C, 
                      int N, void (*kernel_func)(const float*, const float*, float*, int),
                      int num_warmup = 3, int num_runs = 10) {
    // Warm-up runs (discarded)
    for (int i = 0; i < num_warmup; i++) {
        kernel_func(d_A, d_B, d_C, N);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Timed runs
    float* times = new float[num_runs];
    for (int run = 0; run < num_runs; run++) {
        GpuTimer timer;
        timer.start();
        
        kernel_func(d_A, d_B, d_C, N);
        
        timer.stop();
        times[run] = timer.elapsed_ms();
    }
    
    float median_ms = compute_median(times, num_runs);
    delete[] times;
    
    return median_ms;
}

// Benchmark cuBLAS with explicit CUDA events (more accurate than GpuTimer at small scales)
float benchmark_cublas_robust(const float* d_A, const float* d_B, float* d_C,
                              int N, int num_warmup = 3, int num_runs = 10) {
    // Create CUDA events for precise timing
    cudaEvent_t start_evt, stop_evt;
    CUDA_CHECK(cudaEventCreate(&start_evt));
    CUDA_CHECK(cudaEventCreate(&stop_evt));
    
    // Warm-up runs
    for (int i = 0; i < num_warmup; i++) {
        cublas_sgemm(d_A, d_B, d_C, N);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Timed runs with CUDA events
    float* times = new float[num_runs];
    for (int run = 0; run < num_runs; run++) {
        CUDA_CHECK(cudaEventRecord(start_evt, 0));
        
        cublas_sgemm(d_A, d_B, d_C, N);
        
        CUDA_CHECK(cudaEventRecord(stop_evt, 0));
        CUDA_CHECK(cudaEventSynchronize(stop_evt));
        
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_evt, stop_evt));
        times[run] = ms;
    }
    
    float median_ms = compute_median(times, num_runs);
    delete[] times;
    
    CUDA_CHECK(cudaEventDestroy(start_evt));
    CUDA_CHECK(cudaEventDestroy(stop_evt));
    
    return median_ms;
}

// ============================================================================
// Main benchmarking function
// ============================================================================
void run_benchmark(const std::vector<int>& sizes, bool json_only = false) {
    init_cublas();
    
    printf("[\n");  // Start JSON array
    bool first = true;
    
    for (int N : sizes) {
        // Allocate host memory
        size_t bytes = N * N * sizeof(float);
        float* h_A = (float*)malloc(bytes);
        float* h_B = (float*)malloc(bytes);
        float* h_C_cpu = (float*)malloc(bytes);
        float* h_C_gpu = (float*)malloc(bytes);
        
        init_matrix(h_A, N, N);
        init_matrix(h_B, N, N);
        
        // Allocate device memory
        float* d_A, * d_B, * d_C;
        CUDA_CHECK(cudaMalloc(&d_A, bytes));
        CUDA_CHECK(cudaMalloc(&d_B, bytes));
        CUDA_CHECK(cudaMalloc(&d_C, bytes));
        
        // Copy to device
        CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));
        
        // CPU reference (only for small N to save time)
        float cpu_ms = 0.0f;
        if (N <= 256) {
            CpuTimer cpu_timer;
            cpu_timer.start();
            cpu_matmul(h_A, h_B, h_C_cpu, N, N, N);
            cpu_ms = cpu_timer.elapsed_ms();
        } else {
            cpu_ms = -1.0f;  // Not measured
        }
        
        // Benchmark each kernel with warm-up + median timing
        float naive_ms = benchmark_kernel(d_A, d_B, d_C, N, kernel_v1_naive, 3, 10);
        
        CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, bytes, cudaMemcpyDeviceToHost));
        bool naive_correct = (N <= 256) ? verify_result(h_C_gpu, h_C_cpu, N, N) : true;
        
        float tiled_16_ms = benchmark_kernel(d_A, d_B, d_C, N, kernel_v2_tiled_16, 3, 10);
        CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, bytes, cudaMemcpyDeviceToHost));
        bool tiled_16_correct = (N <= 256) ? verify_result(h_C_gpu, h_C_cpu, N, N) : true;
        
        float tiled_32_ms = benchmark_kernel(d_A, d_B, d_C, N, kernel_v2_tiled_32, 3, 10);
        CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, bytes, cudaMemcpyDeviceToHost));
        bool tiled_32_correct = (N <= 256) ? verify_result(h_C_gpu, h_C_cpu, N, N) : true;
        
        float register_blocked_ms = benchmark_kernel(d_A, d_B, d_C, N, kernel_v3_register_blocked, 3, 10);
        CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, bytes, cudaMemcpyDeviceToHost));
        bool reg_blocked_correct = (N <= 256) ? verify_result(h_C_gpu, h_C_cpu, N, N) : true;
        
        // cuBLAS with robust CUDA event-based timing
        float cublas_ms = benchmark_cublas_robust(d_A, d_B, d_C, N, 3, 10);
        CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, bytes, cudaMemcpyDeviceToHost));
        bool cublas_correct = (N <= 256) ? verify_result(h_C_gpu, h_C_cpu, N, N) : true;
        
        // Output JSON
        if (!first) printf(",\n");
        first = false;
        
        printf("  {\n");
        printf("    \"N\": %d,\n", N);
        printf("    \"CPU\": { \"ms\": %.4f, \"gflops\": %.2f },\n", 
               cpu_ms, calculate_gflops(N, cpu_ms));
        printf("    \"Naive CUDA\": { \"ms\": %.4f, \"gflops\": %.2f, \"correct\": %s },\n",
               naive_ms, calculate_gflops(N, naive_ms), naive_correct ? "true" : "false");
        printf("    \"Tiled (16)\": { \"ms\": %.4f, \"gflops\": %.2f, \"correct\": %s },\n",
               tiled_16_ms, calculate_gflops(N, tiled_16_ms), tiled_16_correct ? "true" : "false");
        printf("    \"Tiled (32)\": { \"ms\": %.4f, \"gflops\": %.2f, \"correct\": %s },\n",
               tiled_32_ms, calculate_gflops(N, tiled_32_ms), tiled_32_correct ? "true" : "false");
        printf("    \"Reg-Blocked\": { \"ms\": %.4f, \"gflops\": %.2f, \"correct\": %s },\n",
               register_blocked_ms, calculate_gflops(N, register_blocked_ms), reg_blocked_correct ? "true" : "false");
        printf("    \"cuBLAS\": { \"ms\": %.4f, \"gflops\": %.2f, \"correct\": %s }\n",
               cublas_ms, calculate_gflops(N, cublas_ms), cublas_correct ? "true" : "false");
        printf("  }");
        
        // Cleanup
        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_C));
        free(h_A);
        free(h_B);
        free(h_C_cpu);
        free(h_C_gpu);
    }
    
    printf("\n]\n");  // End JSON array
    cleanup_cublas();
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char** argv) {
    bool json_only = false;
    
    // Check for --json or --json-only flag
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--json") == 0 || strcmp(argv[i], "--json-only") == 0) {
            json_only = true;
            break;
        }
    }
    
    if (!json_only) {
        printf("CUDA Matrix Multiplication Benchmark\n");
        printf("GPU: NVIDIA GeForce GTX 1050 Ti (Pascal, sm_61)\n");
        printf("======================================\n\n");
    }
    
    std::vector<int> sizes = {128, 256, 512, 1024, 2048};
    run_benchmark(sizes, json_only);
    
    return 0;
}
