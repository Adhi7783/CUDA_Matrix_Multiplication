// ============================================================================
// Kernel V1: Naive Matrix Multiplication
// ============================================================================
// Each thread computes one element of C.
// Each iteration reads from global memory (high latency, low reuse).
// 
// Performance characteristics:
//  - 0 reuse of global data (arithmetic intensity = 2N/(4*N^2) ≈ 1/2N)
//  - For N=1024: AI ~ 0.0005, memory-bound
//  - Expected: ~0.8-1 GFLOP/s

#include "utils.h"

__global__ void kernel_v1_naive_impl(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

void kernel_v1_naive(const float* d_A, const float* d_B, float* d_C, int N) {
    dim3 block(16, 16);  // 256 threads per block
    dim3 grid((N + 15) / 16, (N + 15) / 16);
    
    kernel_v1_naive_impl<<<grid, block>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}
