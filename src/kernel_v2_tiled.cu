// ============================================================================
// Kernel V2: Tiled Matrix Multiplication with Shared Memory
// ============================================================================
// Uses shared memory to cache tiles of A and B.
// Each block loads a TILE_SIZE × TILE_SIZE tile from both matrices,
// then computes a TILE_SIZE × TILE_SIZE block of C.
// 
// Performance characteristics (TILE=16):
//  - Arithmetic intensity = (2*TILE^3)/(TILE^2*4*4*2) = TILE/16 ≈ 1 (for TILE=16)
//  - Global memory per operation: reduced by TILE times
//  - Expected: ~5-8 GFLOP/s (5x speedup over naive)
//
// Performance characteristics (TILE=32):
//  - Arithmetic intensity ≈ 2 (for TILE=32)
//  - Expected: ~25-30 GFLOP/s (30x speedup over naive)

#include "utils.h"

// ─────────────────────────────────────────────────────────────────────────
// Tiled kernel with TILE_SIZE = 16
// ─────────────────────────────────────────────────────────────────────────
__global__ void kernel_v2_tiled_16_impl(const float* A, const float* B, float* C, int N) {
    const int TILE = 16;
    
    __shared__ float tileA[TILE][TILE];
    __shared__ float tileB[TILE][TILE];
    
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    
    float sum = 0.0f;
    
    // Iterate over tiles along the k dimension
    for (int tile_k = 0; tile_k < N; tile_k += TILE) {
        // Load tiles from global memory
        if (row < N && tile_k + threadIdx.x < N) {
            tileA[threadIdx.y][threadIdx.x] = A[row * N + (tile_k + threadIdx.x)];
        } else {
            tileA[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        if (tile_k + threadIdx.y < N && col < N) {
            tileB[threadIdx.y][threadIdx.x] = B[(tile_k + threadIdx.y) * N + col];
        } else {
            tileB[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial sum for this tile
        for (int k = 0; k < TILE; k++) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

void kernel_v2_tiled_16(const float* d_A, const float* d_B, float* d_C, int N) {
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (N + 15) / 16);
    
    kernel_v2_tiled_16_impl<<<grid, block>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ─────────────────────────────────────────────────────────────────────────
// Tiled kernel with TILE_SIZE = 32
// ─────────────────────────────────────────────────────────────────────────
__global__ void kernel_v2_tiled_32_impl(const float* A, const float* B, float* C, int N) {
    const int TILE = 32;
    
    __shared__ float tileA[TILE][TILE];
    __shared__ float tileB[TILE][TILE];
    
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    
    float sum = 0.0f;
    
    // Iterate over tiles along the k dimension
    for (int tile_k = 0; tile_k < N; tile_k += TILE) {
        // Load tiles from global memory
        if (row < N && tile_k + threadIdx.x < N) {
            tileA[threadIdx.y][threadIdx.x] = A[row * N + (tile_k + threadIdx.x)];
        } else {
            tileA[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        if (tile_k + threadIdx.y < N && col < N) {
            tileB[threadIdx.y][threadIdx.x] = B[(tile_k + threadIdx.y) * N + col];
        } else {
            tileB[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial sum for this tile
        for (int k = 0; k < TILE; k++) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

void kernel_v2_tiled_32(const float* d_A, const float* d_B, float* d_C, int N) {
    dim3 block(32, 32);
    dim3 grid((N + 31) / 32, (N + 31) / 32);
    
    kernel_v2_tiled_32_impl<<<grid, block>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}
