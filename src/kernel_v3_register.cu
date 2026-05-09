// ============================================================================
// Kernel V3: Register-Blocked Matrix Multiplication
// ============================================================================
// Each thread computes a block of output elements using registers.
// Combines tiling with intra-warp blocking.
// 
// Strategy:
//  - Each block processes TILE_SIZE × TILE_SIZE output
//  - Each thread processes BX × BY elements (stored in registers)
//  - Load tiles into shared memory
//  - Reduce global memory pressure further
//
// Performance characteristics:
//  - Arithmetic intensity ≈ 3-4 (much better than tiled)
//  - Better utilization of register file and L1 cache
//  - Expected: ~15-20 GFLOP/s (20x speedup over naive)

#include "utils.h"

__global__ void kernel_v3_register_blocked_impl(const float* A, const float* B, float* C, int N) {
    const int TILE = 32;
    const int BX = 2;  // Each thread computes BX outputs along columns
    const int BY = 2;  // Each thread computes BY outputs along rows
    
    __shared__ float tileA[TILE][TILE];
    __shared__ float tileB[TILE][TILE];
    
    // Thread indices within block
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Global output position for this thread's block
    int block_row = blockIdx.y * TILE;
    int block_col = blockIdx.x * TILE;
    
    // This thread's row and column within the output block
    int row_base = block_row + ty * BY;
    int col_base = block_col + tx * BX;
    
    // Register array to accumulate results
    float result[BY][BX];
    #pragma unroll
    for (int i = 0; i < BY; i++) {
        #pragma unroll
        for (int j = 0; j < BX; j++) {
            result[i][j] = 0.0f;
        }
    }
    
    // Process tiles of K dimension
    for (int tile_k = 0; tile_k < N; tile_k += TILE) {
        // Load tileA: distribute 32x32 tile across 16x16=256 threads
        // Each thread loads 4 elements (32*32 / 256 = 4)
        int thread_idx = ty * 16 + tx;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int linear_idx = thread_idx + i * 256;
            int sm_row = linear_idx / TILE;
            int sm_col = linear_idx % TILE;
            int global_row = block_row + sm_row;
            int global_col = tile_k + sm_col;
            
            if (global_row < N && global_col < N) {
                tileA[sm_row][sm_col] = A[global_row * N + global_col];
            } else {
                tileA[sm_row][sm_col] = 0.0f;
            }
        }
        
        // Load tileB: distribute 32x32 tile across 16x16=256 threads
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int linear_idx = thread_idx + i * 256;
            int sm_row = linear_idx / TILE;
            int sm_col = linear_idx % TILE;
            int global_row = tile_k + sm_row;
            int global_col = block_col + sm_col;
            
            if (global_row < N && global_col < N) {
                tileB[sm_row][sm_col] = B[global_row * N + global_col];
            } else {
                tileB[sm_row][sm_col] = 0.0f;
            }
        }
        
        __syncthreads();
        
        // Compute: multiply this thread's A and B tiles
        for (int k = 0; k < TILE; k++) {
            float valA[BY];
            float valB[BX];
            
            // Load A values for this thread's output rows
            #pragma unroll
            for (int i = 0; i < BY; i++) {
                int a_row = (row_base - block_row + i);  // Relative to block
                if (a_row < TILE) {
                    valA[i] = tileA[a_row][k];
                } else {
                    valA[i] = 0.0f;
                }
            }
            
            // Load B values for this thread's output columns
            #pragma unroll
            for (int j = 0; j < BX; j++) {
                int b_col = (col_base - block_col + j);  // Relative to block
                if (b_col < TILE) {
                    valB[j] = tileB[k][b_col];
                } else {
                    valB[j] = 0.0f;
                }
            }
            
            // Accumulate products
            #pragma unroll
            for (int i = 0; i < BY; i++) {
                #pragma unroll
                for (int j = 0; j < BX; j++) {
                    result[i][j] += valA[i] * valB[j];
                }
            }
        }
        
        __syncthreads();
    }
    
    // Write results back to global memory
    #pragma unroll
    for (int i = 0; i < BY; i++) {
        #pragma unroll
        for (int j = 0; j < BX; j++) {
            int out_row = row_base + i;
            int out_col = col_base + j;
            if (out_row < N && out_col < N) {
                C[out_row * N + out_col] = result[i][j];
            }
        }
    }
}

void kernel_v3_register_blocked(const float* d_A, const float* d_B, float* d_C, int N) {
    // Block size: (16, 16) = 256 threads
    // Each thread computes 2x2 output block
    // Each block handles 32x32 output tile
    dim3 block(16, 16);
    dim3 grid((N + 31) / 32, (N + 31) / 32);
    
    kernel_v3_register_blocked_impl<<<grid, block>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}
