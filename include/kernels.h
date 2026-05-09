#pragma once

#include <cuda_runtime.h>

// ============================================================================
// Kernel Function Declarations
// ============================================================================

// V1: Naive CUDA - each thread reads from global memory per operation
void kernel_v1_naive(const float* d_A, const float* d_B, float* d_C, int N);

// V2a: Tiled with TILE_SIZE = 16
void kernel_v2_tiled_16(const float* d_A, const float* d_B, float* d_C, int N);

// V2b: Tiled with TILE_SIZE = 32
void kernel_v2_tiled_32(const float* d_A, const float* d_B, float* d_C, int N);

// V3: Register-blocked - each thread computes multiple output elements
void kernel_v3_register_blocked(const float* d_A, const float* d_B, float* d_C, int N);

// cuBLAS wrapper
void cublas_sgemm(const float* d_A, const float* d_B, float* d_C, int N);
