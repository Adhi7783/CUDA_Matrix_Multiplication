// ============================================================================
// cuBLAS Wrapper for Comparison
// ============================================================================
// Links against NVIDIA's optimized cuBLAS library for the gold standard.

#include "utils.h"
#include <cublas_v2.h>

static cublasHandle_t handle = nullptr;

void init_cublas() {
    if (handle == nullptr) {
        cublasCreate(&handle);
    }
}

void cleanup_cublas() {
    if (handle != nullptr) {
        cublasDestroy(handle);
        handle = nullptr;
    }
}

void cublas_sgemm(const float* d_A, const float* d_B, float* d_C, int N) {
    if (handle == nullptr) init_cublas();
    
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // cuBLAS uses column-major (Fortran) convention by default
    // C = alpha * A * B + beta * C
    // We have row-major, so swap A and B and use CUBLAS_OP_T
    cublasStatus_t status = cublasSgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, N, N,          // m, n, k
        &alpha,
        d_A, N,            // A, lda
        d_B, N,            // B, ldb
        &beta,
        d_C, N             // C, ldc
    );
    
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS error: %d\n", status);
        exit(EXIT_FAILURE);
    }
    
    CUDA_CHECK(cudaDeviceSynchronize());
}
