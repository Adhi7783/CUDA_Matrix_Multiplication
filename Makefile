# ============================================================================
# Makefile for CUDA Matrix Multiplication
# ============================================================================
# GTX 1050 Ti uses Pascal architecture (sm_61)
# Usage: make [all|clean|profile|bench|test]

NVCC := nvcc
CUDA_ARCH ?= sm_61
NVCC_FLAGS := -O3 -arch=$(CUDA_ARCH) -std=c++17 -lineinfo
CUBLAS_FLAGS := -lcublas
INCLUDE_FLAGS := -I./include

# Source files
SRC_DIR := src
TEST_DIR := tests
BIN_DIR := bin

# Create bin directory if it doesn't exist
$(shell mkdir -p $(BIN_DIR))

# Kernel source files
KERNEL_SOURCES := $(SRC_DIR)/kernel_v1_naive.cu \
                  $(SRC_DIR)/kernel_v2_tiled.cu \
                  $(SRC_DIR)/kernel_v3_register.cu \
                  $(SRC_DIR)/kernel_cublas.cu

# Targets
BENCH := $(BIN_DIR)/matmul_bench
TEST := $(BIN_DIR)/test_correctness

all: $(BENCH) $(TEST)

# ─────────────────────────────────────────────────────────────────────────
# Benchmark executable
# ─────────────────────────────────────────────────────────────────────────
$(BENCH): $(SRC_DIR)/benchmark.cu $(KERNEL_SOURCES)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDE_FLAGS) $(CUBLAS_FLAGS) -o $@ $^
	@echo "Build successful: $@"

# ─────────────────────────────────────────────────────────────────────────
# Test executable
# ─────────────────────────────────────────────────────────────────────────
$(TEST): $(TEST_DIR)/test_correctness.cu $(KERNEL_SOURCES)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDE_FLAGS) $(CUBLAS_FLAGS) -o $@ $^
	@echo "Build successful: $@"

# ─────────────────────────────────────────────────────────────────────────
# Run tests
# ─────────────────────────────────────────────────────────────────────────
test: $(TEST)
	@echo "Running correctness tests..."
	./$(TEST)

# ─────────────────────────────────────────────────────────────────────────
# Run benchmark
# ─────────────────────────────────────────────────────────────────────────
bench: $(BENCH)
	@echo "Running benchmark (output to benchmark_results.json)..."
	./$(BENCH) > results/benchmark_results.json
	@echo "Benchmark complete. Results saved to results/benchmark_results.json"

# ─────────────────────────────────────────────────────────────────────────
# Profile with nvprof
# ─────────────────────────────────────────────────────────────────────────
profile: $(BENCH)
	@echo "Profiling with nvprof..."
	nvprof --metrics all ./$(BENCH)

# ─────────────────────────────────────────────────────────────────────────
# Clean
# ─────────────────────────────────────────────────────────────────────────
clean:
	rm -f $(BENCH) $(TEST)
	rm -f *.o *.ptx
	@echo "Clean complete."

# ─────────────────────────────────────────────────────────────────────────
# Help
# ─────────────────────────────────────────────────────────────────────────
help:
	@echo "CUDA Matrix Multiplication Build Targets:"
	@echo "  make all       - Build benchmark and test executables"
	@echo "  make all CUDA_ARCH=sm_75 - Override target arch if needed"
	@echo "  make bench     - Run benchmarks (outputs JSON)"
	@echo "  make test      - Run correctness tests"
	@echo "  make profile   - Profile with nvprof"
	@echo "  make clean     - Remove built files"
	@echo "  make help      - Show this help message"

.PHONY: all test bench profile clean help
