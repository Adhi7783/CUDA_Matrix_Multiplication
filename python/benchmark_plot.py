#!/usr/bin/env python3
# ============================================================================
# Benchmark Plotting Script
# ============================================================================
# Reads benchmark JSON output and generates publication-quality plots
# showing performance comparison across all implementations.
#
# Usage: python3 benchmark_plot.py

import subprocess
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import sys

def run_benchmark():
    """Run the benchmark and return parsed JSON results."""
    print("Running benchmark...")
    result = subprocess.run(["./bin/matmul_bench"], capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error running benchmark: {result.stderr}")
        sys.exit(1)
    
    # The benchmark prints a human-readable header before emitting JSON.
    # Extract the first JSON array/object from stdout to be robust to that header.
    out = result.stdout
    # Find the start of the JSON payload (either '[' or '{')
    idx = min([i for i in [out.find('['), out.find('{')] if i != -1] or [ -1 ])
    if idx == -1:
        print("Error: no JSON payload found in benchmark output")
        print(f"Output was:\n{out}")
        sys.exit(1)

    json_text = out[idx:]
    try:
        data = json.loads(json_text)
        return data
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON payload: {e}")
        print(f"Extracted payload was:\n{json_text}")
        sys.exit(1)

def main():
    # Run benchmark
    results = run_benchmark()
    
    # Extract data
    sizes = []
    implementations = {
        "CPU": [],
        "Naive CUDA": [],
        "Tiled (16)": [],
        "Tiled (32)": [],
        "Reg-Blocked": [],
        "cuBLAS": []
    }
    
    for result in results:
        N = result["N"]
        sizes.append(N)
        
        # Handle CPU (may be -1 for large N)
        cpu_data = result.get("CPU", {})
        implementations["CPU"].append(cpu_data.get("gflops", 0))
        
        implementations["Naive CUDA"].append(result["Naive CUDA"]["gflops"])
        implementations["Tiled (16)"].append(result["Tiled (16)"]["gflops"])
        implementations["Tiled (32)"].append(result["Tiled (32)"]["gflops"])
        implementations["Reg-Blocked"].append(result["Reg-Blocked"]["gflops"])
        implementations["cuBLAS"].append(result["cuBLAS"]["gflops"])
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("CUDA Matrix Multiplication Performance — GTX 1050 Ti (Pascal)", 
                 fontsize=16, fontweight='bold')
    
    colors = {
        "CPU": "#DC2626",
        "Naive CUDA": "#EA580C",
        "Tiled (16)": "#CA8A04",
        "Tiled (32)": "#16A34A",
        "Reg-Blocked": "#2563EB",
        "cuBLAS": "#7C3AED"
    }
    
    # ─────────────────────────────────────────────────────────────────────────
    # Plot 1: Execution Time vs Matrix Size (log scale)
    # ─────────────────────────────────────────────────────────────────────────
    ax = axes[0]
    for impl, gflops_list in implementations.items():
        if impl == "CPU" and gflops_list[0] <= 0:
            continue  # Skip CPU if not measured
        times = [(2 * N**3) / (gf * 1e9 * 1e-3) if gf > 0 else 0 
                 for N, gf in zip(sizes, gflops_list)]
        ax.semilogy(sizes, times, marker='o', label=impl, color=colors[impl], linewidth=2, markersize=8)
    
    ax.set_xlabel("Matrix Size N", fontsize=12, fontweight='bold')
    ax.set_ylabel("Execution Time (ms, log scale)", fontsize=12, fontweight='bold')
    ax.set_title("Execution Time vs Matrix Size", fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    
    # ─────────────────────────────────────────────────────────────────────────
    # Plot 2: Throughput Comparison at N=1024
    # ─────────────────────────────────────────────────────────────────────────
    ax = axes[1]
    idx_1024 = sizes.index(1024) if 1024 in sizes else -1
    
    if idx_1024 >= 0:
        impl_names = list(implementations.keys())
        gflops_at_1024 = [implementations[impl][idx_1024] for impl in impl_names]
        
        bars = ax.bar(impl_names, gflops_at_1024, color=[colors[impl] for impl in impl_names])
        
        # Peak theoretical throughput for GTX 1050 Ti
        peak_fp32 = 1900  # GFLOP/s
        ax.axhline(y=peak_fp32, color='black', linestyle='--', linewidth=2, alpha=0.7, label='Theoretical Peak (1900 GFLOP/s)')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_ylabel("Throughput (GFLOP/s)", fontsize=12, fontweight='bold')
        ax.set_title(f"Peak Throughput at N=1024", fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # ─────────────────────────────────────────────────────────────────────────
    # Plot 3: Speedup over CPU Baseline
    # ─────────────────────────────────────────────────────────────────────────
    ax = axes[2]
    
    if implementations["CPU"][0] > 0:  # Only if CPU was measured
        cpu_gflops = implementations["CPU"]
        for impl in ["Naive CUDA", "Tiled (16)", "Tiled (32)", "Reg-Blocked", "cuBLAS"]:
            speedups = [cpu_gflops[i] / implementations[impl][i] if implementations[impl][i] > 0 else 0
                       for i in range(len(sizes))]
            ax.plot(sizes, speedups, marker='o', label=impl, color=colors[impl], 
                   linewidth=2, markersize=8)
        
        ax.set_xlabel("Matrix Size N", fontsize=12, fontweight='bold')
        ax.set_ylabel("Speedup over CPU", fontsize=12, fontweight='bold')
        ax.set_title("GPU Speedup vs CPU Baseline", fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("results/matmul_benchmark.png", dpi=150, bbox_inches='tight')
    print("✓ Saved plot to results/matmul_benchmark.png")
    
    # Also save raw data for further analysis
    with open("results/benchmark_data.json", 'w') as f:
        json.dump({
            "sizes": sizes,
            "implementations": implementations
        }, f, indent=2)
    print("✓ Saved data to results/benchmark_data.json")

if __name__ == "__main__":
    main()
