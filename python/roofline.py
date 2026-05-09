#!/usr/bin/env python3
# ============================================================================
# Roofline Model Visualization
# ============================================================================
# Plots the roofline model for GTX 1050 Ti, showing:
# 1. Compute throughput ceiling (1900 GFLOP/s)
# 2. Memory bandwidth ceiling (~112 GB/s for float32)
# 3. Compute and memory bounds as functions of arithmetic intensity
#
# This helps explain why kernels are compute-bound vs memory-bound.
#
# Usage: python3 roofline.py

import numpy as np
import matplotlib.pyplot as plt
import json

def main():
    # GTX 1050 Ti specifications
    peak_compute = 1900      # GFLOP/s (F32)
    peak_bandwidth = 112     # GB/s (memory bandwidth)
    
    # Convert bandwidth to GFLOP/s per unit arithmetic intensity
    # Bandwidth limit (GFLOP/s) = AI (FLOP/byte) * peak_bandwidth (GB/s)
    # 1 GB/s = 1e9 bytes/s, so 1 GFLOP = 4 bytes (for float32)
    # bandwidth_gflops = AI * peak_bandwidth
    
    # Arithmetic intensity range (FLOP/byte)
    ai = np.logspace(-1, 2, 1000)  # 0.1 to 100 FLOP/byte
    
    # Memory-bound region: GFLOP/s = AI * bandwidth
    memory_bound = ai * peak_bandwidth
    
    # Compute-bound region: constant peak compute
    compute_bound = np.full_like(ai, peak_compute)
    
    # Roofline is the minimum of the two
    roofline = np.minimum(memory_bound, compute_bound)
    
    # Inflection point (where memory and compute bounds meet)
    ai_peak = peak_compute / peak_bandwidth
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot roofline
    ax.loglog(ai, roofline, 'k-', linewidth=3, label='Roofline')
    
    # Shade regions
    ax.fill_between(ai, memory_bound, roofline, where=(memory_bound <= compute_bound),
                    alpha=0.3, color='red', label='Memory-Bound Region')
    ax.fill_between(ai, compute_bound, roofline, where=(compute_bound <= memory_bound),
                    alpha=0.3, color='blue', label='Compute-Bound Region')
    
    # Mark the peak compute line
    ax.axhline(y=peak_compute, color='blue', linestyle='--', linewidth=2, alpha=0.7)
    ax.text(0.15, peak_compute * 1.2, f'Peak Compute: {peak_compute} GFLOP/s', 
           fontsize=11, color='blue', fontweight='bold')
    
    # Mark the inflection point
    ax.plot(ai_peak, peak_compute, 'go', markersize=10, label=f'Inflection: {ai_peak:.2f} FLOP/byte')
    
    # Typical AI values for different kernel types
    ai_naive = 0.001       # 1/(2*N) for large N
    ai_tiled_16 = 1.0      # Approximate for TILE=16
    ai_tiled_32 = 2.0      # Approximate for TILE=32
    ai_regblocked = 3.0    # Approximate for register-blocked
    
    # Plot kernel operating points (hypothetical)
    kernel_points = [
        (ai_naive, 0.8, 'Naive CUDA', 'red'),
        (ai_tiled_16, 5, 'Tiled (16)', 'orange'),
        (ai_tiled_32, 30, 'Tiled (32)', 'green'),
        (ai_regblocked, 50, 'Reg-Blocked', 'blue'),
    ]
    
    for ai_val, gflops, label, color in kernel_points:
        # Clamp to roofline
        interp_gflops = min(ai_val * peak_bandwidth, peak_compute)
        ax.plot(ai_val, gflops, 'o', markersize=12, color=color, markeredgecolor='black', markeredgewidth=2)
        ax.annotate(label, xy=(ai_val, gflops), xytext=(10, 10), textcoords='offset points',
                   fontsize=10, fontweight='bold', color=color,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Formatting
    ax.set_xlabel('Arithmetic Intensity (FLOP/Byte)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Throughput (GFLOP/s)', fontsize=13, fontweight='bold')
    ax.set_title('Roofline Model: GTX 1050 Ti\nPeak Compute: 1900 GFLOP/s | Peak Bandwidth: 112 GB/s',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, which='both', alpha=0.3, linestyle=':')
    
    # Set axis limits
    ax.set_xlim(0.1, 100)
    ax.set_ylim(0.1, 5000)
    
    plt.tight_layout()
    plt.savefig('results/roofline_model.png', dpi=150, bbox_inches='tight')
    print("✓ Saved roofline plot to results/roofline_model.png")
    
    # Print analysis
    print("\nRoofline Analysis:")
    print(f"  Peak Compute (FP32): {peak_compute} GFLOP/s")
    print(f"  Peak Bandwidth: {peak_bandwidth} GB/s")
    print(f"  Inflection Point: {ai_peak:.4f} FLOP/byte")
    print(f"\nTo achieve peak compute, need AI >= {ai_peak:.4f} FLOP/byte")
    print(f"This means for matrix multiply (2N^3 / (4N^2) = N/2 effective AI):")
    print(f"  - Need N >= {2*ai_peak:.1f} to be compute-bound")
    print(f"\nKernel Analysis:")
    for ai_val, gflops, label, _ in kernel_points:
        region = "Memory-Bound" if ai_val < ai_peak else "Compute-Bound"
        print(f"  {label:20} AI={ai_val:.2f}, {region}")

if __name__ == "__main__":
    main()
