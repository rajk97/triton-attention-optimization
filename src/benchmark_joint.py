"""
Joint roofline benchmark: naive vs tiled Triton attention kernels.

Plots both kernels on the same roofline so the optimization win is visible:
naive sits in the memory-bound region (AI ~0.5) while tiled jumps to the
compute-bound region (AI ~48 for BLOCK_M=64).
"""
import numpy as np
import torch
import matplotlib.pyplot as plt

from triton_native import naive_attention
from triton_tiled import tiled_attention

# ======= Hardware specs (RTX 5060 Laptop) =======
PEAK_BW_GB_S = 448
PEAK_TFLOPS_FP32 = 15

# ======= Test parameters =======
SEQ_LENS = [512, 1024, 2048, 4096]
D_K = 64
BLOCK_M_TILED = 64
WARMUP = 10
ITERS = 100


# ---------- Theoretical models ----------
def compute_flops(N, d_k):
    # Same for both kernels: 6 * N^2 * d_k
    return 6 * N * N * d_k


def compute_bytes_naive(N, d_k):
    # Naive: 12 * N^2 * d_k  (K loaded twice + V loaded once per query row)
    return 12 * N * N * d_k


def compute_bytes_tiled(N, d_k, BLOCK_M):
    # Tiled: 8 * N^2 * d_k / BLOCK_M  (K + V loaded once per program; N/BLOCK_M programs)
    return 8 * N * N * d_k / BLOCK_M


# ---------- Benchmark helper ----------
def benchmark(kernel_fn, N, d_k, label):
    Q = torch.randn(N, d_k, device='cuda', dtype=torch.float32)
    K = torch.randn(N, d_k, device='cuda', dtype=torch.float32)
    V = torch.randn(N, d_k, device='cuda', dtype=torch.float32)

    for _ in range(WARMUP):
        kernel_fn(Q, K, V)

    times = []
    for _ in range(ITERS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        kernel_fn(Q, K, V)
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))

    median_ms = sorted(times)[ITERS // 2]
    return median_ms


# ---------- Run both kernels ----------
naive_results = []
tiled_results = []

print(f"{'N':>6} | {'kernel':>6} | {'ms':>9} | {'TFLOPS':>7} | {'GB/s':>7} | {'AI':>6}")
print("-" * 60)

for N in SEQ_LENS:
    # Naive
    ms = benchmark(naive_attention, N, D_K, 'naive')
    s = ms / 1000.0
    flops = compute_flops(N, D_K)
    bytes_ = compute_bytes_naive(N, D_K)
    naive_results.append({
        'N': N, 'ms': ms,
        'tflops': flops / s / 1e12,
        'bw': bytes_ / s / 1e9,
        'ai': flops / bytes_,
    })
    r = naive_results[-1]
    print(f"{N:>6} | {'naive':>6} | {ms:>9.3f} | {r['tflops']:>7.3f} | "
          f"{r['bw']:>7.1f} | {r['ai']:>6.2f}")

    # Tiled
    ms = benchmark(tiled_attention, N, D_K, 'tiled')
    s = ms / 1000.0
    bytes_ = compute_bytes_tiled(N, D_K, BLOCK_M_TILED)
    tiled_results.append({
        'N': N, 'ms': ms,
        'tflops': flops / s / 1e12,
        'bw': bytes_ / s / 1e9,
        'ai': flops / bytes_,
    })
    r = tiled_results[-1]
    print(f"{N:>6} | {'tiled':>6} | {ms:>9.3f} | {r['tflops']:>7.3f} | "
          f"{r['bw']:>7.1f} | {r['ai']:>6.2f}")

# ---------- Joint roofline plot ----------
fig, ax = plt.subplots(figsize=(11, 6.5))

ai_range = np.logspace(-2, 3, 200)
mem_ceiling = PEAK_BW_GB_S * ai_range / 1e3
compute_ceiling = np.full_like(ai_range, PEAK_TFLOPS_FP32)
roofline = np.minimum(mem_ceiling, compute_ceiling)

ax.loglog(ai_range, roofline, 'k-', linewidth=2, label='Roofline (RTX 5060 Laptop)')

# Ridge point
ridge = PEAK_TFLOPS_FP32 * 1e3 / PEAK_BW_GB_S
ax.axvline(ridge, color='gray', linestyle='--', alpha=0.5,
           label=f'Ridge point ≈ {ridge:.1f} FLOP/byte')

# Naive points
ais = [r['ai'] for r in naive_results]
tflops = [r['tflops'] for r in naive_results]
ax.loglog(ais, tflops, 'ro', markersize=10, label='Naive Triton kernel')
for r in naive_results:
    ax.annotate(f"N={r['N']}", (r['ai'], r['tflops']),
                textcoords="offset points", xytext=(8, -12), fontsize=9, color='red')

# Tiled points
ais = [r['ai'] for r in tiled_results]
tflops = [r['tflops'] for r in tiled_results]
ax.loglog(ais, tflops, 'b^', markersize=10, label=f'Tiled Triton kernel (BLOCK_M={BLOCK_M_TILED})')
for r in tiled_results:
    ax.annotate(f"N={r['N']}", (r['ai'], r['tflops']),
                textcoords="offset points", xytext=(8, 5), fontsize=9, color='blue')

ax.set_xlabel('Arithmetic Intensity (FLOP/byte)')
ax.set_ylabel('Performance (TFLOPS)')
ax.set_title('Roofline: Naive vs Tiled FlashAttention (Triton, RTX 5060 Laptop)')
ax.legend(loc='lower right')
ax.grid(True, which='both', alpha=0.3)
plt.tight_layout()

out_path = '../roofline_comparison.png'
plt.savefig(out_path, dpi=150)
print(f"\nSaved {out_path}")
