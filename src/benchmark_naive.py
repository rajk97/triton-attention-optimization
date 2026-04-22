import torch
import matplotlib.pyplot as plt
from triton_native import naive_attention

# ======= Hardware specs (RTX 5060 Laptop) =======
PEAK_BW_GB_S = 448        # GB/s memory bandwidth
PEAK_TFLOPS_FP32 = 15     # TFLOPS FP32

# ======= Test parameters =======
SEQ_LENS = [512, 1024, 2048, 4096]
D_K = 64
WARMUP = 10
ITERS = 100

# ======= FILL IN: Theoretical model =======
def compute_flops(N, d_k):
    """Total FLOPs for the full kernel (all N query rows)."""
    # BLANK 1: You derived FLOPs per query row = 6 * N * d_k
    # There are N query rows, so total = ???
    return 6*N*d_k*N

def compute_bytes(N, d_k):
    """Total bytes moved from global memory (all N query rows)."""
    # BLANK 2: You derived bytes per query row = 12 * N * d_k
    # Total = ???
    return 12*N*d_k*N

# ======= Benchmark loop =======
results = []

for N in SEQ_LENS:
    Q = torch.randn(N, D_K, device='cuda', dtype=torch.float32)
    K = torch.randn(N, D_K, device='cuda', dtype=torch.float32)
    V = torch.randn(N, D_K, device='cuda', dtype=torch.float32)

    # Warmup
    for _ in range(WARMUP):
        naive_attention(Q, K, V)

    # FILL IN: Timed iterations using CUDA events
    times = []
    for _ in range(ITERS):
        # BLANK 3: Create start and end events
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        # BLANK 4: Record start, run kernel, record end, synchronize
        start.record()
        naive_attention(Q, K, V)
        end.record()
        
        end.synchronize() #CPU blocks until GPU hits 'end'
        # BLANK 5: Get elapsed time (events give milliseconds)
        elapsed_ms = start.elapsed_time(end)
        times.append(elapsed_ms)

    median_ms = sorted(times)[ITERS // 2]
    median_s = median_ms / 1000.0

    flops = compute_flops(N, D_K)
    total_bytes = compute_bytes(N, D_K)

    achieved_tflops = flops / median_s / 1e12
    achieved_bw = total_bytes / median_s / 1e9  # GB/s
    arith_intensity = flops / total_bytes        # FLOP/byte

    results.append({
        'N': N,
        'median_ms': median_ms,
        'achieved_tflops': achieved_tflops,
        'achieved_bw': achieved_bw,
        'arith_intensity': arith_intensity,
    })

    print(f"N={N:5d} | {median_ms:8.3f} ms | {achieved_tflops:.4f} TFLOPS | "
          f"{achieved_bw:.1f} GB/s | AI={arith_intensity:.2f} FLOP/byte")

# ======= FILL IN: Roofline Plot =======
fig, ax = plt.subplots(figsize=(10, 6))

# BLANK 6: Create the roofline ceiling lines
# X-axis range for arithmetic intensity
import numpy as np
ai_range = np.logspace(-2, 3, 200)

# Memory ceiling: performance = bandwidth * arithmetic_intensity
# (convert GB/s to TFLOPS: 448 GB/s = 0.448 TB/s)
mem_ceiling = PEAK_BW_GB_S * ai_range / 1e3  # array: PEAK_BW * ai_range, in TFLOPS

# Compute ceiling: flat line at peak TFLOPS
compute_ceiling = np.full_like(ai_range, PEAK_TFLOPS_FP32)  # array: constant PEAK_TFLOPS for all ai_range

# The roofline is the MIN of the two ceilings
roofline = np.minimum(mem_ceiling, compute_ceiling)

ax.loglog(ai_range, roofline, 'k-', linewidth=2, label='Roofline')

# BLANK 7: Plot your measured points
ais = [r['arith_intensity'] for r in results]
tflops = [r['achieved_tflops'] for r in results]
bw = [r['achieved_bw'] for r in results]
print(f"Achieved bandwidths (GB/s): {bw}")

ax.loglog(ais, tflops, 'ro', markersize=10, label='Naive kernel')

# Labels for each point
for r in results:
    ax.annotate(f"N={r['N']}", (r['arith_intensity'], r['achieved_tflops']),
                textcoords="offset points", xytext=(10, 5))

ax.set_xlabel('Arithmetic Intensity (FLOP/byte)')
ax.set_ylabel('Performance (TFLOPS)')
ax.set_title('Roofline Analysis — Naive Attention Kernel')
ax.legend()
ax.grid(True, which='both', alpha=0.3)
plt.tight_layout()
plt.savefig('../roofline_naive.png', dpi=150)
print("\nSaved roofline_naive.png")
plt.show()
