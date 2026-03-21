# Project 1: Triton Attention Kernel — Profile-Driven Optimization

## Mission
Build a FlashAttention-style fused attention kernel in Triton from scratch, progressively optimize it using profiling data, and produce a polished Jupyter notebook that tells the optimization story. This is the crown jewel project — the single most impressive artifact for an inference engineering job.

## Environment (DO NOT reinstall anything)
- **Activate**: `source ~/deeplearn_env/bin/activate`
- **Working dir**: `/home/raj/Documents/Projects/inference_projects/project1_triton_attention/`
- **GPU**: NVIDIA RTX 5060 Laptop (Blackwell, sm_120, 8GB VRAM, ~448 GB/s bandwidth, ~15 TFLOPS measured)
- **PyTorch**: 2.10.0+cu128
- **Triton**: 3.6.0
- **Also available**: matplotlib, plotly, numpy, pandas, tabulate, gpustat
- **Key hardware numbers for roofline**:
  - Peak memory bandwidth: ~448 GB/s (GDDR7, 128-bit)
  - Peak compute (FP32): ~15 TFLOPS (measured via matmul)
  - Peak compute (FP16/BF16 Tensor Cores): estimate ~30+ TFLOPS
  - Shared memory per SM: 100 KB, per block: 48 KB
  - SMs: 26, Registers/SM: 65536

## Time Budget: 100 minutes total
| Phase | Minutes | Description |
|---|---|---|
| Phase 1: Naive attention | 20 | Correct but slow baseline |
| Phase 2: Benchmark + roofline | 15 | Prove it's memory-bound |
| Phase 3: Tiled online softmax | 30 | The FlashAttention core insight |
| Phase 4: Autotune + re-profile | 15 | Data-driven optimization |
| Phase 5: Integrate + compare | 20 | Real-world comparison |

## Phase 1: Naive Attention Kernel (20 min)

Write a Triton kernel that computes: `O = softmax(Q @ K^T / sqrt(d_k)) @ V`

Implementation:
1. Each program instance computes one row of the output (one query token)
2. Load the full Q row, compute dot products with ALL K rows to get scores
3. Apply scaling by 1/sqrt(d_k)
4. Compute numerically stable softmax: subtract max, exp, divide by sum
5. Multiply softmax weights by V rows, accumulate output

Key parameters:
- Batch size: 1 (keep simple), Heads: 1 (keep simple)
- Seq lengths to test: [512, 1024, 2048, 4096]
- d_k (head dim): 64 (standard)
- Use float32 first for correctness, then switch to float16

Correctness check: compare output against `torch.nn.functional.scaled_dot_product_attention`

**Checkpoint**: Working kernel, correct output (atol=1e-2 for fp16, 1e-5 for fp32)

## Phase 2: Benchmark + Roofline Analysis (15 min)

For each seq_len in [512, 1024, 2048, 4096]:
1. Measure kernel latency (use torch.cuda.Event for timing, 100 iterations, skip 10 warmup)
2. Compute FLOPS: For attention, approximately `4 * N * N * d + 3 * N * N` (two matmuls + softmax)
3. Compute bytes moved: Naive reads Q (N*d), K (N*d), V (N*d) from HBM, plus materializes S=QK^T (N*N) to HBM, reads it back for softmax, writes O (N*d). Total ≈ `3*N*d + 2*N*N + N*d` elements * bytes_per_element
4. Compute arithmetic intensity = FLOPS / bytes
5. Plot on roofline: x-axis = arithmetic intensity (FLOP/byte), y-axis = achieved TFLOPS. Draw the roofline ceiling lines for memory bandwidth and compute.

**Checkpoint**: Roofline plot showing naive kernel is far below compute ceiling (memory-bound)

## Phase 3: Tiled Online Softmax — The FlashAttention Core (30 min)

This is the hard part and the most valuable skill demonstration.

The key insight: Instead of materializing the full N×N attention matrix to HBM, process K and V in BLOCKS (tiles) along the sequence dimension. Maintain running statistics for numerically stable softmax:
- `m_i` = running max of scores seen so far
- `l_i` = running sum of exp(scores - m_i) seen so far
- `o_i` = running weighted sum, rescaled each time m_i updates

Algorithm per query block (BLOCK_M queries):
```
For each key block k (BLOCK_N keys):
    1. Load Q_block (BLOCK_M × d), K_block (BLOCK_N × d)
    2. Compute S_block = Q_block @ K_block^T / sqrt(d_k)  [BLOCK_M × BLOCK_N]
    3. Compute block_max = max(S_block, axis=1)  [BLOCK_M]
    4. Update running max: m_new = max(m_old, block_max)
    5. Correction factor: alpha = exp(m_old - m_new)
    6. P_block = exp(S_block - m_new[:, None])  [BLOCK_M × BLOCK_N]
    7. l_new = alpha * l_old + sum(P_block, axis=1)
    8. O = alpha[:, None] * O + P_block @ V_block
    9. m_old = m_new, l_old = l_new
Final: O = O / l_new[:, None]
```

Triton implementation notes:
- Use `tl.program_id(0)` for the query block index
- BLOCK_M, BLOCK_N, d_k as `tl.constexpr`
- Use `tl.dot` for the matmuls (maps to Tensor Cores on Blackwell)
- Use `tl.max`, `tl.sum`, `tl.exp` for softmax operations
- Process in float32 accumulators even if inputs are float16

**Checkpoint**: Correct output matching naive kernel + PyTorch reference. Measure memory — should NOT allocate N×N intermediate.

## Phase 4: Autotune + Re-profile (15 min)

Add `@triton.autotune` with these config candidates:
```python
configs = [
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=4, num_stages=2),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
]
```

Key function: `key=['seq_len']` so it retunes per sequence length.

After autotuning:
1. Record which config won for each seq_len
2. Re-benchmark the tiled kernel with optimal config
3. Recompute arithmetic intensity (now much lower bytes since no N×N materialization)
4. Update roofline plot — show the tiled kernel moved to a better position

**Checkpoint**: Autotuning table + updated roofline showing improvement

## Phase 5: Integrate + Compare (20 min)

1. Wrap the best Triton kernel in a PyTorch-compatible function
2. Compare against:
   - `torch.nn.functional.scaled_dot_product_attention` (cuDNN Flash backend)
   - `torch.compile`-d naive Python attention
   - The naive Triton kernel from Phase 1
3. For each, measure at seq_len = [512, 1024, 2048, 4096]:
   - Latency (ms)
   - Peak GPU memory allocated
4. Generate comparison charts

**Checkpoint**: Final comparison chart + analysis of where your kernel wins/loses and why

## Final Notebook Structure

Create `project1_triton_attention/attention_kernel_optimization.ipynb` (or .py if jupyter has issues) with these sections:

1. **Introduction** (2-3 sentences: what we're building and why)
2. **Phase 1: Naive Attention** — code + correctness validation
3. **Phase 2: Profiling the Naive Kernel** — benchmark code + roofline plot + analysis paragraph ("The naive kernel achieves X TFLOPS at seq_len=Y, sitting at Z FLOP/byte arithmetic intensity. The roofline shows it's firmly memory-bound because...")
4. **Phase 3: Tiled Online Softmax** — the algorithm explanation + code + correctness validation + memory comparison
5. **Phase 4: Autotuning** — config table + re-profiling + updated roofline + analysis ("The autotuner chose config X for seq_len Y because...")
6. **Phase 5: Final Comparison** — bar charts + analysis of crossover points
7. **Key Takeaways** — 3-4 bullets on what you learned about GPU performance engineering

## Critical Rules
- Every number must come from actual measurement, not made up
- Include ugly intermediate results — failed attempts are more impressive than polished lies
- Write analysis paragraphs explaining WHY, not just WHAT
- Use `torch.cuda.synchronize()` before timing measurements
- Use `torch.cuda.Event` for accurate GPU timing
- Clear cache between benchmarks: `torch.cuda.empty_cache()`
- The narrative matters more than the code — this is an optimization STORY

## What NOT to do
- Don't implement backward pass
- Don't implement multi-head (single head is fine for demonstrating the concept)
- Don't implement warp specialization (that's a week-long endeavor)
- Don't spend time on pretty formatting — content over form
- Don't use FlashAttention library — the point is writing it yourself
