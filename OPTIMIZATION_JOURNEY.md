# Triton Attention Kernel — Optimization Journey

## Overview
Building a FlashAttention-style fused attention kernel in Triton from scratch, with profile-driven optimization on NVIDIA RTX 5060 Laptop (Blackwell, sm_120, 8GB VRAM).

**Goal**: Demonstrate deep GPU performance engineering — from naive baseline to optimized tiled kernel, guided entirely by profiling data and first-principles reasoning.

**Hardware**:
| Spec | Value |
|------|-------|
| GPU | NVIDIA RTX 5060 Laptop (Blackwell) |
| SM Count | 26 |
| VRAM | 8 GB GDDR7 |
| Peak Bandwidth | ~448 GB/s |
| Peak FP32 | ~15 TFLOPS |
| Peak FP16 (Tensor Cores) | ~30+ TFLOPS |
| Shared Mem / SM | 100 KB |

---

## Phase 1: Naive Attention Kernel
**Status**: Complete ✓

### Objective
Implement `O = softmax(Q @ K^T / sqrt(d_k)) @ V` as a single Triton kernel. One program instance per query row. Correct but slow — materializes full N×N attention matrix.

### Key Decisions & Learnings
- Two-pass approach: Triton can't store an N-length scratch array, so pass 1 finds max score, pass 2 uses it for stable softmax + output accumulation in one shot
- Element-wise Q*K with `tl.sum` for dot product (no `tl.dot` for vector-vector)
- Float32 accumulators throughout for correctness

### Results
- Max error vs PyTorch SDPA: **3.87e-07** (fp32) — well within tolerance

---

## Phase 2: Benchmark + Roofline Analysis
**Status**: Complete ✓

### Objective
Profile the naive kernel. Compute FLOPS, bytes moved, arithmetic intensity. Place on roofline to prove it's memory-bound.

### Key Decisions & Learnings
- **FLOP counting per query row**: Pass 1 = 2·N·d_k (dot product + max), Pass 2 = 4·N·d_k (dot + exp + weighted V sum). Total per row = 6·N·d_k
- **Byte counting per query row**: Pass 1 loads N K rows (4·N·d_k bytes), Pass 2 loads N K + N V rows (8·N·d_k bytes). Total = 12·N·d_k bytes
- **Arithmetic Intensity** = 6Nd_k / 12Nd_k = **0.5 FLOP/byte** — constant regardless of N
- GPU ridge point = 15 TFLOPS / 0.448 TB/s = **33.5 FLOP/byte** → kernel is 67x below ridge
- Used `torch.cuda.Event(enable_timing=True)` for GPU-side timing (not `time.time()` — kernel launches are async)
- 10 warmup iterations (JIT compilation + GPU clock ramp), 100 timed iterations, median latency
- Points appeared slightly above theoretical roofline at AI=0.5 → L2 cache effect: multiple program instances reading same K/V rows get cache hits, reducing actual bytes from global memory

### Results
- All seq_lens confirmed **memory-bound** (far left of ridge on roofline)
- Larger N achieves slightly higher TFLOPS (better SM occupancy with more program instances)
- See `roofline_naive.png` for plot
- **Conclusion**: Optimization must reduce memory traffic, not compute → motivation for tiled FlashAttention (Phase 3)

---

## Phase 3: Tiled Online Softmax (FlashAttention Core)
**Status**: Not started

### Objective
Eliminate N×N materialization via tiled processing with online softmax (running max, running sum, rescaling). The core algorithmic insight.

### Key Decisions & Learnings

### Results
<!-- Correctness, memory comparison, performance -->

---

## Phase 4: Autotune + Re-profile
**Status**: Not started

### Objective
Use `@triton.autotune` to search over BLOCK_M, BLOCK_N, num_warps, num_stages. Re-benchmark and update roofline.

### Key Decisions & Learnings

### Results
<!-- Winning configs per seq_len, updated roofline -->

---

## Phase 5: Integrate + Final Comparison
**Status**: Not started

### Objective
Compare against PyTorch's `scaled_dot_product_attention` (cuDNN Flash backend), `torch.compile`, and naive Triton kernel. Latency + memory across seq_lens.

### Key Decisions & Learnings

### Results
<!-- Final comparison charts, analysis -->

---

## Key Takeaways
<!-- Fill at the end: 3-5 bullets on GPU performance engineering lessons -->

---

## Resume-Ready Summary
<!-- Distilled version for resume/portfolio use -->
