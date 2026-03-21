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
**Status**: Not started

### Objective
Implement `O = softmax(Q @ K^T / sqrt(d_k)) @ V` as a single Triton kernel. One program instance per query row. Correct but slow — materializes full N×N attention matrix.

### Key Decisions & Learnings
<!-- Add entries as you progress -->

### Results
<!-- Correctness checks, initial timing -->

---

## Phase 2: Benchmark + Roofline Analysis
**Status**: Not started

### Objective
Profile the naive kernel. Compute FLOPS, bytes moved, arithmetic intensity. Place on roofline to prove it's memory-bound.

### Key Decisions & Learnings

### Results
<!-- Roofline plot, benchmark numbers -->

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
