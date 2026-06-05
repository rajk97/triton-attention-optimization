"""
Debug runner for triton_tiled.py using Triton's interpreter mode.

How it works:
- TRITON_INTERPRET=1 makes @triton.jit kernels execute in pure Python on the CPU
  instead of compiling to GPU. This lets you use breakpoint(), print(), pdb, etc.
  INSIDE the kernel.
- Must be set BEFORE `import triton` anywhere in the process.

Usage:
    python src/debug_tiled.py

Tips:
- Use tiny N, d_k, BLOCK_M, BLOCK_N so prints/breakpoints don't explode.
- Add `breakpoint()` inside the kernel (in triton_tiled.py) where you want to
  inspect state. You can then print Q_block, S, m_i, l_i, etc. as numpy arrays.
- Each program_id iteration runs sequentially in interpreter mode.
"""
import os

# MUST be set before importing triton anywhere in the process tree.
os.environ["TRITON_INTERPRET"] = "1"

import torch
from triton_tiled import tiled_attention


def main():
    torch.manual_seed(0)
    # Tiny sizes so interpreter mode is fast and prints are readable.
    N = 8
    d_k = 4
    BLOCK_M = 4
    BLOCK_N = 4

    # Interpreter mode runs on CPU. Use CPU tensors.
    device = "cpu"

    Q = torch.randn((N, d_k), device=device, dtype=torch.float32)
    K = torch.randn((N, d_k), device=device, dtype=torch.float32)
    V = torch.randn((N, d_k), device=device, dtype=torch.float32)

    print(f"Running tiled_attention in INTERPRET mode: N={N}, d_k={d_k}, "
          f"BLOCK_M={BLOCK_M}, BLOCK_N={BLOCK_N}")
    print("(Add breakpoint() inside the kernel to step through.)")

    O = tiled_attention(Q, K, V, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
    O_ref = torch.nn.functional.scaled_dot_product_attention(Q, K, V)

    max_err = (O - O_ref).abs().max().item()
    print(f"\nMax error vs PyTorch SDPA: {max_err:.3e}")
    print("Q:\n", Q)
    print("O (ours):\n", O)
    print("O_ref:\n", O_ref)


if __name__ == "__main__":
    main()
