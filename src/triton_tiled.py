"""
Tiled FlashAttention-style kernel.

KEY IDEA vs naive:
- Naive: 1 program = 1 query row, loops over ALL N keys twice
- Tiled: 1 program = BLOCK_M query rows, loops over K/V in tiles of BLOCK_N
- Uses ONLINE SOFTMAX to avoid the two-pass problem

ONLINE SOFTMAX UPDATE (per K/V tile):
    For each tile of BLOCK_N keys:
        S     = Q_block @ K_tile^T / sqrt(d_k)           # [BLOCK_M, BLOCK_N]
        m_new = max(m_old, rowmax(S))                    # [BLOCK_M]
        alpha = exp(m_old - m_new)                       # correction factor
        P     = exp(S - m_new[:, None])                  # softmax numerators
        l_new = alpha * l_old + rowsum(P)                # denominator
        O_new = alpha[:, None] * O_old + P @ V_tile      # weighted values
    After all tiles:
        O = O / l[:, None]
"""
import torch
import triton
import triton.language as tl


@triton.jit
def tiled_attention_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    N,
    stride_qm, stride_kn, stride_vn, stride_om,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    d_k: tl.constexpr,
):
    # Which BLOCK_M-sized chunk of queries am I handling?
    pid = tl.program_id(0)

    # Row indices for this block's queries: [BLOCK_M]
    m_offsets = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    # Column indices for d_k: [d_k]
    d_offsets = tl.arange(0, d_k)

    # ---------- Load Q block: [BLOCK_M, d_k] ----------
    # Q_ptr[m, d] = Q_ptr + m*stride_qm + d
    q_ptrs = Q_ptr + m_offsets[:, None] * stride_qm + d_offsets[None, :]
    q_mask = m_offsets[:, None] < N
    Q_block = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # ---------- Initialize running state ----------
    # m_i: running max per query row; start at -inf
    m_i = tl.full((BLOCK_M,), float('-inf'), dtype=tl.float32)
    # l_i: running denominator sum; start at 0
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    # O_block: running weighted sum; start at 0
    O_block = tl.zeros((BLOCK_M, d_k), dtype=tl.float32)

    scale = 1.0 / (d_k ** 0.5)

    # ---------- Loop over K/V tiles ----------
    for start_n in range(0, N, BLOCK_N):
        n_offsets = start_n + tl.arange(0, BLOCK_N)

        # Load K tile: [BLOCK_N, d_k]
        k_ptrs = K_ptr + n_offsets[:, None] * stride_kn + d_offsets[None, :]
        k_mask = n_offsets[:, None] < N
        K_tile = tl.load(k_ptrs, mask=k_mask, other=0.0)

        # Load V tile: [BLOCK_N, d_k]
        v_ptrs = V_ptr + n_offsets[:, None] * stride_vn + d_offsets[None, :]
        V_tile = tl.load(v_ptrs, mask=k_mask, other=0.0)

        # ---------- FILL IN: Online softmax update ----------
        # BLANK A: Compute scores S = Q_block @ K_tile^T * scale
        #   Shape: [BLOCK_M, BLOCK_N]
        #   Hint: tl.dot(Q_block, tl.trans(K_tile)) * scale
        S = ???

        # BLANK B: Mask out-of-bounds keys (set their scores to -inf)
        #   so they don't affect the max or contribute to softmax
        S = tl.where(n_offsets[None, :] < N, S, float('-inf'))

        # BLANK C: Find this tile's per-row max
        #   Shape: [BLOCK_M]
        m_tile = ???

        # BLANK D: Update running max
        m_new = ???

        # BLANK E: Compute correction factor alpha = exp(m_old - m_new)
        #   Shape: [BLOCK_M]
        alpha = ???

        # BLANK F: Compute P = exp(S - m_new[:, None])
        #   Shape: [BLOCK_M, BLOCK_N]
        P = ???

        # BLANK G: Update running denominator l
        #   l_new = alpha * l_old + sum(P, axis=1)
        l_i = ???

        # BLANK H: Update running output O
        #   O_new = alpha[:, None] * O_old + P @ V_tile
        #   Hint: P needs to be cast for tl.dot; use P.to(V_tile.dtype) if needed
        O_block = ???

        # Shift: m_old becomes m_new for next iteration
        m_i = m_new

    # ---------- Final normalization ----------
    O_block = O_block / l_i[:, None]

    # ---------- Store output ----------
    o_ptrs = O_ptr + m_offsets[:, None] * stride_om + d_offsets[None, :]
    tl.store(o_ptrs, O_block, mask=q_mask)


def tiled_attention(Q, K, V, BLOCK_M=64, BLOCK_N=64):
    N, d_k = Q.shape
    O = torch.empty(N, d_k, device=Q.device, dtype=torch.float32)
    grid = (triton.cdiv(N, BLOCK_M),)
    tiled_attention_kernel[grid](
        Q, K, V, O,
        N,
        Q.stride(0), K.stride(0), V.stride(0), O.stride(0),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        d_k=d_k,
    )
    return O


if __name__ == "__main__":
    torch.manual_seed(0)
    N = 1024
    d_k = 64

    Q = torch.randn((N, d_k), device='cuda', dtype=torch.float32)
    K = torch.randn((N, d_k), device='cuda', dtype=torch.float32)
    V = torch.randn((N, d_k), device='cuda', dtype=torch.float32)

    O = tiled_attention(Q, K, V)
    O_ref = torch.nn.functional.scaled_dot_product_attention(Q, K, V)

    max_err = (O - O_ref).abs().max().item()
    print(f"Max error vs PyTorch SDPA: {max_err:.3e}")
    assert max_err < 1e-4, f"Correctness FAILED: max error {max_err}"
    print("PASSED ✓")
