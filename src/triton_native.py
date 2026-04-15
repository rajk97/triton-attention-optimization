import torch
import triton
import triton.language as tl

@triton.jit
def naive_attention_kernel(Q_ptr, K_ptr, V_ptr, O_ptr, N, stride_qn, stride_kn, stride_vn, stride_on, d_k: tl.constexpr):

    # Step 1: Which row am I?
    row_i = tl.program_id(0)

    # Create the column offsets to load the entire row at once
    col_offsets = tl.arange(0, d_k)
    
    # Step 2: Load Q[row_i, :] — a vector of length d_k
    # ... you fill this in
    Q_row = tl.load(Q_ptr + row_i * stride_qn + col_offsets, mask = col_offsets < d_k, other=0.0)

    # Two phase approact to avoid indexing as it is not possible in Triton 
    # Pass 1: Find max score across all the elements in the row
    max_score = float('-inf')
    for j in range(N):
        K_row = tl.load(K_ptr + j*stride_kn + col_offsets, mask = col_offsets<d_k, other = 0.0)
        s = tl.sum(Q_row*K_row, axis=0)/(d_k ** 0.5)
        max_score = tl.maximum(max_score, s)
    
    # Pass 2: compute exp, sum, weighted V - all in one pass directly 
    exp_sum = 0.0
    O_row = tl.zeros((d_k,), dtype=tl.float32)

    for j in range(N):
        
        K_row = tl.load(K_ptr + j*stride_kn + col_offsets, mask = col_offsets<d_k, other = 0.0)
        s = tl.sum(Q_row*K_row, axis=0)/(d_k ** 0.5)
        exp_s = tl.exp(s - max_score)
        exp_sum += exp_s
        V_row = tl.load(V_ptr + j*stride_vn + col_offsets, mask = col_offsets<d_k, other = 0.0)
        O_row += exp_s * V_row

    O_row = O_row/(exp_sum)
    tl.store(O_ptr+row_i*stride_on+col_offsets, O_row, mask = col_offsets<d_k)        
    
    return

def naive_attention(Q, K, V):

    N, d_k = Q.shape
    O = torch.empty(N, d_k, device=Q.device, dtype = torch.float32)
    grid = (N,)
    naive_attention_kernel[grid](Q, K, V, O, N, Q.stride(0), K.stride(0), V.stride(0), O.stride(0), d_k)

    return O

if __name__ == "__main__":

    # Define dimensions 
    N = 1024
    d_k = 64

    Q = torch.randn((N, d_k), device='cuda')
    K = torch.randn((N, d_k), device='cuda')
    V = torch.randn((N, d_k), device='cuda')

    O = naive_attention(Q, K, V)

    # Validate against PyTorch 
    O_ref = torch.nn.functional.scaled_dot_product_attention(Q, K, V)
    print(f"Max error: {(O-O_ref).abs().max().item()}")
