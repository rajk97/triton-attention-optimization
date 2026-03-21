import torch 

def attention(K, Q, V):

    d_k = K.size(-1)
    output = torch.matmul(torch.softmax(((torch.matmul(Q, K.transpose(-2, -1)))/d_k**0.5), dim=-1), V)
    return output


if __name__ == "__main__":

    Q = torch.randn(1024, 64)
    K = torch.randn(1024, 64)
    V = torch.randn(1024, 64)
    O = attention(K, Q, V)

    ideal_output = torch.nn.functional.scaled_dot_product_attention(Q, K, V)
    
    # Compare the two outputs by getting some element wise error metric, for example mean squared error
    mse = torch.mean((O - ideal_output) ** 2)
    print(f'Mean Squared Error between custom attention and PyTorch implementation: {mse.item()}')

