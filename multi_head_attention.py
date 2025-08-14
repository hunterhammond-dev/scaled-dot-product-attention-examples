
"""
Scaled Dot-Product Attention Example (No Mask)

This script demonstrates a basic, dependency-free implementation of the Scaled Dot-Product Attention mechanism,
as described in Figure 2 (left) of the "Attention Is All You Need" paper.

Key Features:
- Pure Python: No external libraries required.
- Step-by-step breakdown of matrix operations for clarity.
- Well-commented for educational use and easy understanding.

Author: Hunter L. Hammond
Date: 08/12/2025
"""

import math
from utils.attention_utils import matmul, transpose, softmax

def split_heads(x, num_heads):
    """
    Split input matrix into multiple heads.

    Args:
        x (List[List[float]]): Input matrix (e.g., Q, K, V)
        num_heads (int): Number of attention heads

    Returns:
        List[List[List[float]]]: List of matrices per head
    """
    head_dim = len(x[0]) // num_heads
    return [
        [[row[i + h * head_dim] for i in range(head_dim)] for row in x]
        for h in range(num_heads)
    ]

def merge_heads(outputs):
    """
    Merge multiple attention head outputs.

    Args:
        outputs (List[List[List[float]]]): List of output matrices per head

    Returns:
        List[List[float]]: Concatenated output
    """
    merged = []
    for tokens in zip(*outputs):  # zip across tokens
        merged.append([val for head in tokens for val in head])
    return merged

def apply_mask(attention, mask):
    masked_value = -1e9

    masked_attention_scores = []
    for attention_score_row, mask_row in zip(attention, mask):
        masked_row = [
            attention_score if keep else masked_value
            for attention_score, keep in zip(attention_score_row, mask_row)
        ]
        masked_attention_scores.append(masked_row)

    return masked_attention_scores

def scaled_dot_product_attention(q, k, v, mask=None):
    """
    Compute Scaled Dot-Product Attention.

    Args:
        q (List[List[float]]): Query matrix
        k (List[List[float]]): Key matrix
        v (List[List[float]]): Value matrix

    Returns:
        output (List[List[float]]): Attention output
        attn_weights (List[List[float]]): Attention weights (softmax scores)
    """
    d_k = len(k[0])  # Dimensionality of key vectors
    t_k = transpose(k)

    # Step 1: Calculate raw attention scores (dot product of Q and K^T)
    raw_attn = matmul(q, t_k)

    # Step 2: Scale the attention scores
    scaling = math.sqrt(d_k)
    s_attn = [[score / scaling for score in row] for row in raw_attn]

    # Step 3: Apply mask
    if mask is not None:
        s_attn = apply_mask(s_attn, mask)

    # Step 4: Apply softmax to get attention weights
    attn_weights = [softmax(row) for row in s_attn]

    # Step 5: Multiply attention weights by values to get output
    output = matmul(attn_weights, v)
    return output, attn_weights

def multi_head_attention(q, k, v, num_heads=2, mask=None):
    """
    Compute Multi-Head Attention.

    Args:
        q, k, v: Input matrices
        num_heads: Number of attention heads
        mask: Optional mask

    Returns:
        output: Final concatenated attention output
        all_weights: List of attention weights per head
    """
    q_heads = split_heads(q, num_heads)
    k_heads = split_heads(k, num_heads)
    v_heads = split_heads(v, num_heads)

    head_outputs = []
    all_weights = []

    for Q, K, V in zip(q_heads, k_heads, v_heads):
        out, weights = scaled_dot_product_attention(Q, K, V, mask)
        head_outputs.append(out)
        all_weights.append(weights)

    final_output = merge_heads(head_outputs)
    return final_output, all_weights

if __name__ == '__main__':
    try:
        # Input sequence (each row is a token embedding)
        input_seq = [
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
            [0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4],
            [0.4, 0.6, 0.2, 0.8, 0.1, 0.9, 0.3, 0.7],
        ]

        # Weight matrices for Q, K, V (for standalone attention)
        W_Q = [
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
        ]

        W_K = [
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
        ]

        W_V = [
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
        ]

        # Step 1: Compute Q, K, V matrices
        Q = matmul(input_seq, W_Q)
        K = matmul(input_seq, W_K)
        V = matmul(input_seq, W_V)

        mask = [
            [1, 0, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 1, 0],
            [1, 1, 1, 1],
        ]

        # Step 2: Run multi-head attention
        output, attn_weights = multi_head_attention(Q, K, V, num_heads=2, mask=mask)

        # Step 3: Display results
        print("\nInput Tokens:")
        for o in input_seq:
            print(["{:.3f}".format(x) for x in o])

        print("\nMulti-Head Attention Output:")
        for o in output:
            print(["{:.3f}".format(x) for x in o])

        print("\nAttention Weights Per Head (with input tokens):")
        for head_idx, head in enumerate(attn_weights):
            print(f"\nHead {head_idx + 1}:")
            for token_idx, attn_row in enumerate(head):
                token_embedding = input_seq[token_idx]
                print(
                    f"{['{:.3f}'.format(x) for x in attn_row]}")
    except Exception as e:
        print("An error occurred:", e)