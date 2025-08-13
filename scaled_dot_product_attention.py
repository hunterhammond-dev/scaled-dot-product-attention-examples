
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


if __name__ == '__main__':
    try:
        # Input sequence (each row is a token embedding)
        input_seq = [
            [0.375, 0.951, 0.732],
            [0.599, 0.156, 0.058],
            [0.866, 0.601, 0.708],
            [0.021, 0.832, 0.212],
        ]

        # Weight matrices for Q, K, V (for standalone attention)
        W_Q = [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [1.0, 0.0],
        ]
        W_K= [
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
        ]
        W_V = [
            [1.0, 1.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]

        # Step 1: Compute Q, K, V matrices
        Q = matmul(input_seq, W_Q)
        K = matmul(input_seq, W_K)
        V = matmul(input_seq, W_V)

        mask = [
            [1, 0, 1],
            [1, 1, 1],
            [1, 0, 1],
        ]

        # Step 2: Run scaled dot-product attention
        output, attn_weights = scaled_dot_product_attention(Q, K, V, mask)

        # Step 3: Display results for inspection
        print("Queries (Q):")
        for q in Q:
            print(q)
        print("\nKeys (K):")
        for k in K:
            print(k)
        print("\nValues (V):")
        for v in V:
            print(v)
        print("\nAttention Weights (Softmax Scores):")
        for row in attn_weights:
            print(["{:.3f}".format(x) for x in row])
        print("\nAttention Output:")
        for o in output:
            print(["{:.3f}".format(x) for x in o])
    except Exception as e:
        print("An error occurred:", e)