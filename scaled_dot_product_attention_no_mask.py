
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


def matmul(a, b):
    """
    Matrix Multiplication (manual implementation).

    Args:
        a (List[List[float]]): Left matrix.
        b (List[List[float]]): Right matrix.

    Returns:
        List[List[float]]: Resulting matrix after multiplication.

    Note:
        This function avoids using numpy for educational clarity.
    """
    b_transposed = list(zip(*b))  # Transpose b for easier column access
    result = []
    for a_row in a:
        new_row = []
        for b_col in b_transposed:
            dot_product = sum(ai * bj for ai, bj in zip(a_row, b_col))
            new_row.append(dot_product)
        result.append(new_row)
    return result


def transpose(matrix):
    """
    Transpose a matrix (swap rows and columns).
    """
    return [list(row) for row in zip(*matrix)]


def softmax(x):
    """
    Numerically stable softmax function.
    Args:
        x (List[float]): Input vector.
    Returns:
        List[float]: Softmax probabilities.
    """
    max_val = max(x)  # For numerical stability
    exps = [math.exp(i - max_val) for i in x]
    sum_exps = sum(exps)
    return [j / sum_exps for j in exps]


def scaled_dot_product_attention(q, k, v):
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
    # Step 3: Apply softmax to get attention weights
    attn_weights = [softmax(row) for row in s_attn]
    # Step 4: Multiply attention weights by values to get output
    output = matmul(attn_weights, v)
    return output, attn_weights


if __name__ == '__main__':
    # Example usage: Small matrices for demonstration

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
    W_K = [
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

    # Step 2: Run scaled dot-product attention
    output, attn_weights = scaled_dot_product_attention(Q, K, V)

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