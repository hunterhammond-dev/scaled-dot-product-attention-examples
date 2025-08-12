"""
Basic Implementation of an Attention Mechanism

Goals
 - Implement Scaled Dot-Product Attention from Figure 2 (left) of Attention is all you need.
 - The script is to be dependency free and only use built-in features

Author: Hunter L. Hammond
Date: 08/12/2025
"""

import math

def matmul(a, b):
    """
    Matrix Multiplication.

    Can use shorthand of matrix multiplication as follows:
        return [[sum(ai * bj for ai, bj in zip(arow, bcol)) for bcol in zip(*b)] for arow in a]

    For readability, I have chosen to show the math in a easier to digest snippet.
    """

    b_transposed = list(zip(*b))

    result = []
    for a_row in a:
        new_row = []
        for b_col in b_transposed:
            dot_product = sum(ai * bj for ai, bj in zip(a_row, b_col))
            new_row.append(dot_product)
        result.append(new_row)

    # Return value: List of List[float], shape (m, n)
    return result

def transpose(matrix):
    return [list(row) for row in zip(*matrix)]

def softmax(x):
    max_val = max(x)
    exps = [math.exp(i - max_val) for i in x]
    sum_exps = sum(exps)
    return [j / sum_exps for j in exps]

def scaled_dot_product_attention(q, k, v):
    d_k = len(k[0]) # First key vector dimensionality

    t_k = transpose(k)
    raw_attn = matmul(Q, t_k) # Multiplying the transposed key and query vectors to get raw attention scores

    scaling = math.sqrt(d_k) # Scaling is applied to normalize the values of the dot product so values are better distributed
    s_attn = [[raw_attn / scaling for raw_attn in row] for row in raw_attn]

    # Applying softmax to the scaled attention scores
    attn_weights = [softmax(row) for row in s_attn]

    # Final multiplication shown in Scaled Dot-Product Attention
    output = matmul(attn_weights, v)

    return output, attn_weights

if __name__ == '__main__':

    # Small numbers and matrices for simple use case
    input_seq = [
        [0.375,0.951,0.732],
        [0.599,0.156,0.058],
        [0.866,0.601,0.708],
        [0.021,0.832,0.212],
    ]

    # Defining weights as we are using as a standalone Attention Mechanism
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

    # Applying weights to inputs
    Q = matmul(input_seq, W_Q)
    K = matmul(input_seq, W_K)
    V = matmul(input_seq, W_V)

    output, attn_weights = scaled_dot_product_attention(Q, K, V)

    # Step 3: Display results
    print("Queries (Q):")
    for q in Q:
        print(q)
    print("\nKeys (K):")
    for k in K:
        print(k)
    print("\nValues (V):")
    for v in V:
        print(v)
    print("\nAttention Weights:")
    for row in attn_weights:
        print(["{:.3f}".format(x) for x in row])
    print("\nAttention Output:")
    for o in output:
        print(["{:.3f}".format(x) for x in o])