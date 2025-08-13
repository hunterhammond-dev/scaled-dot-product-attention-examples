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
    
    # Using zip to transpose the matrix
    return [list(row) for row in zip(*matrix)]


def softmax(x):
    """
    Numerically stable softmax function.
    Args:
        x (List[float]): Input vector.
    Returns:
        List[float]: Softmax probabilities.
    """
    # Check if the input is a non-empty list of numbers
    if not x or not all(isinstance(i, (int, float)) for i in x):
        raise ValueError("Input to softmax must be a non-empty list of numbers.")
    
    max_val = max(x)  # For numerical stability
    exps = [math.exp(i - max_val) for i in x]
    sum_exps = sum(exps)
    return [j / sum_exps for j in exps]