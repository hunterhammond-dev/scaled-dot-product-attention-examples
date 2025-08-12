# Scaled Dot-Product Attention (Vanilla Python)

A minimal, dependency-free implementation of Scaled Dot-Product Attention, inspired by Figure 2 (left) from the seminal paper “Attention Is All You Need”. This script is designed for educational clarity and reproducibility, using only Python’s built-in features.

## Goals
- Implement the core logic of Scaled Dot-Product Attention without external libraries.
- Showcase matrix operations (multiplication, transpose, softmax) from scratch.
- Provide a readable and transparent walkthrough of attention mechanics.

## Features
- Pure Python (no NumPy, no PyTorch)
- Custom matrix multiplication and softmax
- Step-by-step computation of attention scores
- Sample input and weight matrices for demonstration

## File Structure
attention.py         - Main script with attention implementation

## How It Works
The script follows these steps:

- **Matrix Multiplication:** Implements matmul(a, b) for dot products.

 - **Softmax Function:** Normalizes attention scores.

 - **Scaled Dot-Product Attention:**

   - Computes raw attention scores:![Equation](https://latex.codecogs.com/svg.image?&space;Q\cdot&space;K^{T})

   - Scales by ![Equation](https://latex.codecogs.com/svg.image?\sqrt{d_{k}})

   - Applies softmax to get attention weights

   - Multiplies weights with values:![Equation](https://latex.codecogs.com/svg.image?softmax\left(QK^{T}/\sqrt{d_{k}}\right)\cdot&space;V)

Example Output
```
Queries (Q):
[0.951, 1.683]
[0.156, 0.214]
...

Attention Weights:
['0.266', '0.240', '0.248', '0.246']
...

Attention Output:
['0.951', '1.317']
```

## Author
Hunter L. Hammond 📅 August 12, 2025 🔗 LinkedIn (optional) 📧 hunter@example.com (optional)

## Reference
Vaswani et al., Attention Is All You Need, 2017. arXiv:1706.03762
