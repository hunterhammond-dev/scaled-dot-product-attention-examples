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
[0.732, 1.6829999999999998]
[0.058, 0.214]
[0.708, 1.309]
[0.212, 1.044]

Keys (K):
[0.951, 1.107]
[0.156, 0.657]
[0.601, 1.5739999999999998]
[0.832, 0.23299999999999998]

Values (V):
[1.107, 1.326]
[0.657, 0.755]
[1.5739999999999998, 1.467]
[0.23299999999999998, 0.853]

Attention Weights:
['0.315', '0.122', '0.458', '0.105']
['0.261', '0.236', '0.276', '0.227']
['0.317', '0.140', '0.410', '0.133']
['0.286', '0.182', '0.384', '0.148']

Attention Output:
['1.174', '1.271']
['0.931', '1.123']
['1.119', '1.241']
['1.075', '1.206']
```

## Author
Hunter L. Hammond 📅 August 12, 2025 🔗 [LinkedIn](https://www.linkedin.com/in/hunter-hammond-a4399919a/) 📧 [hunter.hammond.dev@gmail.com](mailto:hunter.hammond.dev@gmail.com)

## Reference
Vaswani et al., Attention Is All You Need, 2017. [Source Link](https://arxiv.org/abs/1706.03762)

Give a ⭐ if you find this helpful!
[![GitHub stars](https://img.shields.io/github/stars/hunterhammond-dev/attention-mechanisms-in-transformers.svg?style=social)](https://github.com/hunterhammond-dev/attention-mechanisms-in-transformers)
