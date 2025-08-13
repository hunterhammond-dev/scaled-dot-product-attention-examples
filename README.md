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

## When to Use Masking
Masking in attention isn’t random, it’s a deliberate part of model design. It controls what each token is allowed to “see” during training, which is especially important in tasks like language modeling where future tokens need to be hidden to prevent cheating. But it goes deeper than that.

Masking is a way to enforce structure. It’s not just about removing noise it’s about removing known distractions. You’re telling the model: “Don’t even consider these positions. They’re not valid in this context.”

This repo includes two versions of the attention implementation:
| File   | Description   | Use Case   |
|------------|------------|------------|
| ```scaled_dot_product_attention.py``` | Includes support for masking. Applies a mask before softmax to zero out invalid positions. | A simpler version without masking. Computes attention across all positions. |
| ```scaled_dot_product_attention_no_mask.py``` | A simpler version without masking. Computes attention across all positions. | Use this for encoder blocks, non-causal tasks, or when masking isn’t needed.|

### What I Learned
Masking isn’t just a technical trick, it’s a behavioral constraint that shapes how the model learns. Without it, attention can get spread across irrelevant tokens, or worse, future tokens that break the training objective. By applying a mask (usually a matrix of large negative values), we zero out those positions in softmax and force the model to focus only where it should.

One thing that clicked for me: masking isn’t about removing data, it’s about controlling influence. If a token like "The" is being over-attended and skewing results, you don’t have to delete it from your dataset. You can mask it in specific contexts to suppress its impact while still letting the model learn from its presence. That’s a cleaner, more targeted way to guide behavior.

If you're working on autoregressive generation, decoder blocks, or anything where attention needs to be restricted, masking is essential. For encoder-only setups or educational demos, the no-mask version keeps things simple and transparent.

## Author
Hunter L. Hammond 📅 August 12, 2025 🔗 [LinkedIn](https://www.linkedin.com/in/hunter-hammond-a4399919a/) 📧 [hunter.hammond.dev@gmail.com](mailto:hunter.hammond.dev@gmail.com)

## Reference
Vaswani et al., Attention Is All You Need, 2017. [Source Link](https://arxiv.org/abs/1706.03762)

Give a ⭐ if you find this helpful!
[![GitHub stars](https://img.shields.io/github/stars/hunterhammond-dev/attention-mechanisms-in-transformers.svg?style=social)](https://github.com/hunterhammond-dev/attention-mechanisms-in-transformers)
