import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scaled_dot_product_attention import scaled_dot_product_attention

class TestMaskedAttention(unittest.TestCase):

    def test_scaled_dot_product_attention_with_mask(self):
        q = [[1, 0], [0, 1]]
        k = [[1, 0], [0, 1]]
        v = [[1, 2], [3, 4]]
        mask = [
            [1, 0],
            [1, 1]
        ]

        output, attn_weights = scaled_dot_product_attention(q, k, v, mask)

        # Basic shape checks
        self.assertEqual(len(output), 2)
        self.assertEqual(len(output[0]), 2)
        self.assertEqual(len(attn_weights), 2)
        self.assertEqual(len(attn_weights[0]), 2)

        # Check that masked values are near zero after softmax
        self.assertAlmostEqual(attn_weights[0][1], 0.0, places=3)

if __name__ == '__main__':
    unittest.main()