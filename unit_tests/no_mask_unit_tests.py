import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scaled_dot_product_attention_no_mask import scaled_dot_product_attention

class TestNoMaskFunctions(unittest.TestCase):

    def test_scaled_dot_product_attention(self):
        q = [[1, 0], [0, 1]]
        k = [[1, 0], [0, 1]]
        v = [[1, 2], [3, 4]]
        output, attn_weights = scaled_dot_product_attention(q, k, v)
        self.assertEqual(len(output), 2)
        self.assertEqual(len(attn_weights), 2)

if __name__ == '__main__':
    unittest.main()