import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils')))

from attention_utils import matmul, transpose, softmax

class TestAttentionUtilsFunctions(unittest.TestCase):

    def test_matmul(self):
        a = [[1, 2], [3, 4]]
        b = [[5, 6], [7, 8]]
        expected = [[19, 22], [43, 50]]
        self.assertEqual(matmul(a, b), expected)

    def test_transpose(self):
        matrix = [[1, 2], [3, 4]]
        expected = [[1, 3], [2, 4]]
        self.assertEqual(transpose(matrix), expected)

    def test_softmax(self):
        x = [1, 2, 3]
        result = softmax(x)
        self.assertAlmostEqual(sum(result), 1.0)
        self.assertTrue(all(0 <= v <= 1 for v in result))

    def test_softmax_error(self):
        with self.assertRaises(ValueError):
            softmax([])

if __name__ == '__main__':
    unittest.main()