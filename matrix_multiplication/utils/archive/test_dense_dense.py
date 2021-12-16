import unittest
import dense_dense
import numpy as np

class TestDenseDenseMultiplication(unittest.TestCase):

    def test_dense_dense(self):
        mat1 = [[3,2,1], [3,2,1]]
        mat2 = [[3], [2], [1]]
        result = dense_dense.dense_dense_multiplication(mat1, mat2)
        np_result = np.matmul(mat1, mat2)
        print(result)
        print(list(np_result))
        self.assertEqual(result, np_result)


if __name__ == '__main__':
    unittest.main()