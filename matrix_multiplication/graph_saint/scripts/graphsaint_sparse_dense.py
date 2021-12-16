from numpy.lib.function_base import diff
import numpy as np
import sys
from scipy import sparse
sys.path.append('../../utils')
from util import sparse_dense_multiplication

def main():

    number_of_nodes = int(sys.argv[1])
    dataset = sys.argv[2]
    numba = sys.argv[3] == 'True'
    print("number_of_nodes: ", number_of_nodes)
    print("dataset: ", dataset)
    print("run with numba: ", numba)

    # matrix1 - sparse matrix
    # matrix2 - dense matrix

    adj = sparse.load_npz('../../../../data/' + dataset + '/' + dataset + '_adj.npz')

    sparse_array = adj+adj.T
    sparse_array = sparse_array[:, :number_of_nodes][:number_of_nodes,:]

    if dataset == 'yelp':
        # dense_array = np.random.rand(number_of_nodes, 16)
        dense_array = np.random.rand(number_of_nodes, 1024)
    else:
        dense_array = np.random.rand(number_of_nodes, 128)

    print("sparse array shape: ", sparse_array.shape)
    print("dense array shape: ", dense_array.shape)

    matrix1 = sparse_array.todense()
    matrix2 = np.array(dense_array)
    res = np.matmul(matrix1, matrix2)


    first_dimension = sparse_array.get_shape()[0]
    second_dimension = len(dense_array[0])
    if not numba:
        result_matrix = np.zeros((first_dimension, second_dimension))
        result_matrix = sparse_dense_multiplication(result_matrix, first_dimension, second_dimension, sparse_array, dense_array)
        np.testing.assert_almost_equal(result_matrix, res)


    if numba:
        output = True
        result_matrix = np.zeros((first_dimension, second_dimension))
        result_matrix = sparse_dense_multiplication(result_matrix, first_dimension, second_dimension, sparse_array, dense_array, numba, output)
        np.testing.assert_almost_equal(result_matrix, res)
        output = False
        sparse_dense_multiplication(result_matrix, first_dimension, second_dimension, sparse_array, dense_array, numba, output)

        output = True
        parallel = True
        result_matrix = np.zeros((first_dimension, second_dimension))
        result_matrix = sparse_dense_multiplication(result_matrix, first_dimension, second_dimension, sparse_array, dense_array, numba, output, parallel)
        np.testing.assert_almost_equal(result_matrix, res)
        output = False
        sparse_dense_multiplication(result_matrix, first_dimension, second_dimension, sparse_array, dense_array, numba, output, parallel)

    return 

if __name__ == "__main__":
    main()