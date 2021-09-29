from unittest import result
from numpy.lib.function_base import diff
import numpy as np
import sys
import scipy.sparse as sp
import time
from numba import njit

def sparse_dense_multiplication(matrix1, matrix2):
    first_dimension = matrix1.get_shape()[0]
    second_dimension = len(matrix2[0])
    value = matrix1.data
    column_idx = matrix1.indices
    ind_ptr = matrix1.indptr

    return sparse_dense_multiplication_operation(first_dimension, second_dimension, value, column_idx, ind_ptr, matrix2)

@njit
def sparse_dense_multiplication_operation(first_dimension, second_dimension, value, column_idx, ind_ptr, matrix2):
    result_matrix = np.zeros((first_dimension, second_dimension))
    
    for i in range(first_dimension):
        for k in range(ind_ptr[i + 1] - ind_ptr[i]):
            row = ind_ptr[i]
            for j in range(second_dimension):
                result_matrix[i][j] += value[row + k] * matrix2[column_idx[row + k]][j]
    return result_matrix



def main():

    number_of_nodes = int(sys.argv[1])
    dataset = sys.argv[2]
    print("number_of_nodes: ", number_of_nodes)
    print("dataset: ", dataset)

    # matrix1 - sparse matrix
    # matrix2 - dense matrix

    adj = sp.load_npz('../../../../data/' + dataset + '/' + dataset + '_adj.npz')

    sparse_array = adj+adj.T
    sparse_array = sparse_array[:, :number_of_nodes][:number_of_nodes,:]

    if dataset == 'yelp':
        dense_array = np.random.rand(number_of_nodes, 16)
    else:
        dense_array = np.random.rand(number_of_nodes, 128)

    print("sparse array shape: ", sparse_array.shape)
    print("dense array shape: ", dense_array.shape)

    # matrix1 = sp.rand(20, 20).tocsr()
    # matrix2 = makelist()

    # print("sparse_array_main: ", sparse_array)

    start = time.time()
    result_mat = sparse_dense_multiplication(sparse_array, dense_array)
    end = time.time()
    # print("first run: ", end-start)
    # sparse_dense_multiplication(sparse_array, dense_array)
    # print(result_mat)

    matrix1 = sparse_array.todense()
    matrix2 = np.array(dense_array)
    res = np.matmul(matrix1, matrix2)
    
    np.testing.assert_almost_equal(result_mat, res)

    start = time.time()
    sparse_dense_multiplication(sparse_array, dense_array)
    end = time.time()

    print("Numba time taken: ", end-start)

    return 

if __name__ == "__main__":
    main()