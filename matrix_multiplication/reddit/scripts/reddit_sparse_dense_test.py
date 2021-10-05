import scipy.sparse as sp
import numpy as np
import sys
from numba import njit
import time

def loadRedditFromNPZ(dataset_dir):
    adj = sp.load_npz(dataset_dir+"reddit_adj.npz")
    data = np.load(dataset_dir+"reddit.npz")
    return adj, data['feats'], data['y_train'], data['y_val'], data['y_test'], data['train_index'], data['val_index'], data['test_index']

def sparse_dense_multiplication(matrix1, matrix2, numba=False, output=True): # result_matrix, first_dimension, second_dimension,
    first_dimension = matrix1.get_shape()[0]
    second_dimension = len(matrix2[0])
    value = matrix1.data
    column_idx = matrix1.indices
    ind_ptr = matrix1.indptr

    if output and numba:
        return sparse_dense_multiplication_operation_numba(first_dimension, second_dimension, value, column_idx, ind_ptr, matrix2)
    if output and not numba:
        return sparse_dense_multiplication_operation(first_dimension, second_dimension, value, column_idx, ind_ptr, matrix2)
    if numba:
        start = time.time()
        sparse_dense_multiplication_operation_numba(first_dimension, second_dimension, value, column_idx, ind_ptr, matrix2)
        end = time.time()
        print("Numba Total Time: ", end - start)
    else:
        start = time.time()
        sparse_dense_multiplication_operation(first_dimension, second_dimension, value, column_idx, ind_ptr, matrix2)
        end = time.time()
        print("Total Time: ", end - start)

@njit
def sparse_dense_multiplication_operation_numba(first_dimension, second_dimension, value, column_idx, ind_ptr, matrix2):
    result_matrix = np.zeros((first_dimension, second_dimension))
    for i in range(first_dimension):
        for k in range(ind_ptr[i + 1] - ind_ptr[i]):
            row = ind_ptr[i]
            for j in range(second_dimension):
                result_matrix[i][j] += value[row + k] * matrix2[column_idx[row + k]][j]
    return result_matrix


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
    print("number_of_nodes: ", number_of_nodes)

    # matrix1 - sparse matrix
    # matrix2 - dense matrix

    adj, features, y_train, y_val, y_test, train_index, val_index, test_index = loadRedditFromNPZ("../../../../data/reddit/")

    sparse_array = adj+adj.T
    sparse_array = sparse_array[:, :number_of_nodes][:number_of_nodes,:]

    dense_array = np.random.rand(number_of_nodes, 256)

    print("sparse array shape: ", sparse_array.shape)
    print("dense array shape: ", dense_array.shape)

    # matrix1 = sp.rand(20, 20).tocsr()
    # matrix2 = makelist()

    # checking results are what is expected
    result_mat = sparse_dense_multiplication(sparse_array, dense_array)
    # print(result_mat)

    matrix1 = sparse_array.todense()
    matrix2 = np.array(dense_array)
    res = np.matmul(matrix1, matrix2)
    
    np.testing.assert_almost_equal(result_mat, res)

    # timing the numba implementation
    print("Not Numba: ")
    sparse_dense_multiplication(sparse_array, dense_array, False, False)

    print("Numba: ")
    sparse_dense_multiplication(sparse_array, dense_array, True, False)
    return 

if __name__ == "__main__":
    main()