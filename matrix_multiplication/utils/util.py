import numpy as np
from numba import njit
import time

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