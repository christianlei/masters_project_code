import numpy as np
from numba import njit, prange, typed
import time

def sparse_dense_multiplication(result_matrix, first_dimension, second_dimension, matrix1, matrix2, numba=False, output=True, parallel=False): # result_matrix, first_dimension, second_dimension,
    # first_dimension = matrix1.get_shape()[0]
    # second_dimension = len(matrix2[0])
    # result_matrix = np.zeros((first_dimension, second_dimension))
    value = matrix1.data
    column_idx = matrix1.indices
    ind_ptr = matrix1.indptr

    if output and numba and parallel:
        return sparse_dense_multiplication_operation_numba_parallel(result_matrix, first_dimension, second_dimension, value, column_idx, ind_ptr, matrix2)
    if output and numba and not parallel:
        return sparse_dense_multiplication_operation_numba(result_matrix, first_dimension, second_dimension, value, column_idx, ind_ptr, matrix2)
    if output and not numba:
        start = time.time()
        return_matrix = sparse_dense_multiplication_operation(result_matrix, first_dimension, second_dimension, value, column_idx, ind_ptr, matrix2)
        end = time.time()
        print("Total Time: ", end - start)
        return return_matrix
    if numba and parallel:
        start = time.time()
        sparse_dense_multiplication_operation_numba_parallel(result_matrix, first_dimension, second_dimension, value, column_idx, ind_ptr, matrix2)
        end = time.time()
        print("Parallel Numba Total Time: ", end - start)
        return
    if numba:
        start = time.time()
        sparse_dense_multiplication_operation_numba(result_matrix, first_dimension, second_dimension, value, column_idx, ind_ptr, matrix2)
        end = time.time()
        print("Numba Total Time: ", end - start)
        return
    

def sparse_dense_multiplication_manual_allocation(result_matrix, second_dimension, matrix1, matrix2, allocation_tuples, output=True):
    value = matrix1.data
    column_idx = matrix1.indices
    ind_ptr = matrix1.indptr

    allocation_tuples = typed.List(allocation_tuples)
    
    if output:
        return sparse_dense_multiplication_operation_numba_parallel_allocated_manual(result_matrix, second_dimension, value, column_idx, ind_ptr, matrix2, allocation_tuples)
    else:
        start = time.time()
        sparse_dense_multiplication_operation_numba_parallel_allocated_manual(result_matrix, second_dimension, value, column_idx, ind_ptr, matrix2, allocation_tuples)
        end = time.time()
        print("Manual Allocaition Numba Total Time: ", end - start)
    return


@njit
def sparse_dense_multiplication_operation_numba(result_matrix, first_dimension, second_dimension, value, column_idx, ind_ptr, matrix2):
    for i in range(first_dimension):
        row = ind_ptr[i]
        for j in range(second_dimension):
            tmp = 0.0
            for k in range(ind_ptr[i + 1] - ind_ptr[i]):
                    tmp += value[row + k] * matrix2[column_idx[row + k]][j]
                    result_matrix[i][j] = tmp
    return result_matrix

@njit(parallel=True)
def sparse_dense_multiplication_operation_numba_parallel(result_matrix, first_dimension, second_dimension, value, column_idx, ind_ptr, matrix2):
    for i in prange(first_dimension):
        row = ind_ptr[i]
        for j in range(second_dimension):
                tmp = 0.0
                for k in range(ind_ptr[i + 1] - ind_ptr[i]):
                    tmp += value[row + k] * matrix2[column_idx[row + k]][j]
                result_matrix[i][j]	= tmp
    return result_matrix

@njit(parallel=True)
def sparse_dense_multiplication_operation_numba_parallel_allocated_manual(result_matrix, second_dimension, value, column_idx, ind_ptr, matrix2, allocation_tuples):
    num_threads = len(allocation_tuples)
    for tid in prange(num_threads): #launch 8 threads
        lower_bound = allocation_tuples[tid][0]
        upper_bound = allocation_tuples[tid][1]
        for i in range(lower_bound, upper_bound): 
            row = ind_ptr[i]
            for j in range(second_dimension):
                tmp = 0.0
                for k in range(ind_ptr[i + 1] - ind_ptr[i]):
                    tmp += value[row + k] * matrix2[column_idx[row + k]][j]
                result_matrix[i][j]	= tmp
    return result_matrix

def sparse_dense_multiplication_operation(result_matrix, first_dimension, second_dimension, value, column_idx, ind_ptr, matrix2):
    for i in range(first_dimension):
        for k in range(ind_ptr[i + 1] - ind_ptr[i]):
            row = ind_ptr[i]
            for j in range(second_dimension):
                result_matrix[i][j] += value[row + k] * matrix2[column_idx[row + k]][j]
    return result_matrix