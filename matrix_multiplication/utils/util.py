import numpy as np
from numba import njit, prange, typed, cuda, float32
import time
import math

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
        time_list = []
        for _ in range(10):
            start = time.time()
            sparse_dense_multiplication_operation_numba_parallel(result_matrix, first_dimension, second_dimension, value, column_idx, ind_ptr, matrix2)
            end = time.time()
            time_list.append(end-start)
        print("Parallel Numba Total Time: ", sum(time_list)/len(time_list))
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
        return sparse_dense_multiplication_operation_numba_parallel_allocated_manual_2(result_matrix, second_dimension, value, column_idx, ind_ptr, matrix2, allocation_tuples)
    else:
        time_list = []
        for _ in range(10):
            start = time.time()
            sparse_dense_multiplication_operation_numba_parallel_allocated_manual_2(result_matrix, second_dimension, value, column_idx, ind_ptr, matrix2, allocation_tuples)
            end = time.time()
            time_list.append(end-start)
        avg = sum(time_list)/len(time_list)
        print("Manual Allocation Numba Total Time: ", avg)
    return


def sparse_dense_multiplication_cuda(result_matrix, first_dimension, second_dimension, matrix1, matrix2, output=True):
    value = matrix1.data
    column_idx = matrix1.indices
    ind_ptr = matrix1.indptr

    TPB = 32

    threadsperblock = (TPB, TPB)
    blockspergrid_x = int(math.ceil(first_dimension / threadsperblock[0]))
    blockspergrid_y = int(math.ceil(second_dimension / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    
    mat2_global_mem = cuda.to_device(matrix2)
    value_global_mem = cuda.to_device(value)
    ind_ptr_global_mem = cuda.to_device(ind_ptr)
    column_idx_global_mem = cuda.to_device(column_idx)
    result_global_mem = cuda.device_array((first_dimension, second_dimension))

    if output:
        sparse_dense_multiplication_numba_parallel_cuda[blockspergrid, threadsperblock](result_global_mem, first_dimension, second_dimension, value_global_mem, column_idx_global_mem, ind_ptr_global_mem, mat2_global_mem)
        result_matrix = result_global_mem.copy_to_host()
        return result_matrix
    else:
        start = time.time()
        sparse_dense_multiplication_numba_parallel_cuda[blockspergrid, threadsperblock](result_global_mem, first_dimension, second_dimension, value_global_mem, column_idx_global_mem, ind_ptr_global_mem, mat2_global_mem)
        end = time.time()
        print("Cuda Numba Total Time: ", end - start)
    return

@cuda.jit
def sparse_dense_multiplication_numba_parallel_cuda(result_matrix, first_dimension, second_dimension, value, column_idx, ind_ptr, matrix2):
    x, y = cuda.grid(2)

    if x >= result_matrix.shape[0] and y >= result_matrix.shape[1]:
        # Quit if (x, y) is outside of valid C boundary
        return

    if x < first_dimension and y < second_dimension:
        for k in range(ind_ptr[x + 1] - ind_ptr[x]):
            row = ind_ptr[x]
            result_matrix[x][y] += value[row + k] * matrix2[column_idx[row + k]][y]


def sparse_dense_multiplication_operation(result_matrix, first_dimension, second_dimension, value, column_idx, ind_ptr, matrix2):
    for i in range(first_dimension):
        for k in range(ind_ptr[i + 1] - ind_ptr[i]):
            row = ind_ptr[i]
            for j in range(second_dimension):
                result_matrix[i][j] += value[row + k] * matrix2[column_idx[row + k]][j]
    return result_matrix

@njit
def sparse_dense_multiplication_operation_numba(result_matrix, first_dimension, second_dimension, value, column_idx, ind_ptr, matrix2):
    for i in range(first_dimension):
        row = ind_ptr[i]
        for j in range(second_dimension):
            tmp = 0.0
            for k in range(ind_ptr[i + 1] - ind_ptr[i]):
                tmp += value[row + k] * matrix2[column_idx[row + k]][j]
            #add an activation right here (ex. sqrt) to increase the runtime of the kernals, you can also add a bias in right here, too. Might mess up answer, but its ok if correct beforehand.
            result_matrix[i][j] = tmp
    return result_matrix

@njit(parallel=True)
def sparse_dense_multiplication_operation_numba_parallel(result_matrix, first_dimension, second_dimension, value, column_idx, ind_ptr, matrix2):
    for i in prange(first_dimension):
        row = ind_ptr[i]
        for j in range(second_dimension):
                tmp = 0.0
                for k in range(ind_ptr[i + 1] - ind_ptr[i]):
                    # value[row+k] *= matrix2[column_idx[row+k]][j]
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
                    # value[row+k] *= matrix2[column_idx[row+k]][j]
                    tmp += value[row + k] * matrix2[column_idx[row + k]][j]
                result_matrix[i][j]	= tmp
    return result_matrix


@njit(parallel=True)
def sparse_dense_multiplication_operation_numba_parallel_allocated_manual_2(result_matrix, second_dimension, value, column_idx, ind_ptr, matrix2, allocation_tuples):
    num_threads = len(allocation_tuples)
    for tid in prange(num_threads): #launch 8 threads
        lower_bound = allocation_tuples[tid][0]
        upper_bound = allocation_tuples[tid][1]
        for i in range(lower_bound, upper_bound): 
            row = ind_ptr[i]
            for j in range(second_dimension):
                tmp = 0.0
                for k in range(ind_ptr[i + 1] - ind_ptr[i]):
                    # value[row+k] *= matrix2[column_idx[row+k]][j]
                    tmp += value[row + k] * matrix2[j][column_idx[row + k]]
                result_matrix[i][j]	= tmp
    return result_matrix