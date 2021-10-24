from numba import njit, prange
import numpy as np
import time

@njit(parallel=True)
def vector_addition(result_matrix, matrix1, matrix2):
    for i in prange(len(matrix1)):
        result_matrix[i] = (matrix1[i] + matrix2[i])


def main():
    matrix1 = np.random.rand(1024,1)
    matrix2 = np.random.rand(1024,1)
    result_matrix = np.zeros((1024, 1))
    vector_addition(result_matrix, matrix1, matrix2)
    start = time.time()
    vector_addition(result_matrix, matrix1, matrix2)
    end = time.time()
    total_time = end - start 
    print("Python Total Time: ", total_time)

if __name__ == "__main__":
    main()