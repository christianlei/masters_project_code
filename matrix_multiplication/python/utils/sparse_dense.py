import scipy.sparse as sp
import random
import numpy as np

def makelist():
    random_int = random.randint(6, 12)
    num_list = []
    for row in range(0, 20):
        row_list = []
        for count in range(0, 20):
            row_list.append(random_int)
        num_list.append(row_list)
    return num_list

def sparse_dense_multiplication(matrix1, matrix2):
    first_dimension = matrix1.get_shape()[0]
    second_dimension = len(matrix2[0])
    
    result_matrix = [[0] * second_dimension for i in range(first_dimension)]

    value = matrix1.data
    column_idx = matrix1.indices
    ind_ptr = matrix1.indptr
    
    for i in range(first_dimension):
        for k in range(ind_ptr[i + 1] - ind_ptr[i]):
            row = ind_ptr[i]
            for j in range(second_dimension):
                result_matrix[i][j] += value[row + k] * matrix2[column_idx[row + k]][j]
    return result_matrix

def main():
    # matrix1 - sparse matrix
    # matrix2 - dense matrix

    matrix1 = sp.rand(20, 20).tocsr()
    matrix2 = makelist()


    result_mat = sparse_dense_multiplication(matrix1, matrix2)

    matrix1 = matrix1.todense()
    matrix2 = np.array(matrix2)
    res = np.matmul(matrix1, matrix2)
    
    np.testing.assert_array_equal(result_mat, res)
    return 

if __name__ == "__main__":
    main()