import scipy.sparse as sp
import random
import numpy as np

def sparse_sparse_multiplication(matrix1, matrix2):
    first_dimension = matrix1.get_shape()[0]
    # second_dimension = matrix2.get_shape()[1]
    second_dimension = len(matrix2[0])
    
    result_matrix = [[0] * second_dimension for i in range(first_dimension)]
    # print("result_matrix", result_matrix)
    # print("result_matrix.shape", np.array(result_matrix).shape)

    value = matrix1.data
    column_idx = matrix1.indices
    ind_ptr = matrix1.indptr
    
    # print("value", value)
    # print("column_idx", column_idx)
    # print("ind_ptr", ind_ptr)
    
    row = 0
    for i in range(first_dimension):
        for k in range(ind_ptr[i + 1] - ind_ptr[i]):
            for j in range(second_dimension):
                # print("i: ", i, "j: ", j, "k:", k, "row: ", row)
                result_matrix[i][j] += value[row + k] * matrix2[column_idx[row + k]][j]
            row += 1

    print(result_matrix)
    

def makelist():
    random_int = random.randint(6, 12)
    num_list = []
    for row in range(0, 20):
        row_list = []
        for count in range(0, 2):
            row_list.append(random_int)
        num_list.append(row_list)
    return num_list



def main():
    matrix1 = sp.rand(20, 20).tocsr()
    matrix2 = makelist()

    # print("matrix1", matrix1)
    # print("matrix2", matrix2)

    sparse_sparse_multiplication(matrix1, matrix2)

    matrix1 = matrix1.todense()
    matrix2 = np.array(matrix2)
    res = np.matmul(matrix1, matrix2)
    print("results from numpy mat mul: ", res)
    return 

if __name__ == "__main__":
    main()