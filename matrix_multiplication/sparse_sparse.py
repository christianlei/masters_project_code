import scipy.sparse as sp
import random
import numpy as np

def sparse_sparse_multiplication(matrix1, matrix2):
    first_dimension = matrix1.get_shape()[0]
    # second_dimension = matrix2.get_shape()[1]
    second_dimension = len(matrix2)
    
    result_matrix = [0 * second_dimension for i in range(first_dimension)]
    
    value = matrix1.data
    column_idx = matrix1.indices
    ind_ptr = matrix1.indptr
    
    print("value", value)
    print("column_idx", column_idx)
    print("row_ptr", ind_ptr)
    
    row = 0
    for i in range(second_dimension):
        for k in range(ind_ptr[i + 1] - ind_ptr[i]):
            result_matrix[i] += value[row + k] * matrix2[column_idx[row + k]]
            row += 1
    print(result_matrix)


def makelist():
    random_int = random.randint(6, 12)
    num_list = []
    for count in range(1, 21):
        num_list.append(random_int)
    return num_list



def main():
    matrix1 = sp.rand(20, 20).tocsr()
    matrix2 = makelist()

    sparse_sparse_multiplication(matrix1, matrix2)

    matrix1 = matrix1.todense()
    matrix2 = np.array(matrix2)
    res = np.matmul(matrix1, matrix2)
    print(res[0])
    return 

if __name__ == "__main__":
    main()