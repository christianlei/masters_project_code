import scipy.sparse as sp
import random

def sparse_sparse_multiplication(matrix1, matrix2):
    first_dimension = matrix1.get_shape()[0]
    # second_dimension = matrix2.get_shape()[1]
    second_dimension = 1
    
    result_matrix = [[0] * second_dimension for i in range(first_dimension)]
    
    value1 = matrix1.data
    column_idx1 = matrix1.indices
    row_ptr1 = matrix1.indptr
    
    print("value1", value1)
    print("column_idx1", column_idx1)
    print("row_ptr1", row_ptr1)
    
    for i in range(second_dimension):
        for k in range(row_ptr1[i], row_ptr1[i + 1]):
            result_matrix[i] += value1[k] * matrix2[column_idx1[k]]
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

    print("matrix1 from main: ", matrix1)

    return sparse_sparse_multiplication(matrix1, matrix2)

if __name__ == "__main__":
    main()