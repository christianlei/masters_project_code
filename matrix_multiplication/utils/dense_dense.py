
def dense_dense_multiplication(matrix1, matrix2):
    first_dimension = len(matrix1)
    second_dimension = len(matrix2[0])

    if len(matrix1[0]) != len(matrix2):
        print("ERROR! - Matrix needs to be correct shape.")

    result_matrix = [[0] * second_dimension for i in range(first_dimension)]

    for i in range(len(matrix1)):
        for j in range(len(matrix2[0])):
            for k in range(len(matrix2)):
                result_matrix[i][j] += matrix1[i][k] * matrix2[k][j]
    
    return result_matrix

def main():
    matrix_1 = [[0,0,0],[0,0,0]]
    matrix_2 = [[0,0,0],[0,0,0],[0,0,0]]
    return dense_dense_multiplication(matrix_1, matrix_2)



if __name__ == "__main__":
    main()