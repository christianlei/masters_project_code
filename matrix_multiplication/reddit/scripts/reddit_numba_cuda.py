import sys
import scipy.sparse as sp
import numpy as np
sys.path.append('../../utils')
from util import sparse_dense_multiplication_cuda

def loadRedditFromNPZ(dataset_dir):
    adj = sp.load_npz(dataset_dir+"reddit_adj.npz")
    data = np.load(dataset_dir+"reddit.npz")
    return adj, data['feats'], data['y_train'], data['y_val'], data['y_test'], data['train_index'], data['val_index'], data['test_index']

def main():
    number_of_nodes = int(sys.argv[1])
    print("________CUDA Numba Code_________")
    print("number_of_nodes: ", number_of_nodes)

    adj, features, y_train, y_val, y_test, train_index, val_index, test_index = loadRedditFromNPZ("../../../../data/reddit/")

    sparse_array = adj+adj.T
    sparse_array = sparse_array[:, :number_of_nodes][:number_of_nodes,:]

    dense_array = np.random.rand(number_of_nodes, 256)

    matrix1 = sparse_array.todense()
    matrix2 = np.array(dense_array)
    ground_truth = np.matmul(matrix1, matrix2)

    first_dimension = sparse_array.get_shape()[0]
    second_dimension = len(dense_array[0])
    print("first_dimension", first_dimension)
    print("second_dimension", second_dimension)

    result_matrix = np.zeros((first_dimension, second_dimension))

    output = True
    result_matrix = sparse_dense_multiplication_cuda(result_matrix, first_dimension, second_dimension, sparse_array, dense_array, output)
    np.testing.assert_almost_equal(result_matrix, ground_truth)
    output = False
    result_matrix = np.zeros((first_dimension, second_dimension))
    sparse_dense_multiplication_cuda(result_matrix, first_dimension, second_dimension, sparse_array, dense_array, output)


    return 

if __name__ == "__main__":
    main()