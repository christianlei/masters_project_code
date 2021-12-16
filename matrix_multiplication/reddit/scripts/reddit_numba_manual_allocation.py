import sys
import scipy.sparse as sp
import numpy as np
sys.path.append('../../utils')
from util import sparse_dense_multiplication_manual_allocation, sparse_dense_multiplication, sparse_dense_multiplication_operation_numba_parallel_allocated_manual_2
sys.path.append('../../../profiling/utils')
from utils import count_nodes_and_edges, determine_nodes_per_thread, determine_edges_per_thread

def loadRedditFromNPZ(dataset_dir):
    adj = sp.load_npz(dataset_dir+"reddit_adj.npz")
    data = np.load(dataset_dir+"reddit.npz")
    return adj, data['feats'], data['y_train'], data['y_val'], data['y_test'], data['train_index'], data['val_index'], data['test_index']

def main():
    numba = True
    number_of_nodes = int(sys.argv[1])
    number_of_threads = int(sys.argv[2])
    width = int(sys.argv[3])
    print("________manually allocated threads_________")
    print("number_of_nodes: ", number_of_nodes)

    adj, features, y_train, y_val, y_test, train_index, val_index, test_index = loadRedditFromNPZ("../../../../data/reddit/")

    sparse_array = adj+adj.T
    sparse_array = sparse_array[:, :number_of_nodes][:number_of_nodes,:]

    node_count, edge_count  = count_nodes_and_edges(sparse_array)
    allocation_tuples_nodes = determine_nodes_per_thread(sparse_array, edge_count, number_of_threads)
    print("allocation_tuples_equal_edges: ", allocation_tuples_nodes)

    allocation_tuples_edges = determine_edges_per_thread(sparse_array, number_of_threads)
    print("allocation_tuples_equal_nodes: ", allocation_tuples_edges)
    # dense_array = np.random.rand(number_of_nodes, 256)
    dense_array = np.random.rand(number_of_nodes, width)
    dense_array_T = dense_array.T

    matrix1 = sparse_array.todense()
    matrix2 = np.array(dense_array)
    ground_truth = np.matmul(matrix1, matrix2)

    first_dimension = sparse_array.get_shape()[0]
    second_dimension = len(dense_array[0])
    print("first_dimension", first_dimension)
    print("second_dimension", second_dimension)

    # Equal Node Count per Thread
    print("Equal Node Count Matrix Multiplication: ")
    result_matrix = np.zeros((first_dimension, second_dimension))
    output = True
    result_matrix = sparse_dense_multiplication_manual_allocation(result_matrix, second_dimension, sparse_array, dense_array_T, allocation_tuples_edges, output)
    np.count_nonzero(ground_truth)
    np.testing.assert_almost_equal(result_matrix, ground_truth)
    output = False
    result_matrix = np.zeros((first_dimension, second_dimension))
    sparse_dense_multiplication_manual_allocation(result_matrix, second_dimension, sparse_array, dense_array_T, allocation_tuples_edges, output)

    # Equal Edge Count per Thread
    # print("Equal Edge Count: ")
    print("Equal Edge Count Matrix Multiplication with Dense Array Transposed")
    result_matrix = np.zeros((first_dimension, second_dimension))
    output = True
    result_matrix = sparse_dense_multiplication_manual_allocation(result_matrix, second_dimension, sparse_array, dense_array_T, allocation_tuples_nodes, output)
    np.count_nonzero(ground_truth)
    np.testing.assert_almost_equal(result_matrix, ground_truth)
    output = False
    result_matrix = np.zeros((first_dimension, second_dimension))
    sparse_dense_multiplication_manual_allocation(result_matrix, second_dimension, sparse_array, dense_array_T, allocation_tuples_nodes, output)

    #Non Manual 
    output = True
    result_matrix = np.zeros((first_dimension, second_dimension))
    result_matrix = sparse_dense_multiplication(result_matrix, first_dimension, second_dimension, sparse_array, dense_array, numba, output)
    np.testing.assert_almost_equal(result_matrix, ground_truth)
    output = False
    sparse_dense_multiplication(result_matrix, first_dimension, second_dimension, sparse_array, dense_array, numba, output)

    output = True
    parallel = True
    result_matrix = np.zeros((first_dimension, second_dimension))
    result_matrix = sparse_dense_multiplication(result_matrix, first_dimension, second_dimension, sparse_array, dense_array, numba, output, parallel)
    np.testing.assert_almost_equal(result_matrix, ground_truth)
    output = False
    sparse_dense_multiplication(result_matrix, first_dimension, second_dimension, sparse_array, dense_array, numba, output, parallel)

    return 

if __name__ == "__main__":
    main()