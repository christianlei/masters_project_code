import sys
from scipy import sparse
import numpy as np
sys.path.append('../../utils')
from util import sparse_dense_multiplication_manual_allocation, sparse_dense_multiplication
sys.path.append('../../../profiling/utils')
from utils import count_nodes_and_edges, determine_nodes_per_thread, determine_edges_per_thread

def main():
    numba = True
    number_of_nodes = int(sys.argv[1])
    dataset = sys.argv[2]
    number_of_threads = int(sys.argv[3])
    width = int(sys.argv[4])

    print("________manually allocated threads_________")
    print("number_of_nodes: ", number_of_nodes)
    print("dataset: ", dataset)

    adj = sparse.load_npz('../../../../data/' + dataset + '/' + dataset + '_adj.npz')

    sparse_array = adj+adj.T
    sparse_array = sparse_array[:, :number_of_nodes][:number_of_nodes,:]

    node_count, edge_count  = count_nodes_and_edges(sparse_array)
    allocation_tuples_nodes = determine_nodes_per_thread(sparse_array, edge_count, number_of_threads)
    print("allocation_tuples_nodes: ", allocation_tuples_nodes)

    allocation_tuples_edges = determine_edges_per_thread(sparse_array, number_of_threads)
    print("allocation_tuples_edges: ", allocation_tuples_edges)
    # if dataset == 'yelp':
    #     # dense_array = np.random.rand(number_of_nodes, 16)
    #     dense_array = np.random.rand(number_of_nodes, 1024)
    # else:
    #     dense_array = np.random.rand(number_of_nodes, 128)

    dense_array = np.random.rand(number_of_nodes, width)
    dense_array_T = dense_array.T

    print("sparse array shape: ", sparse_array.shape)
    print("dense array shape: ", dense_array.shape)

    matrix1 = sparse_array.todense()
    matrix2 = np.array(dense_array)
    ground_truth = np.matmul(matrix1, matrix2)

    first_dimension = sparse_array.get_shape()[0]
    second_dimension = len(dense_array[0])

    # Equal Edge Count per Thread
    print("Equal Edge Count: ")
    result_matrix = np.zeros((first_dimension, second_dimension))
    output = True
    result_matrix = sparse_dense_multiplication_manual_allocation(result_matrix, second_dimension, sparse_array, dense_array_T, allocation_tuples_nodes, output)
    np.testing.assert_almost_equal(result_matrix, ground_truth)
    output = False
    result_matrix = np.zeros((first_dimension, second_dimension))
    sparse_dense_multiplication_manual_allocation(result_matrix, second_dimension, sparse_array, dense_array_T, allocation_tuples_nodes, output)

    # Equal Node Count per Thread
    print("Equal Node Count: ")
    result_matrix = np.zeros((first_dimension, second_dimension))
    output = True
    result_matrix = sparse_dense_multiplication_manual_allocation(result_matrix, second_dimension, sparse_array, dense_array_T, allocation_tuples_edges, output)
    np.count_nonzero(ground_truth)
    np.testing.assert_almost_equal(result_matrix, ground_truth)
    output = False
    result_matrix = np.zeros((first_dimension, second_dimension))
    sparse_dense_multiplication_manual_allocation(result_matrix, second_dimension, sparse_array, dense_array_T, allocation_tuples_edges, output)


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