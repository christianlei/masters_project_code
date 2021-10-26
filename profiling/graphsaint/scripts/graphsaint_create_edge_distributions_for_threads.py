import matplotlib.pyplot as plt
import scipy.sparse as sp
import sys

def count_nodes_and_edges(sparse_mat):
    edge_count = 0
    node_count = 0
 
    for i in range(sparse_mat.shape[0]):
        node_count += 1
        edge_count+=(sparse_mat.getrow(i).count_nonzero())

    print("edge_count", edge_count)
    print("node_count", node_count)
    return node_count, edge_count

def determine_nodes_per_thread(sparse_mat, edge_count, number_of_threads):
    edges_per_thread = edge_count / int(number_of_threads)
    print("edges_per_thread", edges_per_thread)
    nodes_per_thread = []
    node_count = 0
    edge_count = 0
    for i in range(sparse_mat.shape[0]):
        node_count += 1
        edge_count+=(sparse_mat.getrow(i).count_nonzero())
        if edge_count >= edges_per_thread:
            nodes_per_thread.append(node_count)
            node_count = 0
            edge_count = 0
    nodes_per_thread.append(node_count)

    print(nodes_per_thread)
    return nodes_per_thread

def main():
    dataset = sys.argv[1] 
    sparse_mat = sp.load_npz('../../../../data/' +  dataset + '/' + dataset + '_adj.npz')
    #Call with the argument of the dataset to use, either yelp or amazon
    print("running....")
    number_of_threads = sys.argv[2]
    node_count, edge_count  = count_nodes_and_edges(sparse_mat)
    determine_nodes_per_thread(sparse_mat, edge_count, number_of_threads)

if __name__ == "__main__":
    main()