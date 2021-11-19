import matplotlib.pyplot as plt
import scipy.sparse as sp
import sys
sys.path.append('../../utils')
from utils import count_nodes_and_edges, determine_nodes_per_thread, determine_edges_per_thread

def main():
    dataset = sys.argv[1] 
    sparse_mat = sp.load_npz('../../../../data/' +  dataset + '/' + dataset + '_adj.npz')
    #Call with the argument of the dataset to use, either yelp or amazon
    print("running....")
    number_of_threads = sys.argv[2]
    node_count, edge_count  = count_nodes_and_edges(sparse_mat)
    determine_nodes_per_thread(sparse_mat, edge_count, number_of_threads)
    determine_edges_per_thread(sparse_mat, number_of_threads)

if __name__ == "__main__":
    main()