import matplotlib.pyplot as plt
import scipy.sparse as sp
import numpy as np
import sys
sys.path.append('../../utils')
from utils import count_nodes_and_edges, determine_nodes_per_thread, determine_edges_per_thread

def loadRedditFromNPZ(dataset_dir):
    adj = sp.load_npz(dataset_dir+"reddit_adj.npz")
    data = np.load(dataset_dir+"reddit.npz")
    return adj, data['feats'], data['y_train'], data['y_val'], data['y_test'], data['train_index'], data['val_index'], data['test_index']


def main():
    sparse_mat, features, y_train, y_val, y_test, train_index, val_index, test_index = loadRedditFromNPZ("../../../../data/reddit/")
    sparse_mat = sparse_mat+sparse_mat.T
    #Call with the argument of the dataset to use, either yelp or amazon
    print("running....")
    number_of_threads = sys.argv[1]
    if len(sys.argv) == 3:
        edge_count = int(sys.argv[2])
        sparse_mat = sparse_mat[:, :edge_count][:edge_count,:]
        node_count, edge_count  = count_nodes_and_edges(sparse_mat)
    else:
        node_count, edge_count  = count_nodes_and_edges(sparse_mat)
    determine_nodes_per_thread(sparse_mat, edge_count, number_of_threads)
    determine_edges_per_thread(sparse_mat, number_of_threads)

if __name__ == "__main__":
    main()