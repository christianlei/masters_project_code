import matplotlib.pyplot as plt
import scipy.sparse as sp
import numpy as np
import sys
sys.path.append('../../utils')
from util import count_nodes_and_edges, determine_nodes_per_thread

def loadRedditFromNPZ(dataset_dir):
    adj = sp.load_npz(dataset_dir+"reddit_adj.npz")
    data = np.load(dataset_dir+"reddit.npz")
    return adj, data['feats'], data['y_train'], data['y_val'], data['y_test'], data['train_index'], data['val_index'], data['test_index']


def count_nodes_and_edges(sparse_mat):
    edge_count = 0
    node_count = 0
 
    for i in range(sparse_mat.shape[0]):
        node_count += 1
        edge_count+=(sparse_mat.getrow(i).count_nonzero())

    print("edge_count", edge_count)
    print("node_count", node_count)
    return node_count, edge_count


def main():
    sparse_mat, features, y_train, y_val, y_test, train_index, val_index, test_index = loadRedditFromNPZ("../../../../data/reddit/")
    sparse_mat = sparse_mat+sparse_mat.T
    #Call with the argument of the dataset to use, either yelp or amazon
    print("running....")
    number_of_threads = sys.argv[1]
    node_count, edge_count  = count_nodes_and_edges(sparse_mat)
    determine_nodes_per_thread(sparse_mat, edge_count, number_of_threads)

if __name__ == "__main__":
    main()