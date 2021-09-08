import statistics
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp


def loadRedditFromNPZ(dataset_dir):
    adj = sp.load_npz(dataset_dir+"reddit_adj.npz")
    data = np.load(dataset_dir+"reddit.npz")
    return adj, data['feats'], data['y_train'], data['y_val'], data['y_test'], data['train_index'], data['val_index'], data['test_index']

def create_node_degree_graph(adj_mat):
    counts = []

    for row in adj_mat:
        counts.append(int(row.count_nonzero()))

    # print("median: ", statistics.median(counts))
    fig = plt.figure()
    fig.suptitle('Degrees of Nodes in Reddit Graph', fontsize=20)
    plt.hist(counts, density=True, bins=1000)
    plt.xlabel("Degrees")
    plt.ylabel("Occurrences")
    # plt.ylim(0, 0.02)
    # plt.xlim(0,200)
    plt.yscale("log")
    plt.savefig('graphs/reddit_node_edges.png')

def main():  
    print("running....")
    adj, features, y_train, y_val, y_test, train_index, val_index, test_index = loadRedditFromNPZ("../../data/reddit/")
    adj = adj+adj.T
    create_node_degree_graph(adj)

if __name__ == "__main__":
    main()