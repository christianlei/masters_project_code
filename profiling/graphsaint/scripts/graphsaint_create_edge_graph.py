import statistics
import matplotlib.pyplot as plt
import scipy.sparse as sp
import sys

def create_node_degree_graph(dataset):
    sparse_mat = sp.load_npz('/home/cclei/data/' +  dataset + '/' + dataset + '_adj.npz')
    counts = []
 
    # print(sparse_mat.shape)
    for i in range(sparse_mat.shape[0]):
        counts.append(sparse_mat.getrow(i).count_nonzero())

    # print("median: ", statistics.median(counts))
    # print(yelp_feat)

    # hist, bin_edges = np.histogram(counts)

    # print(hist)

    plt.hist(counts, density=True, bins=1000)
    # plt.ylim(0, 0.02)
    # plt.xlim(0,200)
    plt.suptitle('Degrees of Nodes in ' + dataset.capitalize() + ' Graph', fontsize=20)
    plt.xlabel("Degrees")
    plt.ylabel("Occurrences")
    plt.yscale("log")
    plt.savefig('../' + dataset + '/graphs/' + dataset +  '_node_edges.png')

def main():

    #Call with the argument of the dataset to use, either yelp or amazon
    print("running....")
    dataset = sys.argv[1] 
    create_node_degree_graph(dataset)

if __name__ == "__main__":
    main()