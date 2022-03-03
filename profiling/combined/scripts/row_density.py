import random
import numpy as np
import sys
import scipy.sparse as sp


def main():

    if len(sys.argv) > 1:
        dataset = sys.argv[1] 

    dataset_dir = '/home/cclei/data/reddit/sorted/' + dataset + '.npz'
    sparse_mat = sp.load_npz(dataset_dir)
    sparse_mat = sparse_mat.tocsr()


if __name__ == "__main__":
    main()