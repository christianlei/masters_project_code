import sys
import time
import tensorflow as tf
from tensorflow import sparse
import numpy as np
import scipy
import json

#tf.sparse.sparse_dense_matmul working with first input as the adjacency matrix (adj), then a random second input (can just #be anything).
#For the random second input, you can just do (np.random.rand(?,2048))

dataset = sys.argv[1] #amazon or yelp
number_of_nodes = int(sys.argv[2])
print("dataset: ", dataset)
print("number of nodes: ", number_of_nodes)

if dataset == 'yelp':
    channel_number = 16
else:
    channel_number = 128

X = np.load('../../../data/' + dataset + '/' + dataset + '.npy')
test_index =  np.zeros((X.shape[0]), dtype=bool)

with open('../../../data/' + dataset + '/role.json') as f:
    data = json.load(f)
    train_list = data["tr"]
    test_list = data['te']
    val_list = data['va']

    for i in range(len(test_list)):
        test_index[i] = True


adj = scipy.sparse.load_npz('../../../data/' + dataset + '/' + dataset + '_adj.npz')

adj = adj+adj.T

adj_test = adj[test_index, :][:, test_index]
adj_test = adj_test[:, :number_of_nodes][:number_of_nodes,:]

print(adj_test)
print("adj_test.shape:",  adj_test.shape)

# csr = adj_test.tocsr()
# csr = csr[0:number_of_nodes]
# adj_test = adj_test.tocsr()
# csr_matrix.sort_indices(adj_test)
coo = adj_test.tocoo()
coo = coo.astype('float')
# coo = adj.tocoo()
indices = np.mat([coo.row, coo.col]).transpose()
adj_sp = sparse.SparseTensor(indices, coo.data, coo.shape)
# adj_sp = sparse.transpose(adj_sp)
rand_array = np.random.rand(number_of_nodes, channel_number)


# pdb.set_trace()

start = time.time()
y = sparse.sparse_dense_matmul(adj_sp, rand_array)
end = time.time()

total_time = end - start


print("Total Time: ", total_time)

