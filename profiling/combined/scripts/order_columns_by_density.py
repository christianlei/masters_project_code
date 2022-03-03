from xxlimited import new
import scipy.sparse as sp
import sys
import collections

dataset = sys.argv[1] 
dataset_dir = '/home/cclei/data/' + dataset + '/' + dataset + '_adj.npz'
sparse_mat = sp.load_npz(dataset_dir)
sparse_mat = sparse_mat.tocsr()

print(sparse_mat.shape)

indptr = sparse_mat.indptr
indices = sparse_mat.indices
values = sparse_mat.data


column_frequency = collections.Counter(indices)

freq_row = []
for row, freq in column_frequency.items():
    freq_row.append((freq, row))

freq_row.sort(key=lambda x:x[0], reverse=True)

# print(freq_row)

row_frequency_as_dict = {}

for idx in range(len(freq_row)):
    row_frequency_as_dict[freq_row[idx][1]] = idx

for i in range(len(indices)):
    indices[i] = row_frequency_as_dict[indices[i]]

# print(len(values))
print(indices)


csr_mat = sp.csr_matrix((values, indices, indptr), shape=sparse_mat.shape)
sp.save_npz('/home/cclei/data/'+ dataset +'/sorted/' + dataset + '.npz', csr_mat)
