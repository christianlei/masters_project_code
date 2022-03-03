import collections
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp


def count_zero_blocks(row, row_length):
    ret = []
    if not row.any():
        return collections.Counter(ret)
    for i in range(len(row) - 1):
        if row[i] + 1 != row[i + 1]:
            ret.append(row[i + 1] - row[i] - 1)
    if row[-1] != row_length - 1:        
        ret.append(row_length - row[-1] - 1)
    return collections.Counter(ret)

def count_blocks(row):
    ret = []
    count = 1
    for i in range(len(row) - 1):
        if row[i] + 1 == row[i + 1]:
            count+=1
        else:
            ret.append(count)
            count = 1
    ret.append(count)
    return collections.Counter(ret)

def count_density_from_zero(row):
    count = 1
    for i in range(len(row) - 1):
        if row[i] + 1 == row[i + 1]:
            count+=1
        else:
            break
    return count

def main():
    if len(sys.argv) > 1:
        dataset = sys.argv[1] 
        sorted = sys.argv[2] == 'sorted'
    else:
        print('add a dataset, retry')
        return

    hist = collections.defaultdict(int)
    hist_zero = collections.defaultdict(int)
    row_density_list = []

    figure_directory = '../graphs/' + dataset + '/'
    if not sorted:
        dataset_dir = '../../../../data/' +  dataset + '/' + dataset + '_adj.npz'
    else:
        dataset_dir = '/home/cclei/data/' + dataset + '/sorted/' + dataset + '.npz'
    sparse_mat = sp.load_npz(dataset_dir)
    sparse_mat = sparse_mat.tocsr()

    indptr = sparse_mat.indptr
    indices = sparse_mat.indices

    row_length = sparse_mat.shape[0]

    for i in range(len(indptr) - 1):
        start = indptr[i]
        end = indptr[i + 1]
        row_density_list.append(count_density_from_zero(indices[start:end]))
        counts = count_blocks(indices[start:end])
        zero_counts = count_zero_blocks(indices[start:end], row_length)
        for key, value in counts.items():
            hist[key] += value
        for key1, value1 in zero_counts.items():
            hist_zero[key1] += value1

    # print(hist_zero)

    zero_hist_arr = []

    non_zero_list = list(dict(hist).items())
    non_zero_list.sort(key=lambda x:x[0])
    # print(non_zero_list)

    # non_zero_bar_x = []
    # non_zero_bar_y = []
    # max_val_hist_non_zero = max(hist.keys())
    # for i in range(len(non_zero_list)):
    #     non_zero_bar_x.append(non_zero_list.keys()[i])
    #     non_zero_bar_y.append(non_zero_list.values()[i])

    print("Size of blocks with nonzero values: ", non_zero_list )

    max_val_hist_zero = max(hist_zero.keys())
    for i in range(max_val_hist_zero + 1):
        zero_hist_arr.append(0)

    for k,v in hist_zero.items():
        zero_hist_arr[k] = v

    zero_array = []
    for item in range(len(zero_hist_arr)):
        if zero_hist_arr[item]:
            zero_array.append((item, zero_hist_arr[item]))

    # print("Size of zero islands:", zero_array)
    # print(len(zero_array))

    x_val = [i for i in range(len(zero_hist_arr))]
    x_val_row_density = [i for i in range(len(row_density_list))]

    plt.bar(x_val_row_density[:], row_density_list[:])
    plt.ylim(0, 10)
    plt.ylabel('Row Desity')
    plt.xlabel('Row Number')
    plt.title('Row Density for Row in ' + dataset.capitalize() + ' Graph')
    plt.savefig(figure_directory + 'row-density-graph.png')

    plt.clf()
    plt.bar(x_val[:], zero_hist_arr[:])
    plt.ylim(0, 250)
    plt.xlim(0, max_val_hist_zero)
    plt.ylabel('Frequency')
    plt.xlabel('Gap Size')
    plt.title('Gap Sizes for Zeros in ' + dataset.capitalize() + ' Graph')
    plt.savefig(figure_directory + 'zero-gaps-graph.png')

    # plt.clf()
    # plt.bar(non_zero_bar_x[:], non_zero_bar_y[:])
    # # plt.ylim(0, 250)
    # plt.xlim(0, max_val_hist_non_zero)
    # plt.ylabel('Frequency')
    # plt.xlabel('Gap Size')
    # plt.title('Gap Sizes for Non Zeros in ' + dataset.capitalize() + ' Graph')
    # plt.savefig(figure_directory + 'non-zero-gaps-graph.png')



if __name__ == "__main__":
    main()