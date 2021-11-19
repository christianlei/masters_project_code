# Based on equal number of edges
def determine_nodes_per_thread(sparse_mat, edge_count, number_of_threads):
    edges_per_thread = edge_count / int(number_of_threads)
    print("edges_per_thread", edges_per_thread)
    print("sparse_mat.shape", sparse_mat.shape)
    nodes_per_thread = []
    node_count = 0
    node_start = 0
    edge_count = 0
    for node_number in range(sparse_mat.shape[0]):
        node_count += 1
        edge_count+=(sparse_mat.getrow(node_number).count_nonzero())
        if edge_count >= edges_per_thread:
            nodes_per_thread.append((node_start, node_start + node_count, node_count, edge_count))
            node_start = node_number + 1
            node_count = 0
            edge_count = 0
    if node_start != node_number:
        nodes_per_thread.append((node_start, node_start + node_count, node_count, edge_count))

    print("nodes_per_thread", nodes_per_thread)
    return nodes_per_thread


# Based on equal number of node
def determine_edges_per_thread(sparse_mat, number_of_threads):
    nodes_per_thread = sparse_mat.shape[0] / int(number_of_threads)
    print("nodes_per_thread", nodes_per_thread)
    print("sparse_mat.shape", sparse_mat.shape)    
    edges_per_thread = []
    node_count = 0
    node_start = 0
    edge_count = 0
    for node_number in range(sparse_mat.shape[0]):
        node_count+=1
        edge_count+=(sparse_mat.getrow(node_number).count_nonzero())
        if node_count >= nodes_per_thread:
            edges_per_thread.append((node_start, node_start + node_count, node_count, edge_count))
            node_start = node_number + 1
            node_count = 0
            edge_count = 0
    if node_count:
        edges_per_thread.append((node_start, node_start + node_count, node_count, edge_count))
    print("edges_per_thread", edges_per_thread)
    return nodes_per_thread


def count_nodes_and_edges(sparse_mat):
    edge_count = 0
    node_count = 0
 
    for i in range(sparse_mat.shape[0]):
        node_count += 1
        edge_count+=(sparse_mat.getrow(i).count_nonzero())
    return node_count, edge_count