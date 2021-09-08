import pdb
import os
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import networkx as nx
import tensorflow as tf
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.layers import Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from spektral.layers import GCNConv
import matplotlib.pyplot as plt
import statistics

#loading the data

all_data = []
all_edges = []

for root,dirs,files in os.walk('./cora'):
    for file in files:
        if '.content' in file:
            with open(os.path.join(root,file),'r') as f:
                all_data.extend(f.read().splitlines())
        elif 'cites' in file:
            with open(os.path.join(root,file),'r') as f:
                all_edges.extend(f.read().splitlines())

                
#Shuffle the data because the raw data is ordered based on the label
random_state = 77
all_data = shuffle(all_data,random_state=random_state)

#_____________________________________________________________________


#parse the data
labels = []
nodes = []
X = []

for i,data in enumerate(all_data):
    elements = data.split('\t')
    labels.append(elements[-1])
    X.append(elements[1:-1])
    nodes.append(elements[0])

X = np.array(X,dtype=int)
N = X.shape[0] #the number of nodes
F = X.shape[1] #the size of node features
print('X shape: ', X.shape)

#parse the edge
edge_list=[]
for edge in all_edges:
    e = edge.split('\t')
    edge_list.append((e[0],e[1]))

print('\nNumber of nodes (N): ', N)
print('\nNumber of features (F) of each node: ', F)
print('\nCategories: ', set(labels))

num_classes = len(set(labels))
print('\nNumber of classes: ', num_classes)

#__________________________________________________________

def limit_data(labels,limit=20,val_num=500,test_num=1000):
    '''
    Get the index of train, validation, and test data
    '''
    label_counter = dict((l, 0) for l in labels)
    train_idx = []

    for i in range(len(labels)):
        label = labels[i]
        if label_counter[label]<limit:
            #add the example to the training data
            train_idx.append(i)
            label_counter[label]+=1
        
        #exit the loop once we found 20 examples for each class
        if all(count == limit for count in label_counter.values()):
            break
    
    #get the indices that do not go to traning data
    rest_idx = [x for x in range(len(labels)) if x not in train_idx]
    val_idx = rest_idx[:val_num]
    test_idx = rest_idx[val_num:(val_num+test_num)]
    return train_idx, val_idx,test_idx

def create_node_degree_graph(figure_name, adj_mat):
    node_degrees = {}
    print(adj_mat.shape)
    node_list = []

    for row in adj_mat:
        degrees = row.count_nonzero()
        node_list.append(int(degrees))
        if degrees in node_degrees:
            node_degrees[degrees]+=1
        else:
            node_degrees[degrees] = 1

    print("median: ", statistics.median(node_list))
    fig = plt.figure()
    fig.suptitle('Degrees of Nodes in Graph - CORA', fontsize=20)
    plt.bar(list(node_degrees.keys()), node_degrees.values(), width=1.0, color='g')
    plt.xlabel("Degrees")
    plt.ylabel("Occurrences")
    plt.xlim(0,30)
    plt.show()
    plt.savefig(figure_name)

train_idx,val_idx,test_idx = limit_data(labels)

#set the mask
train_mask = np.zeros((N,),dtype=bool)
train_mask[train_idx] = True

val_mask = np.zeros((N,),dtype=bool)
val_mask[val_idx] = True

test_mask = np.zeros((N,),dtype=bool)
test_mask[test_idx] = True

#_____________________________

#build the graph
G = nx.Graph()
G.add_nodes_from(nodes)
G.add_edges_from(edge_list)

#obtain the adjacency matrix (A)
A = nx.adjacency_matrix(G)
print('Graph info: ', nx.info(G))

create_node_degree_graph('cora_degree_graph.png', A)

#__________________________________

def encode_label(labels):
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    labels = to_categorical(labels)
    return labels, label_encoder.classes_

labels_encoded, classes = encode_label(labels)


#____________________________________________________________

# Parameters
channels = 16           # Number of channels in the first layer
dropout = 0.5           # Dropout rate for the features
l2_reg = 5e-4           # L2 regularization rate
learning_rate = 1e-2    # Learning rate
epochs = 200            # Number of training epochs
es_patience = 10        # Patience for early stopping

# Preprocessing operations
A = GCNConv.preprocess(A).astype('f4')

# Model definition
X_in = Input(shape=(F, ))
fltr_in = Input((N, ), sparse=True)

dropout_1 = Dropout(dropout)(X_in)
graph_conv_1 = GCNConv(channels,
                         activation='relu',
                         kernel_regularizer=l2(l2_reg),
                         use_bias=False)([dropout_1, fltr_in])

dropout_2 = Dropout(dropout)(graph_conv_1)
graph_conv_2 = GCNConv(num_classes,
                         activation='softmax',
                         use_bias=False)([dropout_2, fltr_in])

# Build model
model = Model(inputs=[X_in, fltr_in], outputs=graph_conv_2)
optimizer = Adam(lr=learning_rate)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              weighted_metrics=['acc'])
model.summary()

dot_img_file = 'model.png'
plot_model(model, to_file=dot_img_file, expand_nested=True, show_shapes=True)

tbCallBack_GCN = tf.keras.callbacks.TensorBoard(
    log_dir='./Tensorboard_GCN_cora',
)
callback_GCN = [tbCallBack_GCN]

#_________________________________________-

# Train model
# validation_data = ([X, A], labels_encoded, val_mask)
# model.fit([X, A],
#           labels_encoded,
#           sample_weight=train_mask,
#           epochs=epochs,
#           batch_size=N,
#           validation_data=validation_data,
#           shuffle=False,
#           callbacks=[
#               EarlyStopping(patience=es_patience,  restore_best_weights=True),
#               tbCallBack_GCN
#           ])
# # Evaluate model
# X_te = X[test_mask]
# A_te = A[test_mask,:][:,test_mask]
# y_te = labels_encoded[test_mask]

# M = X_te.shape[0]
# # print("batch size:", N)
# tf.profiler.experimental.start('logdir')
# pdb.set_trace()
# y_pred = model.predict([X_te, A_te], batch_size=M)
# tf.profiler.experimental.stop()
# report = classification_report(np.argmax(y_te,axis=1), np.argmax(y_pred,axis=1), target_names=classes)
# print('GCN Classification Report: \n {}'.format(report))