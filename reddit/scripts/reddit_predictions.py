import os
import sys
import time
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dropout
from tensorflow.keras.regularizers import l2
import numpy as np
import scipy.sparse as sp
from spektral.layers import GCNConv
from tensorflow.keras.optimizers import Adam

number_of_nodes = int(sys.argv[1])

def loadRedditFromNPZ(dataset_dir):
    adj = sp.load_npz(dataset_dir+"reddit_adj.npz")
    data = np.load(dataset_dir+"reddit.npz")
    return adj, data['feats'], data['y_train'], data['y_val'], data['y_test'], data['train_index'], data['val_index'], data['test_index']

adj, features, y_train, y_val, y_test, train_index, val_index, test_index = loadRedditFromNPZ("../../../data/reddit/")
adj = adj+adj.T

adj_train = adj[train_index, :][:, train_index]

dropout = 0.2
F = 602
N = 232965
channels = 256
l2_reg = 5e-4
learning_rate = .001
num_classes = 41
epochs = 600

numNode_train = adj_train.shape[0]

checkpoint_path = "../saved_model/training_256_reddit/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

#A = GCNConv.preprocess(adj).astype('f4')
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

model = Model(inputs=[X_in, fltr_in], outputs=graph_conv_2)
optimizer = Adam(lr=learning_rate)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              weighted_metrics=['acc'])
model.summary()
model.load_weights(checkpoint_path)



# N:
# print(N)
# Evaluate model
# (55334,)
test_features = features[test_index]
# print("TEST INDEX SHAPE BEFORE SPLICE:", test_features.shape)

test_features = features[test_index][0:number_of_nodes]
print(test_features.shape)
adj_test = adj[test_index, :][:, test_index]
print("ADJ MATRIX BEFORE", adj_test.shape)
adj_test = adj_test[:, :number_of_nodes][:number_of_nodes,:]
print("ADJ MATRIX AFTER", adj_test.shape)


# create_node_degree_graph('degree_test.png', adj_test)





# test_features = csr_matrix(test_features)
adj_test = sp.csr_matrix(adj_test)
# csr_matrix.sort_indices(test_features)
sp.csr_matrix.sort_indices(adj_test)


print("TEST INDEX SHAPE AFTER SPLICE:", test_features.shape)
# print("ADJ MATRIX AFTER", adj_test.shape)
M = test_features.shape[0]
print("M: ", M)

# tf.profiler.experimental.start('prediction_logs')
start = time.time()
y_pred = model.predict([test_features, adj_test], batch_size=M)
end = time.time()
# tf.profiler.experimental.stop()

# print("shape: ", y_pred.shape)

print("overall time: ", end-start)