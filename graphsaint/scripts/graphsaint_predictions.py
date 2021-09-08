import sys
import numpy as np
import scipy
import os
import json
import time
import tensorflow as tf
from tensorflow.keras.layers import Input, Dropout
from spektral.layers import GCNConv
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2


number_of_nodes = int(sys.argv[1])
dataset = sys.argv[2]

A = scipy.sparse.load_npz('../../../data/' + dataset + '/' + dataset + '_adj.npz')
X = np.load('../../../data/' + dataset + '/' + dataset + '.npy')
A = A + A.T

print(X.shape)
print(A.shape)

train_mask =  np.zeros((X.shape[0]), dtype=bool)
test_mask =  np.zeros((X.shape[0]), dtype=bool)
val_mask =  np.zeros((X.shape[0]), dtype=bool)

with open('../../../data/' + dataset + '/role.json') as f:
    data = json.load(f)
    train_list = data["tr"]
    test_list = data['te']
    val_list = data['va']


    for i in range(len(train_list)):
        train_mask[i] = True

    for i in range(len(test_list)):
        test_mask[i] = True

    for i in range(len(val_list)):
        val_mask[i] = True

with open('../../../data/' + dataset + '/class_map.json') as f:
  data = json.load(f)
labels_encoded = np.array(list(data.values()))

labels_test = labels_encoded[test_list, :]

X_test = X[test_list,:]
A_test = A[test_list, :][:, test_list]

dropout = 0.2
F = X.shape[1]
N = X.shape[0]
if dataset == 'yelp':
    channels = 16
else:
    channels = 128
l2_reg = 5e-4
learning_rate = .001
num_classes = labels_encoded.shape[1]
epochs = 600

if dataset == 'yelp':
    checkpoint_path = "../" + dataset + "/saved_model/training_16_" + dataset + "/cp.ckpt"
else:
    checkpoint_path = "../" + dataset + "/saved_model/training_128_" + dataset + "/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

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
              loss=tf.nn.softmax_cross_entropy_with_logits,
              weighted_metrics=['acc'])
model.summary()
model.load_weights(checkpoint_path)

test_features = X[test_mask]

test_features = X[test_mask][0:number_of_nodes]
print(test_features.shape)
adj_test = A[test_mask, :][:, test_mask]
print("ADJ MATRIX BEFORE", adj_test.shape)
adj_test = adj_test[:, :number_of_nodes][:number_of_nodes,:]
print("ADJ MATRIX AFTER", adj_test.shape)




M = test_features.shape[0]

start = time.time()
y_pred = model.predict([test_features, adj_test], batch_size=M)
end = time.time()

print("overall time: ", end-start)
