import time
import scipy.sparse as sp
import os
import numpy as np
import pdb
import sys
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dropout
from tensorflow.keras.regularizers import l2
from spektral.layers import GCNConv
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

def nontuple_preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features

def transferLabel2Onehot(labels, N):
    y = np.zeros((len(labels),N))
    for i in range(len(labels)):
        pos = labels[i]
        y[i,pos] =1
    return y

def loadRedditFromNPZ(dataset_dir):
    adj = sp.load_npz(dataset_dir+"reddit_adj.npz")
    data = np.load(dataset_dir+"reddit.npz")
    return adj, data['feats'], data['y_train'], data['y_val'], data['y_test'], data['train_index'], data['val_index'], data['test_index']

adj, features, y_train, y_val, y_test, train_index, val_index, test_index = loadRedditFromNPZ("../../../data/reddit/")
adj = adj+adj.T




#adj = tf.sparse.reorder(adj)
channel_number = int(sys.argv[1])
print("number of channels: ", channel_number)

y_train = transferLabel2Onehot(y_train, 41)
y_val = transferLabel2Onehot(y_val, 41)
y_test = transferLabel2Onehot(y_test, 41)

features = sp.lil_matrix(features)

features = nontuple_preprocess_features(features).todense()

adj_train = adj[train_index, :][:, train_index]
adj_val = adj[val_index, :][:, val_index]


checkpoint_path = "../saved_model/training_" + str(channel_number) + "_reddit/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

dropout = 0.2
F = 602
N = 232965
channels = channel_number
l2_reg = 5e-4
learning_rate = .001
num_classes = 41
epochs = 600

numNode_train = adj_train.shape[0]

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

# Build model
model = Model(inputs=[X_in, fltr_in], outputs=graph_conv_2)
optimizer = Adam(lr=learning_rate)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              weighted_metrics=['acc'])
model.summary()

train_features = features[train_index]
val_features = features[val_index]


# train_mask = np.zeros((N,),dtype=bool)
# for i in train_index:
#     train_mask[i] = True

#full_labels = []
#j = 0
#for i in range(N):
#    if i in train_index:
#        full_labels.append(y_train[j])
#        j += 1;
#    else:
#        full_labels.append([0]*41)

#def flatten(t):
#    return [item for sublist in t for item in sublist]

#fl = np.array(flatten(full_labels)).reshape(N,41)
adj_train.sort_indices()
adj_val.sort_indices()

model.fit([train_features, adj_train],
          y_train,
          epochs=epochs,
          validation_data=([val_features, adj_val],y_val),
          validation_batch_size = len(val_features),
          batch_size=len(train_features),
          shuffle=False,
          callbacks=[cp_callback]
          )


#pdb.set_trace()
