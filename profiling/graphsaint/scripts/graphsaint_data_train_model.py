import json
import os
import sys
import tensorflow as tf
import numpy as np
import scipy
from scipy.sparse import csr_matrix
from tensorflow.keras.layers import Input, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model
from spektral.layers import GCNConv
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt


channel_number = int(sys.argv[1])
dataset = sys.argv[2] #amazon or yelp
print("channel_number: ", channel_number)
print("dataset: ", dataset)

checkpoint_path = "../" + dataset + "/width_results/training_" + str(channel_number) + "_" + dataset + "/cp.ckpt"
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)


A = scipy.sparse.load_npz('../../../data/' + dataset + '/' + dataset + '_adj.npz')
X = np.load('../../../data/' + dataset + '/' + dataset + '.npy')
A = A + A.T

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

labels_train = labels_encoded[train_list, :]
labels_val = labels_encoded[val_list, :]

X_train = X[train_list,:]
A_train = A[train_list, :][:, train_list]

X_val  = X[val_list, :]
A_val = A[val_list, :][:, val_list]
print("All Shapes")
print(X.shape)
print(A.shape)


print("Train Shapes")
print(X_train.shape)
print(A_train.shape)
print(labels_train.shape)
print(train_mask.shape)

print("Val Shapes")
print(X_val.shape)
print(A_val.shape)
print(labels_val.shape)
print(val_mask.shape)
csr_matrix.sort_indices(A_train)
csr_matrix.sort_indices(A_val)


log_dir = 'logs'
tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

tbCallBack_GCN = tf.keras.callbacks.TensorBoard(
    log_dir='../' + dataset + '/profiling/Tensorboard_GCN_' + dataset + '_' + str(channel_number),
)

checkpoint_path = "../"+ dataset +"/saved_model/training_" + str(channel_number) + "_" + dataset + "/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

dropout = 0.2
F = X.shape[1]
N = X.shape[0]
channels = channel_number
l2_reg = 5e-4
learning_rate = .001
num_classes = labels_encoded.shape[1]
if dataset == 'amazon':
    epochs = 150
else:
    epochs = 600
# es_patience = 10        # Patience for early stopping

# A = GCNConv.preprocess(adj).astype('f4')
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


validation_data = ([X_val, A_val], labels_val) #, val_mask
# print(validation_data)
print(X.shape)
print(A.shape)
print(labels_encoded.shape)
print(val_mask.shape)
model.fit([X_train, A_train],
        labels_train,
        # sample_weight=train_mask,
        epochs=epochs,
        batch_size=N,
        validation_data=validation_data,
        shuffle=True,
        callbacks=[
            # EarlyStopping(patience=es_patience,  restore_best_weights=True),
            # tbCallBack_GCN,
            cp_callback
        ])
