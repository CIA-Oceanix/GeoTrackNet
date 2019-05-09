#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 13:10:40 2018

@author: vnguye04
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
import pickle

LEARNING_RATE =  0.0003
BATCH_SIZE = 100
DISPLAY_INTERS = 30000
N_CLASSES = 4
lambda_loss_amount = 0.005
NUM_EPOCHS = 100

LABELS = [
    "Cargo", 
    "Passenger", 
    "Tanker", 
    "Tug"
]

with open("/users/local/dnguyen/Datasets/AIS_datasets/MarineC/MarineC_Jan2014_norm/MarineC_Jan2014_norm_train_rnn_state.pkl","rb") as f:
    l_data_train = pickle.load(f)

with open("/users/local/dnguyen/Datasets/AIS_datasets/MarineC/MarineC_Jan2014_norm/MarineC_Jan2014_norm_test_rnn_state.pkl","rb") as f:
    l_data_test = pickle.load(f)

X_train = []
Y_train = []
X_test = []
Y_test = []
for D in l_data_train:
    X_train.append(D['rnn_state'][:,1,:]) #(144, 400)
    Y_train.append(D['vessel_type'])
X_train = np.array(X_train)
Y_train = np.array(Y_train)
for D in l_data_test:
    X_test.append(D['rnn_state'][:,1,:]) #(144, 400)
    Y_test.append(D['vessel_type'])
X_test = np.array(X_test)
Y_test = np.array(Y_test)


x = tf.placeholder(tf.float32, [None, 144, 400])
# dynamically reshape the input
x_shaped = tf.reshape(x, [-1, 144, 400, 1])
y = tf.placeholder(tf.float32, [None, N_CLASSES])
keep_prob = tf.placeholder(tf.float32) 

def one_hot(y_):
    # Function to encode output labels from number indexes 
    # e.g.: [[5], [0], [3]] --> [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]

    y_ = y_.reshape(len(y_))
    n_values = int(np.max(y_)) + 1
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]  # Returns FLOATS

def extract_batch_size(_train, v_idx, step, batch_size):
    # Function to fetch a "batch_size" amount of data from "(X|y)_train" data. 
    
    shape = list(_train.shape)
    shape[0] = batch_size
    batch_s = np.empty(shape)

    for i in range(batch_size):
        # Loop index
        index = v_idx[step*batch_size + i]
        batch_s[i] = _train[index] 

    return batch_s


training_data_count = len(X_train)  # 7689 training series 
test_data_count = len(X_test)  # 1890 testing series


## Convolution layers
###############################################################################

def create_new_conv_layer(input_data, num_input_channels, num_filters, filter_shape, pool_shape, name):
    # setup the filter input shape for tf.nn.conv_2d
    conv_filt_shape = [filter_shape[0], filter_shape[1], num_input_channels,
                      num_filters]

    # initialise weights and bias for the filter
    weights = tf.Variable(tf.truncated_normal(conv_filt_shape, stddev=0.03),
                                      name=name+'_W')
    bias = tf.Variable(tf.truncated_normal([num_filters]), name=name+'_b')

    # setup the convolutional layer operation
    out_layer = tf.nn.conv2d(input_data, weights, [1, 1, 1, 1], padding='SAME')

    # add the bias
    out_layer += bias

    # apply a ReLU non-linear activation
    out_layer = tf.nn.relu(out_layer)

    # now perform max pooling
    ksize = [1, pool_shape[0], pool_shape[1], 1]
    strides = [1, 2, 2, 1]
    out_layer = tf.nn.max_pool(out_layer, ksize=ksize, strides=strides, 
                               padding='SAME')

    return out_layer
    
layer1 = create_new_conv_layer(x_shaped, 1, 32, [5, 5], [2, 2], name='layer1')
layer2 = create_new_conv_layer(layer1, 32, 64, [5, 5], [2, 2], name='layer2')
layer3 = create_new_conv_layer(layer2, 64, 128, [3, 3], [2, 2], name='layer3')
layer4 = create_new_conv_layer(layer3, 128, 256, [3, 3], [2, 2], name='layer4')

## Fully connected layers
###############################################################################
flattened = tf.reshape(layer4, [-1, 9*25*256])
# setup some weights and bias values for this layer, then activate with ReLU
wd1 = tf.Variable(tf.truncated_normal([9*25*256, 1000], stddev=0.03), name='wd1')
bd1 = tf.Variable(tf.truncated_normal([1000], stddev=0.01), name='bd1')
dense_layer1 = tf.matmul(flattened, wd1) + bd1
dense_layer1 = tf.nn.relu(dense_layer1)
# apply DropOut to hidden layer
drop_out1 = tf.nn.dropout(dense_layer1, keep_prob)

# another layer with softmax activations
wd2 = tf.Variable(tf.truncated_normal([1000, N_CLASSES], stddev=0.03), name='wd2')
bd2 = tf.Variable(tf.truncated_normal([N_CLASSES], stddev=0.01), name='bd2')
#dense_layer2 = tf.matmul(dense_layer1, wd2) + bd2
dense_layer2 = tf.matmul(drop_out1, wd2) + bd2
y_ = tf.nn.softmax(dense_layer2)


## Cost function and Optimizer
###############################################################################
l2 = lambda_loss_amount * sum(
    tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables()) # L2 loss prevents this overkill neural network to overfit the data

#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=dense_layer2, labels=y))
cost = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=dense_layer2, targets=y, pos_weight = np.array([1, 1, 1, 1]))) + l2


# add an optimiser
train_op = tf.train.AdamOptimizer(LEARNING_RATE)
gradients = train_op.compute_gradients(cost)
capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients]
optimizer = train_op.apply_gradients(capped_gradients)

# define an accuracy assessment operation
correct_pred = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy_tf = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


## TRAINING
###############################################################################

test_losses = []
test_accuracies = []
train_losses = []
train_accuracies = []

# Launch the graph
#sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=False))
init = tf.global_variables_initializer()
sess.run(init)

# Perform Training steps with "batch_size" amount of example data at each loop
for epoch in range(NUM_EPOCHS):
    v_idx_permu = np.random.permutation(training_data_count)
    epoch_loss = 0
    epoch_acc = 0
    for step in range(int(training_data_count/BATCH_SIZE)):
        batch_xs =         extract_batch_size(X_train, v_idx_permu, step, BATCH_SIZE)
        batch_ys = one_hot(extract_batch_size(Y_train, v_idx_permu, step, BATCH_SIZE))

        # Fit training using batch data
        _, loss, acc = sess.run([optimizer, cost, accuracy_tf],
                                feed_dict={
                                        x: batch_xs, 
                                        y: batch_ys,
                                        keep_prob : 0.5
                                        }
                                )
        epoch_loss += loss
        epoch_acc += acc
    epoch_loss /= float(step+1)
    epoch_acc /= float(step+1)
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc)

    # Evaluate network only at some steps for faster training:       
    # To not spam console, show training accuracy/loss in this "if"
    print("Training epoch #" + str(epoch) + \
          ":   Batch Loss = " + "{:.6f}".format(epoch_loss) + \
          ", Accuracy = {}".format(epoch_acc))


print("Optimization Finished!")

# Accuracy for test data


#one_hot_predictions, accuracy, final_loss = sess.run(
#    [logits, accuracy_tf, cost],
#    feed_dict={
#        x: X_test,
#        y: one_hot(Y_test)
#    }
#)
step = 1
final_loss = 0
accuracy = 0
one_hot_predictions = np.empty((0,4))
v_idx_permu = np.arange(test_data_count)
for step in range(int(test_data_count/BATCH_SIZE)):
    batch_xs =         extract_batch_size(X_test, v_idx_permu, step, BATCH_SIZE)
    batch_ys = one_hot(extract_batch_size(Y_test, v_idx_permu, step, BATCH_SIZE))

    # Fit training using batch data
    one_hot_temp, accuracy_tmp, final_loss_tmp = sess.run([y_, accuracy_tf, cost],
                                                feed_dict={
                                                        x: batch_xs, 
                                                        y: batch_ys,
                                                        keep_prob : 1
                                                    }
                                                )
    one_hot_predictions = np.concatenate((one_hot_predictions,one_hot_temp),axis=0)
    accuracy += accuracy_tmp
    final_loss += final_loss_tmp
accuracy /= float(step+1)
final_loss /= float(step+1) 

print("FINAL RESULT: " + \
      "Batch Loss = {}".format(final_loss) + \
      ", Accuracy = {}".format(accuracy))
      

### VISUALISATION
################################################################################
import matplotlib
import matplotlib.pyplot as plt

width = 12
height = 12
plt.figure(figsize=(width, height))

indep_train_axis = np.array(list(range(BATCH_SIZE, (len(train_losses)+1)*BATCH_SIZE, BATCH_SIZE)))
plt.plot(indep_train_axis, np.array(train_losses),     "b--", label="Train losses")
plt.plot(indep_train_axis, np.array(train_accuracies), "g--", label="Train accuracies")

plt.title("Training session's progress over iterations")
plt.legend(loc='upper right', shadow=True)
plt.ylabel('Training Progress (Loss or Accuracy values)')
plt.xlabel('Training iteration')

plt.show()


### CONFUSION MATRIX
################################################################################
## Results
from sklearn import metrics

predictions = one_hot_predictions.argmax(1)

print("Testing Accuracy: {}%".format(100*accuracy))

print("")
print("Number of trainable parameters: " + str(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])))
print("Precision: {}%".format(100*metrics.precision_score(Y_test[:len(predictions)], predictions, average="weighted")))
print("Recall: {}%".format(100*metrics.recall_score(Y_test[:len(predictions)], predictions, average="weighted")))
print("f1_score: {}%".format(100*metrics.f1_score(Y_test[:len(predictions)], predictions, average="weighted")))

print("")
print("Confusion Matrix:")
confusion_matrix = metrics.confusion_matrix(Y_test[:len(predictions)], predictions)
print(confusion_matrix)
normalised_confusion_matrix = np.array(confusion_matrix, dtype=np.float32)/np.sum(confusion_matrix)*100

print("")
print("Confusion matrix (normalised to % of total test data):")
print(normalised_confusion_matrix)


