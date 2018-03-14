#!/usr/bin/python3

import tensorflow as tf

#load input data
files = open("query_testset.txt","r")
queries = files.readlines()
files.close()

files = open("document_lengths.txt","r")
doc_lengths = f.readlines()
files.close()

files = open("scores.txt","r")
scores = files.readlines()
files.close()

#Dense representation
#TODO pass input data into tensors
q = np.array([])
d = np.array([])
s = np.array([])


x_train = tf.placeholder(shape=,dtype=tf.float)
y_train = tf.placeholder(shape=,dtype=tf.float)

#input layer
#TODO create a vector as the one mentioned in the paper
z = 

#initial weights and biases
W = tf.Variable(initial_value = tf.zeros([]),dtype = tf.float32)
b = tf.Variable(initial_value = tf.zeros([]) ,dtype = tf.float32)

#RELU activation function
activationf = tf.nn.relu(tf.matmul(W,z)+b)

#1 hidden fully connected layer with 16 units and dropout for now 
hidden = tf.contrib.layers.fully_connected(inputs,num_outputs=16,activation_fn=activationf)
dropout = tf.layers.dropout(inputs=hidden)

#output fully connected layer with linear activation
S = tf.contrib.layers.fully_connected(dropout,num_outputs=1,activation_fn=None)

#loss to optimize
loss = tf.losses.mean_squared_error(s,S)

#optimize loss
train_op = tf.train.AdamOptimizer().minimize(loss)

sess = tf.Session()

#initialize variables
sess.run(tf.global_variables_initializers)

#training first
#TODO add display of final loss for training and test data. get output scores in
#order to run trec_eval on them
for step in range(10000):
    sess.run(fetches=[train_op], feed_dict={x_train: train_data, y_train: train_labels})
#    sess.run([metrics], feed=train_data) 
#    sess.run([metrics], feed=test_data)

#print(sess.run(fetches=output, feed_dict={})
sess.close()


