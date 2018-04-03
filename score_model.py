import tensorflow as tf
import numpy as np
import pandas as pd

def input_fn(data_set):
    labels = data_set['document_score'].values

    FEATURES = ['document_count', 'mean_document_length', 'document_length', 'term_frequency_1', 'term_frequency_2',
                'term_frequency_3', 'term_frequency_4', 'term_frequency_5', 'document_frequency_1',
                'document_frequency_2',
                'document_frequency_3', 'document_frequency_4', 'document_frequency_5']

    feature_cols = data_set[FEATURES].as_matrix()

    return feature_cols, labels

# read training data
filename = 'query_training_set_00.csv'
training_input = 'data/10000queries/' + filename
df = pd.read_csv(training_input)

# get features and labels
feature_cols, labels = input_fn(df)

# parameters
input_size = len(feature_cols)
batch_size = 512
hidden_units = 1024
dropout = 0.5
learning_rate = 1e-3

# define tf placeholders
x_train = tf.placeholder(shape=[None,13],dtype=tf.float32)
y_train = tf.placeholder(dtype=tf.float32)

# initial weights and biases
W1 = tf.Variable(tf.zeros([13,hidden_units]),dtype = tf.float32)
b1 = tf.Variable(tf.zeros([]),dtype = tf.float32)
W2 = tf.Variable(tf.zeros([hidden_units,hidden_units]),dtype = tf.float32)
b2 = tf.Variable(tf.zeros([]),dtype = tf.float32)
W3 = tf.Variable(tf.zeros([hidden_units,hidden_units]),dtype = tf.float32)
b3 = tf.Variable(tf.zeros([]),dtype = tf.float32)
W4 = tf.Variable(tf.zeros([hidden_units,1]),dtype = tf.float32)
b4 = tf.Variable(tf.zeros([]),dtype = tf.float32)

# 1st hidden fully connected layer with ReLU
inputs1 = tf.matmul(x_train,W1)+b1
dropout1 = tf.layers.dropout(inputs=inputs1,rate=dropout)
hidden1 = tf.contrib.layers.fully_connected(dropout1,num_outputs=hidden_units)

# 2nd hidden fully connected layer with ReLU
inputs2 = tf.matmul(hidden1,W2)+b2
dropout2 = tf.layers.dropout(inputs=inputs2,rate=dropout)
hidden2 = tf.contrib.layers.fully_connected(dropout2,num_outputs=hidden_units)

# 3th hidden fully connected layer with ReLU
inputs3 = tf.matmul(hidden2,W3)+b3
dropout3 = tf.layers.dropout(inputs=inputs3,rate=dropout)
hidden3 = tf.contrib.layers.fully_connected(dropout3,num_outputs=hidden_units)

# 4th hidden fully connected layer with ReLU
inputs4 = tf.matmul(hidden3,W4)+b4
dropout4 = tf.layers.dropout(inputs=inputs4,rate=dropout)
hidden4 = tf.contrib.layers.fully_connected(dropout4,num_outputs=hidden_units)

# output fully connected layer with linear activation
S = tf.contrib.layers.fully_connected(hidden4,num_outputs=1,activation_fn=None)

# loss to optimize
loss = tf.losses.mean_squared_error(y_train,S)

# optimize loss
train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# start tf session with gpu options
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

# for 1 epoch
for epoch in range(1):
    permutation = np.random.permutation(input_size)

    for k in range(0, input_size, batch_size):
        batches = permutation[k:k + batch_size]

        batch = [feature_cols[batches], labels[batches]]
        _, loss_val = sess.run([train_op, loss], feed_dict={x_train: batch[0], y_train: batch[1]})

        if k % 1000 == 0:
            print('k = ' + str(k) + ' loss = ' + loss_val.astype('str'))

sess.close()