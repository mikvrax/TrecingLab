import tensorflow as tf
import numpy as np
import math
import scipy.sparse
import random


# read query document pairs
filename = 'query_training_set_00.csv'

a = open(filename,'r')
b = a.readlines()
a.close()

# read compressed sparse representation of queries
c = open('queries_sparse.csv','r')
d = c.readlines()
c.close()

# read compressed sparse representation of documents
h = open('docs_vec9.txt','r')
o = h.readlines()
h.close()

# read compressed sparse representation of corpus
jj = open('corpus_sparse.txt','r')
kk = jj.readline()
jj.close()

# create sparse vector for corpus
ll1 = []
ll2 = []
ll3 = []

mm = kk.split(',')
for i in range(len(mm)):
    ll1.append(0)
    ll2.append(i)
    ll3.append(int(mm[i]))

ll = scipy.sparse.coo_matrix((ll3,(ll1,ll2)), shape=(1,2534163))

# link document ids with document names
aa = open('docs_ids.txt','r')
pp = aa.readlines()
aa.close()
bb = set(pp)
cc = list(bb)


p=len(b)
labels=np.empty(shape=(50,1))
feature_cols=np.empty(shape=(50,7602489))

# create training sparse vectors
for i in range(50):
    t = b[i].split('\n')
    s = t[0].split(',')
    labels[i]= (float(s[2]))
    if i%100 == 0:
        q = math.floor(i/100)
        r1 = []
        r2 = []
        r3 = []
        u = d[q]
        v = u.split('\n')
        z = v[0].split(',')
        j = 0
        while(j<len(z)-1):
            if (z[j] == '0'):
                r1.append(0)
                r2.append(int(z[j+1]))
                r3.append(int(z[j+2]))
                j = j + 3
            else:
                j = j+1
        r = scipy.sparse.coo_matrix((r3,(r1,r2)), shape=(1,2534163))
    ee = s[1]+'\n'
    ff = cc.index(ee)
    gg = o[ff]
    hh = gg.split(',')
    j = 0
    ii1 = []
    ii2 = []
    ii3 = []
    while(j<len(hh)):
        if (hh[j] == '0'):
            ii1.append(0)
            ii2.append(int(hh[j+1]))
            ii3.append(int(hh[j+2]))
            j = j + 3
        else:
            j = j + 1
    ii = scipy.sparse.coo_matrix((ii3,(ii1,ii2)), shape=(1,2534163))
    feature_cols[i] = np.concatenate((np.asarray(ll.todense()),np.asarray(r.todense()),np.asarray(ii.todense())),axis=1)
                
# release memory
del r
del ii
del cc
del bb
del pp
del mm
del kk
del b
del d

#parameters
input_size = 50
batch_size = 50
hidden_units = 16
dropout = 0
learning_rate = 1e-5

# define tf placeholders
x_train = tf.placeholder(shape=[None,7602489],dtype=tf.float32)
y_train = tf.placeholder(dtype=tf.float32)

# initial weights and biases
W1 = tf.Variable(tf.zeros([7602489,hidden_units]),dtype = tf.float32,name='W1')
b1 = tf.Variable(tf.zeros([]),dtype = tf.float32,name='b1')
# 1st hidden fully connected layer with ReLU
inputs1 = tf.matmul(x_train,W1)+b1
dropout1 = tf.layers.dropout(inputs=inputs1,rate=dropout,name='dropout1')
hidden1 = tf.contrib.layers.fully_connected(dropout1,num_outputs=hidden_units,scope='hidden1')
# output fully connected layer with linear activation
S = tf.contrib.layers.fully_connected(hidden1,num_outputs=1,activation_fn=None,scope='S')

# loss to optimize
loss = tf.losses.mean_squared_error(y_train,S)

# optimize loss
train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

sess = tf.Session()
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
    # read compressed sparse test representations
    e = open('sparse_rep2.txt','r')
    g = e.readlines()
    e.close()
    
    # create test sparse vectors
    valid = np.empty(shape=(50,7602489))
    for i in range(50):
        vv1 =[]
        vv2 = []
        vv3 = []
        qq = g[i].split('\n')
        zz = qq[0].split(',')
        j=0
        while j<len(zz)-1:
            if (zz[j] == '0'):
                vv1.append(0)
                vv2.append(int(zz[j+1]))
                vv3.append(int(zz[j+2]))
                j=j+3
            else:
                j=j+1
        vv = scipy.sparse.coo_matrix((vv3,(vv1,vv2)), shape=(1,2534163))
        yy =[]
        for j in range(1):
            xx =round( random.random() * len(o)) - 1
            while (xx in yy):
                xx =round( random.random() * len(o)) - 1
            yy.append(xx)
            gg = o[xx]
            print(xx)
            hh = gg.split(',')
            j = 0
            ii1 = []
            ii2 = []
            ii3 = []
            while(j<len(hh)):
                if (hh[j] == '0'):
                    ii1.append(0)
                    ii2.append(int(hh[j+1])) 
                    ii3.append(int(hh[j+2]))
                    j = j + 3
                else:
                    j = j + 1
                    ii = scipy.sparse.coo_matrix((ii3,(ii1,ii2)), shape=(1,2534163))
        valid[i]=(np.concatenate((np.asarray(ll.todense()),np.asarray(vv.todense()),np.asarray(ii.todense())),axis=1))
    scores = sess.run([S], feed_dict={x_train: valid})


print(scores)
sess.close()
