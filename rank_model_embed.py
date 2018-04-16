import tensorflow as tf
import numpy as np
import pandas as pd
import json
import time

def input_fn(data_set, query_terms, document_terms, batch):
    queries = data_set['query']
    documents = data_set['document_id']

    # find document2 indices
    batch_pairs = []
    for x in batch:
        query = queries.loc[x]
        if queries.get(x-1, 0) == query:
            batch_pairs.append(x-1)
        else:
            batch_pairs.append(x+1)

    # bm25 scores of document1 and document2
    labels1 = data_set['document_score'].loc[batch].values
    labels2 = data_set['document_score'].loc[batch_pairs].values

    # query and document1 terms
    queries_terms = []
    documents_terms1 = []
    for i in batch:
        query_split = queries.loc[i].split()
        query_split_terms = []
        for j in range(3):
            if j < len(query_split_terms):
                query_term_id = query_terms[query_split[j]]
                query_split_terms.append(query_term_id)
            else:
                query_split_terms.append(0)

        queries_terms.append(query_split_terms)

        document_id1 = documents.loc[i]
        document_split_terms1 = []
        for k in range(100): #479
            doc_terms = document_terms[str(document_id1)]
            if k < len(doc_terms):
                document_split_terms1.append(doc_terms[k])
            else:
                document_split_terms1.append(0)

        documents_terms1.append(document_split_terms1)

    #document2 terms
    documents_terms2 = []
    for l in batch_pairs:
        document_id2 = documents.loc[l]
        document_split_terms2 = []
        for j in range(100):
            doc_terms = document_terms[str(document_id2)]
            if j < len(doc_terms):
                document_split_terms2.append(doc_terms[j])
            else:
                document_split_terms2.append(0)

        documents_terms2.append(document_split_terms2)

    return np.array(queries_terms, dtype=np.uint32), np.array(documents_terms1, dtype=np.uint32), \
           np.array(documents_terms2, dtype=np.uint32), labels1, labels2

print('Reading query and document term ids')
t = time.time()

query_terms = json.load(open('data/query_terms.json'))
document_terms = json.load(open('data/document_terms.json'))

elapsed = time.time() - t
print('Elapsed time: %.2f' % elapsed)

print('Reading training data')
t = time.time()

training_set = []
for i in range(0, 1):
    filename = 'query_training_set_' + str(i) + '.csv'
    training_input = 'data/training_splitted/pointwise/' + filename
    print('Reading: ' + filename)
    df = pd.read_csv(training_input, dtype={'document_id': np.uint32, 'document_score': np.float32})
    training_set.append(df)

training_set = pd.concat(training_set, axis=0)
elapsed = time.time() - t
print('Elapsed time: %.2f' % elapsed)

# parameters
input_size = training_set.shape[0]
vocabulary_size = len(query_terms)
embedding_size = 100
batch_size = 1
hidden_units = 512
dropout = 0.2
learning_rate = 1e-3

print('Input size: ' + str(input_size))
print('Vocabulary size: ' + str(vocabulary_size))

#with tf.device('/cpu:0'):
# word embeddings
epsilon_embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0), name='epsilon')
omega_embeddings = tf.Variable(tf.random_uniform([vocabulary_size, 1], -1.0, 1.0), name='omega')

# define tf placeholders
query_train = tf.placeholder(shape=[None, None], dtype=tf.int32, name='query')
document_train1 = tf.placeholder(shape=[None, None], dtype=tf.int32, name='document1')
document_train2 = tf.placeholder(shape=[None, None], dtype=tf.int32, name='document2')
y_train1 = tf.placeholder(dtype=tf.float32, name='label1')
y_train2 = tf.placeholder(dtype=tf.float32, name='label2')

# query embeddings lookup
epsilon_query = tf.nn.embedding_lookup(epsilon_embeddings, query_train)
omega_query = tf.nn.embedding_lookup(omega_embeddings, query_train)

# document1 embeddings lookup
epsilon_document1 = tf.nn.embedding_lookup(epsilon_embeddings, document_train1)
omega_document1 = tf.nn.embedding_lookup(omega_embeddings, document_train1)

# document2 embeddings lookup
epsilon_document2 = tf.nn.embedding_lookup(epsilon_embeddings, document_train2)
omega_document2 = tf.nn.embedding_lookup(omega_embeddings, document_train2)

# compositionality function query
print(epsilon_query.get_shape().as_list())
exp_omega_query = tf.exp(omega_query)
print(exp_omega_query.get_shape().as_list())
embed_query = tf.divide(tf.reduce_sum(tf.matmul(exp_omega_query, epsilon_query, transpose_a=True),2), tf.reduce_sum(exp_omega_query,1))
print(embed_query.get_shape().as_list())

# compositionality function document1
exp_omega_document1 = tf.exp(omega_document1)
embed_document1 = tf.divide(tf.reduce_sum(tf.matmul(exp_omega_document1, epsilon_document1, transpose_a=True),2), tf.reduce_sum(exp_omega_document1,1))
print(embed_document1.get_shape().as_list())

# compositionality function document2
exp_omega_document2 = tf.exp(omega_document2)
embed_document2 = tf.divide(tf.reduce_sum(tf.matmul(exp_omega_document2, epsilon_document1, transpose_a=True),2), tf.reduce_sum(exp_omega_document2,1))
print(embed_document2.get_shape().as_list())

# input vectors
embed1 = tf.concat([embed_query, embed_document1], 1)
embed2 = tf.concat([embed_query, embed_document2], 1)
print(embed1.get_shape().as_list())
print(embed2.get_shape().as_list())

def build_layers(embed):
    hidden1 = tf.layers.dense(inputs=embed, units=hidden_units, activation=tf.nn.relu, name='hidden1', reuse=tf.AUTO_REUSE)
    dropout1 = tf.layers.dropout(inputs=hidden1,rate=dropout, name='dropout1')

    hidden2 = tf.layers.dense(inputs=dropout1, units=hidden_units, activation=tf.nn.relu, name='hidden2', reuse=tf.AUTO_REUSE)
    dropout2 = tf.layers.dropout(inputs=hidden2,rate=dropout, name='dropout2')

    hidden3 = tf.layers.dense(inputs=dropout2, units=hidden_units, activation=tf.nn.relu, name='hidden3', reuse=tf.AUTO_REUSE)
    dropout3 = tf.layers.dropout(inputs=hidden3,rate=dropout, name='dropout3')

    return tf.layers.dense(inputs=dropout3, units=1, activation=tf.nn.tanh, name='output', reuse=tf.AUTO_REUSE)

# scoring
S1 = build_layers(embed1)
S2 = build_layers(embed2)

# loss to optimize
margin = 1.0
loss = tf.reduce_mean(tf.maximum(0.0, margin - tf.multiply(tf.sign(y_train1 - y_train2),(S1 - S2))))

# optimize loss
train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# start tf session with gpu options
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
run_options = tf.RunOptions(report_tensor_allocations_upon_oom = True)
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# for 1 epoch
for epoch in range(1):
    permutation = np.random.permutation(input_size)

    counter = 0
    for k in range(0, input_size, batch_size):
        counter += 1

        batch = permutation[k:k + batch_size]
        queries_terms, documents_terms1, documents_terms2, labels1, labels2 = input_fn(training_set, query_terms, document_terms, batch)

        loss_val, predictions1, predictions2 = sess.run([loss, S1, S2], feed_dict={query_train: queries_terms,
                                               document_train1: documents_terms1, document_train2: documents_terms2,
                                               y_train1: labels1, y_train2: labels2}, options=run_options)

        if k % 1000 == 0:
            print('counter = ' + str(counter) + ' k = ' + str(k) + ' loss = ' + loss_val.astype('str'))

        if counter == 1000000:
            break

save_path = saver.save(sess, "saved_models/rank_model_embed.ckpt")
print("Model saved in path: %s" % save_path)
sess.close()