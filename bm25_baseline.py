import pyndri
import pandas as pd
import numpy as np
import os
import time

def getDocuments(index, bm25_query_env, query):
    list_of_series = []
    topic = query[:3]
    query = query[4:]

    # retrieve 2000 documents per query
    query_results = bm25_query_env.query(query, results_requested=2000)

    # iterate over query results
    for document_id, document_score in query_results:
        document_name, _ = index.document(document_id)
        series = pd.Series([topic, query, document_name, document_score])
        list_of_series.append(series)

    return list_of_series

t = time.time()

# read query validation set
filename = "data/validation_set/query_validation_set.txt"
base_filename, file_extension = os.path.splitext(filename)
output = f'{base_filename}.csv'
input = open(filename, "r")
lines = input.readlines()
input.close()

# index of corpus
index = pyndri.Index('Vol45/Vol45-index')

# define bm25 query environment
bm25_query_env = pyndri.OkapiQueryEnvironment(index, k1=1.2, b=0.75, k3=1000)

# retrieve documents and bm25 score
df = pd.DataFrame()
for i in range(len(lines)):
    query = lines[i].rstrip()
    list_of_series = getDocuments(index, bm25_query_env, query)
    df = pd.concat([df, pd.DataFrame(list_of_series)])

df.columns = ['topic', 'query', 'document_name', 'document_score']

# uncomment if you want to write queries and documents to csv
#df.to_csv(output, index=False, chunksize=1000)

# format output for trec_eval
input_size = df.shape[0]
df['Q0'] = np.array(['Q0'] * input_size)
df['rank'] = df.groupby('topic')['document_score'].rank(ascending=False)
df['run_tag'] = np.array(['bm25'] * input_size)

columns = ['topic', 'Q0', 'document_name', 'rank', 'document_score', 'run_tag']
bm25_ranking = df[columns]
bm25_ranking.to_csv(path_or_buf='data/validation_set/bm25_ranking.txt', sep=' ', index=False, header=False)

# check that documents are returned for all 250 queries
print(df.topic.nunique())

elapsed = time.time() - t
print('Elapsed time: %.2f' % elapsed)