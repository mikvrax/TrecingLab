# Partial reproduction of Neural Ranking Models with Weak Supervision

## Files 

- filter\_queries.py - keeps only important queries from dataset AOL-user-ct-collection
- diff\_training\_validation.py - removes duplicate queries between training and validation set
- parametrize\_queries.py - creates the xml file needed by the Indri indexing and query engine to run queries
- docs\_sparse.py - creates compressed sparse representation of corpus documents 
- queries\_sparse.py - creates compressed sparse representation of training queries
- embeddings.py - create word embeddings of documents
- query\_embeddings.py - create word embeddings of training queries
- score\_sparse.py - uses sparse representation as input for the score neural network
- rank\_embed.py - uses embeddings as input for rank neural network
- bm25\_baseline.py - runs BM25 on the queries and corpus
- score\_model.py - uses dense representation on score neural network
