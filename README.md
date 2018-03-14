# Partial reproduction of Neural Ranking Models with Weak Supervision

## Files 

- filter\_queries.py - keeps only important queries from dataset AOL-user-ct-collection
- create\_validation\_set.py - copies queries from topics 301-450, 601-700 of the TREC Robust 04 track
- diff\_training\_validation.py - removes duplicate queries between training and validation set
- parametrize\_queries.py - creates the xml file needed by the Indri indexing and query engine to run queries
- create\_documents.py - uses the term index table for every document in the Robust04-Title dataset to recreate its contents
- document\_lengths.py - uses the term index table to calculate the length of each document in the Robust04-Title dataset and get the average length
- scorenn.py - not finished implementation of the score model for a neural network as described in the paper

## Remaining

- Get document frequency for each query term and query term frequency for all documents
- Run queries on corpus index to keep only those that have more than 10 hits and keep their scores to be used as an input for the neural networks
- Get all features needed for the 3 input data representations
- Finish neural networks' implementations


