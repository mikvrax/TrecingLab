#!/usr/bin/python3

files = open("query_testset.txt","r")
training_queries = files.readlines()
files.close()

files = open("query_validationset.txt","r")
validation_queries = files.readlines()
files.close()

out = open("query_test_set.txt","w")

for i in range(0,len(training_queries)):
    if training_queries[i] not in validation_queries:
        out.write(training_queries[i])

out.close()

