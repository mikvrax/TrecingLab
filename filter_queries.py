#!/usr/bin/python3.4
import re

path = "data/raw/"
filename = path + "user-ct-test-collection-0"
out = open("data/aol_queries.txt","w")
substrings = ["http", "www.", ".com", ".net", ".org", ".edu"]
queries = []


#iterate over the lines of the first 9 files, get queries that don't contain substrings and make them contain only alpharithmetic characters
for i in range(1,10):
    files = open(filename + str(i)+".txt","r")
    lines = files.readlines()
    files.close()
    for j in range(0,len(lines)):
        query = lines[j].split("\t")[1]
        flag = 0
        for string in substrings:
            try:
                query.index(string)
                flag = 1
            except ValueError:
                continue
        if flag == 0:
            query = re.sub("[^a-zA-Z0-9 ]", '', query)
            query = " ".join(query.split())
            if (len(query) != 0):
                queries.append(query+"\n")
      
#do the same for the tenth file
files = open(path + "user-ct-test-collection-10.txt","r")
lines = files.readlines()
files.close()

for j in range(0,len(lines)):
    query = lines[j].split("\t")[1]
    flag = 0
    for string in substrings:
        try:
            query.index(string)   
            flag = 1
        except ValueError:
            continue
    if flag == 0:
        query = re.sub("[^a-zA-Z0-9 ]", '', query)
        query = " ".join(query.split())
        if (len(query) != 0):
            queries.append(query+"\n")
 
#remove duplicates by using a set
queries_set = set(queries)

for i in queries_set:
    out.write(i)

out.close()
