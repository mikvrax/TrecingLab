#!/usr/bin/python3

files = open("rob04.title.krovetz.txt","r")
lines = files.readlines()
files.close()

out = open("query_validationset.txt","w")

#get only the text of each query
for i in range(0,len(lines)):
    query = lines[i][4:]
    query = query.replace("\t"," ")
    out.write(query)

out.close()
    


