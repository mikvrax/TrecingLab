#!/usr/bin/python3

import json

f= open("terms.json","r")
j = json.load(f)

f.close()

f=open("queries_training_set.txt","r")
l = f.readlines()
f.close()

o=open("query_terms.txt","w")
for i in range(len(l)):
    a = l[i].split("\n")
    b = a[0].split(" ")
    s=""
    for c in b:
        s+=str(j[c])+","

    o.write(s[:len(s)-1]+"\n")

o.close()
