#!/usr/bin/python3


import pyndri
import json

index = pyndri.Index("FinalIndex")

p = open("queries_training_set.txt","r")
l= p.readlines()
p.close()

o=open("terms.json","w")

t,i,d = index.get_dictionary() 
del i
del d
m = max(t.values())

for i in range(len(l)):
    a = l[i].split("\n")
    b = a[0].split(" ")
    for c in b:
        if c not in t.keys():
            t[c] = len(t.keys()) + 1
        
json.dump(t,o)


