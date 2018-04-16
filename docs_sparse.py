#!/usr/bin/python3


import pyndri

index = pyndri.Index("FinalIndex")

f = index.get_term_frequencies()
dm = index.maximum_document()
t = index.document_base()
m = max(f.keys())

o=open("docs_vec.txt","w")

for i in range(t,dm):
    d = index.document(i)
    c = 0
    s=""
    for j in range(1,m):
        if d[1].count(j) == 0:
           c = c+1
        else:
           if c > 0:
              s+= "0," +str(c)+","
           s += str(d[1].count(j)) +","
           c = 0
    o.write(s[:len(s)-1] +"\n")

