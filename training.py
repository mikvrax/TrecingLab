#!/usr/bin/python3

N = 260712.0
avgD =  571.2232693546903

f = open("query_test_set.txt","r")
l = f.readlines()
f.close()

f = open("test_results10","r")
p = f.readlines()
f.close()

f = open("rob04-title/rob04.title.galago.docset","r")
t = f.readlines()
f.close()

f = open("document_frequencies","r")
u = f.readlines()
f.close()

o = open("training10","w")

for i in range(0,10):
    q = l[i]
    c  = q.split("\n")
    a = c.split(" ")
    if (len(a) > 5):
       b = a[:5]
    else:
       b = a
    g = 0
    o.write(str(N)+","+str(avgD)+",")
    for j in range(0,len(p)):
        d = p[j].split("\n")
        e = d[0].split("\t")
        if (int(e[0]) > i):
            break
        elif int(e[0]) == i:
            g = c+1
            if (g<=1000):
                for k in range(0,len(t)):
                    h = t[k].split("\n")
                    l = h[0].split("\t")
                    if l[0] == e[1]:
                        o.write(l[1]+",")
                        m = l[2].split(" ")
                        for s in range(0,len(b)):
                            for n in range(0,len(m)):
                                r = m[n].split(":")
                                if (r[0] == b[s]):
                                     o.write(u[i][s]+","+r[1]+",") 
                        if (len(b) < 5):
                            for v in range(len(b),5):
                                o.write(+str(0.0)+","+str(0.0)+",")   
                        o.write("\n")
            else:
                break            

o.close()

