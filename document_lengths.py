#!/usr/bin/python3.4

files = open("rob04.title.galago.docset","r")
lines = files.readlines()
files.close()

document_length = []

for i in range(0,len(lines)):
    line = lines[i].split("\n")
    parts = line[0].split("\t")
    term_indexes = parts[2].split(" ")
    length = 0
    for j in term_indexes:
        frequencies = j.split(":")  
        length += int(frequencies[1])
    document_length.append(length)

out = open("document_lengths.txt","w")
avg_document_length = 0
for i in range(0,len(document_length)):
    avg_document_length += document_length[i]
    out.write(str(document_length[i])+|"\n")

avg_document_length /= len(document_length)
out.write(str(avg_document_length))
out.close()


