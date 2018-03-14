#!/usr/bin/python3.4

files = open("rob04.title.galago.docset","r")
lines = files.readlines()
files.close()

out = open("rob04-title.trectext","w")

for i in range(0,len(lines)):
    out.write("<DOC>\n")
    line = lines[i].split("\n")
    parts = line[0].split("\t")
    document_number = parts[0]
    out.write("<DOCNO>"+document_number+"</DOCNO>\n")
    terms_frequencies = parts[2]
    terms = terms_frequencies.split(" ")
    out.write("<TEXT>\n")
    for j in terms:
        frequencies = j.split(":")
        for k in range(0,int(frequencies[1])):
            out.write(frequencies[0]+" ")
    out.write("\n</TEXT>\n</DOC>\n")
 

out.close()
