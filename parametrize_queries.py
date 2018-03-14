#!/usr/bin/python3.4

files = open("query_testset.txt","r")
lines = files.readlines()
files.close()

out = open("training_queries.xml","w")
out.write("<parameters>\n")
out.write("<index>/home/user/Test/Index</index>\n<runID>test</runID>\n<trecFormat>true</trecFormat>\n")
part1 = "<query>\n<number>"
part2 = "</number>\n<text>"
part3 = "</text>\n</query>\n<count>10</count>\n<baseline>okapi</baseline>\n"

for i in range(0,len(lines)):
    line = lines[i].split("\n")
    words = line[0].split(" ")
    query = ""
    for j in range(0,len(words)):
        query = query + " " + words[j]
    out.write(part1+str(i)+part2+query+part3)

out.write("</parameters>\n")
out.close()

