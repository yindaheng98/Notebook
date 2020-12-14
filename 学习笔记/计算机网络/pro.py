import os
with open(os.path.split(os.path.abspath(__file__))[0]+"\\计算机网络复习.md", 'r', encoding='utf8') as f:
    t = ""
    for l in f:
        t += l
t=t.replace("\n\n","\n")
t=t.replace("\n\n","\n")
with open(os.path.split(os.path.abspath(__file__))[0]+"\\计算机网络复习.txt", 'w', encoding='utf8') as f:
    f.write(t)