import time
import os


def getDate(path):
    t = os.path.getmtime(path)
    t = time.localtime(t)
    t = time.strftime('%Y-%m-%d %H:%M:%S', t)
    return t


def appendFILE(path, content):
    with open(path, 'r+', encoding='utf-8') as f:
        old = f.read()
        f.seek(0)
        f.write(content)
        f.write(old)


def getHEAD(head):
    HEAD = "---\n"
    for k in head:
        HEAD += "%s: %s\n" % (k, head[k])
    HEAD += "---\n"
    return HEAD


def processMD(path, tags):
    date = getDate(path)
    with open(path, 'r', encoding='utf-8') as f:
        title = f.readline()[1:]
    appendFILE(path, getHEAD({'title': title, 'date': date}))
    print('Processed file %s' % path)


def processMDIR(path):
    for i in os.listdir(path):
        p = path+'/'+i
        if os.path.isdir(p):
            processMDIR(p)
        elif p[-3:] == '.md' and os.path.isfile(p):
            processMD(p, None)
    print('Processed directory %s' % path)


processMDIR('./学习笔记')
