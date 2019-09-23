import os
import json
import time
path = '学习笔记'
meta_data = {}
try:
    with open('meta.json', 'r', encoding='utf-8') as f:
        meta_data = json.load(f)
except:
    pass


def getDate(path):
    t = os.path.getmtime(path)
    t = time.localtime(t)
    t = time.strftime('%Y-%m-%d %H:%M:%S', t)
    return t


def processMD(path):
    meta = meta_data[path] if path in meta_data else {}
    date = getDate(path)
    with open(path, 'r', encoding='utf-8') as f:
        title = f.readline()[1:-1]
    if not 'date' in meta:
        meta['date'] = date
    if not 'title' in meta:
        meta['title'] = title
    if not 'tags' in meta:
        meta['tags'] = path.split('/')[1:-1]
    meta_data[path] = meta
    print('Meta data collected in file %s' % path)


def processMDIR(path):
    for i in os.listdir(path):
        p = path+'/'+i
        if os.path.isdir(p):
            processMDIR(p)
        elif p[-3:] == '.md' and os.path.isfile(p):
            processMD(p)
    print('Meta data collected in directory %s' % path)


if __name__ == "__main__":
    processMDIR(path)
    with open('meta.json', 'w', encoding='utf-8') as f:
        json.dump(meta_data, f)
