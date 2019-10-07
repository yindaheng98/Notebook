import os
import json
import time
path = '学习笔记'
metas_filename = '_meta.json'
meta_data = {}

#获取文件创建时间
def getDate(path):
    t = os.path.getmtime(path)
    t = time.localtime(t)
    t = time.strftime('%Y-%m-%d %H:%M:%S', t)
    return t

#获取某个目录下的meta数据
def getDIRMetas(filedir):
    metas = {}
    metasfile = os.path.join(filedir, metas_filename)
    if (not os.path.exists(metasfile)) or os.path.isdir(metasfile):
        with open(metasfile, 'w', encoding='utf-8') as f:
            json.dump(metas, f, ensure_ascii=False, indent=4)#没有此文件则创建文件
    else:
        with open(metasfile, 'r', encoding='utf-8') as f:
            metas = json.load(f)#有此文件则读取文件
    print('Meta data loaded from dir %s' % filedir)
    return metas

#设置某个目录下的meta数据
def setDIRMetas(filedir,metas):
    metasfile = os.path.join(filedir, metas_filename)
    with open(metasfile, 'w', encoding='utf-8') as f:
        json.dump(metas, f, ensure_ascii=False, indent=4)
    print('Meta data dumped to dir %s' % filedir)

# 获取某个.md文件的meta数据
def getMDMeta(filedir, filename, metas):
    #先从metas里面找meta，找不到就用{}
    meta = metas[filename] if filename in metas else {}
    filepath = os.path.join(filedir, filename)

    with open(filepath, 'r', encoding='utf-8') as f:
        meta['title'] = f.readline()[1:-1]#文章标题数据直接覆盖
    if not 'date' in meta:#日期数据
        meta['date'] = getDate(filepath)
    
    path_splitted = filedir.split('\\')[1:]
    if not 'tags' in meta:#tag数据
        meta['tags'] = path_splitted
    meta['categories'] = path_splitted#目录数据直接覆盖

    print('Meta data of file %s collected' % filename)
    meta_data[filepath] = meta
    return meta


def processMDIR(path):
    metas = getDIRMetas(path)
    for i in os.listdir(path):
        p = os.path.join(path, i)
        if os.path.isdir(p):
            processMDIR(p)
        elif p[-3:] == '.md' and os.path.isfile(p):
            metas[i] = getMDMeta(path,i,metas)
    setDIRMetas(path,metas)
    print('Meta data processed in dir %s' % path)

processMDIR(path)
