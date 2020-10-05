import os
import json
import time
import re
from pprint import pprint
path = '学习笔记'
metas_filename = '_meta.json'
meta_data = {}

#获取文件创建时间
def getCreatedTime(path):
    if path in created_time:
        return created_time[path]
    print("%s not found in commit history"%path)
    t = os.path.getmtime(path)
    t = time.localtime(t)
    t = time.strftime('%Y-%m-%d %H:%M:%S', t)
    return t

#获取文件修改时间
def getLastUpdated(path):
    if path in updated_time:
        return updated_time[path]
    print("%s not found in commit history"%path)
    t = os.path.getmtime(path)
    t = time.localtime(t)
    t = time.strftime('%Y-%m-%d %H:%M:%S', t)
    return t

#获取某个目录下的meta数据
def getDIRMetas(filedir):
    metas = {}
    metasfile = os.path.join(filedir, metas_filename)
    if os.path.exists(metasfile):
        with open(metasfile, 'r', encoding='utf-8') as f:
            metas = json.load(f)#有此文件则读取文件
    print('Meta data loaded from dir %s' % filedir)
    return metas

#设置某个目录下的meta数据
def setDIRMetas(filedir,metas):
    if metas == {}:
        print('No meta data in %s' % filedir)
        return
    metasfile = os.path.join(filedir, metas_filename)
    with open(metasfile, 'w', encoding='utf-8') as f:
        json.dump(metas, f, ensure_ascii=False, indent=4)
    print('Meta data dumped to dir %s' % filedir)

# 获取某个.md文件的meta数据
titler = re.compile(r'#\s+(.*?)\n', re.S)
coverr = re.compile(r'!\[.*?\]\((.*?)\)\n', re.S)
titletagr = re.compile(r'^\((.*?)\)', re.S)
def updateMDMeta(filedir, filename, metas):
    print('Meta data collecting: %s' % filename)
    #先从metas里面找meta，找不到就用{}
    meta = metas[filename] if filename in metas else {}
    filepath = os.path.join(filedir, filename)
    with open(filepath, 'r', encoding='utf-8') as f:
        s = f.read() + '\n'

    title = re.findall(titler,s)
    if len(title)>0:
        meta['title'] = title[0]#文章标题数据直接覆盖

    if __name__=="__main__":#TODO:CI系统pygit2中的Object not found问题仍未解决
        meta['created'] = getCreatedTime(filepath)#日期数据直接覆盖
        meta['updated'] = getLastUpdated(filepath)#日期数据直接覆盖
        meta['date'] = meta['updated']
        if 'lastUpdated' in meta:
            del meta['lastUpdated']
    
    path_splitted = filedir.replace('\\','/').split('/')[1:]
    if not 'tags' in meta:#tag数据
        meta['tags'] = path_splitted
    
    meta['categories'] = path_splitted#目录数据直接覆盖

    for c in meta['categories']:
        if not c in meta['tags']:
            meta['tags'].append(c)
        
    titletag = re.findall(titletagr,meta['title']) #找文章开头的括号加入到tag中，比如“(未完成)”
    if len(titletag)>0 and not titletag[0] in meta['tags']:
        meta['tags'].append(titletag[0])
    
    cover = re.findall(coverr,s)
    if len(cover)>0:
        meta['cover'] = '/'+'/'.join(path_splitted)+'/'+cover[0]#封面数据直接覆盖
    pprint(meta)
    print('Meta data collected : %s' % filename)
    meta_data[filepath] = meta
    metas[filename] = meta
    return meta


def processMDIR(path):
    old_metas = getDIRMetas(path)#获取该目录下文件的meta数据
    new_metas = {}
    for file in os.listdir(path):#遍历当前目录下的每个子路径
        p = os.path.join(path, file)
        if os.path.isdir(p):#如果是文件夹就递归
            processMDIR(p)
        elif p[-3:] == '.md' and os.path.isfile(p):#是md文件就更新meta
            new_metas[file] = updateMDMeta(path,file,old_metas)
    setDIRMetas(path,new_metas)#最后将这个文件夹的meta数据写入到对应文件中
    print('Meta data processed in dir %s' % path)

#TODO:CI系统pygit2中的Object not found问题仍未解决
if __name__=="__main__":
    from getDate import created_time,updated_time

processMDIR(path)
