import os
import time
import json
import pygit2
from pygit2 import GIT_SORT_TOPOLOGICAL,GIT_SORT_REVERSE

STATUS={pygit2.GIT_DELTA_ADDED:"A",
        pygit2.GIT_DELTA_COPIED:"C",
        pygit2.GIT_DELTA_DELETED:"D",
        pygit2.GIT_DELTA_IGNORED:"I",
        pygit2.GIT_DELTA_MODIFIED:"M",
        pygit2.GIT_DELTA_RENAMED:"R",
        pygit2.GIT_DELTA_TYPECHANGE:"T",
        pygit2.GIT_DELTA_UNMODIFIED:"UM",
        pygit2.GIT_DELTA_UNREADABLE:"UNREADABLE",
        pygit2.GIT_DELTA_UNTRACKED:"UNTRACKED"}

data={}
def parseDiff(diff):
    diff.find_similar()
    files={s:set() for s in STATUS}
    for patch in diff:
        delta=patch.delta
        files[delta.status].add(
            (delta.old_file.path,delta.new_file.path))
    md_files={}
    for status in files:
        fs=set()
        for file in files[status]:
            if file[0][-3:]=='.md':
                fs.add(file)
        md_files[status]=fs
    return md_files

def constructDiff(diff):
    files=parseDiff(diff)
    renamed_files=files[pygit2.GIT_DELTA_RENAMED]
    created_files=files[pygit2.GIT_DELTA_ADDED]
    deleted_files=files[pygit2.GIT_DELTA_DELETED]
    for created_file in created_files:
        created_name=os.path.split(created_file[1])#获取新建文件名
        for deleted_file in deleted_files:
            deleted_name=os.path.split(deleted_file[1])#获取删除文件名
            if created_name==deleted_name:#比对文件名，相同则更新
                created_files.remove(created_file)#删
                deleted_files.remove(deleted_file)#删
                renamed_files.add((deleted_file,created_file))#加
    files[pygit2.GIT_DELTA_RENAMED]=renamed_files
    files[pygit2.GIT_DELTA_ADDED]=created_files
    files[pygit2.GIT_DELTA_DELETED]=deleted_files
    return files

def updateData(date,diff):
    files=constructDiff(diff)
    renamed_files=files[pygit2.GIT_DELTA_RENAMED]
    created_files=files[pygit2.GIT_DELTA_ADDED]
    deleted_files=files[pygit2.GIT_DELTA_DELETED]
    modifed_files=files[pygit2.GIT_DELTA_MODIFIED]
    for file in renamed_files:
        if file[0] in data:
            print('\trename file: %s->%s'%(file[0],file[1]))
            data[file[1]]=data[file[0]]
        else:
            print('\t[not exists]rename file: %s->%s'%(file[0],file[1]))
            data[file[1]]=date
    for file in created_files:
        if not file[1] in data:
            print('\tcreate file: %s'%file[1])
            data[file[1]]=date
        else:
            print('\t[already exists]create file: %s'%file[1])
    for file in modifed_files:
        print('\tmodify file: %s'%file[1])
        data[file[1]]=date
    for file in deleted_files:
        if file[1] in data:
            print('\tdelete file: %s'%file[1])
        else:
            print('\t[not exists]delete file: %s'%file[1])

#获取代码库
repo = pygit2.Repository('./')
#获取commit记录
commits=repo.walk(repo.head.target,GIT_SORT_TOPOLOGICAL|GIT_SORT_REVERSE)

last_tree=None
for commit in commits:#开始遍历
    date=time.gmtime(commit.commit_time)#获取时间
    date=time.strftime('%Y-%m-%d %H:%M:%S',date)#格式化时间
    print("\nscanning the commit at %s: %s"%(date,commit.message))

    tree=commit.tree#获取文件树
    if last_tree is None:
        last_tree=tree
        continue

    diff=last_tree.diff_to_tree(tree,interhunk_lines=10)#比较文件树
    updateData(date,diff)
    last_tree=tree

data={k.replace("/",'\\'):v for k,v in data.items()}

    
