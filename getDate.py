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
    files={s:set() for s in STATUS}
    for patch in diff:
        delta=patch.delta
        files[delta.status].add(
            (delta.old_file.path,delta.new_file.path))
    return files

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
            data[file[1]]=data[file[0]]
        else:
            data[file[1]]=date
    for file in created_files:
        if not file[1] in data:
            data[file[1]]=date
    for file in deleted_files:
        data[file[1]]=date
    for file in modifed_files:
        data[file[1]]=date

#获取代码库
repo = pygit2.Repository('./')
#获取commit记录
commits=repo.walk(repo.head.target,GIT_SORT_TOPOLOGICAL|GIT_SORT_REVERSE)
import traceback
try:
    print(list(repo.walk(repo.head.target,GIT_SORT_TOPOLOGICAL|GIT_SORT_REVERSE)))
except:
    print("Unexpected error:", traceback.print_exc())

last_tree=None
for commit in commits:#开始遍历
    date=time.gmtime(commit.commit_time)#获取时间
    date=time.strftime('%Y-%m-%d %H:%M:%S',date)#格式化时间
    print("scanning commit %s at %s"%(commit.message,date))

    tree=commit.tree#获取文件树
    if last_tree is None:
        last_tree=tree
        continue

    diff=tree.diff_to_tree(last_tree)#比较文件树
    updateData(date,diff)
    last_tree=tree

data={k.replace("/",'\\'):v for k,v in data.items()}

    
