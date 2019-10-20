from meta import meta_data, path
import time
import os
import shutil

#在文件开头进行添加
def appendFILE(path, content):
    with open(path, 'r+', encoding='utf-8') as f:
        old = f.read()
        f.seek(0)
        f.write(content)
        f.write(old)

#获取文件头信息
def getHEAD(head):
    HEAD = "---\n"
    for k in head:
        if isinstance(head[k], list):
            content = '\n'
            for i in head[k]:
                content += ' - '+i+'\n'
            HEAD += "%s: %s\n" % (k, content[0:-1])
        else:
            HEAD += "%s: %s\n" % (k, head[k])
    HEAD += "---\n"
    print(HEAD)
    return HEAD

#处理.md文件
def processMD(path, meta):
    appendFILE(path, getHEAD(meta))
    print('Processed file %s' % path)

#进行MD文件目录的处理
#扫描文件目录
#如果遇到文件目录，则processMDIR递归
#如果遇到.md文件则调用processMD处理
def processMDIR(path, meta_data):
    for i in os.listdir(path):
        p = os.path.join(path, i)
        if os.path.isdir(p):
            processMDIR(p, meta_data)
        elif p[-3:] == '.md' and os.path.isfile(p):
            processMD(p, meta_data[p] if p in meta_data else {})
    print('Processed directory %s' % path)


if __name__ == "__main__":
    shutil.copytree(path, path+'.bak')
    processMDIR(path, meta_data)
