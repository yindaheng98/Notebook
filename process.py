from meta import meta_data, path
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
        if isinstance(head[k], list):
            content = ''
            for i in head[k]:
                content += i+','
            HEAD += "%s: %s\n" % (k, content[0:-1])
        else:
            HEAD += "%s: %s\n" % (k, head[k])
    HEAD += "---\n"
    return HEAD


def processMD(path, meta):
    appendFILE(path, getHEAD(meta))
    print('Processed file %s' % path)


def processMDIR(path, meta_data):
    for i in os.listdir(path):
        p = path+'/'+i
        if os.path.isdir(p):
            processMDIR(p, meta_data)
        elif p[-3:] == '.md' and os.path.isfile(p):
            processMD(p, meta_data[p] if p in meta_data else {})
    print('Processed directory %s' % path)


if __name__ == "__main__":
    print(meta_data)
    processMDIR(path, meta_data)
