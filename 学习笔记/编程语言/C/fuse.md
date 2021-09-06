# FUSE（用户空间文件系统）

## 原理概述

## 基本使用方法

FUSE本质是一套C语言库文件，更具体一点，是一个作为接口的结构体`fuse_operations`和一个用在主函数里加载接口的函数`fuse_mail`。

在使用是，先引入库文件：

```c
#include <fuse.h>
```

然后对结构体进行实现：

```c
int XXfs_getattr (const char *, struct stat *) {
    //自己实现的接口函数
}
int XXfs_readlink (const char *, char *, size_t) {
    //自己实现的接口函数
}
int XXfs_getdir (const char *, fuse_dirh_t, fuse_dirfil_t) {
    //自己实现的接口函数
}
int XXfs_mknod (const char *, mode_t, dev_t) {
    //自己实现的接口函数
}
int XXfs_mkdir (const char *, mode_t) {
    //自己实现的接口函数
}
int XXfs_unlink (const char *) {
    //自己实现的接口函数
}
int XXfs_rmdir (const char *) {
    //自己实现的接口函数
}
int XXfs_symlink (const char *, const char *) {
    //自己实现的接口函数
}
int XXfs_rename (const char *, const char *) {
    //自己实现的接口函数
}
//接口函数还有很多，不一一列举

//各接口函数都实现后将其一一填入结构体中
struct fuse_operations setfs_oper = {
    .getattr = XXfs_getattr,
    .readlink = XXfs_readlink,
    .getdir = XXfs_getdir,
    .mknod = XXfs_mknod,
    .mkdir = XXfs_mkdir,
    .unlink = XXfs_unlink,
    .rmdir = XXfs_rmdir,
    .symlink = XXfs_symlink,
    .rename = XXfs_symlink
    //接口函数还有很多，不一一列举
}
```

实现了结构体之后把它填入`fuse_main`即可：

```c
int main(int argc, char *argv[])
{
    int fuse_stat;
    void *user_data; 、、这里可以随便整点自定义的数据
    fuse_stat = fuse_main(argc, argv, &setfs_oper, user_data);
    return fuse_stat;
}
```

## FUSE接口解析（待续）

