# UTS(UNIX Time-sharing System) namespace

[Uts_namespace分析](https://blog.csdn.net/tanzhe2017/article/details/81004164)

## UTS是什么

当前一个系统的uts是linux主机所用的操作系统的版本、硬件名称等基本信息。uts信息在操作系统内核中定义为一个结构体,其值可以通过一些`set*`函数设置：

```c
struct new_utsname {
    char sysname[__NEW_UTS_LEN + 1];
    char nodename[__NEW_UTS_LEN + 1];
    char release[__NEW_UTS_LEN + 1];
    char version[__NEW_UTS_LEN + 1];
    char machine[__NEW_UTS_LEN + 1];
    char domainname[__NEW_UTS_LEN + 1];
};
```

bash中可以用`uname -a`输出全部的utsname：

```
yin@DESKTOP-IG564I6:~$ uname -a
Linux DESKTOP-IG564I6 4.4.0-18362-Microsoft #836-Microsoft Mon May 05 16:04:00 PST 2020 x86_64 x86_64 x86_64 GNU/Linux
```

其中：

UTS名称 | 值 | 含义
-|-|-
sysname | Linux | 内核名称
nodename | DESKTOP-IG564I6 | 主机在网络节点上的名称或主机名称
release | 4.4.0-18362-Microsoft | linux操作系统内核版本号
version | #836-Microsoft Mon May 05 16:04:00 PST 2020 | 操作系统版本
machine | x86_64 | 主机的硬件(CPU)名
domainname | 无 | 域名
processor | x86_64 | 处理器类型
hardware-platform | x86_64 | 硬件平台类型
operating-system | GNU/Linux | 操作系统名

## UTS namespace是什么

Uts命名空间的数据结构是uts_namespace：

```c
struct uts_namespace {
    struct kref kref; // 引用计数
    struct new_utsname name;
    struct user_namespace *user_ns;
    struct ucounts *ucounts;
    struct ns_common ns;
};
```

核心结构体是 uts_namespace，创建的过程和前文讲的 clone 调用相关，来自 copy_utsname。另外从前文介绍可以看到 new_utsname 结构体内容是 UTS namespace 隔离的所有内容。user_namepace是user命名空间的一个指针。

## UTS namespace的创建、克隆、复制

* 创建：创建uts_namespace的过程比较简单，主要就是分配一个uts_namespace实例的内核空间(`kmalloc`)，并增加一个引用计数(`kref_init`)。
* 克隆：克隆一个uts_namspace首先创建一个uts_namespace，然后将旧的命名空间的内容复制到新创建的uts_namespace中。
* 复制：如果不是CLONE_NEWUTS标志，直接返回旧的命名空间，但是增加了旧的命名空间的引用计数(`kref_init`)，否则就克隆一个UTS命名空间。

# User namespace

一个普通用户的进程通过clone()创建的新进程在新user namespace中可以拥有不同的用户和用户组。这意味着一个进程在容器外属于一个没有特权的普通用户，但是**他创建的容器进程却属于拥有所有权限的超级用户**，这个技术为容器提供了极大的自由。

## User namespace 与其它 namespace 的关系

如果你要把user namespace与其他namespace混合使用，那么依旧需要root权限。解决方案可以是**先以普通用户身份创建user namespace**，然后**在新建的namespace中作为root再clone()进程加入其他类型的namespace隔离**。

当使用包含User在内的多个namespace时，**内核会保证 CLONE_NEWUSER 先被执行**，然后执行剩下的其他 CLONE_NEW*，这样就使得不用 root 用户而创建新的容器成为可能。

Linux 下的每个 namespace，都有一个 user namespace 与之关联，这个 user namespace 就是创建相应 namespace 时进程所属的 user namespace，这样保证对任何 namespace 的操作都受到 user namespace 权限的控制。例如在上面的UTS namespace结构体中有一个User namespace指针`struct user_namespace *user_ns;`。