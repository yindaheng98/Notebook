# namespace

## 引入：Linux上的资源隔离指令`chroot`

`chroot`，即 change root directory，更改 root 目录。在 linux 系统中，系统默认的目录结构都是以`/`，即是以根 (root) 开始的。而在使用 `chroot` 之后，系统的目录结构将以指定的位置作为`/`位置。

在经过 chroot 之后，系统读取到的目录和文件将不在是旧系统根下的而是新根下(即被指定的新的位置)的目录结构和文件。

### `chroot`的作用与用法

1. 增加了系统的安全性，限制了用户的权力；
   * 在经过 chroot 之后，在新根下将访问不到旧系统的根目录结构和文件
   * 一般是在登录 (login) 前使用 chroot，使得用户不能访问一些特定的文件
2. 建立一个与原系统隔离的系统目录结构，方便用户的开发；
3. 切换系统的根目录位置，引导 Linux 系统启动以及急救系统等；
   * 最为明显的是在系统初始引导磁盘的处理过程中使用，从初始 RAM 磁盘 (initrd) 切换系统的根位置并执行真正的 init

### `chroot`没有隔离的东西

1. chroot没有在进程层面上进行隔离
   * 在原系统下面可以看到新根目录下执行的进程
2. chroot没有在网络层面上进行隔离
   * 在新根目录下执行ifconfig等网络操作可以看到网络信息跟原系统是完全一样的

## namespace API

namespace的API包括clone()、setns()以及unshare()，和/proc下的部分文件。

### `clone()`：创建一个namespace然后在其中创建一个新进程

`clone()`实际上是传统UNIX系统调用fork()的一种更通用的实现方式，它可以通过flags来控制使用多少功能。Linux namespace所指的功能实际上只是`clone()`功能的子集，一共有二十多种`CLONE_*`形式的flag（标志位）参数用来控制clone进程的方方面面，与namespace相关的只有以下7种：

| namespace | 系统调用参数    | 隔离内容                   |
| --------- | --------------- | -------------------------- |
| Mount     | CLONE_NEWNS     | 文件系统                   |
| User      | CLONE_NEWUSER   | 用户和用户组               |
| PID       | CLONE_NEWPID    | 进程编号                   |
| IPC       | CLONE_NEWIPC    | 信号量、消息队列、共享内存 |
| Network   | CLONE_NEWNET    | 网络设备。协议栈、端口     |
| UTS       | CLONE_NEWUTS    | 主机名与域名               |
| Cgroup    | CLONE_NEWCGROUP | Cgroup根目录               |

```C
int clone(int (*child_func)(void *), void *child_stack, int flags, void *arg);
```

各参数定义如下：

* `child_func`：指定一个由新进程执行的函数。当这个函数返回时，子进程终止。该函数返回一个整数，表示子进程的退出代码
* `child_stack`：传入子进程使用的栈空间，也就是把用户态堆栈指针赋给子进程的 esp 寄存器
* `flags`：前文所述的`CLONE_*`形式的标志位参数，多个标志位通过`|`（位或）操作来实现
* `arg`：指向传递给`child_func`函数的参数

#### `clone()`的用法：类比`fork()`

`clone()`的运行方式与`fork()`基本相同，它们都会创建一个新的进程；不同的是，`clone()`的子进程栈空间和所运行的函数由`child_stack`和`child_func`指定，而`fork()`直接将当前进程的当前栈空间和运行过程复制一份。

例如：

```C
#include <unistd.h>
#include <stdio.h>
int main (){
    pid_t fpid; //fpid表示fork函数返回的值
    int count=0;
    fpid=fork();
    if (fpid < 0)printf("error in fork!");
    else if (fpid == 0) {
        printf("我是父进程%d的输出/n",getpid());
    }
    else {
        printf("我是子进程%d的输出/n",getpid());
    }
    return 0;
}
```

`fork()`在父进程中返回`fpid`为0，而其创建的子进程从`if (fpid < 0)`处开始运行（栈空间和运行过程被复制），返回的`fpid`为1。因此输出：

```sh
root@local:~$ gcc -Wall fork_example.c && ./a.out
我是父进程28365的输出
我是子进程28366的输出
```

### `/proc/[pid]/ns`文件夹

查看当前进程的`/proc/[pid]/ns`文件夹（`$$`表示当前bash进程的pid）：

```sh
yin@yin-v:~$ ll /proc/$$/ns
total 0
dr-x--x--x 2 yin yin 0 Jun 11 13:44 ./
dr-xr-xr-x 9 yin yin 0 Jun 11 13:03 ../
lrwxrwxrwx 1 yin yin 0 Jun 11 13:44 cgroup -> 'cgroup:[4026531835]'
lrwxrwxrwx 1 yin yin 0 Jun 11 13:44 ipc -> 'ipc:[4026531839]'
lrwxrwxrwx 1 yin yin 0 Jun 11 13:44 mnt -> 'mnt:[4026531840]'
lrwxrwxrwx 1 yin yin 0 Jun 11 13:44 net -> 'net:[4026531993]'
lrwxrwxrwx 1 yin yin 0 Jun 11 13:44 pid -> 'pid:[4026531836]'
lrwxrwxrwx 1 yin yin 0 Jun 11 13:44 pid_for_children -> 'pid:[4026531836]'
lrwxrwxrwx 1 yin yin 0 Jun 11 13:44 user -> 'user:[4026531837]'
lrwxrwxrwx 1 yin yin 0 Jun 11 13:44 uts -> 'uts:[4026531838]'
```

这些 namespace 文件都是链接文件。从名字上就能看出，它们和前面所讲的7种namespace对应。链接文件的内容的格式为`namespace 的类型:[Inode number]`（`Inode number`是namespace 的标识符，相当于ID）。如果两个进程的某个 namespace 文件指向同一个链接文件，说明其相关资源在同一个 namespace 中。

#### 为什么是链接文件？

在 `/proc/[pid]/ns` 里放置链接文件的作用是：一旦这些链接文件被打开，只要打开的文件描述符(fd)存在，那么就算该 namespace 下的所有进程都已结束，这个 namespace 也会一直存在，后续的进程还可以再加入进来。

除了打开文件外，通过文件挂载的方式也可以阻止 namespace 被删除。

### `setns()`：将当前进程加入到一个已有的namespace中

将当前进程加入到已有的namespace中。

```C
int setns(int fd, int nstype);
```

参数如下：

* `fd`：要加入 namespace 的文件描述符。它是一个指向 `/proc/[pid]/ns` 目录中文件的文件描述符（可以通过直接打开该目录下的链接文件得到）
* `nstype`：让调用者可以检查 fd 指向的 namespace 类型是否符合实际要求。若把该参数设置为 0 表示不检查

#### 用法示例

```C
fd = open(argv[1], O_RDONLY);   /* 获取namespace文件描述符 */
setns(fd, 0);                   /* 加入新的namespace */
execve(argv[2], &argv[2]);      /* 创建子进程执行程序 */
```

### `unshare()`：创建新的 namespace 并将当前进程加入其中

```C
int unshare(int flags);
```

参数：`flags`同`clone()`的`flags`参数

## namespace类型

### [Mount](./namespaces/Mount.md)

### [UTS(UNIX Time-sharing System)](./namespaces/UTSandUser.md)

### [User](./namespaces/UTSandUser.md)

### [PID](./namespaces/PIDandIPC.md)

### [IPC(Interprocess Communication)](./namespaces/PIDandIPC.md)

### Network

### Cgroup