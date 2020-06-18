# PID namespace

内核为所有的PID namespace维护了一个树状结构，最顶层的是系统初始时创建的，我们称之为root namespace。他创建的新PID namespace就称之为child namespace（树的子节点），而原先的PID namespace就是新创建的PID namespace的parent namespace（树的父节点）。通过这种方式，不同的PID namespaces会形成一个等级体系。所属的**父节点可以看到子节点中的进程**，并可以通过**信号量等方式对子节点中的进程产生影响**。反过来，**子节点不能看到父节点PID namespace中的任何内容**。

## Linux中的1号进程与Docker Daemon

当我们新建一个PID namespace时，默认启动的进程PID为1。

在传统的UNIX系统中，PID为1的进程是init，地位非常特殊。他作为所有进程的父进程，维护一张进程表，不断检查进程的状态，一旦有某个子进程因为程序错误成为了“孤儿”进程，init就会负责回收资源并结束这个子进程。

在Docker中，所有的容器进程均在Docker Daemon下的子PID namespace中。而Docker启动时的第一个进程“dockerinit”即负责init进程的工作，借助PID namespace的树状结构监控所有Docker容器的运行情况，实现进程监控和资源回收。

## PID namespace中的信号量屏蔽

在传统的UNIX系统中，init进程有一种防止误杀的特权：信号量屏蔽。如果init中没有写处理某个信号量的代码逻辑，那么与init在同一个PID namespace下的进程（即使有超级权限）发送给它的该信号量都会被屏蔽。

若PID namespace父节点发送SIGKILL或SIGSTOP，子节点的init会强制终止；其他信号量一律忽略（没有写处理某个信号量的代码逻辑时）。

## proc文件系统

PID namespace只影响部分PID特性。比如，在子PID namespace中执行`aux`或`top`之类的命令还是能看到父PID namespace中的进程，因为`aux`和`top`是通过读取`/proc`中的文件完成的。若要完全隔离PID，需要在新建PID namespace和Mount namespace后重新挂载一个新的`/proc`。

## unshare()和setns()的PID namespace行为

与[《namespace》](../namespace.md)中所述不同，unshare()和setns()并不会使已有进程进入新的PID namespace，而是随后创建的子进程进入。

**一旦程序进程创建以后，那么它的PID namespace的关系就确定下来了，进程不会变更他们对应的PID namespace。**

# IPC(Interprocess Communication) namespace

容器中进程间通信采用的方法包括常见的信号量、消息队列和共享内存。然而与虚拟机不同的是，容器内部进程间通信对宿主机来说，实际上是具有相同PID namespace中的进程间通信，因此需要一个唯一的标识符来进行区别。申请IPC资源就申请了这样一个全局唯一的32位ID，所以IPC namespace中实际上**包含了系统IPC标识符**以及**实现POSIX消息队列的文件系统**。在同一个IPC namespace下的进程彼此可见，而与其他的IPC namespace下的进程则互相不可见。
