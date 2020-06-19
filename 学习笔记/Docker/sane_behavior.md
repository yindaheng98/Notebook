# [纯转载]下一代的CGroup

[原文](https://blog.csdn.net/zhajianc/article/details/88842663)

上面，我们可以看到，CGroup的一些常用方法和相关的术语。一般来说，这样的设计在一般情况下还是没什么问题的，除了操作上的用户体验不是很好，但基本满足我们的一般需求了。

不过，对此，有个叫Tejun Heo的同学非常不爽，他在Linux社区里对cgroup吐了一把槽，还引发了内核组的各种讨论。

对于Tejun Heo同学来说，cgroup设计的相当糟糕。他给出了些例子，大意就是说，如果有多种层级关系，也就是说有多种对进程的分类方式，比如，我们可以按用户来分，分成Professor和Student，同时，也有按应用类似来分的，比如WWW和NFS等。那么，当一个进程即是Professor的，也是WWW的，那么就会出现多层级正交的情况，从而出现对进程上管理的混乱。另外，一个case是，如果有一个层级A绑定cpu，而层级B绑定memory，还有一个层级C绑定cputset，而有一些进程有的需要AB，有的需要AC，有的需要ABC，管理起来就相当不易。

层级操作起来比较麻烦，而且如果层级变多，更不易于操作和管理，虽然那种方式很好实现，但是在使用上有很多的复杂度。你可以想像一个图书馆的图书分类问题，你可以有各种不同的分类，分类和图书就是一种多对多的关系。

所以，在Kernel 3.16后，引入了unified hierarchy的新的设计，这个东西引入了一个叫__DEVEL__sane_behavior的特性（这个名字很明显意味目前还在开发试验阶段），它可以把所有子系统都挂载到根层级下，只有叶子节点可以存在tasks，非叶子节点只进行资源控制。

我们mount一下看看：

```sh
$ sudo mount -t cgroup -o __DEVEL__sane_behavior cgroup ./cgroup

$ ls ./cgroup
cgroup.controllers  cgroup.procs  cgroup.sane_behavior  cgroup.subtree_control

$ cat ./cgroup/cgroup.controllers
cpuset cpu cpuacct memory devices freezer net_cls blkio perf_event net_prio hugetlb
```

我们可以看到有四个文件，然后，你在这里mkdir一个子目录，里面也会有这四个文件。上级的cgroup.subtree_control控制下级的cgroup.controllers。

举个例子：假设我们有以下的目录结构，b代表blkio，m代码memory，其中，A是root，包括所有的子系统（）。

```sh
# A(b,m) - B(b,m) - C (b)
#               \ - D (b) - E
# 下面的命令中， +表示enable， -表示disable
# 在B上的enable blkio
# echo +blkio > A/cgroup.subtree_control
# 在C和D上enable blkio
# echo +blkio > A/B/cgroup.subtree_control
# 在B上enable memory 
# echo +memory > A/cgroup.subtree_control
```

在上述的结构中，

* cgroup只有上线控制下级，无法传递到下下级。所以，C和D中没有memory的限制，E中没有blkio和memory的限制。而本层的cgroup.controllers文件是个只读的，其中的内容就看上级的subtree_control里有什么了。
* 任何被配置过subtree_control的目录都不能绑定进程，根结点除外。所以，A,C,D,E可以绑上进程，但是B不行。

我们可以看到，这种方式干净的区分开了两个事，一个是进程的分组，一个是对分组的资源控制（以前这两个事完全混在一起），在目录继承上增加了些限制，这样可以避免一些模棱两可的情况。

当然，这个事还在演化中，cgroup的这些问题这个事目前由cgroup的吐槽人Tejun Heo和华为的Li Zefan同学负责解决中。总之，这是一个系统管理上的问题，而且改变会影响很多东西，但一旦方案确定，老的cgroup方式将一去不复返。