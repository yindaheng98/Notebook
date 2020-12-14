# CSI插件相关知识简介

CSI(Container Storage Interface, 容器存储接口)是K8S定义的进行容器存储配置的接口标准。CSI插件是指符合CSI标准的存储配置工具。

CSI支持目前主流的大多数存储方案，包括Local等各种本地存储方案和NFS等网络存储方案。

## CSI之前的容器存储配置方式：FlexVolume

FlexVolume插件也定义了一组用于进行存储配置的接口，它的运行方式与CSI有所不同：
* 用户安装：将FlexVolume可执行文件放入指定位置（默认位于`/usr/libexec/kubernetes/kubelet-plugins/volume/exec`）
  * FlexVolume插件本质上是一个存在于Node宿主机空间中的可执行文件
* K8S调用：当有配置存储的请求到达Node时，kubelet会按照FlexVolume标准定义执行指定的插件，由插件完成容器的存储配置
  * FlexVolume插件的调用本质上是在Node宿主机空间中运行一个指令（程序）
  * FlexVolume标准本质上是对运行指令时的命令行输入格式的规定

FlexVolume的缺点：

* 在宿主机上运行的指令必然存在需要在宿主机上安装依赖的情况，这些依赖可能会对Node中运行的容器产生不好的影响
* FlexVolume只能在K8S中用，但是世界上不只有K8S一种编排系统
* FlexVolume的设计没有考虑容器化部署

## CSI特点

* 不止适用于K8S，在Swarm和Mesos等编排系统里面也能用
* 纯容器化部署