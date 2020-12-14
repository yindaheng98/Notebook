# CNI插件相关知识简介

CNI(Container Network Interface, 容器网络接口)是K8S定义的进行容器网络配置的接口标准。CNI插件是指符合CNI标准的网络配置工具。

CNI插件是K8S插件系统中数量最多、实现花样最多的插件类型。

## K8S如何调用CNI插件？

### 用户配置

1. 在节点的`/etc/cni/net.d/xxnet.conf`中写上CNI插件的配置信息
2. 将CNI插件配置工具（可执行文件）放入节点的`/opt/cni/bin/xxnet`中
3. 启动CNI插件后台程序

### K8S调用

1. 用户创建了Pod，这个Pod被K8S调度到了当前节点
2. Kubelet创建了Pod中要求的容器
3. Kubelet按照`/etc/cni/net.d/xxnet.conf`中的配置信息和CNI标准定义的方式执行CNI插件（输入“你应该把网络配置成什么样”）
4. CNI插件执行网络配置过程

## CNI插件如何运行？

### 给Pod“插网线”：CNI插件配置工具配置Pod的网卡和IP

1. 创建虚拟网卡
   * 通常使用veth-pair，一端在Pod的Network namespace中，一端在根namespace中（相关介绍见[Network namespace](../Docker/namespaces/Network.md)）
2. 给Pod分配集群中唯一的IP地址
   * 通常把Pod网段按Node分段，每个Pod再从Node网段中分配IP
3. 给Pod配置上分配的IP和路由
   * 将分配到的IP配置到Pod的网卡上
   * 再Pod的网卡上配置集群网段的路由表
   * Node上Pod的对端网卡配置IP地址路由表
4. 将Pod和分配的IP反馈给K8S

### 给Pod“连网络”：CNI插件后台程序维护集群内部的转发规则

1. CNI Daemon进程获取到集群中所有的Pod和Node的IP地址
   * 监听K8S APIServer获取Pod和Node的网络信息
2. CNI Daemon进程配置网络打通Pod间的IP访问
   * 创建到所有Node的通道，有三种方法：
     * 靠隧道进行通信，即控制Node组建Overlay
       * 所有流量都通过隧道到达其他Node
       * 不依赖底层网络
       * 协议转换很耗时，效率低
     * 靠路由进行通信（例：VPC路由表）
       * CNI插件直接控制网络中的路由器写路由表实现Node间的连通
       * 部分依赖底层网络，要求CNI插件有直接控制网络中路由器的能力
       * 标准的TCP/IP协议实现，速度中等
     * 靠底层网络进行通信（例：BGP路由）
       * CNI插件直接控制底层网络的转发规则实现Node间的连通
       * 完全依赖底层网络
       * 底层协议甚至可以是定制的协议，效率最高
   * 根据上一步获取的Pod和Node的网络信息将Pod的IP与通道相关联
     * Linux路由（最常见）、FDB转发表、OVS流表等