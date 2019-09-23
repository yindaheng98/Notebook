# Kubernetes(k8s)的使用方法

## 创建集群

来自[官方互动教程](https://kubernetes.io/docs/tutorials/kubernetes-basics/create-cluster/cluster-interactive/)和[官方教程](https://kubernetes.io/docs/tutorials/hello-minikube/)。

### 启动minikube

k8s的一种实现，可以看作是一种集群管理的服务端

```shell
minikube start
```

## Kubernetes 对象管理

k8s提供了kubectl工具用于管理集群。kubectl工具支持三种方式进行对象的管理:

* 命令式的方式
* 命令式的对象配置
* 声明式的对象配置

这三种技术之间的对比见[官方教程](https://kubernetes.io/zh/docs/tutorials/object-management-kubectl/object-management/)

管理技术|操作|推荐环境|支持撰写|学习曲线
-|-|-|-|-
命令式的方式|活动对象|开发项目|1+|最低
命令式对象配置|单文件|生产项目|1|中等
声明式对象配置|文件目录|生产项目|1+|最高

**警告**: Kubernetes 对象应该只使用一种技术进行管理。混合使用不同的技术，会导致相同对象出现未定义的行为。
