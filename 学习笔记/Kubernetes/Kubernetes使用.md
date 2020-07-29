# Kubernetes(k8s)的安装和基本使用

上一篇：[Kubernetes(k8s)介绍](./Kubernetes介绍.md)

## 安装

### k8s不支持所有版本的docker，安装之前要先安装支持版本的docker

[安装方法](https://kubernetes.io/docs/setup/production-environment/container-runtimes/)

### 学习环境minikube

安装比较简单但是支持单节点，只能用于学习。

### 生产环境

[安装kubeadm](https://kubernetes.io/docs/setup/production-environment/tools/kubeadm/install-kubeadm/)

[用kubeadm安装Kubernetes](https://kubernetes.io/docs/setup/production-environment/tools/kubeadm/create-cluster-kubeadm/)

安装比较复杂，生产环境用。

安装完成后启动前要先禁用Linux交换分区：

```sh
sudo swapoff -a
sysctl -p
```

还得设置好代理翻墙：

```sh
export http_proxy=http://proxyhost[:port]
export https_proxy=https://proxyhost[:port]
```

docker也得翻墙：[方法](https://docs.docker.com/config/daemon/systemd/#httphttps-proxy)

## 创建集群

来自[官方互动教程](https://kubernetes.io/docs/tutorials/kubernetes-basics/create-cluster/cluster-interactive/)和[官方教程](https://kubernetes.io/docs/tutorials/hello-minikube/)。

### 启动minikube

k8s的一种实现，可以看作是一种集群管理的服务端

```shell
minikube start
```

## 用kubectl指令控制Kubernetes机器

### 与docker指令功能类似的在kubectl指令

[Docker 用户使用 kubectl 命令指南](https://kubernetes.io/zh/docs/reference/kubectl/docker-cli-to-kubectl/)

#### 预备知识

* docker容器和k8s pods的对应关系：
  * 在docker中容器是操作的最小单位，而在k8s中pod是最小单位
  * 一个pod会包含多个容器，因此k8s中所有对容器的操作都需要额外指定是哪个pod的容器
* docker创建的容器和k8s创建的pod里的容器之间的差别：
  * 默认情况下如果 pod 中的容器退出 pod 也不会终止，相反它将会被k8s重启。就相当于在 `docker run`加了`--restart=always` 选项

#### 运行一个 nginx Deployment 并将其暴露出来

* docker：运行单个容器+端口映射

```shell
docker run -d --restart=always -e DOMAIN=cluster --name nginx-app -p 80:80 nginx
```

* kubectl：以deployment的形式运行应用的pod然后在Service里面暴露端口

```shell
#启动运行 nginx 的 pod
kubectl run --image=nginx nginx-app --port=80 --env="DOMAIN=cluster"
#通过Service暴露端口
kubectl expose deployment nginx-app --port=80 --name=nginx-http
```

#### 列出正在运行的容器/pods

* docker：列出正在运行的容器

```shell
docker ps
```

* kubectl：列出正在运行的pods

```shell
kubectl get po
```

#### 连接到正在运行的容器

* docker：直接连接到正在运行的容器

```shell
docker ps #找要连接的容器编号
docker attach [容器编号]
```

* kubectl：连接到正在运行的pod的某个容器

```shell
kubectl get po #找要连接的容器编号（一个pod里面可能有多个容器）
kubectl attach -it [容器编号]
```

#### 在容器中执行命令

* docker：直接在容器中执行命令

```shell
docker ps #找要执行的容器编号
docker exec -it [容器编号] [指令]
```

* kubectl：在正在运行的pod的某个容器中执行命令

```shell
kubectl get po #找要执行的容器编号
kubectl exec -it [容器编号] -- [指令]
```

#### 查看运行中容器的 stdout/stderr

* docker：直接在容器中执行命令

```shell
docker ps #找要查看的容器编号
docker logs -f [容器编号]
```

* kubectl：在正在运行的pod的某个容器中执行命令

```shell
kubectl get po #找要查看的容器编号
kubectl logs -f [容器编号]
```

在 docker 中，每次`logs`输出的都是到当前时刻为止的全部输出，但是对于 kubernetes，每次次`logs`仅从当前时刻开始输出。要查看到当前时刻为止的全部输出，请执行：

```shell
kubectl logs --previous [容器编号]
```

#### 停止和删除

* docker：直接的容器停止删除

```shell
docker stop [容器编号]
docker rm [容器编号]
```

* kubectl：删除拥有某个 pod 的 Deployment
  * pod是k8s最小操作单元，其中的容器不能直接删除
  * 如果直接删除 pod，Deployment 将会重新创建该 pod

```shell
kubectl get deployment nginx-app #找拥有要删除的pod的Deployment名称
kubectl delete deployment [Deployment名称]
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

### kubectl对象管理指令

[官方教程](https://kubernetes.io/zh/docs/tutorials/object-management-kubectl/imperative-object-management-command/)

下一篇：[Kubernetes(k8s)对象](./Kubernetes对象.md)