# kubectl指令

## 与docker指令功能类似的在kubectl指令

[Docker 用户使用 kubectl 命令指南](https://kubernetes.io/zh/docs/reference/kubectl/docker-cli-to-kubectl/)

### 预备知识

* docker容器和k8s pods的对应关系：
  * 在docker中容器是操作的最小单位，而在k8s中pod是最小单位
  * 一个pod会包含多个容器，因此k8s中所有对容器的操作都需要额外指定是哪个pod的容器
* docker创建的容器和k8s创建的pod里的容器之间的差别：
  * 默认情况下如果 pod 中的容器退出 pod 也不会终止，相反它将会被k8s重启。就相当于在 `docker run`加了`--restart=always` 选项

### 运行一个 nginx Deployment 并将其暴露出来

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

### 列出正在运行的容器/pods

* docker：列出正在运行的容器

```shell
docker ps
```

* kubectl：列出正在运行的pods

```shell
kubectl get po
```

### 连接到正在运行的容器

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

### 在容器中执行命令

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

### 查看运行中容器的 stdout/stderr

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

### 停止和删除

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

## kubectl对象管理指令

[官方教程](https://kubernetes.io/zh/docs/tutorials/object-management-kubectl/imperative-object-management-command/)
