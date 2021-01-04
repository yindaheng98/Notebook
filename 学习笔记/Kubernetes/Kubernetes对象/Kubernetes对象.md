# Kubernetes(k8s)对象

>在 Kubernetes 系统中，Kubernetes 对象 是持久化的实体。Kubernetes 使用这些实体去表示整个集群的状态。特别地，它们描述了如下信息：
>
>* 哪些容器化应用在运行（以及在哪个 Node 上）
>* 可以被应用使用的资源
>* 关于应用运行时表现的策略，比如重启策略、升级策略，以及容错策略
>
>Kubernetes 对象是 “目标性记录” —— 一旦创建对象，Kubernetes 系统将持续工作以确保对象存在。通过创建对象，本质上是在告知 Kubernetes 系统，所需要的集群工作负载看起来是什么样子的，这就是 Kubernetes 集群的**期望状态（Desired State）**。

k8s有四种基本对象：

* [Pod](./Pod.md)
* [Service](./Service.md)
* [Volume](./Volume.md)
* [Namespace](./Namespace.md)

在这些基本对象之上，k8s还包含大量的被称作控制器（Controller）的高级抽象，用于提供额外的功能和方便使用的特性，包括：

* Deployment
* DaemonSet
* StatefulSet
* ReplicaSet
* Job

k8s对象有三种属性：

* `metadata`（对象元数据）：帮助识别对象唯一性的附加数据
* `spec`（对象规约）：由用户定义，表示这个对象的**期望状态（Desired State）**
* `status`（对象状态）：由k8s在运行时给出，表示这个对象的**实际状态（Actual State）**

其中，`metadata`、`spec`这两种属性由用户指定，写在一个yaml或json文件中，使用`kubectl apply -f <对象定义文件名>`创建对象。

在使用文件定义对象时，除上述`metadata`和`spec`字段外，还需要：

* `apiVersion`：告诉k8s对象要使用的k8s api版本，不同的版本所支持的k8s对象功能和属性定义有所不同，与k8s对象抽象逻辑无关。
* `kind`：指定对象的类型

例如一个典型Deployment对象的yaml格式对象定义文件`application/deployment.yaml`如下：

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nginx-deployment
spec:
  selector:
    matchLabels:
      app: nginx
  replicas: 2
  template:
    metadata:
      labels:
        app: nginx
    spec:
      containers:
      - name: nginx
        image: nginx:1.14.2
        ports:
        - containerPort: 80
```

输入指令`kubectl apply -f application/deployment.yaml`即可创建这个Deployment。
