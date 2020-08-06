# Service - K8S流量路由

>Kubernetes Pod 是有生命周期的。 它们可以被创建，而且销毁之后不会再启动。 如果您使用 Deployment 来运行您的应用程序，则它可以动态创建和销毁 Pod。
>
>每个 Pod 都有自己的 IP 地址，但是在 Deployment 中，在同一时刻运行的 Pod 集合可能与稍后运行该应用程序的 Pod 集合不同。
>
>这导致了一个问题： 如果一组 Pod（称为“后端”）为群集内的其他 Pod（称为“前端”）提供功能， 那么前端如何找出并跟踪要连接的 IP 地址，以便前端可以使用工作量的后端部分？

Pod的生命是有限的，如果Pod重启，IP很有可能会发生变化。如果我们的服务都是将Pod的IP地址写死，Pod的IP变化时，后端其他服务也将会不可用。当然我们可以通过手动修改如nginx的反向代理配置来适应后端的服务IP改变，但K8S中的Service对象可以帮助我们自动完成这一功能。

>在讨论 Kubernetes 网络连接的方式之前，非常值得与 Docker 中 “正常” 方式的网络进行对比。
>
>默认情况下，Docker 使用私有主机网络连接，只能与同在一台机器上的容器进行通信。 为了实现容器的跨节点通信，必须在机器自己的 IP 上为这些容器分配端口，**为容器进行端口转发或者代理**。
>
>多个开发人员之间协调端口的使用很难做到规模化，那些难以控制的集群级别的问题，都会交由用户自己去处理。 **Kubernetes 假设 Pod 可与其它 Pod 通信，不管它们在哪个主机上**。 我们**给 Pod 分配属于自己的集群私有 IP 地址**，所以没必要在 Pod 或映射到的容器的端口和主机端口之间显式地创建连接。 这表明了在 Pod 内的容器都能够连接到本地的每个端口，集群中的所有 Pod 不需要通过 NAT 转换就能够互相看到。 

## 定义Service

如果现在有一个暴露了8080端口的Pod，它被打上了`app=the-nginx`的标签，有两个副本：

```yml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: just-a-nginx
spec:
  replicas: 2
  template:
    metadata:
      labels:
        app: the-nginx
    spec:
      containers:
      - name: the-nginx
        image: nginx
        ports:
        - containerPort: 8080
```

那么这Pod的两个副本会拥有两个不同的IP地址，在集群中的任何一个容器内都可以通过IP地址访问到这两个副本，但其本身并没有占用其所部署节点的端口，因此无法从外部访问到。

如果我们想从其他Pod（在集群内）通过80端口访问它，那就要创建一个`Service`将外部端口80映射到有标签`app=the-nginx`的Pod的8080端口：

```yml
apiVersion: v1
kind: Service
metadata:
  name: the-service
spec:
  selector:
    app: the-nginx
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8080
```

上述配置创建一个名称为`the-service`的 `Service` 对象，它会将请求代理到使用 TCP 端口 8080，并且具有标签`app=the-nginx`的 Pod 上。Kubernetes为该服务分配一个IP地址（有时称为 "集群IP"），该IP地址由服务代理使用。服务选择算符的控制器不断扫描与其选择器匹配的Pod，然后将所有更新发布到也称为`the-service`的 `Endpoint` 对象。

## 查看Service

查看当前Namespace中的Service

```shell
$ kubectl  get svc
NAME               TYPE        CLUSTER-IP      EXTERNAL-IP   PORT(S)          AGE
httpd-svc          NodePort    10.108.195.47   <none>        8080:30088/TCP   16h
kubernetes         ClusterIP   10.96.0.1       <none>        443/TCP          115d
mysql-production   ClusterIP   10.102.208.69   <none>        3306/TCP         14d
order              NodePort    10.99.99.88     <none>        8080:30080/TCP   17d
```

查看所有Namespace中的Service

```shell
$ kubectl  get svc --all-namespaces
NAMESPACE     NAME                   TYPE        CLUSTER-IP       EXTERNAL-IP   PORT(S)                          AGE
blog          mysql                  ClusterIP   10.101.134.172   <none>        3306/TCP                         30d
blog          wordpress              NodePort    10.107.173.113   <none>        80:32255/TCP                     30d
default       httpd-svc              ClusterIP   10.108.195.47    <none>        8080/TCP                         3m
default       kubernetes             ClusterIP   10.96.0.1        <none>        443/TCP                          114d
default       mysql-production       ClusterIP   10.102.208.69    <none>        3306/TCP                         13d
default       order                  NodePort    10.99.99.88      <none>        8080:30080/TCP                   16d
kube-ops      jenkins2               NodePort    10.111.112.3     <none>        8080:30005/TCP,50000:30340/TCP   17d
kube-system   kube-dns               ClusterIP   10.96.0.10       <none>        53/UDP,53/TCP                    114d
kube-system   kubernetes-dashboard   NodePort    10.108.149.176   <none>        443:30002/TCP                    111d
```

## Service分类

>对一些应用（如前端）的某些部分，可能希望通过外部 Kubernetes 集群外部 IP 地址暴露 Service。

>Kubernetes ServiceTypes 允许指定一个需要的类型的 Service，默认是 `ClusterIP` 类型。

### ClusterIP

>**通过集群的内部 IP 暴露服务**，选择该值，服务只能够在集群内部可以访问，这也是默认的ServiceType。

![ClusterIP](./i/ClusterIP.png)

### NodePort

>**通过每个 Node 节点上的 IP 和静态端口（NodePort）暴露服务**。NodePort 服务会路由到 ClusterIP 服务，这个 ClusterIP 服务会自动创建。通过请求，可以从集群的外部访问一个 NodePort 服务。

![NodePort](./i/NodePort.png)

例如，若要从集群外访问上面那个Pod，可以使用：

```yml
apiVersion: v1
kind: Service
metadata:
  name: the-service
spec:
  selector:
    app: the-nginx
  type: NodePort
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8080
      nodePort: 80
```

>如果将 type 字段设置为 NodePort，则 Kubernetes 控制平面将在 --service-node-port-range 标志指定的范围内分配端口（默认值：30000-32767）。 每个节点将那个端口（每个节点上的相同端口号）代理到您的服务中。 您的服务在其 .spec.ports[*].nodePort 字段中要求分配的端口。

>如果您想指定特定的 IP 代理端口，则可以将 kube-proxy 中的 --nodeport-addresses 标志设置为特定的 IP 块。从 Kubernetes v1.10 开始支持此功能。
>
>该标志采用逗号分隔的 IP 块列表（例如，10.0.0.0/8、192.0.2.0/25）来指定 kube-proxy 应该认为是此节点本地的 IP 地址范围。

### 将服务暴露给指定的外部IP

比如，我想让上面的Pod只能从指定外部IP访问：

```yml
apiVersion: v1
kind: Service
metadata:
  name: the-service
spec:
  selector:
    app: the-nginx
  type: NodePort
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8080
  externalIPs:
    - 80.11.12.10 #只能从这个外部IP访问
```

### LoadBalancer

>**使用云提供商的负载局衡器**，可以向外部暴露服务。外部的负载均衡器可以路由到 NodePort 服务和 ClusterIP 服务，这个需要结合具体的云厂商进行操作。

![LoadBalancer](./i/LoadBalancer.png)

### ExternalName

>**通过返回 CNAME 和它的值**，可以将服务映射到 externalName 字段的内容（例如， foo.bar.example.com）。没有任何类型代理被创建，这只有 Kubernetes 1.7 或更高版本的 kube-dns 才支持。

>大多数 Kubernetes 用户都有可能用到集群外部的服务。例如，您可能使用 Twillio API 发送短信，或使用 Google Cloud Vision API 进行图像分析。
>
>如果位于不同环境中的应用连接相同的外部端点，并且您不打算将外部服务引入 Kubernetes 集群，那么在代码中直接使用外部服务端点是完全可以的。然而，很多时候情况并非如此。

一个典型的情况是，我的集群中需要使用一个托管在外部网络中的数据库，而我想在集群中通过一个固定的域名访问它，比如这个数据库是一个MySQL数据库`mysql://my.sql.database.example.com`，那么我们就可以在集群中定义一个`ExternalName`类型的Service：

```yml
apiVersion: v1
kind: Service
metadata:
  name: the-mysql
spec:
  type: ExternalName
  externalName: my.sql.database.example.com
```

然后我就可以在集群中通过`mysql://the-mysql`访问它了。

注：因为是基于DNS的，所以`ExternalName`没有和端口有关的功能。

### 使用IP访问外部服务

除了上面的`ExternalName`外，还不得不提使用外部IP访问外部服务的情况。当我有多个外部数据库且部署在同一个位置的不同端口的时候，使用ExternalName就有点麻烦了。这个时候就可以使用一个Headless Services加一个`Endpoints`将外部IP绑定到集群内：

```yml
kind: Service
apiVersion: v1
metadata:
name: mongo
spec:
  ports:
    - port: 27017
      targetPort: 49763
---
kind: Endpoints
apiVersion: v1
metadata:
name: mongo
subsets:
  - addresses:
    - ip: 35.188.8.12
      ports:
        - port: 49763
```

#### 附加知识：Headless Services

>有时不需要或不想要负载均衡，以及单独的 Service IP。 遇到这种情况，可以通过指定 Cluster IP（spec.clusterIP）的值为 "None" 来创建 Headless Service。
>
>您可以使用无头 Service 与其他服务发现机制进行接口，而不必与 Kubernetes 的实现捆绑在一起。
>
>对这无头 Service 并不会分配 Cluster IP，kube-proxy 不会处理它们， 而且平台也不会为它们进行负载均衡和路由。 DNS 如何实现自动配置，依赖于 Service 是否定义了选择算符。
>
>##### 带选择算符的服务
>
>对定义了选择算符的无头服务，Endpoint 控制器在 API 中创建了 Endpoints 记录， 并且修改 DNS 配置返回 A 记录（地址），通过这个地址直接到达 Service 的后端 Pod 上。
>
>##### 无选择算符的服务
>对没有定义选择算符的无头服务，Endpoint 控制器不会创建 Endpoints 记录。 然而 DNS 系统会查找和配置，无论是：
>
>* ExternalName 类型 Service 的 CNAME 记录
>* 记录：与 Service 共享一个名称的任何 Endpoints，以及所有其它类型

## 原理简介——虚拟IP实施