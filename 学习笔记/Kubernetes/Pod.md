# Pod - K8S最小管理单位

Pod模型可以理解为应用程序特定的“**逻辑主机**”，并且可以包含**相对紧密耦合的不同应用程序容器**。例如，Pod 可能包含带有 Node.js 应用程序的容器以及另一个要吸收 Node.js Web 服务器提供的数据的不同容器。Pod 中的容器有如下特点：

* 共享 IP 地址和端口空间（共用一个Network Namespace）
* 始终位于同一位置并且统一调度
* 在相同的Node上运行
* 共享上下文环境

如图所示是一些大小不同的Pod：

![各种Pod](i/module_03_pods.png)

**Pod是Kubernetes平台管理的最小单位**，每个Pod在Kubernetes中都有一个唯一的 IP 地址，且能从Pod外部进行访问的容器端口由用户明确指定。当我们在Kubernetes上创建一个部署(Deployment)时，该部署将在其中创建包含容器的 Pod（而不是直接创建容器）。

（其实就相当于一个docker-compose，用户指定expose端口）

## Pod定义

使用K8s时一般不建议直接创建Pod，而是通过控制器和模版配置来管理和调度。比如在`Deployment`中的`template`字段定义一个Pod模板：

```yml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: deployment_a
spec:
  ...
  template: #Pod模板
    metadata:
      name: pod_a
    spec:
      ...
      containers:
        - image: XXX
          name: XXX
          ...
        - image: XXXX
          name: XXXX
        ...
```

其中的`template`等效于一个Pod定义：

```yml
apiVersion: apps/v1
kind: Deployment
metadata:
    name: pod_a
spec:
    ...
    containers:
    - image: XXX
        name: XXX
        ...
    - image: XXXX
        name: XXXX
    ...
```

## Pod重启策略

Pod的重启策略由`restartPolicy`字段指定，策略有三种：

* `Always`：只要退出就会重启
* `OnFailure`：只有在失败退出（exit code不等于0）时，才会重启。
* `Never`：只要退出，就不再重启

```yml
apiVersion: apps/v1
kind: Deployment
metadata:
  ...
spec:
  ...
  template: #Pod模板
    metadata:
      ...
    spec:
      ...
      restartPolicy: Always #重启策略
      containers:
        ...
```

默认地，Pod重启策略为`Always`

（相当于docker-compose中的`restart: always`）。

## 资源请求和资源限制

资源请求和资源限制由`resources`字段指定，其中：

* `requests`：资源请求，调度器会保证这个Pod调度到资源充足的Node上
* `limits`：资源限制，资源使用上限

```yml
apiVersion: apps/v1
kind: Deployment
metadata:
  ...
spec:
  ...
  template: #Pod模板
    metadata:
      ...
    spec:
      containers:
        - image: XXX
          name: XXX
          resources:
            requests: #资源请求
              cpu: "300m"
              memory: "64Mi"
            limits: #资源限制
              cpu: "500m"
              memory: "128Mi"
          ...
        - image: XXXX
          name: XXXX
          ...
        ...
```

## 健康检查

对于每个Pod，k8s会定期进行健康检查，若检查未通过，则Kill这个Pod的进程，是否重启取决于`restartPolicy`。

### 重要概念：Probe（探针）

探针是一类**用于检测容器状态的程序**的统称。

探针可以是一个指令。比如一个容器中的主程序会在完全启动后创建一个文件，那么健康检查探针就可以是`cat [文件]`指令，当程序正常启动后，文件被创建，执行此探针（指令）会正常返回文件内容（exit 0），否则会报错（exit 1）。由指令返回值即可知道程序是否正常启动。

探针可以是一个TCP连接程序或Http请求程序。这种探针多用于服务端应用，其目的很明显：像一个用户一样向服务端发起请求，通过是否正常返回以及其返回内容可以很容易地判断服务卡死或出错。

### 健康检查类型

* `livenessProbe`：存活性检查。检查容器是否处于运行状态
  * 如果检测失败，kubelet将会杀掉掉容器，并根据重启策略进行下一步的操作
* `readnessProbe`：可读性检查。检查容器是否已经处于可接受服务请求的状态。
  * 如果Readiness Probe失败，端点控制器将会从服务端点（与Pod匹配的）中移除容器的IP地址

```yml
apiVersion: apps/v1
kind: Deployment
metadata:
  ...
spec:
  ...
  template: #Pod模板
    metadata:
      ...
    spec:
      containers:
        - image: XXX
          name: XXX
          livenessProbe: #存活性检查
            ...
          readnessProbe: #可读性检查
            ...
          ...
        - image: XXXX
          name: XXXX
          livenessProbe:
            ...
          ...
        ...
```

### 健康检查参数

* `initialDelaySeconds`：检查开始执行的时间，以容器启动完成为起点计算
* `periodSeconds`：检查执行的周期，默认为10秒，最小为1秒
* `timeoutSeconds`：每次检查的时间限制，默认为1秒，最小为1秒
* `successThreshold`：从上次检查失败后重新认定检查成功的检查次数阈值（必须是连续成功），默认为1
* `failureThreshold`：从上次检查成功后认定检查失败的检查次数阈值（必须是连续失败），默认为1

### 健康检查方法（探针）

* `exec`：在容器中执行特定的命令，命令退出返回0表示成功
  * 使用`command`字段指定指令
* `tcpSocket`：根据容器IP地址及特定的端口进行TCP检查，端口开放表示成功
  * `host`：主机名或IP
  * `port`：请求端口
* `httpGet`：根据容器IP、端口及访问路径发起HTTP请求，如果返回码在200到400之间表示成功
  * `host`：主机名或IP
  * `scheme`：链接类型，HTTP或HTTPS，默认为HTTP
  * `path`：请求路径
  * `httpHeaders`：自定义请求头
  * `port`：请求端口

一个健康检查设置实例如下：

```yml
apiVersion: apps/v1
kind: Deployment
metadata:
  ...
spec:
  ...
  template: #Pod模板
    metadata:
      ...
    spec:
      containers:
        - image: XXX
          name: XXX
          livenessProbe: #存活性检查
            exec: #检查方式是在容器中执行指定命令
              command: #指定命令
                - cat
                - /tmp/healthy
              initialDelaySeconds: 5 #容器启动5s后检查开始
              periodSeconds: 5 #每5s检查一次
          readnessProbe: #可读性检查
            httpGet: #检查方式是在容器中发起HTTP请求
              path: /index.html #请求路径
              port: 80 #请求端口
              httpHeaders: #请求头
                - name: X-Custom-Header
                  value: Awesome
        - image: XXXX
          name: XXXX
          livenessProbe: #存活性检查
            tcpSocket: #检查方式是TCP检查
              port: 8080 #检查的端口是8080
          ...
        ...
```

## 初始化容器`initContainers`

顾名思义，初始化容器用于Pod的初始化操作，比如向某个文件夹（volumes）中写入重要的运行时文件。初始化容器的配置由`initContainers`字段指定，主要用处：

* 初始化容器可以包含不能随普通容器一起发布出去的敏感信息。
* 初始化容器可以包含用户自定义的代码、工具，如sed、awk、python等方便完成初始化、设置工作。
* 因为初始化逻辑与主体业务逻辑分布在不同的image中，因此image构建者与主体业务逻辑开发者可以各自独立的工作。
* 初始化容器使用Linux namespace，不同于普通应用容器，具有不同的文件系统视图，并且对于低层的操作系统有更大的访问权限。
* 当应用启动的前置条件不具备时，初始化容器可以阻止普通应用容器启动，避免在条件不具备时反复启动注定会失败的容器而浪费系统资源。


```yml
apiVersion: apps/v1
kind: Deployment
metadata:
  ...
spec:
  ...
  template: #Pod模板
    metadata:
      ...
    spec:
      ...
      containers:
        - image: XXX
          name: XXX
          ...
      initContainers:
        - image: XXX #初始化容器
          name: XXX
          command: #初始化容器中的指令
            - wget #把一个文件下载到Pod里面
            - "-O"
            - "/some/dir/some.file"
            - "http://some.site/some/url"
          volumeMounts: #文件下载到这个volume里
            - name: workdir
              mountPath: "/work-dir"
            ...
          ...
        - image: XXXX #另一个初始化容器
          name: XXXX
          ...
      volumes:
        - name: workdir #上面那个volume的定义
          ...
```

## Pod调度

### 简单的方式：选择节点 - `nodeSelector`

nodeSelector是目前最为简单的一种pod运行时调度限制，Pod.spec.nodeSelector通过kubernetes的label-selector机制选择节点，**由调度器调度策略匹配label，而后调度pod到目标节点**，该匹配规则属于强制约束。

#### 节点（Node）的标签（label）是什么

节点的label是节点的一个标签，以键值对的形式存在。一个节点可以有多个label，一种label也可以关联到多个节点。

##### 查看节点标签

查看节点标签的指令是`kubectl get nodes --show-labels`。比如在一个minikube节点上：

```sh
$ kubectl get nodes --show-labels
NAME       STATUS    ROLES     AGE       VERSION   LABELS
minikube   Ready     <none>    1m        v1.10.0   beta.kubernetes.io/arch=amd64,beta.kubernetes.io/os=linux,kubernetes.io/hostname=minikube
```

可以看到，`minikube`节点有三个标签`beta.kubernetes.io/arch`、`beta.kubernetes.io/os`和`kubernetes.io/hostname`，其值分别为`amd64`、`linux`和`minikube`。

##### 添加节点标签

查看节点标签的指令是`kubectl label node [节点名] [标签名]=[标签值]`比如在上面那个minikube节点上加个标签`disktype=ssd`：

```sh
$ kubectl label node minikube disktype=ssd
node/minikube labeled
$ kubectl get nodes --show-labels
NAME       STATUS    ROLES     AGE       VERSION   LABELS
minikube   Ready     <none>    5m        v1.10.0   beta.kubernetes.io/arch=amd64,beta.kubernetes.io/os=linux,disktype=ssd,kubernetes.io/host
```

##### 删除节点标签

删除节点标签的指令是标签名加减号`kubectl label node [节点名] [标签名]-`。比如删除上面那个`disktype=ssd`标签：

```sh
$ kubectl label node minikube disktype-
node/minikube labeled
$ kubectl get node --show-labels
NAME       STATUS    ROLES     AGE       VERSION   LABELS
minikube   Ready     <none>    23m       v1.10.0 beta.kubernetes.io/arch=amd64,beta.kubernetes.io/os=linux,kubernetes.io/hostname=minikube
```

#### 用`nodeSelector`字段通过标签选择部署节点

```yml
apiVersion: apps/v1
kind: Deployment
metadata:
  ...
spec:
  ...
  template: #Pod模板
    metadata:
      ...
    spec:
      nodeSelector:
        disktype: ssd #选择在disktype标签值为ssd的节点上部署
      containers:
          ...
        ...
```

### [复杂的方式一：亲和性 - `affinity`](./Pod亲和性调度.md)

### [复杂的方式二：污点（`taints`）与容忍（`tolerations`）](./Pod容忍与Node污点.md)