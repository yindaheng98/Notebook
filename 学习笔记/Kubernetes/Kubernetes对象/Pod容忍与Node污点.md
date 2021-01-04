# Node污点与Pod容忍调度

>节点亲和性（详见[这里](./Pod亲和性调度.md)） 是 Pod 的一种属性，它使 Pod 被吸引到一类特定的节点。 这可能出于一种偏好，也可能是硬性要求。 Taint（污点）则相反，**它使节点能够排斥一类特定的 Pod**。
>
>容忍度（Tolerations）是应用于 Pod 上的，允许（但并不要求）Pod 调度到带有与之匹配的污点的节点上。
>
>污点和容忍度（Toleration）相互配合，可以用来避免 Pod 被分配到不合适的节点上。 每个节点上都可以应用一个或多个污点，这表示对于那些不能容忍这些污点的 Pod，是不会被该节点接受的。

比如用户希望把 Master 节点保留给 Kubernetes 系统组件使用，或者把一组具有特殊资源预留给某些 pod，则污点就很有用了，pod 不会再被调度到 taint 标记过的节点。我们使用kubeadm搭建的集群默认就给 master 节点添加了一个污点标记，所以我们看到我们平时的 pod 都没有被调度到 master 上去：

```shell
$ kubectl describe node master
Name:               master
Roles:              master
Labels:             beta.kubernetes.io/arch=amd64
                    beta.kubernetes.io/os=linux
                    kubernetes.io/hostname=master
                    node-role.kubernetes.io/master=
......
Taints:             node-role.kubernetes.io/master:NoSchedule
Unschedulable:      false
......
```

其中有一条关于 Taints 的信息：`node-role.kubernetes.io/master:NoSchedule`，就是master节点的污点标记。

## 使用方法

### 节点上的`taint`

#### 在节点上添加污点

```shell
kubectl taint nodes <节点名称> <污点名key>=<污点值value>:<污点效果effect>
```

例如在一个节点`node1`上添加三个污点，其中污点`key1`同时使用两种不同的效果：

```shell
kubectl taint nodes node1 key1=value1:NoSchedule
kubectl taint nodes node1 key1=value1:NoExecute
kubectl taint nodes node1 key2=value2:NoSchedule
```

#### 从节点上删除污点

```shell
kubectl taint nodes <节点名称> <污点名key>:<污点效果effect>-
```

注意最后的减号

例如删除掉节点`node1`上的一个污点：

```shell
kubectl taint nodes node1 key2:NoSchedule-
```

### Pod上的`tolerations`

例如，若要调度到上述`node1`节点，Pod必须这么写：

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
      tolerations:
        - key: "key1"
          operator: "Equal"
          value: "value1"
          effect: "NoSchedule"
        - key: "key1"
          operator: "Equal"
          value: "value1"
          effect: "NoExecute"
      containers:
          ...
        ...
```

一个容忍度和一个污点相“匹配”是指它们有一样的键名和效果，并且：

* 如果 `operator` 是 `Exists` （此时容忍度不能指定 `value`），或者
* 如果 `operator` 是 `Equal` ，则它们的 `value` 应该相等

>说明：
>
>存在两种特殊情况：
>
>如果一个容忍度的 `key` 为空且 `operator` 为 `Exists`， 表示这个容忍度与任意的 `key` 、`value` 和 `effect` 都匹配，即这个容忍度能容忍任意 `taint`。
>
>如果 `effect` 为空，则可以与所有键名 key 的效果相匹配。

## 污点效果`effect`详解

>您可以给一个节点添加多个污点，也可以给一个 Pod 添加多个容忍度设置。 Kubernetes 处理多个污点和容忍度的过程就像一个过滤器：从一个节点的所有污点开始遍历， 过滤掉那些 Pod 中存在与之相匹配的容忍度的污点。余下未被过滤的污点的 `effect` 值决定了 Pod 是否会被分配到该节点，特别是以下情况：
>
>* 如果未被过滤的污点中存在至少一个 `effect` 值为 `NoSchedule` 的污点， 则 Kubernetes 不会将 Pod 分配到该节点。
>* 如果未被过滤的污点中不存在 `effect` 值为 `NoSchedule` 的污点， 但是存在 `effect` 值为 `PreferNoSchedule` 的污点， 则 Kubernetes 会 *尝试* 将 Pod 分配到该节点。
>* 如果未被过滤的污点中存在至少一个 `effect` 值为 `NoExecute` 的污点， 则 Kubernetes 不会将 Pod 分配到该节点（如果 Pod 还未在节点上运行）， 或者将 Pod 从该节点驱逐（如果 Pod 已经在节点上运行）。

* `PreferNoSchedule`：软策略，表示尽量不调度到污点节点上去
* `NoSchedule`：硬策略，表示绝对不调度到污点节点上去
* `NoExecute`：比`NoSchedule`更强的硬策略，该选项意味着一旦`taint`生效，如该节点内正在运行的 pod 没有对应`tolerate`设置，会直接被驱逐

## `NoExecute`策略的`tolerationSeconds`

`tolerationSeconds`定义了Pod在`NoExecute`污点策略下受驱逐的时间，例如：

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
      tolerations:
        - key: "key1"
          operator: "Equal"
          value: "value1"
          effect: "NoSchedule"
        - key: "key1"
          operator: "Equal"
          value: "value1"
          effect: "NoExecute"
          tolerationSeconds: 3600 #3600秒后才会被驱逐
      containers:
          ...
        ...
```

>* 如果 Pod 不能忍受 `effect` 值为 `NoExecute` 的污点，那么 Pod 将马上被驱逐
>* 如果 Pod 能够忍受 `effect` 值为 `NoExecute` 的污点，但是在容忍度定义中没有指定 `tolerationSeconds`，则 Pod 还会一直在这个节点上运行。
>* 如果 Pod 能够忍受 `effect` 值为 `NoExecute` 的污点，而且指定了 `tolerationSeconds`， 则 Pod 还能在这个节点上继续运行这个指定的时间长度。

如果在`tolerationSeconds`前污点被删除了，则不会被驱逐。

## K8S字自动添加的污点

>* `node.kubernetes.io/not-ready`：节点未准备好。这相当于节点状态 Ready 的值为 "False"。
>* `node.kubernetes.io/unreachable`：节点控制器访问不到节点. 这相当于节点状态 Ready 的值为 "Unknown"。
>* `node.kubernetes.io/out-of-disk`：节点磁盘耗尽。
>* `node.kubernetes.io/memory-pressure`：节点存在内存压力。
>* `node.kubernetes.io/disk-pressure`：节点存在磁盘压力。
>* `node.kubernetes.io/network-unavailable`：节点网络不可用。
>* `node.kubernetes.io/unschedulable`: 节点不可调度。
>* `node.cloudprovider.kubernetes.io/uninitialized`：如果 kubelet 启动时指定了一个 "外部" 云平台驱动， 它将给当前节点添加一个污点将其标志为不可用。在 cloud-controller-manager 的一个控制器初始化这个节点后，kubelet 将删除这个污点。
>
>使用这个功能特性，结合 `tolerationSeconds`，Pod 就可以**指定当节点出现一个或全部上述问题时还将在这个节点上运行多长的时间**。
>
>比如，一个使用了很多本地状态的应用程序在网络断开时，仍然希望停留在当前节点上运行一段较长的时间， 愿意等待网络恢复以避免被驱逐。在这种情况下，Pod 的容忍度可能是下面这样的：
>
>```yml
>tolerations:
>- key: "node.kubernetes.io/unreachable"
>  operator: "Exists"
>  effect: "NoExecute"
>  tolerationSeconds: 6000
>```
>
>说明：
>
>Kubernetes 会自动给 Pod 添加一个 `key` 为 `node.kubernetes.io/not-ready` 的容忍度 并配置 `tolerationSeconds=300`，除非用户提供的 Pod 配置中已经已存在了 `key` 为 `node.kubernetes.io/not-ready` 的容忍度。
>
>同样，Kubernetes 会给 Pod 添加一个 `key` 为 `node.kubernetes.io/unreachable` 的容忍度 并配置 `tolerationSeconds=300`，除非用户提供的 Pod 配置中已经已存在了 `key` 为 `node.kubernetes.io/unreachable` 的容忍度。