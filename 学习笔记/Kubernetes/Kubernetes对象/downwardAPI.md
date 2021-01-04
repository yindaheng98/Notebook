# downwardAPI - 让容器了解自己的情况

`downwardAPI`用于让程序在容器中获取 Pod 的基本信息。

使用`downwardAPI`让程序在容器中获取 Pod 基本信息的方法有两种：

* 把 Pod 基本信息放到环境变量中
* 把 Pod 基本信息生成文件挂载到容器内部（DownwardAPIVolumeFiles）

>下面这些信息可以通过环境变量和DownwardAPIVolumeFiles提供给容器：
>
>能通过`fieldRef`获得的：
>  * `metadata.name` - Pod名称
>  * `metadata.namespace` - Pod名字空间
>  * `metadata.uid` - Pod的UID, 版本要求 v1.8.0-alpha.2
>  * `metadata.labels['<KEY>']` - 单个 pod 标签值 `<KEY>` (例如, `metadata.labels['mylabel']`); 版本要求 Kubernetes 1.9+
>  * `metadata.annotations['<KEY>']` - 单个 pod 的标注值 `<KEY>` (例如, `metadata.annotations['myannotation']`); 版本要求 Kubernetes 1.9+
>
>能通过`resourceFieldRef`获得的：
>  * 容器的CPU约束值
>  * 容器的CPU请求值
>  * 容器的内存约束值
>  * 容器的内存请求值
>  * 容器的临时存储约束值, 版本要求 v1.8.0-beta.0
>  * 容器的临时存储请求值, 版本要求 v1.8.0-beta.0
>
>此外，以下信息可通过DownwardAPIVolumeFiles从`fieldRef`获得：
>
>* `metadata.labels` - all of the pod’s labels, formatted as `label-key="escaped-label-value"` with one label per line
>* `metadata.annotations` - all of the pod’s annotations, formatted as `annotation-key="escaped-annotation-value"` with one annotation per line
>* `metadata.labels` - 所有Pod的标签，以`label-key="escaped-label-value"`格式显示，每行显示一个label
>* `metadata.annotations` - Pod的注释，以`annotation-key="escaped-annotation-value"`格式显示，每行显示一个标签
>
>以下信息可通过环境变量从`fieldRef`获得：
>
>* `status.podIP` - 节点IP
>* `spec.serviceAccountName` - Pod服务帐号名称, 版本要求 v1.4.0-alpha.3
>* `spec.nodeName` - 节点名称, 版本要求 v1.4.0-alpha.3
>* `status.hostIP` - 节点IP, 版本要求 v1.7.0-alpha.1


## 把 Pod 基本信息放到环境变量中

案例：

```yml
apiVersion: v1
kind: Pod
metadata:
  name: test-env-pod
  namespace: kube-system
spec:
  containers:
  - name: test-env-pod
    image: busybox:latest
    env:
    - name: POD_NAME
      valueFrom:
        fieldRef:
          fieldPath: metadata.name #获取Pod名称
    - name: POD_NAMESPACE
      valueFrom:
        fieldRef:
          fieldPath: metadata.namespace #获取Pod名字空间
    - name: POD_IP
      valueFrom:
        fieldRef:
          fieldPath: status.podIP #获取节点IP
```

在这个容器中查看环境变量：

```shell
$ env | grep POD
POD_IP=172.30.19.24
POD_NAME=test-env-pod
POD_NAMESPACE=kube-system
```

可以看到相关信息都已被写入。

## 把 Pod 基本信息生成文件挂载到容器内部

比如把上面那个加个注释改成挂载文件到容器内部：

```yml
apiVersion: v1
kind: Pod
metadata:
  name: test-env-pod
  namespace: kube-system
  annotations:
    build: test
    own: qikqiak
spec:
  volumes:
  - name: pod-info
    downwardAPI:
      items:
        - path: "labels"
          fieldRef:
            fieldPath: metadata.labels
        - path: "annotations"
          fieldRef:
            fieldPath: metadata.annotations
  containers:
  - name: test-env-pod
    image: busybox:latest
    volumeMounts:
      - name: podinfo
        mountPath: /etc/podinfo
```

这样，我们就能在容器中看到两个文件：

```shell
$ ls /etc/podinfo
labels
annotations
$ cat /etc/podinfo/labels
k8s-app="test-volume"
node-env="test"
$ cat /etc/podinfo/annotations
build="test"
kubernetes.io/config.seen="2018-03-02T17:51:10.856356259+08:00"
kubernetes.io/config.source="api"
own="qikqiak"
```