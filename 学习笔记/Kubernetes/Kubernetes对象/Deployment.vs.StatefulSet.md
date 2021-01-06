# Deployment vs StatefulSet

-|Deployment|StatefulSet
-|-|-|
适合场景|无状态的应用|有状态的应用
Pod启动顺序|pod之间没有顺序|部署、扩展、更新、删除都要有顺序
存储|所有pod共享存储|每个pod都有自己存储，所以都用volumeClaimTemplates，为每个pod都生成一个自己的存储，保存自己的状态
Pod名|pod名字包含随机数字|pod名字始终是固定的
Service|service都有ClusterIP,可以负载均衡|service没有ClusterIP，是headlessservice，所以无法负载均衡，返回的都是pod名，所以pod名字都必须固定

## StatefulSet示例

```yaml
apiVersion: v1
kind: Service
metadata:
  name: nginx
  labels:
    app: nginx
spec:
  ports:
  - port: 80
    name: web
  clusterIP: None
  selector:
    app: nginx
---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: web
spec:
  selector:
    matchLabels:
      app: nginx # has to match .spec.template.metadata.labels
  serviceName: "nginx"  #声明它属于哪个Headless Service.
  replicas: 3 # by default is 1
  template:
    metadata:
      labels:
        app: nginx # has to match .spec.selector.matchLabels
    spec:
      terminationGracePeriodSeconds: 10
      containers:
      - name: nginx
        image: k8s.gcr.io/nginx-slim:0.8
        ports:
        - containerPort: 80
          name: web
        volumeMounts:
        - name: www
          mountPath: /usr/share/nginx/html
  volumeClaimTemplates:   #可看作pvc的模板
  - metadata:
      name: www
    spec:
      accessModes: [ "ReadWriteOnce" ]
      storageClassName: "gluster-heketi"  #存储类名，改为集群中已存在的
      resources:
        requests:
          storage: 1Gi
```

### 为什么需要 headless service 无头服务？

在用Deployment时，每一个Pod名称是没有顺序的，是随机字符串，因此是Pod名称是无序的，但是在statefulset中要求必须是有序 ，每一个pod不能被随意取代，pod重建后pod名称还是一样的。而pod IP是变化的，所以是以Pod名称来识别。pod名称是pod唯一性的标识符，必须持久稳定有效。这时候要用到无头服务，它可以给每个Pod一个唯一的名称 。

### 为什么需要volumeClaimTemplate？

对于有状态的副本集都会用到持久存储，对于分布式系统来讲，它的最大特点是数据是不一样的，所以各个节点不能使用同一存储卷，每个节点有自已的专用存储，但是如果在Deployment中的Pod template里定义的存储卷，是所有副本集共用一个存储卷，数据是相同的，因为是基于模板来的 ，而statefulset中每个Pod都要自已的专有存储卷，所以statefulset的存储卷就不能再用Pod模板来创建了，于是statefulSet使用volumeClaimTemplate，称为卷申请模板，它会为每个Pod生成不同的pvc，并绑定pv， 从而实现各pod有专用存储。这就是为什么要用volumeClaimTemplate的原因。

### 规律

```sh
$ kubectl get pod
NAME                      READY     STATUS    RESTARTS   AGE
web-0                     1/1       Running   0          4m
web-1                     1/1       Running   0          3m
web-2                     1/1       Running   0          1m
$ kubectl get pvc
NAME              STATUS    VOLUME                                  CAPACITY   ACCESS MODES   STORAGECLASS     AGE
www-web-0         Bound     pvc-ecf003f3-828d-11e8-8815-000c29774d39   2G        RWO          gluster-heketi   7m
www-web-1         Bound     pvc-0615e33e-828e-11e8-8815-000c29774d39   2G        RWO          gluster-heketi   6m
www-web-2         Bound     pvc-43a97acf-828e-11e8-8815-000c29774d39   2G        RWO          gluster-heketi   4m
```

* StatefulSet使用Headless服务来控制Pod的域名，这个域名的FQDN为：`$(service name).$(namespace).svc.cluster.local`，其中，`“`cluster.local`”`指的是集群的域名。
* 根据`volumeClaimTemplates`，为每个Pod创建一个pvc，pvc的命名规则匹配模式：`(volumeClaimTemplates.name)-(pod_name)`，比如上面的`volumeMounts.name`是`www`， Pod name是`web-[0-2]`，因此创建出来的PVC是`www-web-0`、`www-web-1`、`www-web-2`。
* 删除Pod不会删除其pvc，手动删除pvc将自动释放pv。

## Statefulset的启停顺序

* 有序部署：部署StatefulSet时，如果有多个Pod副本，它们会被顺序地创建（从0到N-1）并且，在下一个Pod运行之前所有之前的Pod必须都是Running和Ready状态。
* 有序删除：当Pod被删除时，它们被终止的顺序是从N-1到0。
* 有序扩展：当对Pod执行扩展操作时，与部署一样，它前面的Pod必须都处于Running和Ready状态。

## 更新策略

* `OnDelete`：通过`.spec.updateStrategy.type`字段设置为`OnDelete`，StatefulSet控制器不会自动更新StatefulSet中的Pod。用户必须手动删除Pod，以使控制器创建新的Pod。
* `RollingUpdate`：通过`.spec.updateStrategy.type`字段设置为`RollingUpdate`，实现了Pod的自动滚动更新，如果`.spec.updateStrategy`未指定，则此为默认策略。StatefulSet控制器将删除并重新创建StatefulSet中的每个Pod。它将以Pod终止（从最大序数到最小序数）的顺序进行，一次更新每个Pod。在更新下一个Pod之前，必须等待这个Pod Running and Ready。
* `Partitions`：通过指定`.spec.updateStrategy.rollingUpdate.partition`来对`RollingUpdate`更新策略进行分区，如果指定了分区，则当 StatefulSet 的`.spec.template`更新时，具有大于或等于分区序数的所有 Pod 将被更新。
