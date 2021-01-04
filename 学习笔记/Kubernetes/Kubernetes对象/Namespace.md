# Namespaces - K8S虚拟集群单元

>Kubernetes 支持多个**虚拟集群**，它们底层依赖于同一个物理集群。 这些虚拟集群被称为**命名空间（namespace）**。
>
>命名空间**适用于存在很多跨多个团队或项目的用户的场景**。对于只有几到几十个用户的集群，根本不需要创建或考虑命名空间。当需要名称空间提供的功能时，请开始使用它们。
>
>命名空间**为名称提供了一个范围**。资源的名称需要在命名空间内是唯一的，但不能跨命名空间。命名空间不能相互嵌套，每个 Kubernetes 资源只能在一个命名空间中。
>
>命名空间是**在多个用户之间划分集群资源的一种方法（通过资源配额）**。
>
>在 Kubernetes 未来版本中，相同命名空间中的对象默认将具有相同的访问控制策略。
>
>不需要使用多个命名空间来分隔轻微不同的资源，例如同一软件的不同版本：**使用 labels 来区分同一命名空间中的不同资源**。

## Namespaces的使用

### 查看名称空间

#### 查看所有名称空间

```shell
$ kubectl get namespaces
NAME          STATUS    AGE
default       Active    11d
kube-system   Active    11d
kube-public   Active    11d
```

上面这三个名称空间是K8S的初始化名称空间。

* `default`：用户创建的未指定名称空间的Pod统一默认归入此名称空间
* `kube-system`：K8S创建的对象统一归入此名称空间
* `kube-public`：该命名空间被自动创建，并且对所有用户（包括未授权用户）都是可读的。该命名空间一般留作集群使用，用于存放一些始终可以被整个集群访问、读取的公共资源。This namespace is **only a convention**, not a requirement.

#### 查看某个名称空间

简略信息：

```shell
$ kubectl get namespaces <name>
```

详细信息：

```shell
$ kubectl describe namespaces <name>
Name:           default
Labels:         <none>
Annotations:    <none>
Status:         Active

No resource quota.

Resource Limits
 Type       Resource    Min Max Default
 ----               --------    --- --- ---
 Container          cpu         -   -   100m
```

### 创建名称空间

>The name of your namespace **must be a valid DNS label**.

#### 用文件创建

创建文件`my-namespace.yaml`：

```yml
apiVersion: v1
kind: Namespace
metadata:
  name: <insert-namespace-name-here>
```

然后创建：

```shell
kubectl create -f ./my-namespace.yaml
```

注：还有一个可选的字段`finalizers`，该字段用于指定一些finalizer监听该Namespace的删除操作，这些finalizer负责释放Namespace中的资源。如果你指定了不存在的`finalizers`，该命名空间会被创建，但删除时会卡在`Terminating`状态。

#### 用指令创建

直接：

```shell
kubectl create namespace <insert-namespace-name-here>
```

### 删除名称空间

```shell
kubectl delete namespaces <insert-some-namespace-name>
```

删除过程是异步的，立即执行`kubectl get namespaces`你可以看到namespaces并没有被删除而是进入`Terminating`状态，过一会才会真正删除。

## 命名空间的正确使用方法（案例）

>在某组织使用共享的 Kubernetes 集群进行开发和生产的场景中：
>
>开发团队希望在集群中维护一个空间，以便他们可以查看**用于构建和运行其应用程序的** Pods、Services 和 Deployments 列表。在这个空间里，Kubernetes **资源被自由地加入或移除**，对谁能够或不能修改资源的**限制被放宽**，以**实现敏捷开发**。
>
>运维团队希望在集群中维护一个空间，以便他们可以**强制实施一些严格的规程**，**对谁可以或不可以操作运行生产站点的** Pods、Services 和 Deployments 集合进行控制。

实现思路：划分为两个命名空间 - development 和 production，development用于开发、production用于生产。

development命名空间定义`development.yml`：

```yml
apiVersion: v1
kind: Namespace
metadata:
  name: development
  labels:
    name: development
```

development命名空间定义`production.yml`：

```yml
apiVersion: v1
kind: Namespace
metadata:
  name: production
  labels:
    name: production
```

创建：

```shell
$ kubectl create -f ./development.yml
$ kubectl create -f ./production.yml
$ kubectl get namespaces --show-labels
NAME          STATUS    AGE       LABELS
default       Active    32m       <none>
development   Active    29s       name=development
production    Active    23s       name=production
```

定义为名称空间定义context，包括所在集群位置、用户名等：

```shell
kubectl config set-context dev --namespace=development --cluster=<集群名称> --user=<用户名称>
kubectl config set-context prod --namespace=production --cluster=<集群名称> --user=<用户名称>
```

切换到`development`命名空间进行操作：

```shell
$ kubectl config use-context dev
```

此后，除切换名称空间外的所有`kubectl`指令都限定于在`development`命名空间中，不同名称空间内的内容互不可见，用户可以随意操作而不必担心影响到`production`命名空间中的内容。

## 理解使用命名空间的动机

>单个集群应该能满足多个用户及用户组的需求（以下称为 “用户社区”）。
>
>Kubernetes 命名空间 帮助不同的项目、团队或客户去共享 Kubernetes 集群。
名字空间通过以下方式实现这点：
>
>* 为名字设置作用域
>* 为集群中的部分资源关联鉴权和策略的机制
>
>使用多个命名空间是可选的。
>
>每个用户社区都希望能够与其他社区隔离开展工作。
>
>每个用户社区都有:
>
>1. 资源（pods, services, replication controllers, 等等）
>2. 策略（谁能或不能在他们的社区里执行操作）
>3. 约束（该社区允许多少配额，等等）
>4. 集群运营者可以为每个唯一用户社区创建命名空间。
>
>命名空间为下列内容提供唯一的作用域：
>
>1. 命名资源（避免基本的命名冲突）
>1. 将管理权限委派给可信用户
>1. 限制社区资源消耗的能力
>
>用例包括:
>
>1. 作为集群运营者, 我希望能在单个集群上支持多个用户社区。
>1. 作为集群运营者，我希望将集群分区的权限委派给这些社区中的受信任用户。
>1. 作为集群运营者，我希望能限定每个用户社区可使用的资源量，以限制对使用同一集群的其他用户社区的影响。
>1. 作为群集用户，我希望与我的用户社区相关的资源进行交互，而与其他用户社区在该集群上执行的操作无关。