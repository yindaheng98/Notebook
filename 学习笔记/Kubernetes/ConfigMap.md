# ConfigMap - 按照配置文件在容器中生成环境变量

>ConfigMap 允许您将配置文件与镜像文件分离，以使容器化的应用程序具有可移植性。

ConfigMap存在的主要意义是在容器中生成一系列环境变量而达到对容器进行配置的目的。

由于是生成环境变量，所以ConfigMap的输入**以键值对为固定格式**，后缀名一般为`.properties`。就像这样：

```properties
enemies=aliens
lives=3
enemies.cheat=true
enemies.cheat.level=noGoodRotten
secret.code.passphrase=UUDDLRLRBABAS
secret.code.allowed=true
secret.code.lives=30
```

## ConfigMap的查看

以yaml格式输出ConfigMap

```shell
kubectl get configmaps <map-name> -o yaml
```

## ConfigMap的创建

### 用指令创建ConfigMap

>```shell
>kubectl create configmap <map-name> <data-source>
>```
>
>其中， `<map-name>` 是要分配给 ConfigMap 的名称，`<data-source>` 是要从中提取数据的目录，文件或者文字值。
>
>数据源对应于 ConfigMap 中的 key-value (键值对)
>
>* key = 您在命令行上提供的文件名或者密钥
>* value = 您在命令行上提供的文件内容或者文字值

#### 直接在命令行指定ConfigMap

```shell
kubectl create configmap special-config --from-literal=special.how=very --from-literal=special.type=charm
```

这个指令会生成：

```shell
kubectl get configmaps special-config -o yaml
```
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  creationTimestamp: 2016-02-18T19:14:38Z
  name: special-config
  namespace: default
  resourceVersion: "651"
  selfLink: /api/v1/namespaces/default/configmaps/special-config
  uid: dadce046-d673-11e5-8cd0-68f728db1985
data:
  special.how: very
  special.type: charm
```

#### 从文件创建ConfigMap

比如将开头那个配置文件保存为`game.properties`，那么我就可以从文件创建一个ConfigMap，例如命名为`game-config`：

```shell
kubectl create configmap game-config --from-file=./game.properties
```

#### 从文件夹创建ConfigMap

比如我又有一个`ui.properties`的配置文件，我想将它与`game.properties`一起创建配置，那就可以把他们一起放在某个目录下，比如`./conf`，然后从文件夹创建ConfigMap：

```shell
kubectl create configmap game-config --from-file=./conf
```

上面的指令等效于：

```shell
kubectl create configmap game-config --from-file=./conf/game.properties --from-file=./conf/ui.properties
```

它们都生成如下ConfigMap：

```shell
kubectl get configmaps game-config -o yaml
```
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  creationTimestamp: 2016-02-18T18:52:05Z
  name: game-config
  namespace: default
  resourceVersion: "516"
  selfLink: /api/v1/namespaces/default/configmaps/game-config
  uid: b4952dc3-d670-11e5-8cd0-68f728db1985
data:
  game.properties: |
    enemies=aliens
    lives=3
    enemies.cheat=true
    enemies.cheat.level=noGoodRotten
    secret.code.passphrase=UUDDLRLRBABAS
    secret.code.allowed=true
    secret.code.lives=30
  ui.properties: |
    color.good=purple
    color.bad=yellow
    allow.textmode=true
    how.nice.to.look=fairlyNice
```

可以看到`game.properties`和`ui.properties`一起生成了ConfigMap。

#### 指定ConfigMap `data`中的key

当然，最终生成的`data`中配置文件的key是可以改变的：

```shell
kubectl create configmap game-config-1 --from-file=game-special-key=./conf/game.properties
```

它会生成如下ConfigMap：

```shell
kubectl get configmaps game-config-1 -o yaml
```

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  creationTimestamp: 2016-02-18T18:54:22Z
  name: game-config-3
  namespace: default
  resourceVersion: "530"
  selfLink: /api/v1/namespaces/default/configmaps/game-config-3
  uid: 05f8da22-d671-11e5-8cd0-68f728db1985
data:
  game-special-key: |
    enemies=aliens
    lives=3
    enemies.cheat=true
    enemies.cheat.level=noGoodRotten
    secret.code.passphrase=UUDDLRLRBABAS
    secret.code.allowed=true
    secret.code.lives=30
```

### `configMapGenerator`

`configMapGenerator`是一种固定格式的yaml文件，代替指令生成ConfigMap。

使用`configMapGenerator`文件生成 ConfigMap 对象的指令是：

```shell
kubectl apply -k <yaml文件或目录>
```

如果使用目录，则目录下的每个`configMapGenerator`文件都生成一个 ConfigMap。

#### 直接在`configMapGenerator`指定ConfigMap

使用这个文件生成ConfigMap：

```yaml
configMapGenerator:
- name: special-config-2
  literals:
  - special.how=very
  - special.type=charm
```

等效于：

```shell
kubectl create configmap special-config --from-literal=special.how=very --from-literal=special.type=charm
```


#### `configMapGenerator`从文件创建ConfigMap

使用这个文件生成ConfigMap：

```yaml
configMapGenerator:
- name: game-config-4
  files:
  - ./game.properties
```

等效于：

```shell
kubectl create configmap game-config --from-file=./game.properties
```

#### 指定ConfigMap `data`中的key

使用这个文件生成ConfigMap：

```yaml
configMapGenerator:
- name: game-config-5
  files:
  - game-special-key=./conf/game.properties
```

等效于：

```shell
kubectl create configmap game-config-1 --from-file=game-special-key=./conf/game.properties
```

### 用标准方式生成ConfigMap

标准方式就是[《Kubernetes(k8s)对象》](./Kubernetes对象.md)里面介绍的方式：

```yml
apiVersion: v1
kind: ConfigMap
metadata:
  name: <指定ConfigMap名称>
  namespace: <指定ConfigMap作用的namespace>
data:
  <ConfigMap数据>
```

例如：

```yml
apiVersion: v1
kind: ConfigMap
metadata:
  name: special-config
  namespace: default
data:
  special.how: very
  special.type: charm
```

## ConfigMap的使用

以两个ConfigMap为例：

```yml
apiVersion: v1
kind: ConfigMap
metadata:
  name: special-config
  namespace: default
data:
  special.how: very
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: env-config
  namespace: default
data:
  log_level: INFO
```

### 用 ConfigMap 中的数据定义容器环境变量

比如，若要令容器中的环境变量`$SPECIAL_LEVEL_KEY`值为`special-config`中的`special.how`定义值，并令`$LOG_LEVEL`值为`env-config`中的`log_level`定义值，那么应该：

```yml
apiVersion: v1
kind: Pod
metadata:
  name: dapi-test-pod
spec:
  containers:
    - name: test-container
      image: k8s.gcr.io/busybox
      env:
        - name: SPECIAL_LEVEL_KEY #指定环境变量的键
          valueFrom: #指定其值从何而来
            configMapKeyRef: #指定其值来源于ConfigMap
              name: special-config #指定其值来源于ConfigMap中的special-config
              key: special.how #指定其值来源于ConfigMap中的special-config中的special.how
        - name: LOG_LEVEL #指定环境变量的键
          valueFrom: #指定其值从何而来
            configMapKeyRef: #指定其值来源于ConfigMap
              name: env-config #指定其值来源于ConfigMap中的env-config
              key: log_level #指定其值来源于ConfigMap中的env-config中的log_level
      ...
    ...
  ...
```