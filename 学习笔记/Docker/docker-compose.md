# docker-compose的使用方法

[先来一个讲的比较好的博文](https://www.jianshu.com/p/658911a8cff3)

## 什么玩意

前面我们使用 Docker 的时候，定义Dockerfile文件，然后使用docker build、docker run等命令操作容器。然而微服务架构的应用系统一般包含若干个微服务，比如可能要有LAMP+redis+node+python+go+...每个微服务一般都会部署多个实例，如果每个微服务都要手动启停，那么效率之低，维护量之大可想而知。所以就有docker-compose这种比docker image更上一层的工具。

docker-compose基本就是一个操作docker image的脚本，告诉计算机要怎么用docker image构建一个微服务系统。

## 安装docker-compose

windows上面装了docker-desktop就有docker-compose了。Linux上面的docker-compose安装自己找教程。

    docker-compose -v

确认一下👆

## 概念：工程、服务、容器

Docker Compose 将所管理的容器分为三层，分别是工程（project）、服务（service）、容器（container）
Docker Compose 运行目录下的所有文件（docker-compose.yml）组成一个工程,一个工程包含多个服务，每个服务中定义了容器运行的镜像、参数、依赖，一个服务可包括多个容器实例

## docker-compose语法

```docker-compose
version: '3'
services:
  web:
    build: .
    ports:
    - "5000:5000"
    volumes:
    - .:/code
    - logvolume01:/var/log
    links:
    - redis
  redis:
    image: redis
volumes:
  logvolume01: {}
```

### 版本号

第一行那个是docker-compose文件的版本号，不同的版本号受支持的docker-compose不同。

### `services:`

定义了一个工程，一个工程(services)里面会包含多个服务(service)，上面这个工程包含了两个服务web和redis

#### `build`和`image`

在一个容器的定义中，build表示这个容器是通过一个Dockerfile构建出来的，而image表示这个容器直接使用了一个云端的image。

上面那个容器`web:`中的`build:.`表示在用目录的Dockerfile构建容器。`redis:`中的`image: redis`表示直接用redis镜像构建容器。

#### `ports`和`volumes`

端口映射，等同于在docker run启动容器时的`-p`和`-v`参数。

#### `links`

docker-compose启动后会在主机中建立一个DNS服务器，如果是按照上面的docker-compose文件启动，主机除了可以通过`http://127.0.0.1:5000`（`web:ports`指定的主机端口映射）访问`web`容器之外，还可以通过`http://web:5000`和`redis:6379`（就是`[网址]:[端口]`）访问`web`容器和`redis`容器。

>Containers for the linked service are reachable at a hostname identical to the alias, or the service name if no alias was specified.

当使用如上所示的`web:links:-redis`之后，在`web`容器里面也可以用`redis:6379`（6379是redis默认端口和容器设置无关）访问`redis`容器了。`links`还可以定义一个别名，如`links:-"redis:rds"`（注意引号），这时就是用`rds:6379`访问`redis`容器。

常见但已经不建议使用，打上`links`之后docker-compose会按顺序启动容器，上面的例子中`web:links:-redis`之后web容器会在redis容器启动完才启动；但是如果后续开发我又想让`redis`里面能访问`web`呢？那当然是再在`redis`里面写个`redis:links:-web`了？这么写了docker-compose直接给我报了个“循环依赖”错误🙂呵呵。

所以links只是一个简单的容器互联方法，比较复杂的容器互联正确姿势见下一节。

>Warning: The `--link` flag is a legacy feature of Docker. It may eventually be removed. Unless you absolutely need to continue using it, we recommend that you use user-defined networks to facilitate communication between two containers instead of using `--link`. One feature that user-defined networks do not support that you can do with `--link` is sharing environmental variables between containers. However, you can use other mechanisms such as volumes to share environment variables between containers in a more controlled way.

## 容器互联

### Overview

>One of the reasons Docker containers and services are so powerful is that you can connect them together, or connect them to non-Docker workloads. Docker containers and services do not even need to be aware that they are deployed on Docker, or whether their peers are also Docker workloads or not. Whether your Docker hosts run Linux, Windows, or a mix of the two, you can use Docker to manage them in a platform-agnostic way.

容器互联就是将各个独立的Docker容器连接在一起，就好像它们里面的程序运行在一个系统中。容器互联的方式基本上是通过虚拟的网络连接和一些端口映射使容器内部能通过某个端口或IP访问到其他的容器。

### 正确姿势

定义一个`network`取名`net1`，然后把要互联的容器`networks`字段里面写上这个`net1`然后它们就能像互相links过一样互联了👇而且不会有“循环依赖”。不过这时候容器间没有依赖关系而会同时启动，对于确实有依赖关系的容器，就用`depends_on`属性👉[请看这个](https://docs.docker.com/compose/compose-file/#/dependson)

```docker-compose
version: '3'
services:
  web:
    build: .
    ports:
    - "5000:5000"
    volumes:
    - .:/code
    - logvolume01:/var/log
    networks:
    - "net1"
  redis:
    image: redis
    networks:
    - "net1"
volumes:
  logvolume01: {}
networks:
  net1:
    driver: bridge
```

👆这个完事了之后`web`容器里面可以访问到`redis:6379`，`redis`容器里面也可以访问到`web:5000`了。关于`networks`里面的`driver`字段请看下文👇。

### Bridge模式

Bridge是docker网络连接的默认模式，效果和上文中的那个`links`效果一样，设置完之后就是通过`[名称]:[端口]`访问容器。

👆这个比较常用

[具体怎么设置，看这个从命令行开始的范例](https://docs.docker.com/network/network-tutorial-standalone/)

### Host模式

Host模式是一种特殊的Bridge。正常的Birdge都是容器间互联，Host就是容器和主机间互联。效果同`docker run`中的`-p`参数指定端口映射一样。

[这个so easy，点我看范例](https://docs.docker.com/network/network-tutorial-host/)

[翻译一下这个，应该就能用起来了](https://docs.docker.com/network/)

### Overlay和Macvlan模式

* Overlay模式用于有多个物理主机的集群(Swarm)
* Macvlan模式是基于MAC地址和vlan设置的网络

👆暂时用不到，告辞

## 环境变量

一个简单的例子👇在`web`容器中把DEBUG环境变量设为1

```docker-compose
version: '3'
services:
  web:
    environment:
      DEBUG: 1
```