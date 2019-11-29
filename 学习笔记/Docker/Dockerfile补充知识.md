# Dockerfile的一些补充知识

## 向进程发送停止信号

默认情况下，docker容器在停止时会向容器内的1号进程发送`SIGTERM`信号告知应用停止。

在命令行运行docker容器时，在容器stdout输出界面（`docker run`不加`-d`）中Ctrl+C就是向1号进程发送`SIGTERM`信号，然后退出容器的stdout（离开容器的stdout回主机命令行），此时容器可能没有完全退出，继续在后台运行，用`docker ps`可以看到，这种容器就要调用`docker stop`才能完全停止。`docker stop`是不管容器内的进程状态如何都直接强行退出。

用docker-compose运行容器时，在stdout输出界面按Ctrl+C或者用`docker-compose down`就是向1号进程发送`SIGTERM`信号，这时docker-compose会等待容器自己退出，并且下面会显示“按Ctrl+C强制退出”之类的话，这时再按Ctrl+C就是和`docker stop`一样的效果。

`SIGTERM`信号相当于是`kill <PID>`指令，但是有些程序被设置成要接收`SIGINT`信号才退出（`SIGINT`信号就相当于在命令行Ctrl-C）。这时就要在Dockerfile里面设置容器的停止指令：

```dockerfile
STOPSIGNAL SIGINT
```

这样，容器退出时向1号进程发送的信号就不是`SIGTERM`而是`SIGINT`了。

## 多阶段构建 - multi-stage builds

在从源码开始的镜像构建时经常遇到一种情况：源码的编译需要**一个超大的编译用镜像**，而编译完成之后就只需要**一个基础的操作系统镜像就能运行**。以最典型的go语言为例，编译时需要用`golang`镜像，并且还要下一堆go库，但是编译完成之后就只用把可执行文件拿到一个啥软件都没有的基础镜像里面就能运行。

在多阶段构建出现之前，像golang和C这类静态编程语言用在Docker里面至少需要两个镜像，一个用来构建，一个用来运行，构建用的镜像通常很大，运行用的镜像就很小。这种搞法就至少要写两个Dockerfile，很不方便，不够简洁。

多阶段构建就是解决这个问题存在的。

多阶段构建使用一个文件同时编译多个镜像，不同的镜像进行不同的操作，最后汇总到一个镜像中。其用法非常简单：

* 把编译镜像和运行镜像的Dockerfile合并写入一个文件中，最终要生成的运行镜像放在最下面
* 把运行镜像Dockerfile里面原有的复制可执行文件的指令加上`--from`参数，使之从编译镜像中复制文件

### 示例

#### 不使用多阶段构建

构建用镜像：

```Dockerfile
FROM golang:alpine
WORKDIR /go/src
RUN apk --update add git && \
    go get -d -v github.com/kataras/iris && \
    go get -d -v github.com/go-sql-driver/mysql && \
    go get -d -v github.com/gocql/gocql && \
    go get -d -v github.com/garyburd/redigo/redis && \
    go get -d -v gopkg.in/yaml.v3
RUN mkdir /app
WORKDIR /app
VOLUME  ["/app"]
```

构建指令：

```sh
docker run -it --rm -v "$(pwd):/app" yindaheng98/go-iris go build -v -o /app/UserAuth
```

运行用镜像：

```Dockerfile
FROM alpine
RUN mkdir /Config
ADD UserAuth /
RUN chmod u+x /UserAuth
ADD Config /Config
EXPOSE 8080
WORKDIR /
VOLUME [ "/Config" ]
ENTRYPOINT ["/UserAuth" ]
```

### 使用多阶段构建

```Dockerfile
FROM golang:alpine AS builder
WORKDIR /go/src
RUN apk --update add git && \
    go get -d -v github.com/kataras/iris && \
    go get -d -v github.com/go-sql-driver/mysql && \
    go get -d -v github.com/gocql/gocql && \
    go get -d -v github.com/garyburd/redigo/redis && \
    go get -d -v gopkg.in/yaml.v3
ADD ./ /app
WORKDIR /app
RUN go build -v -o /UserAuth

FROM alpine
RUN mkdir /Config
COPY --from=builder /UserAuth /
RUN chmod u+x /UserAuth
ADD Config /Config
EXPOSE 8080
WORKDIR /
VOLUME [ "/Config" ]
ENTRYPOINT ["/UserAuth" ]
```

## 奇怪的错误

当前文件夹下存在`etc`、`/etc/glusterfs`、`etc-etc`、`etc-etc/etc`

下面这个指令没有问题👇

```sh
docker run --rm -v "./etc:/etc/glusterfs" gluster/gluster-centos
docker run --rm -v "./etc/glusterfs:/etc/glusterfs" gluster/gluster-centos
```

下面这个就有问题👇

```sh
docker run --rm -v "./etc/glusterfs:/etc/glusterfs" gluster/gluster-centos
docker run --rm -v "./etc-etc:/etc/glusterfs" gluster/gluster-centos
```

报这种错👇

```log
C:\Program Files\Docker\Docker\Resources\bin\docker.exe: Error response from daemon: create etc/glusterfs: "etc/glusterfs" includes invalid characters for a local volume name, only "[a-zA-Z0-9][a-zA-Z0-9_.-]" are allowed. If you intended to pass a host directory, use absolute path.
See 'C:\Program Files\Docker\Docker\Resources\bin\docker.exe run --help'.
```

这么说Docker只能挂载一层相对目录？？？在docker-compose.yml里面就没这问题。

## Dockerfile的CMD和Shell脚本

1. Shell脚本开头最好加`#! /bin/bash`以免报错
2. 下面这两个CMD效果是不一样的👇

```Dockerfile
CMD [ "sh", "-c", "start.sh" ]
```

这个正常👆

```Dockerfile
CMD [ "start.sh" ]
```

这个会让容器启动时直接`exit 0`。原因？不太懂。

## docker build的网络连接

docker build镜像时，默认使用网桥(bridge)模式，容器时虚拟环境，没有自己的网卡，所以无法连接网络。

好在docker在构建(build)或者运行(run)镜像时都提供了选择网络的参数，我们可以使用宿主机的网络，方式是`--network host`指令。

```shell
docker build -t archieves_center . --network host
```

可是在win上就不用这样？奇怪了。
