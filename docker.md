# Docker笔记

## 安装Docker

### Ubuntu Docker 安装

    uname -r

👆先运行这个命令看Linux内核版本，大于3.10才能装

    wget -qO- https://get.docker.com/ | sh

👆一条命令就能装docker，连相关的设置和依赖都给你整好

    sudo usermod -aG docker [你的用户名]

👆当要以非root用户运行docker时，需要执行这个命令，然后重新登陆，一直用root的就不用管了。具体可以看安装完成后的一个提示：
> If you would like to use Docker as a non-root user, you should now consider adding your user to the "docker" group with something like:
>
>sudo usermod -aG docker your-user
>
>Remember that you will have to log out and back in for this to take effect!

## 运行Docker的hello-world

    sudo service docker start
    docker run hello-world

👆启动docker后台服务然后整一个测试用的hello-world

* `docker`: Docker的二进制执行文件
* `run`: 运行容器的Docker命令
* `hello-world`: 要运行的镜像名，这里是运行测试用的hello-world镜像

Docker首先从本地主机上查找镜像是否存在，如果不存在，Docker 就会从镜像仓库 Docker Hub 下载公共镜像。

## 载一个Ubuntu镜像

    docker pull ubuntu

👆从dockerhub载个ubuntu，这是dockerhub热门镜像之一（这镜像好像挺小的30M多）

## 看本机有哪些镜像

    docker images

👆打了这个命令可以看到本机都下了哪些镜像，有`REPOSITORY`表示镜像的仓库源、镜像的版本标签（不同版本的镜像算作不同镜像，版本标签值就不一样）、镜像的ID`IMAGE ID`、什么时间建的、还有大小。可以看到刚才那个ubuntu镜像只有不到90M

## 开一个新容器并在里面运行命令

    docker run ubuntu /bin/echo "Hello world"

👆比前面运行hello-world的时候多了个`/bin/echo "Hello world"`，这是要在容器里面执行的命令。输出就是个`Hello world`

>注意容器和镜像的区别：
>
>镜像(image)是一个静态文件，里面的内容是不会在运行的时候发生变化的，镜像运行之后会生成一个容器(container)，里面的内容会随着我们运行的程序和操作发生变化。镜像是容器的模板，容器基于镜像而产生。
>
>所以在删镜像的时候，如果有某个容器是基于这个镜像而产生的，那删这个镜像的时候会报“某个容器在用这个镜像”的错误，要先把容器删了才行。

/bin/echo "Hello world": 在启动的容器里执行的命令

以上命令完整的意思可以解释为：Docker以ubuntu镜像创建一个新容器，然后在容器里执行 bin/echo "Hello world"，然后输出结果。

## 交互式地开容器

    docker run -i -t ubuntu /bin/bash

👆加了两个参数 -i -t，直接进了ubuntu系统的命令行，和ssh一样可以打`exit`退出。各个参数解析：

* -i: 允许你对容器内的标准输入 (STDIN) 进行交互。
* -t: 在新容器内指定一个伪终端或终端。

此时我们已进入一个 ubuntu15.10系统的容器
我们尝试在容器中运行命令 cat /proc/version和ls分别查看当前系统的版本信息和当前目录下的文件列表

## 后台模式开容器

    docker run -d ubuntu

👆加一个`-d`把容器放在后台运行。打了这个命令之后会出一串数字，叫做容器ID，对每个容器来说都是唯一的，我们可以通过容器ID来查看对应的容器发生了什么

## 查看容器

    docker ps

👆查看正在运行的容器。看不见刚才挂在后台的ubuntu？应该是没有打要运行的命令然后ubuntu关机了吧。在后台挂一个死循环shell看看👇

    docker run -d ubuntu /bin/sh -c "while true; do echo hello world; sleep 1; done"

👌好现在再打`docker ps`能看见ubuntu在运行了。而且还可以从`COMMAND`里面看ubuntu在运行啥。

另外还有👇查看所有的容器，没在运行容器的也显示出来了

    docker ps -a

👆经过上面一通操作之后打这个命令会发现已经有了好几个容器了，为什么呢？因为`docker run ubuntu`是用ubuntu镜像创建新容器而不是运行已有容器啊小傻瓜

## 运行已有容器

    docker start [容器ID]

👆这个命令用于运行已有容器，[容器ID]里面填之前打`docker ps`出来的那个表里面的`CONTAINER ID`列

    docker start -i [容器ID]

👆start不需要用-t搞终端，终端在run的时候用-t搞好

## 查看容器内的输出

    docker logs [容器ID]

👆这个命令看某个正在运行的容器里面都输出了啥，[容器ID]里面填啥上面运行已有容器一样。输刚才挂后台的那个ubuntu的`CONTAINER ID`出一堆hello world（废话

    docker logs -f [容器ID]

👆不加`-f`的时候输出只输出目前容器里面已经输出的东西，出完就结束，加一个`-f`一直输出，`ctrl+c`退出

## 和在后台的容器交互

    docker run -dit [容器ID]

👆当一个容器以这种方式运行的时候，就可以用这种方式👇再进到运行中的容器的命令行。

    docker attach [容器ID]

## 把容器关了

    docker stop [容器ID]

👆字面意思，不说了。[容器ID]里面填啥上面一样

## 删镜像和删容器

    docker rm [容器ID]
    docker rmi [镜像ID]
    docker rmi [REPOSITORY]:[TAG]

👆rm是删容器，rmi是删镜像，[容器ID]和上面一样，[镜像ID]是打`docker images`出的表里面的`IMAGE ID`列

## 把容器保存为镜像

    docker commit -m="[确认信息]" -a="[作者]" [容器ID] [REPOSITORY]:[TAG]

👆把[容器ID]所指定的容器保存为名称为[REPOSITORY]版本为[TAG]的镜像。[确认信息]和[作者]随便填。

## 把镜像上传Dockerhub

    docker push [REPOSITORY]

👆把[REPOSITORY]镜像上传到Dockerhub里去（这网速也太卡了

## Docker的端口映射

    docker run -p "80:80" ubuntu

👆把ubuntu的80端口映射到主机的80端口

## Docker的外部硬盘挂载

    docker run -v [主机地址]:[虚拟机地址] ubuntu

👆把[主机地址]挂载到ubuntu的[虚拟机地址]上