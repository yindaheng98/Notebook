# Docker 代理总结

## 为docker pull设置代理

给`docker pull`意味着通过代理下载镜像。

### 在windows上

docker desktop上的proxies设置就是设置的`docker pull`代理。默认情况下是`10.0.75.1`而不是`127.0.0.1`，为什么？因为windows上的docker是一个Hyper-V虚拟机，并且有一个虚拟网络，`10.0.75.1`是主机在虚拟机网络中的地址

### 在linux上

创建文件`/etc/systemd/system/docker.service.d/http-proxy.conf`，然后在文件里写：

```conf
[Service]
Environment="HTTP_PROXY=http://proxy.server:port"
Environment="HTTPS_PROXY=http://proxy.server:port"
Environment="NO_PROXY=localhost,127.0.0.1"
```

最后

```shell
sudo systemctl daemon-reload
sudo systemctl restart docker
```

## 为docker build设置代理

```shell
docker build --build-arg http_proxy=http://10.0.75.1:10801 /path/to/Dockerfile
```

在win上，和上文同理，主机地址是`10.0.75.1`。

## 为docker run设置代理

在容器里设置即可。
