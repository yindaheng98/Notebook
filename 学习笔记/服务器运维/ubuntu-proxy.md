# Ubuntu代理使用总结

在Ubuntu系统(14.04)中里，使用代理有一种通用方式：系统设置 –> 网络 –> 网络代理 –> 应用到整个系统，这里设置的代理是全局代理，整个系统都会走这个代理设置。但一般我们不会这样使用，我们需要对我们指定的工具或软件设置代理。

## apt-get

### 全局代理

APT工具集使用的默认配置文件是/etc/apt/apt.conf，打开后发现文件默认是空文件。但是当我们设置了全局代理后，文件的内容变为：

```conf
Acquire::http::proxy "http://127.0.0.1:1080/";
Acquire::https::proxy "https://127.0.0.1:1080/";
```

### 临时代理

当只有某个或某几个包无法下载时就要用临时代理。apt-get工具可以使用-o参数来使用配置字符串，或使用-c参数使用指定配置文件。

#### 使用-o选项

```sh
sudo apt-get -o Acquire::http::proxy="http://127.0.0.1:1080/" update
sudo apt-get -o Acquire::http::proxy="http://127.0.0.1:1080/" install XXX
```

#### 使用-c选项

创建apt-get代理配置文件~/apt_proxy.conf，内容：

```conf
Acquire::http::proxy "http://127.0.0.1:1080/";
Acquire::https::proxy "https://127.0.0.1:1080/";
```

代理的使用命令：

```sh
sudo apt-get -c ~/apt_proxy.conf update
sudo apt-get -c ~/apt_proxy.conf install XXX
```

#### 使用系统变量

如果我们设置了环境变量APT_CONFIG，那么APT工具集将使用APT_CONFIG指向的配置文件。

```sh
export APT_CONFIG=~/apt_proxy.conf
sudo apt-get update
```

## curl

### 临时普通HTTP代理

```sh
curl -x http://[user:password@]proxyhost[:port]/ -I url
```

### 临时socks5代理

```sh
curl -x socks5://[user:password@]proxyhost[:port]/ url
curl --socks5 proxyhost[:port] url
```

### 文件设置代理

编辑 ~/.curlrc 文件：

```sh
proxy = proxyhost[:port]
proxy-user = "user:password"
```

然后就可以按正常的使用方法使用curl。
