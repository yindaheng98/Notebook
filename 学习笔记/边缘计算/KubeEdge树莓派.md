# KubeEdge在树莓派上的安装

## 开始之前

### 装系统

刷入[Ubuntu系统](https://ubuntu.com/download/raspberry-pi/thank-you?version=20.04.1&architecture=arm64+raspi)，按照[教程](https://ubuntu.com/tutorials/how-to-install-ubuntu-on-your-raspberry-pi#1-overview)进行基本设置。

### 设置挂载cgroup

在`/boot`分区中的`/boot/cmdline.txt`文件开头加上`cgroup_enable=memory cgroup_memory=1`。

### 关闭自动更新

```shell
apt autoremove --purge
dpkg-reconfigure -plow unattended-upgrades
reboot
```

### 设置WiFi

将`/etc/netplan/50-cloud-init.yaml`修改为这样：

```yaml
# This file is generated from information provided by the datasource.  Changes
# to it will not persist across an instance reboot.  To disable cloud-init's
# network configuration capabilities, write a file
# /etc/cloud/cloud.cfg.d/99-disable-network-config.cfg with the following:
# network: {config: disabled}
network:
    ethernets:
        eth0:
            dhcp4: true
            optional: true
    version: 2
    wifis:
        wlan0:
            dhcp4: true
            optional: true
            access-points:
                    "HUAWEI-PKAWX9_HiLink":
                            password: "87654321"
                    "yin_Home":
                            password: "409987654321"
```


## 本文所有操作均在`su`账户下完成

## 安装Docker

以下内容摘自[官方教程](https://docs.docker.com/engine/install/ubuntu/)。

```shell
apt-get update
apt-get install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg-agent \
    software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
apt-key fingerprint 0EBFCD88
add-apt-repository \
   "deb [arch=arm64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"
apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io
```

## 编译`keadm`

### 安装Golang

```shell
wget https://golang.org/dl/go1.15.2.linux-arm64.tar.gz
tar -C /usr/local -xzf go1.15.2.linux-arm64.tar.gz
export PATH=$PATH:/usr/local/go/bin
export GOPATH="/root/go" #安装过程中Golang只用一次所以不加到profile中
go version
```

### 安装make

```shell
apt-get install -y make
```

### 下载编译KubeEdge源码

```shell
git clone https://github.com/kubeedge/kubeedge $GOPATH/src/github.com/kubeedge/kubeedge
cd $GOPATH/src/github.com/kubeedge/kubeedge
make all WHAT=keadm
```

## 加入集群

### 从云端获取token

在运行着`cloudcore`的云端执行：

```shell
keadm gettoken
```

记录下输出的token字符串。

### 将树莓派加入集群

```shell
cd $GOPATH/src/github.com/kubeedge/kubeedge/_output/local/bin
./keadm join --cloudcore-ipport=<云端的IP>:<cloudcore端口号，默认为10000> --edgenode-name=<为这个边缘节点取名> --token=<从云端获取到的token>
```

## 量产边缘节点

树莓派安装KubeEdge的关键实际上只是一个`keadm`程序，因此可以在`keadm`编译完成后直接放到其他树莓派上使用而不再下载Golang编译源码。

### 构造原型系统

从编译安装的系统中取出`keadm`程序和`/etc/kubeedge`文件夹，在一个新SD卡中：

1. 刷入经过基本设置的Ubuntu系统
   * 包括在`wpa_supplicant.conf`中设置无线连接
2. 安装Docker
3. 将`keadm`上传到`/bin`
4. 将`/etc/kubeedge`上传到`/etc/kubeedge`

### 生成镜像

选择一个硬盘容量足够的主机，记录下IP地址，在原型系统上执行：

```shell
dd bs=4M if=/dev/mmcblk0 | gzip | ssh <目标主机IP> "cat > rasp.img.gz"
```

### 使用镜像量产边缘节点

对于每个树莓派，将系统镜像刷入SD卡，开机后执行：

```shell
keadm join --cloudcore-ipport=<云端的IP>:<cloudcore端口号，默认为10000> --edgenode-name=<为这个边缘节点取名> --token=<从云端获取到的token>
```

即可完成设置。

### (未完成)设置自动初始化代码