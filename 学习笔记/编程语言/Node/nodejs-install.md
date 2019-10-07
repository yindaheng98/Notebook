# Nodejs学习

## 概述

### 是什么

一种用JavaScript写逻辑的服务器

### 为什么

并发性能好，速度快，易学习（会JavaScript就行）

## 怎么装

### 整一个Ubuntu

（Ubuntu系统跳过此步骤）在windows上开一个管理员PowerShell，运行：

    Enable-WindowsOptionalFeature -Online -FeatureName Microsoft-Windows-Subsystem-Linux

👆开一波windows的linux子系统功能（WSL说的就是这个，Windows Subsystem for Linux）

然后上应用商店整一个Ubuntu

### 装nvm

先装个编译器👇

    sudo apt-get install build-essential

上官网找安装命令👇装就完事了

[点我上官网](https://github.com/creationix/nvm#install-script)

### 用nvm装nodejs

装最新版👇

    nvm install node

或者选一个版本装👇

    nvm install 4.2.2

### 看看装了哪些版本

看本地有些什么版本👇

    nvm ls

看远程都有些什么版本👇（版本超多）

    nvm ls-remote

### 用nvm切版本

切到最新版👇

    nvm use node

切到某个版本👇

    nvm use 4.2.2

### 用npm装扩展包

npm是随nodejs捆绑安装的不用自己装

装包👇加个`-g`在所有下载的nodejs中装

    npm install [要装哪个包] [-g]

我 装 我 自 己（升级）👇

    npm install npm -g

看看都装了啥👇`-g`用法同上

    npm list [-g]

### npm淘宝镜像

阿里出品，必属精品👇

    npm install -g cnpm --registry=https://registry.npm.taobao.org

然后就可以把上面那些命令里面的npm全改成cnpm运行了
