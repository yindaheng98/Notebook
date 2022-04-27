# `pion/ion-sfu`Q&A

## `pion/ion-sfu`和ion中的SFU服务之间的区别和联系？

* ion中的SFU服务是在`pion/ion-sfu`的基础上添加了GRPC信令传输功能得来的
* ion中的SFU服务代码主要是传输信令和根据信令调用`pion/ion-sfu`中的函数

## 可以控制`pion/ion-sfu`主动连接其他SFU吗

* `pion/ion-sfu`主要为被动接收连接请求设计，所以不能`CreateOffer`，ion中的SFU服务只有一个信令服务器，想要发起连接只能用`pion/ion-go-sdk`将本地流推送到SFU服务，而不能控制SFU服务主动向其他SFU发起请求
* 但`pion/ion-sfu`中有`OnOffer`，如果hack一下`pion/ion-go-sdk`
* Session相关的代码都在`pion/ion-sfu`里面，ion中的SFU服务的代码中基本没有操作Session的逻辑

## 可以用本地视频文件创建一个没有上行流的SFU服务吗？