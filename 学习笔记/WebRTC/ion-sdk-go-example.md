# `ion-go-sdk`解析——从例程开始

**`ion-go-sdk`是传统的基于HTTP的GRPC的客户端！不是nats-grpc客户端！`ion-go-sdk`发出的所有GRPC请求响应全部是经过[Signal](ion-signal.md)服务转发到nats里面的！**

## `pion/ion-sdk-go`中的例程简介

截至版本`1ee43f3e8c202e909959a0652505a82483cdb224`，`pion/ion-sdk-go`中的example文件夹下有10个例程：
* `ion-cluster-sample`：这个例程示范了ION集群的用法，包括如何创建Room并添加成员以及如何使用Signal从SFU接入视频流
* `ion-sfu-gstreamer-receive`

## `ion-go-sdk`例程——加入聊天室`ion-cluster-sample`

来源：[github.com/pion/ion-sdk-go/example/ion-cluster-simple/main.go](https://github.com/pion/ion-sdk-go/blob/c7ea02d7059b062806d3873eec2cc1ef6d8e1267/example/ion-cluster-simple/main.go)