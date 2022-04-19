# `ion-go-sdk`解析——以加入聊天室为例

**`ion-go-sdk`是传统的基于HTTP的GRPC的客户端！不是nats-grpc客户端！`ion-go-sdk`发出的所有GRPC请求响应全部是经过[Signal](ion-signal.md)服务转发到nats里面的！**

来源：[github.com/pion/ion-sdk-go/example/ion-cluster-simple/main.go](https://github.com/pion/ion-sdk-go/blob/c7ea02d7059b062806d3873eec2cc1ef6d8e1267/example/ion-cluster-simple/main.go)