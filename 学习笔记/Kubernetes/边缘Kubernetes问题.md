# kubernetes管理边缘设备需要解决什么样的问题

* 轻量化：边缘设备通常计算能力较弱
* 内网穿透：边缘设备通常位于内网中，云端不能直接访问到
  * k8s要求其所部署的所有物理机之间两两都能直接通信
* 断网后仍能运行：边缘设备经常断网
  * k8s的worker节点在需要时只会从master节点查询pod信息，不会在本地维护pod信息，所以连不上master节点时worker节点上的服务无法自动恢复


参考：[OpenYurt](https://openyurt.io/zh-cn/index.html)

参考：[在树莓派上玩转 OpenYurt](https://openyurt.io/zh-cn/blog/Play_with_Openyurt_on_Raspberry_Pi.html)

参考：[KubeEdge](https://kubeedge.io/zh/)