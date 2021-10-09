# interceptor寻踪：总结

## 发送方

主角：`TrackLocal`和`RTPSender`

* `BindRTCPReader`在`NewRTPSender`里被调用，返回的`RTCPReader.Read`在`RTPSender`的`Read`里调用，供用户从`RTPSender`里读取自定义的RTCP包
* `BindLocalStream`在`RTPSender.Send`里被调用，并且在最顶层上都是在`SetLocalDescription`和`SetRemoteDescription`里初始化时调用的。在`RTPSender.Send`里，`RTPSender`构造为`TrackLocalWriter`封装进`TrackLocalContext`然后绑定给用户定义的`TrackLocal`里，实际发送RTP包需要用户在自己实现的`TrackLocal`里调用`TrackLocalWriter.Write`

## 接收方

主角：`TrackRemote`和`RTPReceiver`

* 读取RTP包：`OnTrack`里用户获取到`TrackRemote`，调用`TrackRemote`里的`Read`，`Read`调用`RTPReceiver`里的非导出类执行发RTP包的操作
* 读取RTCP包：`OnTrack`里用户获取到`RTPReceiver`，调用`RTPReceiver`里的`Read`就是实际读取RTCP包的操作
* 初始化：在`SetLocalDescription`和`SetRemoteDescription`里，interceptor相关类被初始化（`BindRemoteStream`和`BindRTCPReader`）后放入`TrackRemote`和`RTPReceiver`里，在`OnTrack`里里用户获取到的就是这些初始化好的类

## 额外

* `BindRTCPWriter`在`NewPeerConnection`里被调用，返回的`RTCPWriter.Write`在`PeerConnection`的`WriteRTCP`里调用，供用户发送一些自定义的RTCP包

## 参考

* `TrackLocal`的介绍：[《pion中的`TrackLocal`》](./TrackLocal.md)
* `RTPSender`的介绍：[《interceptor寻踪：从`TrackLocal`开始深入挖掘`pion/interceptor`的用法》](./interceptor在tracklocal里.md)
* `TrackRemote`的介绍：[《pion中的`TrackRemote`》](./TrackRemote.md)
* `RTPReceiver`的介绍：[《interceptor寻踪：从`TrackRemote`开始深入挖掘`pion/interceptor`的用法》](./interceptor在trackremote里.md)