# 从一张图开始理解WebRTC中的STUN、SDP和ICE过程

前面几篇文章已经总结了很多关于WebRTC数据传输方面的操作过程了，现在是时候解析一下最后一块拼图了：WebRTC通过ICE建立连接的过程。

只需要下面一张图即可理解：

![](./i/ICE.png)

1. 首先，虽然WebRTC的通信主要是通过P2P连接进行的，然而P2P连接建立之前，位于NAT后面的通信双方是找不到对方的，所以还是需要一个有公网IP的中心服务器，转发双方的连接信息。在WebRTC里，信令服务器就是承担的这个功能，所以在连接建立的开始，通信双方都要先连上信令服务器
2. 准备好音视频流的相关参数：通信双方先配置好各自的音视频流（支持的输入流格式、编码器输出流的格式、码率等）
3. 交换音视频流的相关参数（Offer/Answer）：注意下面写的发送SDP的方式是自己定的，WebRTC里没有具体规定，只要能传到对面就行
   1. 发起连接的一方调用PeerConnection的`CreateOffer`方法创建一个用于Offer的SDP对象，SDP对象中保存当前音视频流的相关参数，通过PeerConnection的`SetLocalDescription`的方法保存该SDP对象
   2. 将用于Offer的SDP对象发给对面
   3. 对面接收到用于Offer的SDP后，通过PeerConnection的`SetRemoteDescription`方法将其保存起来
   4. 对面调用PeerConnection的`CreateAnswer`方法创建一个用于Answer的SDP对象，也通过PeerConnection的`SetLocalDescription`的方法保存该SDP对象
   5. 将用于Answer的SDP对象发回发起连接的一方
   6. 发起连接的一方接收到用于Answer的SDP后，也通过PeerConnection的`SetRemoteDescription`方法将其保存起来
   7. 至此，通信双方都通过`SetLocalDescription`保存了自己的音视频流参数、通过`SetRemoteDescription`保存了对方的音视频流参数
4. 获取连接信息：获取本地的IP地址、在STUN服务器处获取到自己的公网地址和端口号等信息
5. 交换连接信息（ICECandidate）：发送连接信息的方式也是自己定的。当连接信息获取好时，将会调用PeerConnection的`OnICECandidate`所指定的函数，你需要在这个函数里实现下面的功能
   1. 发起连接的一方将ICECandidate发给对面
   2. 对面收到ICECandidate，通过PeerConnection的`AddIceCandidate`方法保存
   3. 对面回复自己的ICECandidate
   4. 发起连接的一方收到ICECandidate，也通过PeerConnection的`AddIceCandidate`方法保存
   5. 至此，通信双方都通过`AddIceCandidate`保存了对方的连接信息
6. 于是，通信双方都知道了对方将要收发的音视频参数，也知道了对方的连接信息，之后就能建立P2P连接传音视频了

扩展知识：
* [《NAT和NAT穿透》](../计算机网络/NAT.md)
* [《P2P通信标准协议(一)之STUN》](../计算机网络/STUN.md)
* [《P2P通信标准协议(二)之TURN》](../计算机网络/TURN.md)
* [《P2P通信标准协议(三)之ICE》](../计算机网络/ICE.md)

## pion里的ICE协商过程案例 - 作为发起方

以[《用实例学习pion - `rtp-forwarder`》](rtp-forwarder.md)中的最后一段代码为例。

前面的`webrtc.PeerConnection`已经配置好了音视频流格式，略去，从创建Offer的过程开始。

```go
// Wait for the offer to be pasted
offer := webrtc.SessionDescription{}
signal.Decode(signal.MustReadStdin(), &offer)
```

这里的Offer SDP信息是从命令行读取来的，而命令行里的SDP是从浏览器里粘贴来的，所以这里相当于是手工担任了信令传输的工作。

```go
// Set the remote SessionDescription
if err = peerConnection.SetRemoteDescription(offer); err != nil {
    panic(err)
}
```

可以看到这里对对面的Offer调用了PeerConnection的`SetRemoteDescription`，和前面介绍的一样。

```go
// Create answer
answer, err := peerConnection.CreateAnswer(nil)
if err != nil {
    panic(err)
}
```

调用PeerConnection的`CreateAnswer`创建Answer，和前面介绍的一样。

```go
// Create channel that is blocked until ICE Gathering is complete
gatherComplete := webrtc.GatheringCompletePromise(peerConnection)
```

这里的`GatheringCompletePromise`返回的是一个`context`(本质上是`<-chan struct{}`)，阻塞直到完成ICECandidate信息收集，实际上就是等待STUN和本地IP地址获取完成。

```go
// Sets the LocalDescription, and starts our UDP listeners
if err = peerConnection.SetLocalDescription(answer); err != nil {
    panic(err)
}
```

可以看到这里对方才生成的Answer调用了PeerConnection的`SetLocalDescription`，和前面介绍的一样。

```go
// Block until ICE Gathering is complete, disabling trickle ICE
// we do this because we only can exchange one signaling message
// in a production application you should exchange ICE Candidates via OnICECandidate
<-gatherComplete
```

等待完成ICECandidate信息收集。

```go
// Output the answer in base64 so we can paste it in browser
fmt.Println(signal.Encode(*peerConnection.LocalDescription()))
```

输出SessionDescription，应该是包含收集到的ICECandidate信息和Answer，给用户在浏览器里粘贴，所以这里也相当于是手工担任了信令传输的工作。
