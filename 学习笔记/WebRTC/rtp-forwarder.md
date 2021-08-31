# 用实例学习pion - [`rtp-forwarder`](https://github.com/pion/webrtc/blob/master/examples/rtp-forwarder/main.go#L24)

## 配置`MediaEngine`

```go
// Everything below is the Pion WebRTC API! Thanks for using it ❤️.

// Create a MediaEngine object to configure the supported codec
m := &webrtc.MediaEngine{}
```

>A MediaEngine defines the codecs supported by a PeerConnection, and the configuration of those codecs.
>
>A MediaEngine must not be shared between PeerConnections.

这里的`MediaEngine`是存放编码器和编码器设置的类。用于定义WebRTC可以接收什么编码的流。

```go
// Setup the codecs you want to use.
// We'll use a VP8 and Opus but you can also define your own
if err := m.RegisterCodec(webrtc.RTPCodecParameters{
    RTPCodecCapability: webrtc.RTPCodecCapability{MimeType: webrtc.MimeTypeVP8, ClockRate: 90000, Channels: 0, SDPFmtpLine: "", RTCPFeedback: nil},
}, webrtc.RTPCodecTypeVideo); err != nil {
    panic(err)
}
if err := m.RegisterCodec(webrtc.RTPCodecParameters{
    RTPCodecCapability: webrtc.RTPCodecCapability{MimeType: webrtc.MimeTypeOpus, ClockRate: 48000, Channels: 0, SDPFmtpLine: "", RTCPFeedback: nil},
}, webrtc.RTPCodecTypeAudio); err != nil {
    panic(err)
}
```

`RegisterCodec`是在`MediaEngine`里添加编码器和编码器设置。
`webrtc.RTPCodecParameters`就是这样的编码器设置类。
这里指定了一个`MimeType`为VP8的视频解码器和`MimeType`为Opus的音频解码器，还有时钟频率之类的相关设置。
`webrtc.RTPCodecTypeVideo`和`webrtc.RTPCodecTypeAudio`就是用来指明所添加的编码器设置是针对视频还是音频。

## 配置`interceptor.Registry`

```go
// Create a InterceptorRegistry. This is the user configurable RTP/RTCP Pipeline.
// This provides NACKs, RTCP Reports and other features. If you use `webrtc.NewPeerConnection`
// this is enabled by default. If you are manually managing You MUST create a InterceptorRegistry
// for each PeerConnection.
i := &interceptor.Registry{}
```

[`interceptor`](https://github.com/pion/interceptor)是pion的RTP引擎。

`interceptor.Registry`里面就是一个`Interceptor`类的列表，在运行之前它会将`Interceptor`一个个地串起来。
**`Interceptor`类的逻辑类似于处理流数据的中间件**。在运行时，rtp/rtcp数据包按顺序经过`interceptor.Registry`串起来的每个`Interceptor`，接受其处理；`Interceptor`可以修改这些数据包，也可以生成新的数据包。比如下面会看到的默认设置里的处理NACK和发送方/接收方报告的`Interceptor`就是统计每个数据包的信息，然后生成NACK和发送方/接收方报告数据包的`Interceptor`。

```go
// Use the default set of Interceptors
if err := webrtc.RegisterDefaultInterceptors(m, i); err != nil {
    panic(err)
}
```
`webrtc.RegisterDefaultInterceptors`会注册一些默认的`Interceptor`。

从`webrtc.RegisterDefaultInterceptors`里面的代码上看，**它帮你注册了处理NACK和发送方/接收方报告的`Interceptor`**。

## 构造WebRTC标准API

```go
// Create the API object with the MediaEngine
api := webrtc.NewAPI(webrtc.WithMediaEngine(m), webrtc.WithInterceptorRegistry(i))
```

`webrtc.NewAPI`用于创建完整的WebRTC设置，包括`Interceptor`、`MediaEngine`和`SettingEngine`。其中`Interceptor0`和`MediaEngine`已经在上面讲过了；`SettingEngine`用于设置那些不在WebRTC标准里的设置项。

从代码上看 **`webrtc.NewAPI`的输入参数全是修改`webrtc.API`类的函数**，`webrtc.WithMediaEngine`和`webrtc.WithInterceptorRegistry`是就是将我们的设置项转化为这种函数的函数，当然，还有`webrtc.WithSettingEngine`

```go
// Prepare the configuration
config := webrtc.Configuration{
    ICEServers: []webrtc.ICEServer{
        {
            URLs: []string{"stun:stun.l.google.com:19302"},
        },
    },
}
```

## 构造PeerConnection

`webrtc.Configuration`用于配置标准WebRTC API里的PeerConnection。

```go
// Create a new RTCPeerConnection
peerConnection, err := api.NewPeerConnection(config)
if err != nil {
    panic(err)
}
defer func() {
    if cErr := peerConnection.Close(); cErr != nil {
        fmt.Printf("cannot close peerConnection: %v\n", cErr)
    }
}()
```

注意这里的`api.NewPeerConnection`和`webrtc.NewPeerConnection`的区别：`api.NewPeerConnection`是按照我们定义的`webrtc.API`类生成PeerConnection；`webrtc.NewPeerConnection`相当于是默认PeerConnection配置，是用`webrtc.RegisterDefaultCodecs`和`webrtc.RegisterDefaultInterceptors`生成默认`webrtc.API`类再调用`api.NewPeerConnection`生成PeerConnection。

可以看出，本实例的PeerConnection与`webrtc.NewPeerConnection`生成的默认PeerConnection区别只在`MediaEngine`上，本例使用的`MediaEngine`比默认配置少，默认配置里添加了所有已实现的编码器。

```go
// Allow us to receive 1 audio track, and 1 video track
if _, err = peerConnection.AddTransceiverFromKind(webrtc.RTPCodecTypeAudio); err != nil {
    panic(err)
} else if _, err = peerConnection.AddTransceiverFromKind(webrtc.RTPCodecTypeVideo); err != nil {
    panic(err)
}
```

配置PeerConnection接受一路视频和一路音频。

## 配置并启动UDP连接

```go
// Create a local addr
var laddr *net.UDPAddr
if laddr, err = net.ResolveUDPAddr("udp", "127.0.0.1:"); err != nil {
    panic(err)
}

// Prepare udp conns
// Also update incoming packets with expected PayloadType, the browser may use
// a different value. We have to modify so our stream matches what rtp-forwarder.sdp expects
udpConns := map[string]*udpConn{
    "audio": {port: 4000, payloadType: 111},
    "video": {port: 4002, payloadType: 96},
}
```

创建两个UDP地址，RTP流就是从这个地址里面来的，这里一看就明白，`127.0.0.1:4000`是音频地址，`127.0.0.1:4002`是视频地址。具体是从里面收数据还是往里发数据还不知道。

这个`udpConn`是在本示例的开头定义：

```go
type udpConn struct {
	conn        *net.UDPConn
	port        int
	payloadType uint8
}
```

它包含`port`和`payloadType`两个数字和一个`net.UDPConn`指针。下面会看到它的用处。

```go
for _, c := range udpConns {
    // Create remote addr
    var raddr *net.UDPAddr
    if raddr, err = net.ResolveUDPAddr("udp", fmt.Sprintf("127.0.0.1:%d", c.port)); err != nil {
        panic(err)
    }

    // Dial udp
    if c.conn, err = net.DialUDP("udp", laddr, raddr); err != nil {
        panic(err)
    }
    defer func(conn net.PacketConn) {
        if closeErr := conn.Close(); closeErr != nil {
            panic(closeErr)
        }
    }(c.conn)
}
```

重点在`c.conn, err = net.DialUDP("udp", laddr, raddr)`，这里用`net.DialUDP`向`udpConn`指定的两个端口创建了两个UDP连接，并且把它们放进对应的`udpConn.conn`里；后面的`defer`指定了退出时关闭连接操作。

## 输入流处理函数

```go
// Set a handler for when a new remote track starts, this handler will forward data to
// our UDP listeners.
// In your application this is where you would handle/process audio/video
peerConnection.OnTrack(func(track *webrtc.TrackRemote, receiver *webrtc.RTPReceiver) {
```

OnTrack用于指定被呼叫时的处理函数，处理函数包含两个参数`(track *webrtc.TrackRemote, receiver *webrtc.RTPReceiver)`：
* `track *webrtc.TrackRemote`表示来自对方的track信息
* `receiver *webrtc.RTPReceiver`表示本地的接收器的信息

接下来是输入流处理函数的正文

```go
    // Retrieve udp connection
    c, ok := udpConns[track.Kind().String()]
    if !ok {
        return
    }
```

这里按照对方的track类型名称取出先前生成的UDP连接。

```go
    // Send a PLI on an interval so that the publisher is pushing a keyframe every rtcpPLIInterval
    go func() {
        ticker := time.NewTicker(time.Second * 2)
        for range ticker.C {
            if rtcpErr := peerConnection.WriteRTCP([]rtcp.Packet{&rtcp.PictureLossIndication{MediaSSRC: uint32(track.SSRC())}}); rtcpErr != nil {
                fmt.Println(rtcpErr)
            }
        }
    }()
```

这里用了一个协程每两秒向PeerConnection发一次PLI请求`rtcp.PictureLossIndication`，从而让发送方知道这边的接收情况。
PLI是一种“关键帧请求”，类似的还有SLI／PLI／FIR，作用是在关键帧丢失无法解码时，请求发送方重新生成并发送一个关键帧。相关知识见[《WebRTC 视频通信中的错误恢复机制》](错误恢复.md)。

```go
    b := make([]byte, 1500)
```

创建一个缓冲区

```go
    rtpPacket := &rtp.Packet{}
    for {
```

`for`循环处理对方track发来的数据

```go
        // Read
        n, _, readErr := track.Read(b)
        if readErr != nil {
            panic(readErr)
        }
```

先把数据读进缓冲区

```go
        // Unmarshal the packet and update the PayloadType
        if err = rtpPacket.Unmarshal(b[:n]); err != nil {
            panic(err)
        }
```

然后将缓冲区里的数据解码为`rtp.Packet`

```go
        rtpPacket.PayloadType = c.payloadType
```

修改一下`rtp.Packet`里的`PayloadType`

```go
        // Marshal into original buffer with updated PayloadType
        if n, err = rtpPacket.MarshalTo(b); err != nil {
            panic(err)
        }
```

再编码放回去缓冲区

这一段的最终目的就只是改一下缓冲区里的`PayloadType`

```go
        // Write
        if _, err = c.conn.Write(b[:n]); err != nil {
            // For this particular example, third party applications usually timeout after a short
            // amount of time during which the user doesn't have enough time to provide the answer
            // to the browser.
            // That's why, for this particular example, the user first needs to provide the answer
            // to the browser then open the third party application. Therefore we must not kill
            // the forward on "connection refused" errors
            if opError, ok := err.(*net.OpError); ok && opError.Err.Error() == "write: connection refused" {
                continue
            }
            panic(err)
        }
```

最后把缓冲区数据写到UDP连接里

```go
    }
})
```

结束。

可以看出，上面这个track不断从PeerConnection里读数据然后写进UDP连接里，说明这个示例是将一个WebRTC流转发为RTP流；并且实例里没有从RTP到WebRTC的代码，说明只能从WebRTC流到RTP流单向发。

## 连接状态变化时的处理函数

```go
// Set the handler for ICE connection state
// This will notify you when the peer has connected/disconnected
peerConnection.OnICEConnectionStateChange(func(connectionState webrtc.ICEConnectionState) {
    fmt.Printf("Connection State has changed %s \n", connectionState.String())

    if connectionState == webrtc.ICEConnectionStateConnected {
        fmt.Println("Ctrl+C the remote client to stop the demo")
    }
})
```

当WebRTC流的ICE连接状态改变时输出一些提示信息。

```go
// Set the handler for Peer connection state
// This will notify you when the peer has connected/disconnected
peerConnection.OnConnectionStateChange(func(s webrtc.PeerConnectionState) {
    fmt.Printf("Peer Connection State has changed: %s\n", s.String())

    if s == webrtc.PeerConnectionStateFailed {
        // Wait until PeerConnection has had no network activity for 30 seconds or another failure. It may be reconnected using an ICE Restart.
        // Use webrtc.PeerConnectionStateDisconnected if you are interested in detecting faster timeout.
        // Note that the PeerConnection may come back from PeerConnectionStateDisconnected.
        fmt.Println("Done forwarding")
        os.Exit(0)
    }
})
```

当WebRTC流的PeerConnection连接状态改变时输出一些提示信息。

## 启动PeerConnection

```go
// Wait for the offer to be pasted
offer := webrtc.SessionDescription{}
signal.Decode(signal.MustReadStdin(), &offer)
```

先从stdin读入SessionDescription

```go
// Set the remote SessionDescription
if err = peerConnection.SetRemoteDescription(offer); err != nil {
    panic(err)
}
```

设置PeerConnection的远端SessionDescription，应该是表示只接受SessionDescription相符的传入连接。

```go
// Create answer
answer, err := peerConnection.CreateAnswer(nil)
if err != nil {
    panic(err)
}
```

设置PeerConnection的SDP Answer。本案例中的WebRTC是接收方，是等待连接的一方，其接收请求方发来的Offer返回自己的Answer，所以是CreateAnswer没有CreateOffer。

```go
// Create channel that is blocked until ICE Gathering is complete
gatherComplete := webrtc.GatheringCompletePromise(peerConnection)
```

这里的`GatheringCompletePromise`返回的是一个`context`(本质上是`<-chan struct{}`)，阻塞直到完成ICE信息收集。

```go
// Sets the LocalDescription, and starts our UDP listeners
if err = peerConnection.SetLocalDescription(answer); err != nil {
    panic(err)
}
```

设置PeerConnection的本地SessionDescription，用在Answer里面。

```go
// Block until ICE Gathering is complete, disabling trickle ICE
// we do this because we only can exchange one signaling message
// in a production application you should exchange ICE Candidates via OnICECandidate
<-gatherComplete
```

等待完成ICE信息收集。注释里讲了这个示例的特殊性：只进行一次ICE信令交换，正常情况应该是使用`OnICECandidate`。

```go
// Output the answer in base64 so we can paste it in browser
fmt.Println(signal.Encode(*peerConnection.LocalDescription()))
```

输出SessionDescription，给用户在服务器里粘贴（包含收集到的ICE信息）。

```go
// Block forever
select {}
```
阻塞，让各协程运行。