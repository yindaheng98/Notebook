# 用实例学习pion - [`rtp-to-webrtc`](https://github.com/pion/webrtc/blob/master/examples/rtp-to-webrtc/main.go)

```go
peerConnection, err := webrtc.NewPeerConnection(webrtc.Configuration{
    ICEServers: []webrtc.ICEServer{
        {
            URLs: []string{"stun:stun.l.google.com:19302"},
        },
    },
})
if err != nil {
    panic(err)
}
```

`webrtc.NewPeerConnection`创建默认配置的WebRTC标准PeerConnection。

```go
// Open a UDP Listener for RTP Packets on port 5004
listener, err := net.ListenUDP("udp", &net.UDPAddr{IP: net.ParseIP("127.0.0.1"), Port: 5004})
if err != nil {
    panic(err)
}
defer func() {
    if err = listener.Close(); err != nil {
        panic(err)
    }
}()
```

开一个UDP端口等待RTP数据传入。

```go
// Create a video track
videoTrack, err := webrtc.NewTrackLocalStaticRTP(webrtc.RTPCodecCapability{MimeType: webrtc.MimeTypeVP8}, "video", "pion")
if err != nil {
    panic(err)
}
rtpSender, err := peerConnection.AddTrack(videoTrack)
if err != nil {
    panic(err)
}
```

创建并添加本地的传给请求方的Track，`AddTrack`会根据传入的track构建。

```go
// Read incoming RTCP packets
// Before these packets are returned they are processed by interceptors. For things
// like NACK this needs to be called.
go func() {
    rtcpBuf := make([]byte, 1500)
    for {
        if _, _, rtcpErr := rtpSender.Read(rtcpBuf); rtcpErr != nil {
            return
        }
    }
}()
```

从注释上看，这里的读取只是为了`rtpSender`里面的那些操作能执行，那么也就是说如果没有`rtpSender.Read`发进来的包会阻塞？

```go
// Set the handler for ICE connection state
// This will notify you when the peer has connected/disconnected
peerConnection.OnICEConnectionStateChange(func(connectionState webrtc.ICEConnectionState) {
    fmt.Printf("Connection State has changed %s \n", connectionState.String())

    if connectionState == webrtc.ICEConnectionStateFailed {
        if closeErr := peerConnection.Close(); closeErr != nil {
            panic(closeErr)
        }
    }
})
```

连接状态改变时输出提示信息，不必多讲。

```go
// Wait for the offer to be pasted
offer := webrtc.SessionDescription{}
signal.Decode(signal.MustReadStdin(), &offer)

// Set the remote SessionDescription
if err = peerConnection.SetRemoteDescription(offer); err != nil {
    panic(err)
}

// Create answer
answer, err := peerConnection.CreateAnswer(nil)
if err != nil {
    panic(err)
}

// Create channel that is blocked until ICE Gathering is complete
gatherComplete := webrtc.GatheringCompletePromise(peerConnection)

// Sets the LocalDescription, and starts our UDP listeners
if err = peerConnection.SetLocalDescription(answer); err != nil {
    panic(err)
}

// Block until ICE Gathering is complete, disabling trickle ICE
// we do this because we only can exchange one signaling message
// in a production application you should exchange ICE Candidates via OnICECandidate
<-gatherComplete

// Output the answer in base64 so we can paste it in browser
fmt.Println(signal.Encode(*peerConnection.LocalDescription()))
```

和[《用实例学习pion - `rtp-forwarder`》](rtp-forwarder.md)里一样的等待链接的方式，不必多讲。

```go
// Read RTP packets forever and send them to the WebRTC Client
inboundRTPPacket := make([]byte, 1600) // UDP MTU
for {
    n, _, err := listener.ReadFrom(inboundRTPPacket)
    if err != nil {
        panic(fmt.Sprintf("error during read: %s", err))
    }

    if _, err = videoTrack.Write(inboundRTPPacket[:n]); err != nil {
        if errors.Is(err, io.ErrClosedPipe) {
            // The peerConnection has been closed.
            return
        }

        panic(err)
    }
}
```
和[《用实例学习pion - `rtp-forwarder`》](rtp-forwarder.md)里直接用`select`阻塞不一样，这里是一个死循环不停将UDP发来的RTP数据包写进前面创建的track里面。所以这地方应该就是主要的转发流程了。