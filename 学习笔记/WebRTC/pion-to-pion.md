# 用实例学习pion - [`pion-to-pion`](https://github.com/pion/webrtc/blob/master/examples/pion-to-pion/main.go)

之前的几个案例中，ICE过程中SDP的传输都是手工复制粘贴完成的，没有显式指定监听端口的过程，而`pion-to-pion`就提供了一种自动传输SDP交互的方式。在`pion-to-pion`里面，pion双方都在固定的端口监听POST请求，SDP信息生成好了之后是用POST传给对方的。

`pion-to-pion`分为两方：Offer方和Answer方。它们各是一个Docker容器

## 公共函数

```go
func signalCandidate(addr string, c *webrtc.ICECandidate) error {
	payload := []byte(c.ToJSON().Candidate)
	resp, err := http.Post(fmt.Sprintf("http://%s/candidate", addr), "application/json; charset=utf-8", bytes.NewReader(payload)) //nolint:noctx
	if err != nil {
		return err
	}

	if closeErr := resp.Body.Close(); closeErr != nil {
		return closeErr
	}

	return nil
}
```

这个函数在Offer方和Answer方的代码都用到了，很明显是一个POST请求，发到对方的`/candidate`路径下，之后会看到其用法。

## Offer方主函数

```go
offerAddr := flag.String("offer-address", ":50000", "Address that the Offer HTTP server is hosted on.")
answerAddr := flag.String("answer-address", "127.0.0.1:60000", "Address that the Answer HTTP server is hosted on.")
flag.Parse()
```
指定双方的http server监听地址和端口

```go
var candidatesMux sync.Mutex
pendingCandidates := make([]*webrtc.ICECandidate, 0)
```
pending列表，在之后会看到它的用法

```go
// Everything below is the Pion WebRTC API! Thanks for using it ❤️.

// Prepare the configuration
config := webrtc.Configuration{
    ICEServers: []webrtc.ICEServer{
        {
            URLs: []string{"stun:stun.l.google.com:19302"},
        },
    },
}

// Create a new RTCPeerConnection
peerConnection, err := webrtc.NewPeerConnection(config)
if err != nil {
    panic(err)
}
defer func() {
    if cErr := peerConnection.Close(); cErr != nil {
        fmt.Printf("cannot close peerConnection: %v\n", cErr)
    }
}()
```
初始化PeerConnection，和[《用实例学习pion - `rtp-forwarder`》](rtp-forwarder.md)里一样，不用多讲。

```go
// When an ICE candidate is available send to the other Pion instance
// the other Pion instance will add this candidate by calling AddICECandidate
peerConnection.OnICECandidate(func(c *webrtc.ICECandidate) {
    if c == nil {
        return
    }

    candidatesMux.Lock()
    defer candidatesMux.Unlock()

    desc := peerConnection.RemoteDescription()
    if desc == nil {
        pendingCandidates = append(pendingCandidates, c)
    } else if onICECandidateErr := signalCandidate(*answerAddr, c); onICECandidateErr != nil {
        panic(onICECandidateErr)
    }
})
```
`OnICECandidate`指定了当本地这边ICECandidate信息收集完成时进行的操作。这里指定的操作分两个：
* 如果没有调用过`SetRemoteDescription`，那就将现在这个ICECandidate放进pending列表里
* 如果已经调用过`SetRemoteDescription`，那就直接用前面定义的公用函数`signalCandidate`把ICECandidate发到对面

结合[《从一张图开始理解WebRTC中的STUN、SDP和ICE过程》](./STUN和ICE和SDP.md)里的介绍，已经调用过`SetRemoteDescription`表明对方用于的Answer的SDP已经到达，这里这个if判断的用意就是要先等对面的SDP信息到达之后再发送这边收集好的ICECandidate信息。

```go
// A HTTP handler that allows the other Pion instance to send us ICE candidates
// This allows us to add ICE candidates faster, we don't have to wait for STUN or TURN
// candidates which may be slower
http.HandleFunc("/candidate", func(w http.ResponseWriter, r *http.Request) {
    candidate, candidateErr := ioutil.ReadAll(r.Body)
    if candidateErr != nil {
        panic(candidateErr)
    }
    if candidateErr := peerConnection.AddICECandidate(webrtc.ICECandidateInit{Candidate: string(candidate)}); candidateErr != nil {
        panic(candidateErr)
    }
})
```
这里的`/candidate`正好就是前面的公用函数`signalCandidate`里面发POST指定的路径，结合上面`signalCandidate`在`OnICECandidate`里面被调用的情况，这个里面的操作就是处理对面发来的ICECandidate连接信息。具体看这里面的操作流程就是读取`r.Body`里的ICE候选信息然后用`AddICECandidate`加到PeerConnection里。

结合[《从一张图开始理解WebRTC中的STUN、SDP和ICE过程》](./STUN和ICE和SDP.md)里的介绍和上面`OnICECandidate`里的操作，可以看出这里ICECandidate信令信息是直接通过POST请求传的，没有什么信令服务器，双方都有一个http server在固定端口上监听接收ICECandidate信息。

```go
// A HTTP handler that processes a SessionDescription given to us from the other Pion process
http.HandleFunc("/sdp", func(w http.ResponseWriter, r *http.Request) {
    sdp := webrtc.SessionDescription{}
    if sdpErr := json.NewDecoder(r.Body).Decode(&sdp); sdpErr != nil {
        panic(sdpErr)
    }

    if sdpErr := peerConnection.SetRemoteDescription(sdp); sdpErr != nil {
        panic(sdpErr)
    }

    candidatesMux.Lock()
    defer candidatesMux.Unlock()

    for _, c := range pendingCandidates {
        if onICECandidateErr := signalCandidate(*answerAddr, c); onICECandidateErr != nil {
            panic(onICECandidateErr)
        }
    }
})
```
这里又给http server加上了一个`/sdp`路径下的POST处理函数，里面的操作有两个：
* 接收Offer SDP，调用`SetRemoteDescription`存储之，和[《从一张图开始理解WebRTC中的STUN、SDP和ICE过程》](./STUN和ICE和SDP.md)里讲的一样
* 将pending列表中的ICECandidate信息全部发给对面

结合前面的`OnICECandidate`里面的操作，进一步验证了这个程序处理Offer/Answer和ICECandidate的顺序：先交换Offer/Answer，完成后再交换ICECandidate

```go
// Start HTTP server that accepts requests from the answer process
go func() { panic(http.ListenAndServe(*offerAddr, nil)) }()
```
启动前面定义好的http server，开始接收Offer/Answer和ICECandidate信息

```go
// Create a datachannel with label 'data'
dataChannel, err := peerConnection.CreateDataChannel("data", nil)
if err != nil {
    panic(err)
}

// Set the handler for Peer connection state
// This will notify you when the peer has connected/disconnected
peerConnection.OnConnectionStateChange(func(s webrtc.PeerConnectionState) {
    fmt.Printf("Peer Connection State has changed: %s\n", s.String())

    if s == webrtc.PeerConnectionStateFailed {
        // Wait until PeerConnection has had no network activity for 30 seconds or another failure. It may be reconnected using an ICE Restart.
        // Use webrtc.PeerConnectionStateDisconnected if you are interested in detecting faster timeout.
        // Note that the PeerConnection may come back from PeerConnectionStateDisconnected.
        fmt.Println("Peer Connection has gone to failed exiting")
        os.Exit(0)
    }
})

// Register channel opening handling
dataChannel.OnOpen(func() {
    fmt.Printf("Data channel '%s'-'%d' open. Random messages will now be sent to any connected DataChannels every 5 seconds\n", dataChannel.Label(), dataChannel.ID())

    for range time.NewTicker(5 * time.Second).C {
        message := signal.RandSeq(15)
        fmt.Printf("Sending '%s'\n", message)

        // Send the message as text
        sendTextErr := dataChannel.SendText(message)
        if sendTextErr != nil {
            panic(sendTextErr)
        }
    }
})

// Register text message handling
dataChannel.OnMessage(func(msg webrtc.DataChannelMessage) {
    fmt.Printf("Message from DataChannel '%s': '%s'\n", dataChannel.Label(), string(msg.Data))
})
```
这里定义了连接建立之后用dataChannel发送测试数据的操作，没啥好分析的

```go
// Create an offer to send to the other process
offer, err := peerConnection.CreateOffer(nil)
if err != nil {
    panic(err)
}

// Sets the LocalDescription, and starts our UDP listeners
// Note: this will start the gathering of ICE candidates
if err = peerConnection.SetLocalDescription(offer); err != nil {
    panic(err)
}
```
`CreateOffer`创建用于Offer的SDP对象，`SetLocalDescription`保存SDP对象，和[《从一张图开始理解WebRTC中的STUN、SDP和ICE过程》](./STUN和ICE和SDP.md)里讲的一样

```go
// Send our offer to the HTTP server listening in the other process
payload, err := json.Marshal(offer)
if err != nil {
    panic(err)
}
resp, err := http.Post(fmt.Sprintf("http://%s/sdp", *answerAddr), "application/json; charset=utf-8", bytes.NewReader(payload)) // nolint:noctx
if err != nil {
    panic(err)
} else if err := resp.Body.Close(); err != nil {
    panic(err)
}
```
将用于Offer的SDP对象发给对面

```go
// Block forever
select {}
```
之后的所有操作都是在前面的`OnXXX`等函数里定义的异步操作

## Answer方主函数

根据[《从一张图开始理解WebRTC中的STUN、SDP和ICE过程》](./STUN和ICE和SDP.md)里的介绍，Answer方在ICE中的行为大部分都与Offer方相同，仅有的区别只有两个：
* Offer方主动发起请求，Answer方被动接收请求
* Offer发起请求时发的是Offer SDP，Answer响应时发的是Answer SDP

在pion-to-pion代码里的，上面这两个不同点也就体体现在两处：
* Offer方和Answer方的http server中`/sdp`路径下的处理函数中多了发回Answer SDP的过程
* Answer方的主函数在配置好PeerConnection然后启动http server之后就结束了，没有Offer方主函数最后面往对面`/sdp`接口POST SDP对象的过程

这是Answer方的http server中`/sdp`路径下的处理函数：
```go
// A HTTP handler that processes a SessionDescription given to us from the other Pion process
http.HandleFunc("/sdp", func(w http.ResponseWriter, r *http.Request) {
    sdp := webrtc.SessionDescription{}
    if err := json.NewDecoder(r.Body).Decode(&sdp); err != nil {
        panic(err)
    }

    if err := peerConnection.SetRemoteDescription(sdp); err != nil {
        panic(err)
    }
```
可以看到，前面这个调用`SetRemoteDescription`的过程和Offer方的一样

```go
    // Create an answer to send to the other process
    answer, err := peerConnection.CreateAnswer(nil)
    if err != nil {
        panic(err)
    }

    // Send our answer to the HTTP server listening in the other process
    payload, err := json.Marshal(answer)
    if err != nil {
        panic(err)
    }
    resp, err := http.Post(fmt.Sprintf("http://%s/sdp", *offerAddr), "application/json; charset=utf-8", bytes.NewReader(payload)) // nolint:noctx
    if err != nil {
        panic(err)
    } else if closeErr := resp.Body.Close(); closeErr != nil {
        panic(closeErr)
    }

    // Sets the LocalDescription, and starts our UDP listeners
    err = peerConnection.SetLocalDescription(answer)
    if err != nil {
        panic(err)
    }
```
这里就是Answer方和Offer方不一样的地方，和[《从一张图开始理解WebRTC中的STUN、SDP和ICE过程》](./STUN和ICE和SDP.md)里介绍的一样，这里Answer方调用了`CreateAnswer`生成了用于Answer的SDP对象并调用了`SetLocalDescription`存储SDP对象。

这里中间先把Answer SDP用POST发给了Offer方的`/sdp`里，就是前面说的Answer方的响应。

深入想想，其实Offer方没必要有`/sdp`接口，Answer方可以直接把Answer SDP写到`/sdp`的响应里面，Offer方可以把`/sdp`处理函数里发送ICECandidate的操作放在收到POST请求的响应之后。而不是像现在这样Answer方发起另外一个POST请求返回Answer SDP。

```go
    candidatesMux.Lock()
    for _, c := range pendingCandidates {
        onICECandidateErr := signalCandidate(*offerAddr, c)
        if onICECandidateErr != nil {
            panic(onICECandidateErr)
        }
    }
    candidatesMux.Unlock()
})
```
可以看到，最后面这里发回ICECandidate操作也和Offer方的一样。