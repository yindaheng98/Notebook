# 用实例学习pion - [`play-from-disk`](https://github.com/pion/webrtc/blob/master/examples/play-from-disk/main.go)

```go
// Assert that we have an audio or video file
_, err := os.Stat(videoFileName)
haveVideoFile := !os.IsNotExist(err)

_, err = os.Stat(audioFileName)
haveAudioFile := !os.IsNotExist(err)

if !haveAudioFile && !haveVideoFile {
    panic("Could not find `" + audioFileName + "` or `" + videoFileName + "`")
}
```
先看看本地视频和音频文件在不在

```go
// Create a new RTCPeerConnection
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
defer func() {
    if cErr := peerConnection.Close(); cErr != nil {
        fmt.Printf("cannot close peerConnection: %v\n", cErr)
    }
}()
```
`webrtc.NewPeerConnection`创建默认配置的WebRTC标准PeerConnection。

```go
iceConnectedCtx, iceConnectedCtxCancel := context.WithCancel(context.Background())
```
创建context，后面用到。

## 绑定视频流

```go
if haveVideoFile {
```
如果视频存在就开始绑定视频流

```go
    // Create a video track
    videoTrack, videoTrackErr := webrtc.NewTrackLocalStaticSample(webrtc.RTPCodecCapability{MimeType: webrtc.MimeTypeVP8}, "video", "pion")
    if videoTrackErr != nil {
        panic(videoTrackErr)
    }
```
首先是创建track，pion里的track分为`TrackLocal`和`TrackRemote`两种interface。其中`TrackLocal`是本地发给别人的track，`TrackRemote`是别人发给本地的track，更多分析可见[《pion中的`TrackLocal`》](TrackLocal.md)和[《pion中的`TrackRemote`》](TrackRemote.md)。这里是要发送给远端的，当然是创建`TrackLocal`。

这里用的`webrtc.NewTrackLocalStaticSample`是pion里提供的创建`TrackLocal`的范例。

```go
    rtpSender, videoTrackErr := peerConnection.AddTrack(videoTrack)
    if videoTrackErr != nil {
        panic(videoTrackErr)
    }

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

用`AddTrack`添加刚才创建好的track，和[《用实例学习pion - `rtp-to-webrtc`》](rtp-to-webrtc.md)里一样

## 视频流的发送过程

```go
    go func() {
```

这里一个协程独立运行视频流的发送过程

```go
        // Open a IVF file and start reading using our IVFReader
        file, ivfErr := os.Open(videoFileName)
        if ivfErr != nil {
            panic(ivfErr)
        }
```

先打开文件

```go
        ivf, header, ivfErr := ivfreader.NewWith(file)
        if ivfErr != nil {
            panic(ivfErr)
        }
```

这就是创建一个读取器

```go
        // Wait for connection established
        <-iceConnectedCtx.Done()
```

等待连接建立后正式开始操作

```go
        // Send our video file frame at a time. Pace our sending so we send it at the same speed it should be played back as.
        // This isn't required since the video is timestamped, but we will such much higher loss if we send all at once.
        //
        // It is important to use a time.Ticker instead of time.Sleep because
        // * avoids accumulating skew, just calling time.Sleep didn't compensate for the time spent parsing the data
        // * works around latency issues with Sleep (see https://github.com/golang/go/issues/44343)
        ticker := time.NewTicker(time.Millisecond * time.Duration((float32(header.TimebaseNumerator)/float32(header.TimebaseDenominator))*1000))
        for ; true; <-ticker.C {
            frame, _, ivfErr := ivf.ParseNextFrame()
            if ivfErr == io.EOF {
                fmt.Printf("All video frames parsed and sent")
                os.Exit(0)
            }

            if ivfErr != nil {
                panic(ivfErr)
            }

            if ivfErr = videoTrack.WriteSample(media.Sample{Data: frame, Duration: time.Second}); ivfErr != nil {
                panic(ivfErr)
            }
        }
```

可以看到，操作不过就是将ivf格式的视频数据读出来用`WriteSample`写进track里。

```go
    }()
}
```
协程结束。

## 绑定音频流、音频流的发送过程

```go
if haveAudioFile {
    // Create a audio track
    audioTrack, audioTrackErr := webrtc.NewTrackLocalStaticSample(webrtc.RTPCodecCapability{MimeType: webrtc.MimeTypeOpus}, "audio", "pion")
    if audioTrackErr != nil {
        panic(audioTrackErr)
    }

    rtpSender, audioTrackErr := peerConnection.AddTrack(audioTrack)
    if audioTrackErr != nil {
        panic(audioTrackErr)
    }

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

    go func() {
        // Open a IVF file and start reading using our IVFReader
        file, oggErr := os.Open(audioFileName)
        if oggErr != nil {
            panic(oggErr)
        }

        // Open on oggfile in non-checksum mode.
        ogg, _, oggErr := oggreader.NewWith(file)
        if oggErr != nil {
            panic(oggErr)
        }

        // Wait for connection established
        <-iceConnectedCtx.Done()

        // Keep track of last granule, the difference is the amount of samples in the buffer
        var lastGranule uint64

        // It is important to use a time.Ticker instead of time.Sleep because
        // * avoids accumulating skew, just calling time.Sleep didn't compensate for the time spent parsing the data
        // * works around latency issues with Sleep (see https://github.com/golang/go/issues/44343)
        ticker := time.NewTicker(oggPageDuration)
        for ; true; <-ticker.C {
            pageData, pageHeader, oggErr := ogg.ParseNextPage()
            if oggErr == io.EOF {
                fmt.Printf("All audio pages parsed and sent")
                os.Exit(0)
            }

            if oggErr != nil {
                panic(oggErr)
            }

            // The amount of samples is the difference between the last and current timestamp
            sampleCount := float64(pageHeader.GranulePosition - lastGranule)
            lastGranule = pageHeader.GranulePosition
            sampleDuration := time.Duration((sampleCount/48000)*1000) * time.Millisecond

            if oggErr = audioTrack.WriteSample(media.Sample{Data: pageData, Duration: sampleDuration}); oggErr != nil {
                panic(oggErr)
            }
        }
    }()
}
```

可以看到，操作都和视频流一毛一样。

## 启动PeerConnection

```go

// Set the handler for ICE connection state
// This will notify you when the peer has connected/disconnected
peerConnection.OnICEConnectionStateChange(func(connectionState webrtc.ICEConnectionState) {
    fmt.Printf("Connection State has changed %s \n", connectionState.String())
    if connectionState == webrtc.ICEConnectionStateConnected {
        iceConnectedCtxCancel()
    }
})

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

// Block forever
select {}
```

和[《用实例学习pion - `rtp-forwarder`》](rtp-forwarder.md)里一样的等待链接的方式，不必多讲。