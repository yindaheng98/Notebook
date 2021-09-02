# pion中的`TrackRemote`

上接[《pion中的`TrackLocal`》](TrackLocal.md)，有`TrackLocal`表示表示本地发往远端的track，对应的自然也会有`TrackRemote`表示远端发到本地的track：

```go
// TrackRemote represents a single inbound source of media
type TrackRemote struct {
	mu sync.RWMutex

	id       string
	streamID string

	payloadType PayloadType
	kind        RTPCodecType
	ssrc        SSRC
	codec       RTPCodecParameters
	params      RTPParameters
	rid         string

	receiver         *RTPReceiver
	peeked           []byte
	peekedAttributes interceptor.Attributes
}
```

`TrackRemote`就没有像`TrackLocal`做成接口了，大概是因为参数和数据都是对面给的，用户只需要读取，不需要什么其他侵入pion内部的操作。

那么最核心的就是这个`Read`和`ReadRTP`了，就是读取数据，不必多讲。

```go
// Read reads data from the track.
func (t *TrackRemote) Read(b []byte) (n int, attributes interceptor.Attributes, err error) {
    ...
}

// ReadRTP is a convenience method that wraps Read and unmarshals for you.
func (t *TrackRemote) ReadRTP() (*rtp.Packet, interceptor.Attributes, error) {
	b := make([]byte, t.receiver.api.settingEngine.getReceiveMTU())
	i, attributes, err := t.Read(b)
	if err != nil {
		return nil, nil, err
	}

	r := &rtp.Packet{}
	if err := r.Unmarshal(b[:i]); err != nil {
		return nil, nil, err
	}
	return r, attributes, nil
}
```

因为不需要用户自定义什么操作，所以也没有什么继承之类的操作，最主要的操作就是在`OnTrack`里面用，当有track连接进来时处理之，比如在[用实例学习pion - `gocv-receive`](gocv-receive.md)里面就是：

```go
// Set a handler for when a new remote track starts, this handler copies inbound RTP packets,
// replaces the SSRC and sends them back
peerConnection.OnTrack(func(track *webrtc.TrackRemote, receiver *webrtc.RTPReceiver) {
    // Send a PLI on an interval so that the publisher is pushing a keyframe every rtcpPLIInterval
    go func() {
        ticker := time.NewTicker(time.Second * 3)
        for range ticker.C {
            errSend := peerConnection.WriteRTCP([]rtcp.Packet{&rtcp.PictureLossIndication{MediaSSRC: uint32(track.SSRC())}})
            if errSend != nil {
                fmt.Println(errSend)
            }
        }
    }()

    fmt.Printf("Track has started, of type %d: %s \n", track.PayloadType(), track.Codec().RTPCodecCapability.MimeType)
    for {
        // Read RTP packets being sent to Pion
        rtp, _, readErr := track.ReadRTP()
        if readErr != nil {
            panic(readErr)
        }

        if ivfWriterErr := ivfWriter.WriteRTP(rtp); ivfWriterErr != nil {
            panic(ivfWriterErr)
        }
    }
})
```

`OnTrack`里面直接一个死循环不断用`track.ReadRTP`读数据拿去`ivfWriter.WriteRTP`处理。