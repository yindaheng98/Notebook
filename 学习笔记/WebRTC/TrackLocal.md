# pion中的`TrackLocal`

```go
// TrackLocal is an interface that controls how the user can send media
// The user can provide their own TrackLocal implementations, or use
// the implementations in pkg/media
type TrackLocal interface {
	// Bind should implement the way how the media data flows from the Track to the PeerConnection
	// This will be called internally after signaling is complete and the list of available
	// codecs has been determined
	Bind(TrackLocalContext) (RTPCodecParameters, error)

	// Unbind should implement the teardown logic when the track is no longer needed. This happens
	// because a track has been stopped.
	Unbind(TrackLocalContext) error

	// ID is the unique identifier for this Track. This should be unique for the
	// stream, but doesn't have to globally unique. A common example would be 'audio' or 'video'
	// and StreamID would be 'desktop' or 'webcam'
	ID() string

	// StreamID is the group this track belongs too. This must be unique
	StreamID() string

	// Kind controls if this TrackLocal is audio or video
	Kind() RTPCodecType
}
```

`TrackLocal`表示的是本地发往远端的track，其最核心的功能就是这个`Bind`和`Unbind`。从注释上看，这两个就是把`TrackLocalContext`绑定和解绑到这个Track上面，来看看`TrackLocalContext`是什么：

```go
// TrackLocalContext is the Context passed when a TrackLocal has been Binded/Unbinded from a PeerConnection, and used
// in Interceptors.
type TrackLocalContext struct {
	id          string
	params      RTPParameters
	ssrc        SSRC
	writeStream TrackLocalWriter
}

// CodecParameters returns the negotiated RTPCodecParameters. These are the codecs supported by both
// PeerConnections and the SSRC/PayloadTypes
func (t *TrackLocalContext) CodecParameters() []RTPCodecParameters {
	return t.params.Codecs
}

// HeaderExtensions returns the negotiated RTPHeaderExtensionParameters. These are the header extensions supported by
// both PeerConnections and the SSRC/PayloadTypes
func (t *TrackLocalContext) HeaderExtensions() []RTPHeaderExtensionParameter {
	return t.params.HeaderExtensions
}

// SSRC requires the negotiated SSRC of this track
// This track may have multiple if RTX is enabled
func (t *TrackLocalContext) SSRC() SSRC {
	return t.ssrc
}

// WriteStream returns the WriteStream for this TrackLocal. The implementer writes the outbound
// media packets to it
func (t *TrackLocalContext) WriteStream() TrackLocalWriter {
	return t.writeStream
}

// ID is a unique identifier that is used for both Bind/Unbind
func (t *TrackLocalContext) ID() string {
	return t.id
}
```
一看就能猜出来，最主要的操作应该是和`TrackLocalWriter`的写入有关，这个`TrackLocalWriter`就是个写入的东西：

```go
// TrackLocalWriter is the Writer for outbound RTP Packets
type TrackLocalWriter interface {
	// WriteRTP encrypts a RTP packet and writes to the connection
	WriteRTP(header *rtp.Header, payload []byte) (int, error)

	// Write encrypts and writes a full RTP packet
	Write(b []byte) (int, error)
}
```

上面的绑定操作应该就是让track知道自己要写哪个`TrackLocalWriter`，而具体是怎么写、写什么数据，那就是由用户自己定义，只要数据能写入到`TrackLocalWriter.WriteRTP`或`TrackLocalWriter.Write`里就行。

比如pion就有一个内置的`TrackLocalStaticRTP`，构建了一种编解码方式不变的`TrackLocal`：

```go
// TrackLocalStaticRTP  is a TrackLocal that has a pre-set codec and accepts RTP Packets.
// If you wish to send a media.Sample use TrackLocalStaticSample
type TrackLocalStaticRTP struct {
	mu           sync.RWMutex
	bindings     []trackBinding
	codec        RTPCodecCapability
	id, streamID string
}
```

它的bind就是简单地把`TrackLocalContext`里的东西放进列表：

```go
// Bind is called by the PeerConnection after negotiation is complete
// This asserts that the code requested is supported by the remote peer.
// If so it setups all the state (SSRC and PayloadType) to have a call
func (s *TrackLocalStaticRTP) Bind(t TrackLocalContext) (RTPCodecParameters, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	parameters := RTPCodecParameters{RTPCodecCapability: s.codec}
	if codec, matchType := codecParametersFuzzySearch(parameters, t.CodecParameters()); matchType != codecMatchNone {
		s.bindings = append(s.bindings, trackBinding{
			ssrc:        t.SSRC(),
			payloadType: codec.PayloadType,
			writeStream: t.WriteStream(),
			id:          t.ID(),
		})
		return codec, nil
	}

	return RTPCodecParameters{}, ErrUnsupportedCodec
}
```

然后它当然是提供了写数据的函数：

```go
// WriteRTP writes a RTP Packet to the TrackLocalStaticRTP
// If one PeerConnection fails the packets will still be sent to
// all PeerConnections. The error message will contain the ID of the failed
// PeerConnections so you can remove them
func (s *TrackLocalStaticRTP) WriteRTP(p *rtp.Packet) error {
	ipacket := rtpPacketPool.Get()
	packet := ipacket.(*rtp.Packet)
	defer func() {
		*packet = rtp.Packet{}
		rtpPacketPool.Put(ipacket)
	}()
	*packet = *p
	return s.writeRTP(packet)
}

// writeRTP is like WriteRTP, except that it may modify the packet p
func (s *TrackLocalStaticRTP) writeRTP(p *rtp.Packet) error {
	s.mu.RLock()
	defer s.mu.RUnlock()

	writeErrs := []error{}

	for _, b := range s.bindings {
		p.Header.SSRC = uint32(b.ssrc)
		p.Header.PayloadType = uint8(b.payloadType)
		if _, err := b.writeStream.WriteRTP(&p.Header, p.Payload); err != nil {
			writeErrs = append(writeErrs, err)
		}
	}

	return util.FlattenErrs(writeErrs)
}
```

可以看到，这基本上就是调用所有的`Bind`里添加进数组的`TrackLocalWriter`的`TrackLocalWriter.WriteRTP`。

在用这个类的时候，就是将其用`AddTrack`或者`AddTransceiverFromTrack`加进PeerConnection，然后调用这个`WriteRTP`方法就相当于是在向远端发送数据了。

比如[《用实例学习pion - `rtp-to-webrtc`》](rtp-to-webrtc.md)里介绍的就是`TrackLocalStaticRTP`的典型应用：

在前面`AddTrack`：

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

然后最后一个死循环不断从UDP连接里读数据写进track里：

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