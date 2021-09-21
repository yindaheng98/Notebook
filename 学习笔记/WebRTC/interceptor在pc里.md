# interceptor寻踪：`pion/interceptor`在`pion/webrtc`里的用法解析

## 初始化

本节主要讲解WebRTC标准接口`NewPeerConnection`内部和调用前所需要进行的interceptor初始化操作。在开始前，你首先需要去[《用实例学习pion - `rtp-forwarder`》](./rtp-forwarder.md)和[《pion学习总结：等待传入track的一般流程》](./传入总结.md)里看看`NewPeerConnection`的用法以及在调用`NewPeerConnection`前所需要进行的操作；然后你还需要理解[《pion/interceptor浅析》](./pion-interceptor.md)中关于级联的思想和[《用实例学习pion interceptor - `nack`》](./pion-nack.md)里出现的`interceptor.NewChain`是什么。

### 在`NewPeerConnection`之前

从[《用实例学习pion - `rtp-forwarder`》](./rtp-forwarder.md)中可以看到，在正式调用`api.NewPeerConnection`之前，与`pion/interceptor`有关的操作主要就是创建`interceptor.Registry`并调用`webrtc.NewAPI`创建WebRTC标准API。这个`interceptor.Registry`非常之简单：
```go
// Registry is a collector for interceptors.
type Registry struct {
	interceptors []Interceptor
}

// Add adds a new Interceptor to the registry.
func (i *Registry) Add(icpr Interceptor) {
	i.interceptors = append(i.interceptors, icpr)
}

// Build constructs a single Interceptor from a InterceptorRegistry
func (i *Registry) Build() Interceptor {
	if len(i.interceptors) == 0 {
		return &NoOp{}
	}

	return NewChain(i.interceptors)
}
```
可以看到，类方法就两个，一个`Add`就是添加，然后一个`Build`生成一个`interceptor.Chain`。所以这个`interceptor.Registry`的用处很明显就是构造interceptor的调用链。

从[《用实例学习pion - `rtp-forwarder`》](./rtp-forwarder.md)中还可以看到，这个`interceptor.Registry`并不是直接输入到`webrtc.NewAPI`里的，而是先经过了一个`webrtc.WithInterceptorRegistry`，这个`webrtc.WithInterceptorRegistry`更是简单：
```go
// WithInterceptorRegistry allows providing Interceptors to the API.
// Settings should not be changed after passing the registry to an API.
func WithInterceptorRegistry(interceptorRegistry *interceptor.Registry) func(a *API) {
	return func(a *API) {
		a.interceptor = interceptorRegistry.Build()
	}
}
```
直接就是调用上面那个`interceptor.Registry`里的`Build`函数。

### 在`NewPeerConnection`里

`NewPeerConnection`就是这个函数：
```go
func (api *API) NewPeerConnection(configuration Configuration) (*PeerConnection, error)
```

在这个函数里与interceptor相关的就一句：
```go
pc.interceptorRTCPWriter = api.interceptor.BindRTCPWriter(interceptor.RTCPWriterFunc(pc.writeRTCP))
```
显然这是给interceptor绑了一个实际进行RTCP写操作的函数`pc.writeRTCP`，这个函数显然是要负责把RTCP包发出去。

然后当然也有关闭的操作，在`PeerConnection.Close`里，就是在关闭`PeerConnection`时要关闭interceptor，很好理解。

## 在`TrackLocal`里

本节介绍`TrackLocal`与interceptor之间的关系。在开始前，你首先需要去[《pion中的`TrackLocal`》](./TrackLocal.md)里看看`TrackLocal`是什么和怎么用。

从[《pion中的`TrackLocal`》](./TrackLocal.md)里可以看到`TrackLocal`只是一个接口，interceptor应该是隐藏在`Bind`函数所输入的`TrackLocalContext`的`writeStream`里的：
```go
// TrackLocalWriter is the Writer for outbound RTP Packets
type TrackLocalWriter interface {
	// WriteRTP encrypts a RTP packet and writes to the connection
	WriteRTP(header *rtp.Header, payload []byte) (int, error)

	// Write encrypts and writes a full RTP packet
	Write(b []byte) (int, error)
}

// TrackLocalContext is the Context passed when a TrackLocal has been Binded/Unbinded from a PeerConnection, and used
// in Interceptors.
type TrackLocalContext struct {
	id          string
	params      RTPParameters
	ssrc        SSRC
	writeStream TrackLocalWriter
}

......

// TrackLocal is an interface that controls how the user can send media
// The user can provide their own TrackLocal implementations, or use
// the implementations in pkg/media
type TrackLocal interface {
	// Bind should implement the way how the media data flows from the Track to the PeerConnection
	// This will be called internally after signaling is complete and the list of available
	// codecs has been determined
	Bind(TrackLocalContext) (RTPCodecParameters, error)
	
	......
}
```
从[《pion中的`TrackLocal`》](./TrackLocal.md)里最后面介绍的`TrackLocalStaticRTP`案例可以看到，`TrackLocalWriter`应该就是`TrackLocal`发送数据用的东西。从注释上看，这个`TrackLocalWriter`是由框架构造好了再传进去的，所以与interceptor相关的操作都是在外面定义好了封进`TrackLocalWriter`再传进来的，`TrackLocal`里面本身不涉及interceptor相关的操作。

## 在`TrackRemote`里

本节介绍`TrackRemote`与interceptor之间的关系。在开始前，你首先需要去[《pion中的`TrackRemote`》](./TrackRemote.md)里看看`TrackRemote`是什么和怎么用。

从[《pion中的`TrackRemote`》](./TrackRemote.md)里的调用链可以看到，最核心的函数就只有一个`Read`：
```go
// Read reads data from the track.
func (t *TrackRemote) Read(b []byte) (n int, attributes interceptor.Attributes, err error) {
	t.mu.RLock()
	r := t.receiver
	peeked := t.peeked != nil
	t.mu.RUnlock()

	......

	n, attributes, err = r.readRTP(b, t)
	if err != nil {
		return
	}

	err = t.checkAndUpdateTrack(b)
	return
}
```
而`Read`里最核心的明显就是这句`r.readRTP(b, t)`，而这个`r`就是来自于上面那句`r := t.receiver`。再去看看`TrackRemote`的定义，可以发现这个`t.receiver`是个`RTPReceiver`：
```go
type TrackRemote struct {
	......

	receiver         *RTPReceiver

	......
}
```
所以这个`TrackRemote`里的interceptor相关操作是在外面定义好了封进`RTPReceiver`传进来的，`TrackRemote`里面本身也不涉及interceptor相关的操作。

## 在`RTPSender`里

## 在`RTPReceiver`里