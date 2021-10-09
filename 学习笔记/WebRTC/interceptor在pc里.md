# interceptor寻踪：`pion/interceptor`在`pion/webrtc`里的用法解析

## 初始化：`NewPeerConnection`

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
显然这是给interceptor绑了一个实际进行RTCP写操作的函数`pc.writeRTCP`，这个函数显然是要负责把RTCP包发出去。返回的`RTCPWriter`被记录在了`pc.interceptorRTCPWriter`里。看看这个`pc.interceptorRTCPWriter`被调用的位置：
```go
// WriteRTCP sends a user provided RTCP packet to the connected peer. If no peer is connected the
// packet is discarded. It also runs any configured interceptors.
func (pc *PeerConnection) WriteRTCP(pkts []rtcp.Packet) error {
	_, err := pc.interceptorRTCPWriter.Write(pkts, make(interceptor.Attributes))
	return err
}
```
嗯，直接就是封装在`WriteRTCP`里，很符合直觉。看过[《pion/interceptor浅析》](./pion-interceptor.md)和[《用实例学习pion interceptor - `nack`》](./pion-nack.md)就能明白，系统需要的发送RTCP包的过程都已经封装在interceptor里了，不需要用户手动去调用，这里的`WriteRTCP`只是留给用户自定义RTCP发包过程调用的。

最后当然也有关闭的操作，在`PeerConnection.Close`里，就是在关闭`PeerConnection`时要关闭interceptor，很好理解。

## 中场休息

截至目前，我们在`NewPeerConnection`找到了一堆初始化操作，我们看到：
* `BindRTCPWriter`在`NewPeerConnection`里被调用，返回的`RTCPWriter.Write`在`PeerConnection`的`WriteRTCP`里调用，供用户发送一些自定义的RTCP包

根据[《pion/interceptor浅析》](./pion-interceptor.md)，还差`BindRTCPReader`、`BindRemoteStream`、`BindLocalStream`的相关操作没用找到。

## 准备好，要开始加速了

在PeerConnection里，与RTP包收发相关的操作当属`AddTrack`和`OnTrack`。

其中，`AddTrack`接受一个`TrackLocal`，返回一个`RTPSender`：
```go
func (pc *PeerConnection) AddTrack(track TrackLocal) (*RTPSender, error)
```
而`OnTrack`回调的输入也是一个`TrackRemote`和一个`RTPReceiver`：
```go
func (pc *PeerConnection) OnTrack(f func(*TrackRemote, *RTPReceiver))
```
一眼看去，两个函数，`AddTrack`主发，`OnTrack`主收，其输入输出参数遥相呼应。显然，他们之间必有共通之处。

顺着`AddTrack`和`OnTrack`深入一层，我们就来到了Track的领域，这里的主角是`TrackLocal`和`TrackRemote`，分别主导RTP发送和接收的过程。下面两篇文章分别从`TrackLocal`和`TrackRemote`入手，深挖interceptor在发送RTP包和接收RTP包的场景下的调用方式。在开始前，你首先需要去[《pion中的`TrackLocal`》](./TrackLocal.md)和[《pion中的`TrackRemote`》](./TrackRemote.md)里看看`TrackLocal`和`TrackRemote`是什么以及怎么用。

* [《interceptor寻踪：从`TrackLocal`开始深入挖掘`pion/interceptor`的用法》](./interceptor在tracklocal里.md)

* [《interceptor寻踪：从`TrackRemote`开始深入挖掘`pion/interceptor`的用法》](./interceptor在trackremote里.md)

* [《interceptor寻踪：总结》](./interceptor总结.md)