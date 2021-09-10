# [pion/interceptor](https://github.com/pion/interceptor)浅析

>v3.0.0 introduces a new Pion specific concept known as a interceptor. A Interceptor is a pluggable RTP/RTCP processor. Via a public API users can easily add and customize operations that are run on inbound/outbound RTP. Interceptors are an interface this means A user could provide their own implementation. Or you can use one of the interceptors Pion will have in-tree.
>
>We designed this with the following vision.
>
>* Useful defaults. Nontechnical users should be able to build things without tuning.
>* Don't block unique use cases. We shouldn't paint ourself in a corner. We want to support interesting WebRTC users
>* Allow users to bring their own logic. We should encourage easy changing. Innovation in this area is important.
>* Allow users to learn. Don't put this stuff deep in the code base. Should be easy to jump into so people can learn.
>
>In this release we only are providing a NACK Generator/Responder interceptor. We will be implementing more and you can import the latest at anytime! This means you can update your pion/interceptor version without having to upgrade pion/webrtc!

## 核心接口

```go
// Interceptor can be used to add functionality to you PeerConnections by modifying any incoming/outgoing rtp/rtcp
// packets, or sending your own packets as needed.
type Interceptor interface {
```
`Interceptor`接口是`pion/interceptor`包的核心，`pion/interceptor`包的主要功能代码是`pion/interceptor/pkg`里继承了`Interceptor`类的各种interceptor，其余的代码基本都是这个`Interceptor`的方法参数里用到的类。`pion/interceptor/pkg`里的这些interceptor都是`pion/webrtc`里要用到的。用户也可以自己定义interceptor用在`pion/webrtc`里，比如[《用实例学习pion - `rtp-forwarder`》](./rtp-forwarder.md)。

### `BindRTCPReader`方法

```go
	// BindRTCPReader lets you modify any incoming RTCP packets. It is called once per sender/receiver, however this might
	// change in the future. The returned method will be called once per packet batch.
	BindRTCPReader(reader RTCPReader) RTCPReader
```
* 从函数名上看，这是一个绑定`RTCPReader`的接口
* 从注释上看，这个接口是为了让用户自定义修改输入RTCP数据包的过程
* 从函数的输入输出上看，函数输入一个`RTCPReader`输出一个`RTCPReader`，这个方法在使用时应该是可以级联的

再看看这个`RTPReader`是什么

```go
// RTCPReader is used by Interceptor.BindRTCPReader.
type RTCPReader interface {
	// Read a batch of rtcp packets
	Read([]byte, Attributes) (int, Attributes, error)
}
```
可以看到，`RTPReader`就是一个`Read`函数，输入一段字节数据和`Attributes`，输出整型、`Attributes`和错误。

结合前面`BindRTCPReader`里说的功能，这个`Read`应该就是用户实现自定义修改输入RTCP数据包过程的地方。

输出里的错误自不必多讲，这里输入的字节数据应该就是修改前的RTCP数据包，修改过程应该是直接在这个字节输入上进行，后面输出的整型应该是修改后的数据长度，让之后的操作可以直接从字节数据里读出RTCP包。这个`Attributes`在后面定义的，是一个`map[interface{}]interface{}`，应该是用于传递一些自定义参数的。

那么这么看，`BindRTCPReader`确实是可以级联的，并且`BindRTCPReader`里面要实现的操作也能大概猜得到：
* 以`BindRTCPReader`输入的`RTPReader`构造自己的`RTPReader`作为输出，在自己的`RTPReader`的`Read`函数中：
  1. 调用`BindRTCPReader`输入的`RTPReader`的`Read`函数
  2. 根据返回的整型值，读取修改后的字节数据，反序列化为RTCP包
  3. 修改RTCP包和`Attributes`，或进行一些其他自定义操作（比如记录统计信息、转发、筛选等）
  4. 把修改后RTCP包序列化到字节数据里（可选）
  5. 返回整型值和`Attributes`

`pion/interceptor/pkg`里的几个interceptor都是这样的，不过它们都没有修改字节数据的操作。
* 比如在`pion/interceptor/pkg/nack`里，interceptor从字节数据里获取RTCP包，然后判断是不是NACK包，如果是就按照NACK里汇报的丢包情况重发RTP包
* 再比如在`pion/interceptor/pkg/report`里，interceptor从字节数据里获取RTCP包，然后判断是不是SenderReport包，如果是就存储之

于是，一级一级地调用一串interceptor的`BindRTCPReader`，每个`BindRTCPReader`都以上一个interceptor的`BindRTCPReader`返回的`RTPReader`为输入；输出的`RTPReader`的`Read`里面先调用了输入的`RTPReader`的`Read`，再进行自定义的修改操作，返回修改后的RTCP包字节数据。这样，最后一个interceptor的`BindRTCPReader`输出的`RTPReader`的`Read`就是一个顺序执行所有自定义操作的RTCP包处理函数。

### `BindRTCPWriter`方法

```go
	// BindRTCPWriter lets you modify any outgoing RTCP packets. It is called once per PeerConnection. The returned method
	// will be called once per packet batch.
	BindRTCPWriter(writer RTCPWriter) RTCPWriter
```
* 从函数名上看，这是一个绑定`RTCPWriter`的接口
* 从注释上看，这个接口是为了让用户自定义修改输出RTCP数据包的过程
* 很明显，这个方法在使用时应该和`BindRTCPReader`一样也是可以级联的，级联方式应该也大差不离

```go
// RTCPWriter is used by Interceptor.BindRTCPWriter.
type RTCPWriter interface {
	// Write a batch of rtcp packets
	Write(pkts []rtcp.Packet, attributes Attributes) (int, error)
}
```
一股子`RTCPReader`的既视感，明显也是可以级联的，要实现的操作应该也差不多：
* 以`BindRTCPWriter`输入的`RTCPWriter`构造自己的`RTCPWriter`作为输出，在`RTCPWriter`的`Write`函数里对输入的`rtcp.Packet`列表进行增减（也就是增减要发送的）

但是`pion/interceptor/pkg`里的几个interceptor好像都没这样用，它们的`BindRTCPWriter`都是直接记录下`RTPWriter`，然后开了个协程写数据：
* 比如在`pion/interceptor/pkg/nack`里是定期读取接收日志，找到有哪些缺失的包，收集序列号构造NACK包发送
* 再比如在`pion/interceptor/pkg/report`里是定期发送SenderReport包
* 又比如在`pion/interceptor/pkg/twcc`里是定期发送反馈信息

```go
	// BindLocalStream lets you modify any outgoing RTP packets. It is called once for per LocalStream. The returned method
	// will be called once per rtp packet.
	BindLocalStream(info *StreamInfo, writer RTPWriter) RTPWriter
```


```go
	// UnbindLocalStream is called when the Stream is removed. It can be used to clean up any data related to that track.
	UnbindLocalStream(info *StreamInfo)
```


```go
	// BindRemoteStream lets you modify any incoming RTP packets. It is called once for per RemoteStream. The returned method
	// will be called once per rtp packet.
	BindRemoteStream(info *StreamInfo, reader RTPReader) RTPReader
```


```go
	// UnbindRemoteStream is called when the Stream is removed. It can be used to clean up any data related to that track.
	UnbindRemoteStream(info *StreamInfo)
```


```go
	io.Closer
}
```


```go
// RTCPWriter is used by Interceptor.BindRTCPWriter.
type RTCPWriter interface {
	// Write a batch of rtcp packets
	Write(pkts []rtcp.Packet, attributes Attributes) (int, error)
}

// RTCPReader is used by Interceptor.BindRTCPReader.
type RTCPReader interface {
	// Read a batch of rtcp packets
	Read([]byte, Attributes) (int, Attributes, error)
}

// Attributes are a generic key/value store used by interceptors
type Attributes map[interface{}]interface{}

// RTPWriterFunc is an adapter for RTPWrite interface
type RTPWriterFunc func(header *rtp.Header, payload []byte, attributes Attributes) (int, error)

// RTPReaderFunc is an adapter for RTPReader interface
type RTPReaderFunc func([]byte, Attributes) (int, Attributes, error)

// RTCPWriterFunc is an adapter for RTCPWriter interface
type RTCPWriterFunc func(pkts []rtcp.Packet, attributes Attributes) (int, error)

// RTCPReaderFunc is an adapter for RTCPReader interface
type RTCPReaderFunc func([]byte, Attributes) (int, Attributes, error)

// Write a rtp packet
func (f RTPWriterFunc) Write(header *rtp.Header, payload []byte, attributes Attributes) (int, error) {
	return f(header, payload, attributes)
}

// Read a rtp packet
func (f RTPReaderFunc) Read(b []byte, a Attributes) (int, Attributes, error) {
	return f(b, a)
}

// Write a batch of rtcp packets
func (f RTCPWriterFunc) Write(pkts []rtcp.Packet, attributes Attributes) (int, error) {
	return f(pkts, attributes)
}

// Read a batch of rtcp packets
func (f RTCPReaderFunc) Read(b []byte, a Attributes) (int, Attributes, error) {
	return f(b, a)
}

// Get returns the attribute associated with key.
func (a Attributes) Get(key interface{}) interface{} {
	return a[key]
}

// Set sets the attribute associated with key to the given value.
func (a Attributes) Set(key interface{}, val interface{}) {
	a[key] = val
}
```