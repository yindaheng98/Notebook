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

总的来说，在WebRTC通信中，要收发的包可以分为两类：
* RTP包：传递媒体内容，如音视频片段等
* RTCP包：传递控制信息，如NACK、接收方报告等

`pion/interceptor`也是按照这样的思路进行实现的，其中实现的interceptor都同时要处理RTP包和RTCP包，处理方式基本上都是根据RTP包的收发情况构造RTCP包进行发送，以及根据收到的RTCP包调整RTP包的发送。
* 比如在`pion/interceptor/pkg/nack`里，就是一方根据RTP包的序列号发送NACK信息，另一方根据NACK信息重发RTP包
* 又比如在`pion/interceptor/pkg/nack`里，就是一方统计RTP包的接收情况发送SenderReport，另一方接收并存储之
* 再比如在`pion/interceptor/pkg/twcc`里，就是一方统计RTP包的接收情况反馈丢包信息，另一方接收然后调整RTP包发送窗口

此外，从逻辑上讲，一个协议必须得收发双方都实现了才能正常运行，而且由于WebRTC是对等连接通信，一方可能既是接收方又是发送方，所以`pion/interceptor`里的interceptor都必须得实现收发双方的功能。在程序里，这个思想体现为收发方基础接口不是分`SenderInterceptor`和`ReceiverInterceptor`两个，而是在一个基础接口`Interceptor`中同时包含收发双方的方法。

本文可以和[《用实例学习pion interceptor - `nack`》](./pion-nack.md)搭配着看。

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

再看看这个`RTCPReader`是什么

```go
// RTCPReader is used by Interceptor.BindRTCPReader.
type RTCPReader interface {
	// Read a batch of rtcp packets
	Read([]byte, Attributes) (int, Attributes, error)
}
```
可以看到，`RTCPReader`就是一个`Read`函数，输入一段字节数据和`Attributes`，输出整型、`Attributes`和错误。

结合前面`BindRTCPReader`里说的功能，这个`Read`应该就是用户实现自定义修改输入RTCP数据包过程的地方。

输出里的错误自不必多讲，这里输入的字节数据应该就是修改前的RTCP数据包，修改过程应该是直接在这个字节输入上进行，后面输出的整型应该是修改后的数据长度，让之后的操作可以直接从字节数据里读出RTCP包。这个`Attributes`在后面定义的，是一个`map[interface{}]interface{}`，应该是用于传递一些自定义参数的。

`pion/interceptor`里还提供了一种简便的构造`RTCPReader`的方式：

```go
// RTCPReaderFunc is an adapter for RTCPReader interface
type RTCPReaderFunc func([]byte, Attributes) (int, Attributes, error)

// Read a batch of rtcp packets
func (f RTCPReaderFunc) Read(b []byte, a Attributes) (int, Attributes, error) {
	return f(b, a)
}
```

这样，只要写好`Read`里的代码，可以不用再定义一个`RTCPReader`子类，直接把函数放进`RTCPReaderFunc`就行。函数式编程思想，很妙。`pion/interceptor/pkg`里的几个interceptor都是这么用的。

那么这么看，`BindRTCPReader`确实是可以级联的，并且`BindRTCPReader`里面要实现的操作也能大概猜得到：
* 以`BindRTCPReader`输入的`RTCPReader`构造自己的`RTCPReader`作为输出，在自己的`RTCPReader`的`Read`函数中：
  1. 调用`BindRTCPReader`输入的`RTCPReader`的`Read`函数
  2. 根据返回的整型值，读取修改后的字节数据，反序列化为RTCP包
  3. 修改RTCP包和`Attributes`，或进行一些其他自定义操作（比如记录统计信息、转发、筛选等）
  4. 把修改后RTCP包序列化到字节数据里（可选）
  5. 返回整型值和`Attributes`

`pion/interceptor/pkg`里的几个interceptor都是这样的，不过它们都没有修改字节数据的操作。
* 比如在`pion/interceptor/pkg/nack`里，interceptor从字节数据里获取RTCP包，然后判断是不是NACK包，如果是就按照NACK里汇报的丢包情况重发RTCP包
* 再比如在`pion/interceptor/pkg/report`里，interceptor从字节数据里获取RTCP包，然后判断是不是SenderReport包，如果是就存储之

于是，一级一级地调用一串interceptor的`BindRTCPReader`，每个`BindRTCPReader`都以上一个interceptor的`BindRTCPReader`返回的`RTCPReader`为输入；输出的`RTCPReader`的`Read`里面先调用了输入的`RTCPReader`的`Read`，再进行自定义的修改操作，返回修改后的RTCP包字节数据。这样，最后一个interceptor的`BindRTCPReader`输出的`RTCPReader`的`Read`就是一个顺序执行所有自定义操作的RTCP包处理函数。

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

但是`pion/interceptor/pkg`里的几个interceptor好像都没这样用，它们的`BindRTCPWriter`都是直接记录下`RTCPWriter`，然后开了个协程写数据：
* 比如在`pion/interceptor/pkg/nack`里是定期读取接收日志，找到有哪些缺失的包，收集序列号构造NACK包发送
* 再比如在`pion/interceptor/pkg/report`里是定期发送SenderReport包
* 又比如在`pion/interceptor/pkg/twcc`里是定期发送反馈信息

`pion/interceptor`里也提供了一种简便的构造`RTCPWriter`的方式：

```go
// RTCPWriterFunc is an adapter for RTCPWriter interface
type RTCPWriterFunc func(pkts []rtcp.Packet, attributes Attributes) (int, error)

// Write a batch of rtcp packets
func (f RTCPWriterFunc) Write(pkts []rtcp.Packet, attributes Attributes) (int, error) {
	return f(pkts, attributes)
}
```

和`RTCPReaderFunc`一个道理，不必多讲。

## `BindRemoteStream`方法

```go
	// BindRemoteStream lets you modify any incoming RTP packets. It is called once for per RemoteStream. The returned method
	// will be called once per rtp packet.
	BindRemoteStream(info *StreamInfo, reader RTPReader) RTPReader

	// UnbindRemoteStream is called when the Stream is removed. It can be used to clean up any data related to that track.
	UnbindRemoteStream(info *StreamInfo)
```
绑定和解绑远端流，从方法和注释上看和`BindRTCPReader`是类似的，都是用来绑定处理发送出去的数据包的方法的。

这里绑定的`RTPReader`和`RTCPReader`的`Read`函数里的输入参数是一模一样的：

```go
// RTPReader is used by Interceptor.BindRemoteStream.
type RTPReader interface {
	// Read a rtp packet
	Read([]byte, Attributes) (int, Attributes, error)
}
```

`BindRemoteStream`和`BindRTCPReader`唯一的区别在于包处理的方式：RTP包和RTCP包在逻辑上的不同之处在于，RTP包是从属于一个流的连续序列，而RTCP包是一个个独立的包。因此在`BindRTCPReader`中，输入的数据直接就是一个`RTCPReader`；而`BindRemoteStream`不仅需要指定`RTPReader`，还需要指定一个存储流信息的`StreamInfo`。

这个`StreamInfo`长这样：

```go
// StreamInfo is the Context passed when a StreamLocal or StreamRemote has been Binded or Unbinded
type StreamInfo struct {
	ID                  string
	Attributes          Attributes
	SSRC                uint32
	PayloadType         uint8
	RTPHeaderExtensions []RTPHeaderExtension
	MimeType            string
	ClockRate           uint32
	Channels            uint16
	SDPFmtpLine         string
	RTCPFeedback        []RTCPFeedback
}

// RTPHeaderExtension represents a negotiated RFC5285 RTP header extension.
type RTPHeaderExtension struct {
	URI string
	ID  int
}

// RTCPFeedback signals the connection to use additional RTCP packet types.
// https://draft.ortc.org/#dom-rtcrtcpfeedback
type RTCPFeedback struct {
	// Type is the type of feedback.
	// see: https://draft.ortc.org/#dom-rtcrtcpfeedback
	// valid: ack, ccm, nack, goog-remb, transport-cc
	Type string

	// The parameter value depends on the type.
	// For example, type="nack" parameter="pli" will send Picture Loss Indicator packets.
	Parameter string
}
```

可以看到，这个`StreamInfo`里面放的是一些与流有关的配置信息。由于RTP包承载的是流，流中的包可以看成是一个整体，是一系列相互关联的连续包，不像RTCP包那样是一个个独立的包，一会是NACK、一会又是SenderReport。`StreamInfo`就是这些连续RTP包中与流相关的标记信息，它可以用来区分RTP包属于哪个流、区分媒体的类型、记录时钟频率等等。

一些重要的`StreamInfo`参数：引自《RTP: audio and video for the Internet》
* `SSRC`：Synchronization Source（SSRC）标识RTP会话中的参与者。它是一个临时的，每个会话的标识符通过RTP控制协议映射到一个长期的规范名称CNAME。SSRC是一个32位整数，由参与者加入会话时随机选择。具有相同SSRC的所有数据包均构成单个时序和序列号空间的一部分，因此接收方必须按SSRC对数据包进行分组才能进行播放。如果参加者在一个RTP会话中生成多个流（例如，来自不同的摄像机），每个流都必须标识为不同的SSRC，以便接收方可以区分哪些数据包属于每个流。
* `PayloadType`：有效负载类型。RTP头的负载类型（或者PT）与RTP传输的媒体数据关联。接收者应用检测负载类型来甄别如何处理数据，例如，传递给特定的解压缩器。
* `MimeType`：有效负载格式。有效负载格式是根据MIME名称空间命名的。该名称空间最初是为电子邮件定义的，用于标识附件的内容，但此后它已成为媒体格式的通用名称空间，并在许多应用程序中使用。所有有效负载格式都应该具有MIME类型注册。更新的有效负载格式将其包含在其各自规范中； 在线维护MIME类型的完整列表，网址为：[http://www.iana.org/assignments/media-types](http://www.iana.org/assignments/media-types)。

还有其他的一些参数在RTP包相关的IETF标准里都应该能找到。

`pion/interceptor`里也提供了一种简便的构造`RTPReader`的方式：

```go
// RTPReaderFunc is an adapter for RTPReader interface
type RTPReaderFunc func([]byte, Attributes) (int, Attributes, error)

// Read a rtp packet
func (f RTPReaderFunc) Read(b []byte, a Attributes) (int, Attributes, error) {
	return f(b, a)
}
```

和`RTCPReaderFunc`一个道理，不必多讲。


## `BindLocalStream`方法

```go
	// BindLocalStream lets you modify any outgoing RTP packets. It is called once for per LocalStream. The returned method
	// will be called once per rtp packet.
	BindLocalStream(info *StreamInfo, writer RTPWriter) RTPWriter

	// UnbindLocalStream is called when the Stream is removed. It can be used to clean up any data related to that track.
	UnbindLocalStream(info *StreamInfo)
```
绑定和解绑本地流，从方法和注释上看和`BindRTCPWriter`是类似的，都是用来绑定处理发送出去的数据包的方法的，这里绑定的`RTPWriter`也和`RTCPWriter`大差不离：

```go
// RTPWriter is used by Interceptor.BindLocalStream.
type RTPWriter interface {
	// Write a rtp packet
	Write(header *rtp.Header, payload []byte, attributes Attributes) (int, error)
}
```

可以看到，唯一的区别在于包构建方式：从代码上看，`RTCPWriter.Write`的输入直接就是`rtcp.Packet`的列表；而`RTPWriter.Write`的输入是分开的一个包头`rtp.Header`和`[]byte`格式的内容。这可能是因为RTCP只会传递一些运行状态数据和控制信息，每种包都有自己独特的结构，而RTP包是由一长串媒体数据切开包装而来，结构比较规整，不能给用户随便调整，所以把包头和包内容分了两个变量。

`pion/interceptor`里也提供了一种简便的构造`RTPWriter`的方式：

```go
// RTPWriterFunc is an adapter for RTPWrite interface
type RTPWriterFunc func(header *rtp.Header, payload []byte, attributes Attributes) (int, error)

// Write a rtp packet
func (f RTPWriterFunc) Write(header *rtp.Header, payload []byte, attributes Attributes) (int, error) {
	return f(header, payload, attributes)
}
```

和`RTCPReaderFunc`一个道理，不必多讲。

## `Close`

```go
	io.Closer
}
```
这里好理解，当要销毁这个`Interceptor`的时候，必须要解绑所有的`RTCPReader`、`RTCPWriter`、`RTPReader`、`RTPWriter`，并且停止所有的相关协程，这个只能由实现`Interceptor`的用户来做。所以在这里加上了一个`io.Closer`，要求用户自己实现一个`Close`方法。

## 来自[《用实例学习pion interceptor - `nack`》](./pion-nack.md)的附加知识

以下是一些关于级联和`Interceptor`具体如何调用的知识。[《用实例学习pion interceptor - `nack`》](./pion-nack.md)里的案例很是简洁，一看就能懂。

从[《用实例学习pion interceptor - `nack`》](./pion-nack.md)中的案例看:
* 在级联的开头，用户需要自行调用`Read`把包传进级联的Reader里
* 在级联的末尾，用户需要自行在`Write`里写上发送包的函数，把级联的Writer传来的包发送出去

比如NACK发送方接收RTP包就是首先获取到`RTPReader`：
```go
// Create the writer just for a single SSRC stream
// this is a callback that is fired everytime a RTP packet is ready to be sent
streamReader := chain.BindRemoteStream(&interceptor.StreamInfo{
	SSRC:         ssrc,
	RTCPFeedback: []interceptor.RTCPFeedback{{Type: "nack", Parameter: ""}},
}, interceptor.RTPReaderFunc(func(b []byte, _ interceptor.Attributes) (int, interceptor.Attributes, error) { return len(b), nil, nil }))
```

然后在循环里从UDP处收包之后放进`RTPReader.Read`：
```go
i, addr, err := conn.ReadFrom(buffer)
if err != nil {
	panic(err)
}

log.Println("Received RTP")

if _, _, err := streamReader.Read(buffer[:i], nil); err != nil {
	panic(err)
}
```
由于级联了NACK Interceptor，所以就能执行一些包统计的操作，找出未接收到的RTP包，构造NACK。

然后NACK发送方发NACK包就是写在`RTCPWriter.Write`里的：
```go
chain.BindRTCPWriter(interceptor.RTCPWriterFunc(func(pkts []rtcp.Packet, _ interceptor.Attributes) (int, error) {
	buf, err := rtcp.Marshal(pkts)
	if err != nil {
		return 0, err
	}

	return conn.WriteTo(buf, addr)
}))
```
这样就能完成“收RTP包——统计丢包——发NACK”的操作。

NACK接收方也是一样，先获取`RTCPReader`：
```go
// Set the interceptor wide RTCP Reader
// this is a handle to send NACKs back into the interceptor.
rtcpReader := chain.BindRTCPReader(interceptor.RTCPReaderFunc(func(in []byte, _ interceptor.Attributes) (int, interceptor.Attributes, error) {
	return len(in), nil, nil
}))
```

然后也是在循环里UDP收包之后放进`RTCPReader.Read`：
```go
i, err := conn.Read(rtcpBuf)
if err != nil {
	panic(err)
}

log.Println("Received NACK")

if _, _, err = rtcpWriter.Read(rtcpBuf[:i], nil); err != nil {
	panic(err)
}
```
于是获取到NACK解包出来就知道要重发哪些RTP包了。

然后NACK接收方重发RTP包就是写在`RTPWriter.Write`里的：
```go
// Create the writer just for a single SSRC stream
// this is a callback that is fired everytime a RTP packet is ready to be sent
streamWriter := chain.BindLocalStream(&interceptor.StreamInfo{
	SSRC:         ssrc,
	RTCPFeedback: []interceptor.RTCPFeedback{{Type: "nack", Parameter: ""}},
}, interceptor.RTPWriterFunc(func(header *rtp.Header, payload []byte, attributes interceptor.Attributes) (int, error) {
	headerBuf, err := header.Marshal()
	if err != nil {
		panic(err)
	}

	return conn.Write(append(headerBuf, payload...))
}))
```
这样就能完成“接收NACK包——找出需要重发的RTP包——重发RTP包”的操作了。

## 总结一下`Interceptor`的创建过程

首先是继承`interceptor.NoOp`，因为实际情况下不一定需要把`BindLocalStream`、`BindRTCPWriter`、`BindRemoteStream`、`BindRTCPReader`四个全实现。

### 实现`BindLocalStream`

1. 实现一个`RTPWriter`，在其中保存另一个`RTPWriter`，并在其`Write`函数中调用保存的`RTPWriter.Write`
2. 实现`BindLocalStream`，将输入的`RTPWriter`保存到你实现的`RTPWriter`中并返回
3. (常见操作)让你实现的`RTPWriter`可以读到`Interceptor`里的数据，然后在`BindLocalStream`里开goroutine调用`RTPWriter`定期获取`Interceptor`里的数据并据此调用保存的另一个`RTPWriter`写一些特殊功能的包

### 实现`BindRTCPWriter`

同上，只不过`RTPWriter`变成`RTCPWriter`

### 实现`BindRemoteStream`

1. 实现一个`RTPReader`，在其中保存另一个`RTPReader`，并在其`Read`函数中调用保存的`RTPReader.Read`
2. 实现`BindRemoteStream`，将输入的`RTPReader`保存到你实现的`RTPReader`中并返回
3. (常见操作)让`RTPReader`可以操作`Interceptor`里的数据，从而可以根据`RTPReader.Read`输入的数据修改`Interceptor`里的数据，进而影响其绑定的`RTPWriter`和`RTCPWriter`的行为

### 实现`BindRTCPReader`

同上，只不过`RTPReader`变成`RTCPReader`