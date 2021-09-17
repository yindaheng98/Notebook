# 用实例学习pion interceptor - [`nack`](https://github.com/pion/interceptor/blob/master/examples/nack/main.go)

```go
const (
	listenPort = 6420
	mtu        = 1500
	ssrc       = 5000
)

func main() {
	go sendRoutine()
	receiveRoutine()
}
```
示例的开头，是一些配置文件和主函数。从主函数里看，这个例程是一收一发两个协程，其中接收是在主协程里。

接下来分别解析发送和接收方协程函数。

## 接收方函数

```go
func receiveRoutine() {
```
以下是接收方函数体

```go
	serverAddr, err := net.ResolveUDPAddr("udp4", fmt.Sprintf("127.0.0.1:%d", listenPort))
	if err != nil {
		panic(err)
	}

	conn, err := net.ListenUDP("udp4", serverAddr)
	if err != nil {
		panic(err)
	}
```
首先当然是开UDP监听端口

```go
	// Create NACK Generator
	generator, err := nack.NewGeneratorInterceptor()
	if err != nil {
		panic(err)
	}
```
然后构造一个NACK Interceptor。接收方是负责生成NACK消息的，所以是用`NewGeneratorInterceptor`，顾名思义，这是生成NACK消息的Interceptor。从源码上看，这个`NewGeneratorInterceptor`返回的是`GeneratorInterceptor`，只实现了`BindRTCPWriter`和`BindRemoteStream`；从逻辑上想也确实应该是这样，因为接收方生成NACK消息只需要知道接收到的RTP包序列（用`BindRemoteStream`实现的功能）然后发送NACK包（用`BindRTCPWriter`实现的功能）就行了。

```go
	// Create our interceptor chain with just a NACK Generator
	chain := interceptor.NewChain([]interceptor.Interceptor{generator})
```
这里的`Chain`本质是一个`Interceptor`的列表，并且自身也是`Interceptor`，它的`BindRTCPReader`、`BindRTCPWriter`等方法的实现就是依次调用其`Interceptor`的列表里的对应方法。这种级联思想的解释可以看[《pion/interceptor浅析》](./pion-interceptor.md)。

```go
	// Create the writer just for a single SSRC stream
	// this is a callback that is fired everytime a RTP packet is ready to be sent
	streamReader := chain.BindRemoteStream(&interceptor.StreamInfo{
		SSRC:         ssrc,
		RTCPFeedback: []interceptor.RTCPFeedback{{Type: "nack", Parameter: ""}},
	}, interceptor.RTPReaderFunc(func(b []byte, _ interceptor.Attributes) (int, interceptor.Attributes, error) { return len(b), nil, nil }))
```
这里绑定了一个处理远端输入流的处理函数，看样子是直接返回缓冲区大小，不进行任何操作。

这个绑定操作主要是为了获取到这个`streamReader`变量。这个`streamReader`是一个嵌套了NACK相关操作的`RTPReader`，NACK的相关操作要调用`streamReader.Read`才能触发（具体为什么是这样可以看[《pion/interceptor浅析》](./pion-interceptor.md)里关于`RTCPReader`嵌套操作的介绍）。

```go
	for rtcpBound, buffer := false, make([]byte, mtu); ; {
```

这里一个死循环，在循环中不断进行数据的处理。

```go
		i, addr, err := conn.ReadFrom(buffer)
		if err != nil {
			panic(err)
		}

		log.Println("Received RTP")
```

首先当然是读取到UDP里发来的RTP包。

```go
		if _, _, err := streamReader.Read(buffer[:i], nil); err != nil {
			panic(err)
		}
```

然后把包输入到`streamReader.Read`中，正如前文所述，`streamReader`是一个嵌套了NACK相关操作的`RTPReader`，调用`streamReader.Read`就会触发NACK Interceptor里的相关操作。

```go
		// Set the interceptor wide RTCP Writer
		// this is a callback that is fired everytime a RTCP packet is ready to be sent
		if !rtcpBound {
			chain.BindRTCPWriter(interceptor.RTCPWriterFunc(func(pkts []rtcp.Packet, _ interceptor.Attributes) (int, error) {
				buf, err := rtcp.Marshal(pkts)
				if err != nil {
					return 0, err
				}

				return conn.WriteTo(buf, addr)
			}))

			rtcpBound = true
		}
```

接下来这是一个绑定`RTCPWriter`的操作，按理说这种绑定操作应该是放在读取数据的这个循环外面，这样就不需要这个`rtcpBound`来防止重复操作了。从代码上看，这里绑定的`RTCPWriter`是把输入的RTCP包发送出去。结合NACK Interceptor里面的包发送逻辑，这个操作会定期被调用，也就是定期发送NACK包。所以这个操作被放在循环里面的原因应该是防止这个NACK的定时发送在对面还没有就绪的时候就发包。

```go
	}
}
```

结束。

## 发送方函数

```go
func sendRoutine() {
```
以下是发送方函数体

```go
	// Dial our UDP listener that we create in receiveRoutine
	serverAddr, err := net.ResolveUDPAddr("udp4", fmt.Sprintf("127.0.0.1:%d", listenPort))
	if err != nil {
		panic(err)
	}

	conn, err := net.DialUDP("udp4", nil, serverAddr)
	if err != nil {
		panic(err)
	}
```
接收方那边是监听UDP端口，发送方这边就是连接接收方那边开的端口。

```go
	// Create NACK Responder
	responder, err := nack.NewResponderInterceptor()
	if err != nil {
		panic(err)
	}

	// Create our interceptor chain with just a NACK Responder.
	chain := interceptor.NewChain([]interceptor.Interceptor{responder})
```
然后当然也有和初始化`Interceptor`的操作。这个`NewResponderInterceptor`返回的是一个`ResponderInterceptor`。和`NewGeneratorInterceptor`差不多的道理，NACK接收方进行的操作是接收NACK包然后重发所需的RTP包，所以这个`ResponderInterceptor`里面只实现了`BindRTCPReader`（接收NACK包）和`BindLocalStream`（重发所需的RTP包）。

```go
	// Set the interceptor wide RTCP Reader
	// this is a handle to send NACKs back into the interceptor.
	rtcpWriter := chain.BindRTCPReader(interceptor.RTCPReaderFunc(func(in []byte, _ interceptor.Attributes) (int, interceptor.Attributes, error) {
		return len(in), nil, nil
	}))
```
如果按照这个变量的含义来说，这个变量名应该是写错了，这里明显生成的是一个`RTCPReader`，变量名应该是`rtcpReader`。

这个也是绑定了一个什么都不做的操作，主要也是为了获取`RTCPReader`。后面会调用`RTCPReader.Read`读取RTCP包并触发NACK相关操作。和前面接收方函数差不多，只不过接收方函数里Read的是RTP包，发送方这里Read的是RTCP包。

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
这边`BindLocalStream`绑定了一个发送用的流，流里的操作就是把上层传过来的需要发送的数据通过UDP连接发送出去。

```go
	// Read RTCP packets sent by receiver and pass into Interceptor
	go func() {
		for rtcpBuf := make([]byte, mtu); ; {
			i, err := conn.Read(rtcpBuf)
			if err != nil {
				panic(err)
			}

			log.Println("Received NACK")

			if _, _, err = rtcpWriter.Read(rtcpBuf[:i], nil); err != nil {
				panic(err)
			}
		}
	}()
```
然后开了个协程从UDP里面读数据，交给前面生成的`RTCPReader`去进行读取操作。因为这边是一个发送方，所以这边收到的只会是NACK包。

```go
	for sequenceNumber := uint16(0); ; sequenceNumber++ {
		// Send a RTP packet with a Payload of 0x0, 0x1, 0x2
		if _, err := streamWriter.Write(&rtp.Header{
			Version:        2,
			SSRC:           ssrc,
			SequenceNumber: sequenceNumber,
		}, []byte{0x0, 0x1, 0x2}, nil); err != nil {
			fmt.Println(err)
		}

		time.Sleep(time.Millisecond * 200)
	}
```
最后这里就是发送随机测试数据，不用多讲。

```go
}
```
结束。