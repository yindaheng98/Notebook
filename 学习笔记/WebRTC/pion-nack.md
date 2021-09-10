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
然后构造一个NACK Interceptor。接收方是负责返回NACK消息的，所以是用`NewGeneratorInterceptor`，顾名思义，这是生成NACK消息的Interceptor。

```go
	// Create our interceptor chain with just a NACK Generator
	chain := interceptor.NewChain([]interceptor.Interceptor{generator})
```
这里的`Chain`本质是一个`Interceptor`的列表，并且自身也是`Interceptor`，它的`BindRTCPReader`、`BindRTCPWriter`等方法的实现就是依次调用其`Interceptor`的列表里的对应方法。

```go
	// Create the writer just for a single SSRC stream
	// this is a callback that is fired everytime a RTP packet is ready to be sent
	streamReader := chain.BindRemoteStream(&interceptor.StreamInfo{
		SSRC:         ssrc,
		RTCPFeedback: []interceptor.RTCPFeedback{{Type: "nack", Parameter: ""}},
	}, interceptor.RTPReaderFunc(func(b []byte, _ interceptor.Attributes) (int, interceptor.Attributes, error) { return len(b), nil, nil }))
```
这里的`Chain`本质是一个`Interceptor`的列表，并且自身也是`Interceptor`，它的`BindRTCPReader`、`BindRTCPWriter`等方法的实现就是依次调用其`Interceptor`的列表里的对应方法。

```go
	for rtcpBound, buffer := false, make([]byte, mtu); ; {
		i, addr, err := conn.ReadFrom(buffer)
		if err != nil {
			panic(err)
		}

		log.Println("Received RTP")

		if _, _, err := streamReader.Read(buffer[:i], nil); err != nil {
			panic(err)
		}

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
	}
}
```

## 发送方函数

```go
func sendRoutine() {
	// Dial our UDP listener that we create in receiveRoutine
	serverAddr, err := net.ResolveUDPAddr("udp4", fmt.Sprintf("127.0.0.1:%d", listenPort))
	if err != nil {
		panic(err)
	}

	conn, err := net.DialUDP("udp4", nil, serverAddr)
	if err != nil {
		panic(err)
	}

	// Create NACK Responder
	responder, err := nack.NewResponderInterceptor()
	if err != nil {
		panic(err)
	}

	// Create our interceptor chain with just a NACK Responder.
	chain := interceptor.NewChain([]interceptor.Interceptor{responder})

	// Set the interceptor wide RTCP Reader
	// this is a handle to send NACKs back into the interceptor.
	rtcpWriter := chain.BindRTCPReader(interceptor.RTCPReaderFunc(func(in []byte, _ interceptor.Attributes) (int, interceptor.Attributes, error) {
		return len(in), nil, nil
	}))

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
}
```