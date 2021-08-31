# 用实例学习pion - [`gocv-receive`](https://github.com/yindaheng98/example-webrtc-applications/blob/master/gocv-receive/main.go)

## 主函数

```go
func main() {
```
主函数开始

```go
	ffmpeg := exec.Command("ffmpeg", "-i", "pipe:0", "-pix_fmt", "bgr24", "-s", strconv.Itoa(frameX)+"x"+strconv.Itoa(frameY), "-f", "rawvideo", "pipe:1") //nolint
	ffmpegIn, _ := ffmpeg.StdinPipe()
	ffmpegOut, _ := ffmpeg.StdoutPipe()
	ffmpegErr, _ := ffmpeg.StderrPipe()
```
创建ffmpeg子进程，从指令上看是从stdin读入数据，然后进行解码，将处理后的原始帧输出到stdout。

```go
	if err := ffmpeg.Start(); err != nil {
		panic(err)
	}
```
启动这个子进程。

```go
	go func() {
		scanner := bufio.NewScanner(ffmpegErr)
		for scanner.Scan() {
			fmt.Println(scanner.Text())
		}
	}()
```
有任何错误都直接输出。

```go
	createWebRTCConn(ffmpegIn)
	startGoCVMotionDetect(ffmpegOut)
```
这两个是主要流程，后面介绍。

```go
}
```
主函数结束。

## `createWebRTCConn`将WebRTC输入流放到ffmpeg子进程的输入里

```go
func createWebRTCConn(ffmpegIn io.Writer) {
```
这个函数的输入是`io.Writer`，在主函数里就是把ffmpeg的子进程的stdin放了进来。

```go
	ivfWriter, err := ivfwriter.NewWith(ffmpegIn)
	if err != nil {
		panic(err)
	}
```
用ffmpeg的子进程的stdin创建了`ivfwriter`。

```go
	// Everything below is the pion-WebRTC API! Thanks for using it ❤️.

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
```
`webrtc.NewPeerConnection`创建默认配置的WebRTC标准PeerConnection。

```go
	// Set a handler for when a new remote track starts, this handler copies inbound RTP packets,
	// replaces the SSRC and sends them back
	peerConnection.OnTrack(func(track *webrtc.TrackRemote, receiver *webrtc.RTPReceiver) {
```
OnTrack用于指定被呼叫时的处理函数，在[《用实例学习pion - `rtp-forwarder`》](rtp-forwarder.md)里解读过，不用多讲。

```go
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
```
和[《用实例学习pion - `rtp-forwarder`》](rtp-forwarder.md)里一样，定时向发送端发送PLI。

```go
		fmt.Printf("Track has started, of type %d: %s \n", track.PayloadType(), track.Codec().RTPCodecCapability.MimeType)
```
打印一个启动信息。

```go
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
```
主要的流程，不断读取track里的RTP包，然后用`WriteRTP`写进`ivfwriter`，由前面的`ivfwriter`构造可以看出，`ivfwriter`里面应该是解析出RTP包里的ivf帧写进ffmpeg子进程stdin。

```go
	})
```
OnTrack处理函数结束。

```go
	// Set the handler for ICE connection state
	// This will notify you when the peer has connected/disconnected
	peerConnection.OnICEConnectionStateChange(func(connectionState webrtc.ICEConnectionState) {
		fmt.Printf("Connection State has changed %s \n", connectionState.String())
	})

	// Wait for the offer to be pasted
	offer := webrtc.SessionDescription{}
	signal.Decode(signal.MustReadStdin(), &offer)

	// Set the remote SessionDescription
	err = peerConnection.SetRemoteDescription(offer)
	if err != nil {
		panic(err)
	}

	// Create an answer
	answer, err := peerConnection.CreateAnswer(nil)
	if err != nil {
		panic(err)
	}

	// Create channel that is blocked until ICE Gathering is complete
	gatherComplete := webrtc.GatheringCompletePromise(peerConnection)

	// Sets the LocalDescription, and starts our UDP listeners
	err = peerConnection.SetLocalDescription(answer)
	if err != nil {
		panic(err)
	}

	// Block until ICE Gathering is complete, disabling trickle ICE
	// we do this because we only can exchange one signaling message
	// in a production application you should exchange ICE Candidates via OnICECandidate
	<-gatherComplete

	// Output the answer in base64 so we can paste it in browser
	fmt.Println(signal.Encode(*peerConnection.LocalDescription()))
```
最后这些操作都和[《用实例学习pion - `rtp-forwarder`》](rtp-forwarder.md)里一样。

```go
}
```
`createWebRTCConn`函数结束。

## `startGoCVMotionDetect`从ffmpeg子进程的输出里读出原始帧进行gocv

```go
// This was taken from the GoCV examples, the only change is we are taking a buffer from ffmpeg instead of webcam
// https://github.com/hybridgroup/gocv/blob/master/cmd/motion-detect/main.go
func startGoCVMotionDetect(ffmpegOut io.Reader) {
```
开头的注释告诉我们，这个函数代码是GoCV官方案例里来的，只改了输入方式。

```go
	window := gocv.NewWindow("Motion Window")
	defer window.Close() //nolint

	img := gocv.NewMat()
	defer img.Close() //nolint

	imgDelta := gocv.NewMat()
	defer imgDelta.Close() //nolint

	imgThresh := gocv.NewMat()
	defer imgThresh.Close() //nolint

	mog2 := gocv.NewBackgroundSubtractorMOG2()
	defer mog2.Close() //nolint
```
初始化一些要用到的变量。

```go
	for {
```
以下是主要的循环。

```go
		buf := make([]byte, frameSize)
		if _, err := io.ReadFull(ffmpegOut, buf); err != nil {
			fmt.Println(err)
			continue
		}
```
从ffmpeg的输出缓冲里读取帧数据，前面已经设置了每一帧的大小都是固定的，ffmpeg子进程输出的就是原始帧，所以就固定每次读取`frameSize`就好。

```go
		img, _ := gocv.NewMatFromBytes(frameY, frameX, gocv.MatTypeCV8UC3, buf)
		if img.Empty() {
			continue
		}
```
把读到的缓冲数据解析为像素矩阵。

```go
		status := "Ready"
		statusColor := color.RGBA{0, 255, 0, 0}

		// first phase of cleaning up image, obtain foreground only
		mog2.Apply(img, &imgDelta)

		// remaining cleanup of the image to use for finding contours.
		// first use threshold
		gocv.Threshold(imgDelta, &imgThresh, 25, 255, gocv.ThresholdBinary)

		// then dilate
		kernel := gocv.GetStructuringElement(gocv.MorphRect, image.Pt(3, 3))
		defer kernel.Close() //nolint
		gocv.Dilate(imgThresh, &imgThresh, kernel)

		// now find contours
		contours := gocv.FindContours(imgThresh, gocv.RetrievalExternal, gocv.ChainApproxSimple)
```
主要的CV操作。

```go
		for i := 0; i < contours.Size(); i++ {
			area := gocv.ContourArea(contours.At(i))
			if area < minimumArea {
				continue
			}

			status = "Motion detected"
			statusColor = color.RGBA{255, 0, 0, 0}
			gocv.DrawContours(&img, contours, i, statusColor, 2)

			rect := gocv.BoundingRect(contours.At(i))
			gocv.Rectangle(&img, rect, color.RGBA{0, 0, 255, 0}, 2)
		}
```
画框框。

```go
		gocv.PutText(&img, status, image.Pt(10, 20), gocv.FontHersheyPlain, 1.2, statusColor, 2)
```
画文字。

```go
		window.IMShow(img)
		if window.WaitKey(1) == 27 {
			break
		}
```
窗口上显示结果。

```go
	}

}
```
结束。