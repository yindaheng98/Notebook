# WebRTC源码中的Encoder

**学习要点：**
1. 所有的编码器都继承自`VideoEncoder`类
2. `LibvpxVp9Encoder`继承自`VP9Encoder`，`VP9Encoder`继承自`VideoEncoder`
3. `LibvpxInterface`类对libvpx里面的那些函数接口进行了封装（libvpx是C写的，没有面向对象）
4. `LibvpxVp8Encoder`和`LibvpxVp9Encoder`都是调用`LibvpxInterface`类进行的实现，而不是直接调用libvpx里面的函数接口
5. 因此，`VideoEncoder`就是沟通底层编码器和上层应用的接口。顺着`VideoEncoder`向上可以找到实现自适应编码的过程、向下可以找到自适应编码修改了编码器的哪些参数。

前置知识：先至少要知道WebRTC的一些接口标准，[《pion学习总结：等待传入track的一般流程》](传入总结.md)和[《pion学习总结：等待传入track的一般流程》](传入总结.md)可能会有所帮助。

最近在愁WebRTC MCU相关的事，需要基于WebRTC实现流处理转发单元，并且这个处理流是要处理流的内容，即把视频解码出来放进什么神经网络里处理好再编码回去。如果还要用上WebRTC的自适应码率调节机制的话

偶然在CSDN看到一篇给WebRTC用自定义编解码算法的操作：[《让 WebRTC 使用外部的音视频编解码器》](https://blog.csdn.net/foruok/article/details/70237019)，喜出望外，遂顺着这个学习一下WebRTC里的编解码器都是什么样的。

按照[《让 WebRTC 使用外部的音视频编解码器》](https://blog.csdn.net/foruok/article/details/70237019)，先从最顶层的`CreatePeerConnectionFactory`开始看：

`api/create_peerconnection_factory.h`:
```cpp
// Create a new instance of PeerConnectionFactoryInterface with optional video
// codec factories. These video factories represents all video codecs, i.e. no
// extra internal video codecs will be added.
RTC_EXPORT rtc::scoped_refptr<PeerConnectionFactoryInterface>
CreatePeerConnectionFactory(
    rtc::Thread* network_thread,
    rtc::Thread* worker_thread,
    rtc::Thread* signaling_thread,
    rtc::scoped_refptr<AudioDeviceModule> default_adm,
    rtc::scoped_refptr<AudioEncoderFactory> audio_encoder_factory,
    rtc::scoped_refptr<AudioDecoderFactory> audio_decoder_factory,
    std::unique_ptr<VideoEncoderFactory> video_encoder_factory,
    std::unique_ptr<VideoDecoderFactory> video_decoder_factory,
    rtc::scoped_refptr<AudioMixer> audio_mixer,
    rtc::scoped_refptr<AudioProcessing> audio_processing,
    AudioFrameProcessor* audio_frame_processor = nullptr);

}  // namespace webrtc
```

一看，这函数前面几个参数都是和线程有关的，后面几个参数都是音视频编解码，很好理解。