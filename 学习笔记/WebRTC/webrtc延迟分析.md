# 【纯转载】webrtc延迟分析

原文[《webrtc延迟分析》](https://blog.csdn.net/zhuiyuanqingya/article/details/81080228)

## 1.背景 

webrtc可以用于将一台机器上的桌面投射到另外一台机器上，通过浏览器实现桌面分享。然而其延迟时间是一个关键问题，即我们在源桌面上做一个操作，经过多长时间能够在目的桌面上看到。接下来，将根据查找到的资料对导致延迟的因素做简要介绍。 

想了解更多可以查看：[抖动和延迟之间的区别](http://webrtc.org.cn/jitter-latency/)

 
## 2.延迟时间构成 

以下是计算WebRTC呼叫总等待时间的关键因素： 

1. 网络延时。取决于网络连接质量和通信距离（在一个国家内部应该小于50毫秒，国家之间可能大于100毫秒）。 
1. 网络带宽和服务质量。丢包或者带宽不足可能触发更多的延时。 
1. 声音延迟。取决于操作系统、音频硬件和驱动（在windows和ios上小于20毫秒，在android和linux上可能更多）。 
1. 抖动缓冲。每种VoIP软件维持一个大小不一的抖动缓冲器，以补偿网络延迟（通常在0到100毫秒）。 
1. 回声消除和前向纠错。回声消除和前向纠错可能引入一个数据包的延迟（通常在20毫秒）。 
1. 其他因素。还有其他因素对延迟有影响，例如CPU占用率过高以及软件实现细节等。 

 如果通话双方在一个国家内部，总的延迟应当小于300毫秒，如果通过webrtc打长距离的跨国电话，总的延迟可能高达600毫秒。 

 翻译自：[What’s the average latency of a WebRTC audio in a chat?](https://www.quora.com/Whats-the-average-latency-of-a-WebRTC-audio-in-a-chat)

 
## 3.延迟影响因素分析 

参考网上的[资料](https://stackoverflow.com/questions/21407043/webrtc-remove-reduce-latency-between-devices-that-are-sharing-their-videos-str)，影响webrtc延迟的因素有以下几种： 

1. ICE（Interactive Connectivity Establishment）延迟 
1. 加密延迟 
1. 使用转发服务器 
1. 管道的大小 
1. TCP 
1. 大规模网络问题 
1. 网络拥堵 

分析延迟目前限制于局域网，所以只考虑（1）和（2），其他因素暂时不考虑。

 
## 4.ICE延迟 

ICE延迟的原因是ICE协议栈在收集地址到探测协商过程花费很长时间，这在VOIP里是不可容忍的，有人把ICE功能关掉，这样解决了延迟问题，但是NAT穿越失败，媒体必须走服务器，这在一些webrtc与sip系统互通的系统中有应用价值，但在在两个webrtc客户端之间的呼叫不用ICE则失去了webrtc的价值。 

有人提出一种trical-ice的方案，思路是客户端一边收集candidate一边发送给对端而不是收集后再发送，实际上客户端收集的一些无效candidate，比如多网卡的情况，如果能够在浏览器引擎的ICE部分直接忽略这部分，收集和探测的时间都会大大减少。 

 

参考：[WebRTC 客户端ICE 延迟问题](https://blog.csdn.net/voipmaker/article/details/28989613) 

 [Delay in Starting a Stream - WebRTC](https://red5pro.zendesk.com/hc/en-us/articles/115001422648-Delay-in-Starting-a-Stream-WebRTC)

 
## 5.加密延时 

加密算法一般都是比较耗时的，webrtc在传输过程中采用的加密算法对传输延迟究竟有多大的影响，如果取消传输加密是否能够有效的改善延迟情况，。

 
### 5.1加密对传输延迟的影响 

网上能够找到的关于加密对webrtc传输延迟影响的文章非常少，只能从中找到只言片语，“视频加密在发送端和接收端进行加解密视频数据，密钥由视频双方协商，代价是会影响视频数据处理的性能；也可以不使用视频加密功能，这样在性能上会好些。”，更具体的就看不到了。

 

参考：[WEBRTC](https://blog.csdn.net/swt198852/article/details/8138704) 

从这里可以看出，加密对视频延迟有影响，但是影响有多大并没有具体说明。

 
### 5.2取消加密对传输延迟的影响 
### 5.2.1取消加密是否可行 

看了网上较多的文章，得出结论是webrtc的加密是[强制性的](https://www.html5rocks.com/en/tutorials/webrtc/basics/#toc-security)。 

[Why is WebRTC encrypted?](https://stackoverflow.com/questions/35070473/why-is-webrtc-encrypted)

 

但是也看到了有人说在测试模式下，可以[取消webrtc的加密功能](https://stackoverflow.com/questions/23624382/can-i-turn-off-srtp-when-use-webrtc)，开关选项如下：

 
```
–disable-webrtc-encryption 
``` 

测试模式最好选用Chromium或者Chrome的 Canary版本。 

版本相关信息可以参考：[Chrome浏览器各个版本区别及离线安装包下载](https://blog.csdn.net/u012195214/article/details/78575387)

 

只需要使用命令行-disable-webrtc-encryption启动浏览器，然后就应该看到弹出了一个警示框提示你正在使用一个未受支持的命令行flag。需要注意的是通话的双方都需要在未加密的状态下进行通话；如果不这样做通话就会连接失败。

 

**小结：取消加密在webrtc中是不可行的，但是测试模式下有选项可以尝试一下。**

 
### 5.2.2取消加密对传输的影响 

采用chromium浏览器，根据主机和虚拟机之间的测试，取消加密对传输延迟[影响甚微](https://groups.google.com/forum/#!topic/mozilla.dev.media/tkxeLEnBK9M)。 

**测试方法** 

具体测试方法是：虚拟机打开chromium浏览器，启动一个在线计时器（毫秒级别），发送虚拟机的桌面给主机，主机这边打开chromium浏览器，接收虚拟机发送的桌面，采用windows的print screen截屏（双屏幕，主机和虚拟机各用一个屏幕），比较两边的秒表时差，即可测得延时数据。

 

**测试结果** 

连接建立过程（从一边发起连接，到另一边开始接收到数据）是导致延迟的关键，大概会导致数秒的延时； 

如果屏幕除了秒表之外完全静止，在暂停秒表使两边同步，消除建立连接导致的延时，重新启动秒表最好的情况大概会有两三百毫秒的延时； 

如果屏幕上一直在浏览网页滑动鼠标，延时会很严重，大概会有四五秒的延时，观察的时间很短，如果延长观察时间，估计会延时会累积； 

加密造成的延时按照目前这种粗略的测试方法基本观察不到，相对于建立连接以及屏幕频繁刷新导致的延时，加密的延时基本可以忽略。

 

**小结：取消加密的功能仅仅用于测试模式，抓取未经加密的RTP数据包，对于传输延迟基本没有影响。**

