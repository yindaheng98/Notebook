# 【转载】WebRTC 实现的 RFC 知多少

原文：[[WebRTC架构分析] WebRTC 实现的 RFC 知多少？(RTP/RTCP/FEC 相关)](https://zhuanlan.zhihu.com/p/87879447)


## 前言

WebRTC 作为一个多媒体实时通信系统，实现了很多 RFC 标准，并且针对 WebRTC 本身也制定了相关的标准。要想对 WebRTC 做深入的了解，参考相关标准文献是必不可少的，否则一头扎进源码去分析具体实现逻辑，很难达到预期的效果。如果阅读了相关文献，从基础理论上有一个宏观上的认识，那么再去分析相关源码，你会时不时有“原来是这样”的感叹。

本人在分析源码的过程中参考一系列的 RFC 文档，计划通过几篇文章对相关 RFC 文档做一个整理、分类，并且做出简要介绍。

归纳起来，WebRFC 实现参考的 RFC 标准分如下几类：


* ICE 协议相关部分，媒体描述，offer/answer 通信过程。
* P2P 穿越相关部分，建立一对一通信链路。
* DTLS 相关部分，主要是网络传输相关标准。
* RTP/RTCP/FEC 相关部分，主要是多媒体传输相关标准。

另外，WebRTC 是作为浏览器内核发布的，对外提供的是 JavaScript 接口，所以有一套 JSEP 规范，暂且没有分析计划。

第一篇我们就介绍 RTP/RTCP/FEC 相关部分。

## RFC 1889

此协议主要是描述了 RTP real-time transport protocol。RTP 协议主要是用于多人音视频会议的应用场景下。协议内定义了 RTP、RTCP 的基本报文格式和最初的算法。

链接：[tools.ietf.org/html/rfc](https://tools.ietf.org/html/rfc1889)

## RFC 3550

RFC 3550 是在 RFC 1889 的基础上进行了改进， 对 RTP 包头，RTCP 的 SR、RR、SDES、APP、BYE 做了介绍，对 RTCP 报文的收发算法、RTT 的计算做了规定。

链接：[tools.ietf.org/html/rfc](https://tools.ietf.org/html/rfc3550)

## RFC 3551 RTP/AVP

RFC 3551 叫做 RTP Audio visual profile，是 RFC 3550 的补充，主要体现在以下几方面：

1. 对 RTP/RTCP 头没有变化。对 RTCP 数据包发送时间周期做了补充说明。
2. 对 AV codec 对应的 payload type 做了说明，历史上采用**静态 payload type** 并且和 name 绑定，后来发现 payload type 空间很有限，所以鼓励用**动态 payload type**，范围是 96-127。
3. 对音频、视频 codec 做了一个说明，这些都是比较老的格式。

链接：[tools.ietf.org/html/rfc](https://tools.ietf.org/html/rfc3551)

## rfc 4585 RTP/AVPF

RFC 4585 叫做 RTP/RTCP Audio visual profile based feedback。在 RFC 3550 和 RFC 3551 的基础上，提供了反馈消息的机制(Feedback message)。


* 提出了反馈基于三个层面的概念：

1. Transport layer feedback RTPFB 205
2. Payload-specific feedback PSFB 206
3. Application layer feedback

* 定义了反馈消息的通用格式如下。
* 定义了基于 Transport layer 的 NACK 反馈机制。
* 定义了基于 Payload-specific

1. PLI(Picture Loss Indication) 丢帧请求。
2. SLI(Slice Loss Indication) 丢片请求。
3. RPSI(Reference Picture Selection Indication) 参考帧选择指示器。

对消息在 SDP 中的属性也做了说明。属性表示是 “a=rtcp-fb:”，比如：

```
a=rtcp-fb:101 nack
a=rtcp-fb:101 nack pli
```

链接：[tools.ietf.org/html/rfc](https://tools.ietf.org/html/rfc4585)

## RFC 5104

RFC 4885 中只是定义了简单的反馈机制，比如 NACK。是适合于 P2P 通信模式，或者是小方多人会议模式。

RFC 5104 制定了更适合多人通信模式的相关反馈机制。

链接：[tools.ietf.org/html/rfc](https://tools.ietf.org/html/rfc5104)

## RFC 5285

定义 RTP 中的扩展头。其实 RFC 3550 中也提供了扩展头规范，但是只能有一种扩展头类型，不够灵活。而 RFC 5285 支持多种类型扩展头。

链接：[tools.ietf.org/html/rfc](https://tools.ietf.org/html/rfc5285)

## RFC 5761

解决 rtp 和 rtcp 共用同一个端口，数据包解复用的问题。这是通过 payload type 来解决。

链接：[tools.ietf.org/html/rfc](https://tools.ietf.org/html/rfc5761)

WebRTC 识别 RTCP 的方法，是按照 RFC 指示做的，如下：

```
// Check the RTP payload type. If 63 &lt; payload type &lt; 96, it's RTCP.
// For additional details, see http://tools.ietf.org/html/rfc5761.
bool IsRtcpPacket(const char* data, size_t len) {
  if (len &lt; 2) {
    return false;
  }
  char pt = data[1] &amp; 0x7F;
  return (63 &lt; pt) &amp;&amp; (pt &lt; 96);
}
```

## RFC 6184

此 RFC 主要是讲述了将 H.264 NALU 打包成 RTP 的规范。

webrtc 中实现了 single 模式和 STAP-A 模式。

也实现了将一个 NALU 分片的机制。

链接：[tools.ietf.org/html/rfc](https://tools.ietf.org/html/rfc6184)

## RFC 2198

链接：[tools.ietf.org/html/rfc](https://tools.ietf.org/html/rfc2198)

此 RFC 主要讲述音频冗余。通过实践来看，网络拥塞、带宽限制等都会造成网络丢包，丢包是互联网音视频通信音频差的主要原因。引入冗余可以让接收端根据冗余数据恢复丢失的数据。

音频冗余相对简单，主要思想就是：发送的时候，一个当前要发送的新包(primary packet)携带几个已经发过的包（即，冗余包），组成的一个大包；接收端，收到此包以后，可以解开，得到多个包，如果其中的一个冗余包刚好是之前发送丢掉的，那么此时马上可以恢复出来，达到冗余效果。

带冗余的音频包，RTP 头还是当前新包的头，在 RTP 头后面加入冗余包头，冗余包头后面是冗余数据。注意，primary packet 的数据是在冗余数据的后面。

冗余包头格式如下:

```
    0                   1                    2                   3
    0 1 2 3 4 5 6 7 8 9 0 1 2 3  4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
   |F|   block PT  |  timestamp offset         |   block length    |
   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

F: 一个bit，1表示后面还有冗余包，0 表示最后一个冗余头。

block PT: 冗余包 payload type

timestamp offset：相对于 primary 包的时间戳偏移量。

block length: 数据长度。

带冗余数据包的一个特点是：所有数据包的时间戳，不应该相同。因为相同毫无意义吧！

## RFC 2733

链接：[tools.ietf.org/html/rfc](https://tools.ietf.org/html/rfc2733)

此文档主要是讲 FEC ：Generic Forward Error Correction

文中讨论了 FEC 的主要解决的问题。如何通过原始媒体数据生成 FEC 数据包，FEC 数据包格式，如何通过 FEC 来恢复丢失的媒体包。

FEC 主要是采用 xor 运算来实现。

## RFC 5109

链接：[tools.ietf.org/html/rfc](https://tools.ietf.org/html/rfc5109)

此文也是讨论 FEC，理念是基于 RFC 2733，可以说是一个改进版本。

unequal error protection

Uneven Level Protection

## 后记

WebRTC rtp_rtcp 模块实现了大多数 RTP/RTCP/FEC 相关规范。所以学习本文整理的规范是进一步 rtp_rtcp 模块的基础。后续跟进情况，再决定是否更新。


