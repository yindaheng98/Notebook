# 【转载】P2P通信标准协议(三)之ICE

原文：[《P2P通信标准协议(三)之ICE》](https://evilpan.com/2015/12/20/p2p-standard-protocol-ice/)

在[P2P通信标准协议(二)](https://evilpan.com/2015/12/15/p2p-standard-protocol-turn/)中,介绍了TURN的基本交互流程,在上篇结束部分也有说到,TURN作为STUN协议的一个拓展,保持了STUN的工具性质,而不作为完整的NAT传输解决方案,只提供穿透NAT的功能, 并且由具体的应用程序来使用.虽然TURN也可以独立工作,但其本身就是被设计为[ICE/RFC5245](http://www.rfc-editor.org/info/rfc5245)的一部分,本章就来介绍一下ICE协议的具体内容.

## SDP

ICE信息的描述格式通常采用标准的[SDP](http://www.rfc-editor.org/info/rfc4566),其全称为Session Description Protocol,即会话描述协议.SDP只是一种信息格式的描述标准,不属于传输协议,但是可以被其他传输协议用来交换必要的信息,如SIP和RTSP等.

### SDP信息

一个SDP会话描述包含如下部分:


* 会话名称和会话目的
* 会话的激活时间
* 构成会话的媒体(media)
* 为了接收该媒体所需要的信息(如地址,端口,格式等)


因为在中途参与会话也许会受限制,所以可能会需要一些额外的信息:


* 会话使用的的带宽信息
* 会话拥有者的联系信息


一般来说,SDP必须包含充分的信息使得应用程序能够加入会话,并且可以提供任何非参与者使用时需要知道的资源状况,后者在当SDP同时用于多个会话声明协议时尤其有用.

### SDP格式

SDP是基于文本的协议,使用ISO 10646字符集和UTF-8编码.SDP字段名称和属性名称只使用UTF-8的一个子集US-ASCII,因此不能存在中文.虽然理论上文本字段和属性字段支持全集,但最好还是不要在其中使用中文.


SDP会话描述包含了多行如下类型的文本:

```
type=value
```

其中type是大小写敏感的,其中一些行是必须要有的,有些是可选的,所有元素都必须以固定顺序给出.固定的顺序极大改善了错误检测,同时使得处理端设计更加简单.如下所示,其中可选的元素标记为`*`:

```
    会话描述:
         v=  (protocol version)
         o=  (originator and session identifier)
         s=  (session name)
         i=* (session information)
         u=* (URI of description)
         e=* (email address)
         p=* (phone number)
         c=* (connection information -- not required if included in
              all media)
         b=* (zero or more bandwidth information lines)
         One or more time descriptions (&quot;t=&quot; and &quot;r=&quot; lines; see below)
         z=* (time zone adjustments)
         k=* (encryption key)
         a=* (zero or more session attribute lines)
         Zero or more media descriptions

    时间信息描述:
         t=  (time the session is active)
         r=* (zero or more repeat times)

    多媒体信息描述(如果有的话):
         m=  (media name and transport address)
         i=* (media title)
         c=* (connection information -- optional if included at
              session level)
         b=* (zero or more bandwidth information lines)
         k=* (encryption key)
         a=* (zero or more media attribute lines)
```
所有元素的type都为小写,并且不提供拓展.但是我们可以用a(attribute)字段来提供额外的信息.一个SDP描述的例子如下:

```
      v=0
      o=jdoe 2890844526 2890842807 IN IP4 10.47.16.5
      s=SDP Seminar
      i=A Seminar on the session description protocol
      u=http://www.example.com/seminars/sdp.pdf
      e=j.doe@example.com (Jane Doe)
      c=IN IP4 224.2.17.12/127
      t=2873397496 2873404696
      a=recvonly
      m=audio 49170 RTP/AVP 0
      m=video 51372 RTP/AVP 99
      a=rtpmap:99 h263-1998/90000
```
具体字段的type/value描述和格式可以去参考[RFC4566](http://www.rfc-editor.org/info/rfc4566).

## Offer/Answer模型

上文说到,SDP用来描述多播主干网络的会话信息,但是并没有具体的交互操作细节是如何实现的,因此[RFC3264](http://www.rfc-editor.org/info/rfc3264)定义了一种基于SDP的offer/answer模型.在该模型中,会话参与者的其中一方生成一个SDP报文构成offer,其中包含了一组offerer希望使用的多媒体流和编解码方法,以及offerer用来接收改数据的IP地址和端口信息. offer传输到会话的另一端(称为answerer),由answerer生成一个answer,即用来响应对应offer的SDP报文. answer中包含不同offer对应的多媒体流,并指明该流是否可以接受.


RFC3264只介绍了交换数据过程,而没有定义传递offer/answer报文的方法,后者在[RFC3261/SIP](http://www.rfc-editor.org/info/rfc3261)即会话初始化协议中描述.值得一提的是,offer/answer模型也经常被SIP作为一种基本方法使用. offer/answer模型在SDP报文的基础上进行了一些定义,工作过程不在此描述,需要了解细节的朋友可以参考RFC3261.

## ICE

ICE的全称为**Interactive Connectivity Establishment**,即交互式连接建立.初学者可能会将其与网络编程的ICE弄混,其实那是不一样的东西,在网络编程中,如C++的ICE库,都是指Internate Communications Engine, 是一种用于分布式程序设计的网络通信中间件.我们这里说的只是交互式连接建立.


ICE是一个用于在[offer/answer](http://www.rfc-editor.org/info/rfc3264)模式下的NAT传输协议,主要用于UDP下多媒体会话的建立,其使用了STUN协议以及TURN协议,同时也能被其他实现了offer/answer模型的的其他程序所使用,比如[SIP](http://www.rfc-editor.org/info/rfc3261)(Session Initiation Protocol).


使用offer/answer模型(RFC3264)的协议通常很难在NAT之间穿透,因为其目的一般是建立多媒体数据流,而且在报文中还携带了数据的源IP和端口信息,这在通过NAT时是有问题的.RFC3264还尝试在客户端之间建立直接的通路,因此中间就缺少了应用层的封装.这样设计是为了减少媒体数据延迟,减少丢包率以及减少程序部署的负担.然而这一切都很难通过NAT而完成. 有很多解决方案可以使得这些协议运行于NAT环境之中,包括`应用层网关(ALGs)`,`Classic STUN`以及`Realm Specific IP`+`SDP` 协同工作等方法.不幸的是,这些技术都是在某些网络拓扑下工作很好,而在另一些环境下表现又很差,因此我们需要一个单一的, 可自由定制的解决方案,以便能在所有环境中都能较好工作.

### ICE工作流程

一个典型的ICE工作环境如下,有两个端点L和R,都运行在各自的NAT之后(他们自己也许并不知道),NAT的类型和性质也是未知的. L和R通过交换[SDP](http://www.rfc-editor.org/info/rfc4566)信息在彼此之间建立多媒体会话,通常交换通过一个SIP服务器完成:

```
                     +-----------+
                     |    SIP    |
    +-------+        |    Srvr   |         +-------+
    | STUN  |        |           |         | STUN  |
    | Srvr  |        +-----------+         | Srvr  |
    |       |        /           \         |       |
    +-------+       /             \        +-------+
                   /<- Signaling ->\
                  /                 \
             +--------+          +--------+
             |  NAT   |          |  NAT   |
             +--------+          +--------+
               /                       \
              /                         \
             /                           \
         +-------+                    +-------+
         | Agent |                    | Agent |
         |   L   |                    |   R   |
         |       |                    |       |
         +-------+                    +-------+
```
ICE的基本思路是,每个终端都有一系列`传输地址`(包括传输协议,IP地址和端口)的候选,可以用来和其他端点进行通信.其中可能包括:


* 直接和网络接口联系的传输地址(host address)
* 经过NAT转换的传输地址,即反射地址(server reflective address)
* TURN服务器分配的中继地址(relay address)


虽然潜在要求任意一个L的候选地址都能用来和R的候选地址进行通信.但是实际中发现有许多组合是无法工作的.举例来说,如果L和R都在NAT之后而且不处于同一内网,他们的直接地址就无法进行通信.ICE的目的就是为了发现哪一对候选地址的组合可以工作,并且通过系统的方法对所有组合进行测试(用一种精心挑选的顺序).


为了执行ICE,客户端必须要识别出其所有的地址候选,ICE中定义了三种候选类型,有些是从物理地址或者逻辑网络接口继承而来,其他则是从STUN或者TURN服务器发现的.很自然,一个可用的地址为和本地网络接口直接联系的地址,通常是内网地址, 称为`HOST CANDIDATE`,如果客户端有多个网络接口,比如既连接了WiFi又插着网线,那么就可能有多个内网地址候选.


其次,客户端通过STUN或者TURN来获得更多的候选传输地址,即`SERVER REFLEXIVE CANDIDATES`和`RELAYED CANDIDATES`, 如果TURN服务器是标准化的,那么两种地址都可以通过TURN服务器获得.当L获得所有的自己的候选地址之后,会将其按优先级排序,然后通过signaling通道发送到R.候选地址被存储在SDP offer报文的属性部分.当R接收到offer之后, 就会进行同样的获选地址收集过程,并返回给L.


这一步骤之后,两个对等端都拥有了若干自己和对方的候选地址,并将其配对,组成`CANDIDATE PAIRS`.为了查看哪对组合可以工作,每个终端都进行一系列的检查.每个检查都是一次STUN request/response传输,将request从候选地址对的本地地址发送到远端地址. 连接性检查的基本原则很简单:

1. 以一定的优先级将候选地址对进行排序.
2. 以该优先级顺序发送checks请求
3. 从其他终端接收到checks的确认信息

两端连接性测试,结果是一个4次握手过程:

```
     L                        R
     -                        -
     STUN request ->             \  L's
               <- STUN response  /  check
    
                <- STUN request  \  R's
     STUN response ->            /  check
```
地址都是接下来进多媒体传输(如RTP和RTCP)的地址和端口,所以,
客户端实际上是将STUN协议与RTP/RTCP协议在数据包中进行复用(而不是在端口上复用).</p>

由于STUN Binding request用来进行连接性测试,因此STUN Binding response中会包含终端的实际地址,
如果这个地址和之前学习的所有地址都不匹配,发送方就会生成一个新的candidate,称为`PEER REFLEXIVE CANDIDATE`,和其他candidate一样,也要通过ICE的检查测试.

### 连接性检查(Connectivity Checks)

所有的ICE实现都要求与STUN(RFC5389)兼容,并且废弃Classic STUN(RFC3489).ICE的完整实现既生成checks(作为STUN client),
也接收checks(作为STUN server),而lite实现则只负责接收checks.这里只介绍完整实现情况下的检查过程.

**1、为中继候选地址生成许可(Permissions).**

**2、从本地候选往远端候选发送Binding Request.**

在Binding请求中通常需要包含一些特殊的属性,以在ICE进行连接性检查的时候提供必要信息.


PRIORITY 和 USE-CANDIDATE:


终端必须在其request中包含PRIORITY属性,指明其优先级,优先级由公式计算而得. 如果有需要也可以给出特别指定的候选(即USE-CANDIDATE属性).


ICE-CONTROLLED和ICE-CONTROLLING:


在每次会话中,每个终端都有一个身份,有两种身份,即受控方(controlled role)和主控方(controlling role). 主控方负责选择最终用来通讯的候选地址对,受控方被告知哪个候选地址对用来进行哪次媒体流传输, 并且不生成更新过的offer来提示此次告知.发起ICE处理进程(即生成offer)的一方必须是主控方,而另一方则是受控方. 如果终端是受控方,那么在request中就必须加上ICE-CONTROLLED属性,同样,如果终端是主控方,就需要ICE-CONTROLLING属性.


生成Credential:


作为连接性检查的Binding Request必须使用STUN的短期身份验证.验证的用户名被格式化为一系列username段的联结,包含了发送请求的所有对等端的用户名,以冒号隔开;密码就是对等端的密码.

**3、处理Response.**

当收到Binding Response时,终端会将其与Binding Request相联系,通常通过事务ID.随后将会将此事务ID与
候选地址对进行绑定.

* 失败响应: 如果STUN传输返回487(Role Conflict)错误响应,终端首先会检查其是否包含了ICE-CONTROLLED或ICE-CONTROLLING属性.如果有ICE-CONTROLLED,终端必须切换为controlling role;如果请求包含ICE-CONTROLLING属性, 则必须切换为controlled role.切换好之后,终端必须使产生487错误的候选地址对进入检查队列中, 并将此地址对的状态设置为`Waiting`.
* 成功响应: 一次连接检查在满足下列所有情况时候就被认为成功: 
  * STUN传输产生一个Success Response
  * response的源IP和端口等于Binding Request的目的IP和端口
  * response的目的IP和端口等于Binding Request的源IP和端口


终端收到成功响应之后,先检查其mapped address是否与本地记录的地址对有匹配,如果没有则生成一个新的候选地址. 即对等端的反射地址.如果有匹配,则终端会构造一个可用候选地址对(valid pair).通常很可能地址对不存在于任何检查列表中,检索检查列表中没有被服务器反射的本地地址,这些地址把它们的本地候选转换成服务器反射地址的基地址, 并把冗余的地址去除掉.

## 后记

本文介绍了一种完整的NAT环境通信解决方案ICE,并且对其中涉及到的概念SDP和offer/answer模型也作了简要介绍. ICE是使用STUN/TURN工具性质的最主要协议之一,其中TURN一开始也被设计为ICE协议的一部分.值的一提的是, 本文只是对这几种协议作了概述性的说明,而具体工作过程和详细的属性描述都未包含,因此如果需要根据协议来实现具体的应用程序,还需要对RFC的文档进行仔细阅读.这里给出一些参考:


* [stun](http://www.rfc-editor.org/info/rfc5389)
* [turn](http://www.rfc-editor.org/info/rfc5766)
* [ice](http://www.rfc-editor.org/info/rfc5245)
* [sdp](http://www.rfc-editor.org/info/rfc4566)
* [sip](http://www.rfc-editor.org/info/rfc3261)


而具体的代码以及实现可以参考:


* [TurnServer](https://github.com/evilpan/TurnServer)
* [pjsip](http://www.pjsip.org)
