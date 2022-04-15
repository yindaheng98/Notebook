# ION的基本架构

服务名称|功能
-|-
ISLB|节点发现与负载均衡
Signal|传递信令
Room|主要业务逻辑，包括聊天室、用户验证等
SFU|WebRTC SFU，用于视频流的转发

用到的独立软件|功能
-|-
Redis|缓存数据库。给ISLB存储节点列表等数据；给Room存储业务逻辑数据
nats|分布式消息队列。用于各服务间的消息传递

>Software applications and services need to exchange data. NATS is an infrastructure that allows such data exchange, segmented in the form of messages. We call this a "message oriented middleware".

![](./i/ION-Arch.png)

在连接建立过程中，各组件之间的关系如下：
```
SDK<---grpc--->signal<---nats--->Room/ISLB<---nats--->signal<---grpc--->SDK
                                    ||
                                  [Redis]
```

在视频流传输中，各组件之间的关系如下：
```
SDK<----webrtc---->SFU<----webrtc---->SDK
```

## 启动

### SFU

在官网教程中，单独启动SFU的指令如下：

```sh
docker pull nats
docker run -p 4222:4222 -p 6222:6222 -p 8222:8222 nats
docker run -p 5000:5000/udp --network host -v $PWD/configs/sfu.toml:/configs/sfu.toml pionwebrtc/ion:latest-sfu
```

可以看出，SFU需要有nats的存在才可运行。

`$PWD/configs/sfu.toml`似乎是配置文件。找到[`pion/ion-sfu`中配置文件的位置](https://github.com/pion/ion-sfu/blob/68545cc25230220435ee028d5a0af6e768a0a79a/config.toml)，可以看到这文件长这样：
```toml
[sfu]
# Ballast size in MiB, will allocate memory to reduce the GC trigger upto 2x the
# size of ballast. Be aware that the ballast should be less than the half of memory
# available.
ballast = 0
# enable prometheus sfu statistics
withstats = false

[router]
# Limit the remb bandwidth in kbps
# zero means no limits
maxbandwidth = 1500
# max number of video tracks packets the SFU will keep track
maxpackettrack = 500
# Sets the audio level volume threshold.
# Values from [0-127] where 0 is the loudest.
# Audio levels are read from rtp extension header according to:
# https://tools.ietf.org/html/rfc6464
audiolevelthreshold = 40
# Sets the interval in which the SFU will check the audio level
# in [ms]. If the active speaker has changed, the sfu will
# emit an event to clients.
audiolevelinterval=1000
# Sets minimum percentage of events required to fire an audio level
# according to the expected events from the audiolevelinterval,
# calculated as audiolevelinterval/packetization time (20ms for 8kHz)
# Values from [0-100]
audiolevelfilter = 20

[router.simulcast]
# Prefer best quality initially
bestqualityfirst = true
# EXPERIMENTAL enable temporal layer change is currently an experimental feature,
# enable only for testing.
enabletemporallayer = false

[webrtc]
# Single port, portrange will not work if you enable this
# singleport = 5000

# Range of ports that ion accepts WebRTC traffic on
# Format: [min, max]   and max - min >= 100
portrange = [5000, 5200]
# if sfu behind nat, set iceserver
# [[webrtc.iceserver]]
# urls = ["stun:stun.stunprotocol.org:3478"]
# [[webrtc.iceserver]]
# urls = ["turn:turn.awsome.org:3478"]
# username = "awsome"
# credential = "awsome"

# sdp semantics:
# "unified-plan"
# "plan-b"
# "unified-plan-with-fallback"
sdpsemantics = "unified-plan"
# toggle multicast dns support: https://tools.ietf.org/html/draft-mdns-ice-candidates-00
mdns = true

[webrtc.candidates]
# In case you're deploying ion-sfu on a server which is configured with
# a 1:1 NAT (e.g., Amazon EC2), you might want to also specify the public
# address of the machine using the setting below. This will result in
# all host candidates (which normally have a private IP address) to
# be rewritten with the public address provided in the settings. As
# such, use the option with caution and only if you know what you're doing.
# Multiple public IP addresses can be specified as a comma separated list
# if the sfu is deployed in a DMZ between two 1-1 NAT for internal and
# external users.
# nat1to1 = ["1.2.3.4"]
# icelite = true

[webrtc.timeouts]
# The duration in [sec] without network activity before a ICE Agent is considered disconnected
disconnected = 5
# The duration in [sec] without network activity before a ICE Agent is considered failed after disconnected
failed = 25
# How often in [sec] the ICE Agent sends extra traffic if there is no activity, if media is flowing no traffic will be sent
keepalive = 2

[turn]
# Enables embeded turn server
enabled = false
# Sets the realm for turn server
realm = "ion"
# The address the TURN server will listen on.
address = "0.0.0.0:3478"
# Certs path to config tls/dtls
# cert="path/to/cert.pem"
# key="path/to/key.pem"
# Port range that turn relays to SFU
# WARNING: It shouldn't overlap webrtc.portrange
# Format: [min, max]
# portrange = [5201, 5400]
[turn.auth]
# Use an auth secret to generate long-term credentials defined in RFC5389-10.2
# NOTE: This takes precedence over `credentials` if defined.
# secret = "secret"
# Sets the credentials pairs
credentials = "pion=ion,pion2=ion2"

[log]
# 0 - INFO 1 - DEBUG 2 - TRACE
v = 1
```

可以看到设置还是比较丰富的，比如有`maxbandwidth`这种带宽设置、WebRTC监听端口范围设置`portrange`、给内网端口映射环境部署时用的`nat1to1`、内置turn服务器设置`[turn]`等。

但是并没看到有nats相关的配置？这个SFU是怎么知道如何连接nats的？在仔细找找，发现这个`pion/ion-sfu`并不是直接放进容器里，而是在[`github.com/pion/ion/pkg/node/sfu`](https://github.com/pion/ion/tree/65dbd12eaad0f0e0a019b4d8ee80742930bcdc28/pkg/node/sfu)里面做了一层封装，然后在[`cmd/sfu/main.go`](https://github.com/pion/ion/blob/65dbd12eaad0f0e0a019b4d8ee80742930bcdc28/cmd/sfu/main.go)里面调用成为一个可执行文件的。

进一步，进到[pkg/node/sfu/sfu.go](https://github.com/pion/ion/blob/65dbd12eaad0f0e0a019b4d8ee80742930bcdc28/pkg/node/sfu/sfu.go)中，找到最主要的配置项：

```go
// Config for sfu node
type Config struct {
	Global global   `mapstructure:"global"`
	Log    logConf  `mapstructure:"log"`
	Nats   natsConf `mapstructure:"nats"`
	isfu.Config
}
```
可以看到，SFU中的配置项实际上是几个配置项拼起来的，其中就包含`pion/ion-sfu`中配置项`isfu.Config`。

再找到从这个配置项生成的[`pion/ion`中关于SFU的配置文件](https://github.com/pion/ion/blob/65dbd12eaad0f0e0a019b4d8ee80742930bcdc28/configs/sfu.toml)：
```toml
[global]
# data center id
dc = "dc1"

[nats]
url = "nats://127.0.0.1:4222"


[sfu]
# ......以下配置和上面那个配置文件差不多，省略
```

可以看到这就是[`pion/ion-sfu`中配置文件](https://github.com/pion/ion-sfu/blob/68545cc25230220435ee028d5a0af6e768a0a79a/config.toml)加了几个配置项而已。其中就有nats的配置项，比如地址和数据库名称啥的。

具体原理的解析，且看[《ION中的SFU服务》](sfu-in-ion.md)。

### islb

在官网教程中，单独启动islb的指令如下：

```sh
docker pull nats
docker run -p 4222:4222 -p 6222:6222 -p 8222:8222 nats
docker pull redis
docker run -p 6379:6379 redis
docker run --network host -v $PWD/configs/islb.toml:/configs/islb.toml pionwebrtc/ion:latest-islb
```

可以看出，islb不仅需要有nats，还需要有Redis的存在才可运行。毕竟是负载均衡，肯定要存一下有哪些个节点可供负载均衡。

`$PWD/configs/islb.toml`应该就是配置文件。和上面一样，直接找到对应的[配置文件](https://github.com/pion/ion/blob/65dbd12eaad0f0e0a019b4d8ee80742930bcdc28/configs/islb.toml)：
```toml
[global]
# data center id
dc = "dc1"

[log]
level = "info"

[nats]
url = "nats://127.0.0.1:4222"


[redis]
addrs = [":6379"]
password = ""
db = 0
```
未免也过于简单了，完全就只有连数据库的配置，没有一点自己的运行逻辑方面配置？

从代码结构上看，在pion项目中，这个islb没有单独的包，主要的逻辑都在[`github.com/pion/ion/pkg/node/islb`](https://github.com/pion/ion/tree/65dbd12eaad0f0e0a019b4d8ee80742930bcdc28/pkg/node/islb)这个包里面，然后在[cmd/islb/main.go](https://github.com/pion/ion/blob/65dbd12eaad0f0e0a019b4d8ee80742930bcdc28/cmd/islb/main.go)里面做成可执行文件。找代码里的配置项也很好找得到：
```go
// Config for islb node
type Config struct {
	Global  global    `mapstructure:"global"`
	Log     logConf   `mapstructure:"log"`
	Nats    natsConf  `mapstructure:"nats"`
	Redis   db.Config `mapstructure:"redis"`
	CfgFile string
}
```

这里面确实没有islb相关的配置项。这就怪了，这islb是什么运行逻辑，怎么都没配置的？好歹还是个“负载均衡”来着。

这个问题比较复杂，详见[《ION中的islb服务》](ion-islb.md)。总的来说，islb实际上就是一个服务注册中心，并没有所谓的负载均衡功能，存储和查询流信息的功能不知道被移到哪去了。推测这个模块后面应该会改个名字，比如改成“Registry”啥的，更符合它现在的功能。

### Signal

从[《ION中的SFU服务》](sfu-in-ion.md)中我们可以看出，信令服务都在SFU里面写好了，那这个Signal是干嘛用的？

先看启动过程。在官网教程中，单独启动Signal的指令如下：

```sh
docker pull nats
docker run -p 4222:4222 -p 6222:6222 -p 8222:8222 nats
docker run -p 5551:5551/tcp --network host -v $PWD/configs/signal.toml:/configs/signal.toml pionwebrtc/ion:latest-signal
```
可以看出，Signal的启动只需要有nats就行了

这个配置文件是这样：
```toml
[global]
# data center id
dc = "dc1"

[log]
level = "info"
# level = "debug"

[nats]
url = "nats://127.0.0.1:4222"


[signal.grpc]
#listen ip port
host = "0.0.0.0"
port = "5551"
allow_all_origins = true
# cert= "configs/certs/cert.pem"
# key= "configs/certs/key.pem"

[signal.jwt]
enabled = false 
key_type = "HMAC"
key = "1q2dGu5pzikcrECJgW3ADfXX3EsmoD99SYvSVCpDsJrAqxou5tUNbHPvkEFI4bTS"

[signal.svc]
services = ["rtc", "room"]
```
前面都没啥特殊的。

这个`signal.grpc`应该是指明Signal服务的对外接口；`signal.jwt`应该是验证功能；`signal.svc`这是什么，看着像是什么服务名，不知道有什么用？

打开[Signal的主函数](https://github.com/pion/ion/blob/65dbd12eaad0f0e0a019b4d8ee80742930bcdc28/cmd/signal/main.go)看一眼，服务注册之类的代码都和前面介绍的一样，最重要的代码应该就是这段：
```go
	srv := grpc.NewServer(
		grpc.CustomCodec(nrpc.Codec()), // nolint:staticcheck
		grpc.UnknownServiceHandler(nproxy.TransparentLongConnectionHandler(sig.Director)))

	s := util.NewWrapperedGRPCWebServer(util.NewWrapperedServerOptions(
		addr, conf.Signal.GRPC.Cert, conf.Signal.GRPC.Key, true), srv)

	if err := s.Serve(); err != nil {
		log.Panicf("failed to serve: %v", err)
	}
	select {}
```
滴滴🤯！！捕捉到关键词`proxy`！看这样子Signal应该是个GRPC代理，把外面来的标准GRPC请求转换为`nats-grpc`的请求。

进一步解析详见[《ION中的Signal服务》](ion-signal.md)。